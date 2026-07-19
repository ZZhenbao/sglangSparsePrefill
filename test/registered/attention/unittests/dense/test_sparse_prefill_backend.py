import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention import sparse_prefill_backend as sparse_backend_module
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
    expected_dense_fixture_output,
    replace_backend,
    run_dense_fixture_eager,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TestTritonBackend(TorchNativeAttnBackend):
    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.page_size = self.token_to_kv_pool.page_size
        self.dense_forward_calls = 0
        self.sparse_kv_write_calls = 0
        self.sparse_kernel_calls = 0
        self.last_sparse_kernel_kwargs = None

    def init_forward_metadata(self, forward_batch):
        super().init_forward_metadata(forward_batch)
        if forward_batch.forward_mode.is_decode():
            return
        prefix_len = int(forward_batch.extend_prefix_lens[0].item())
        req_pool_idx = int(forward_batch.req_pool_indices[0].item())
        self.forward_metadata = SimpleNamespace(
            kv_indices=self.req_to_token_pool.req_to_token[
                req_pool_idx, :prefix_len
            ].to(torch.int64),
            swa_out_cache_loc=self.swa_out_cache_loc,
            out_cache_loc_full_physical=None,
        )
        self.initialized_forward_metadata = self.forward_metadata

    def forward_extend(self, *args, **kwargs):
        self.dense_forward_calls += 1
        return super().forward_extend(*args, **kwargs)

    def _set_kv_buffer(
        self,
        forward_batch,
        layer,
        loc_info,
        k,
        v,
        k_scale=None,
        v_scale=None,
    ):
        self.sparse_kv_write_calls += 1
        self.token_to_kv_pool.set_kv_buffer(
            layer, loc_info, k, v, k_scale, v_scale
        )

    def exact_sparse_extend_attention_fwd(
        self,
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        selected_kv_slots,
        selected_lens,
        k_scale,
        v_scale,
        sm_scale=None,
        **kwargs,
    ):
        self.sparse_kernel_calls += 1
        self.last_sparse_kernel_kwargs = kwargs
        self.last_selected_kv_slots = selected_kv_slots
        self.last_selected_lens = selected_lens
        for query_idx in range(q_extend.shape[0]):
            slots = selected_kv_slots[
                query_idx, : int(selected_lens[query_idx].item())
            ]
            key = torch.cat(
                (k_buffer.index_select(0, slots) * k_scale, k_extend[: query_idx + 1]),
                dim=0,
            )
            value = torch.cat(
                (
                    v_buffer.index_select(0, slots) * v_scale,
                    v_extend[: query_idx + 1],
                ),
                dim=0,
            )
            output = scaled_dot_product_attention(
                q_extend[query_idx : query_idx + 1].movedim(0, 1).unsqueeze(0),
                key.movedim(0, 1).unsqueeze(0),
                value.movedim(0, 1).unsqueeze(0),
                enable_gqa=q_extend.shape[1] != key.shape[1],
                scale=sm_scale,
            )
            o_extend[query_idx].copy_(output.squeeze(0).squeeze(1))


class TestSparsePrefillBackend(CustomTestCase):
    def _build_fixture(
        self,
        *,
        forward_mode: ForwardMode,
        prefix_lens: tuple[int, ...],
        extend_lens: tuple[int, ...] = (),
        policy: str = "token_h2o",
        ratio: float = 1.0,
        selection_unit_size: int = 1,
    ):
        case = DenseAttentionCase(
            name=f"sparse_prefill_{forward_mode.name.lower()}_{policy}",
            backend="torch_native",
            forward_mode=forward_mode,
            num_heads=4,
            num_kv_heads=2 if forward_mode == ForwardMode.EXTEND else 4,
            page_size=4,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
        )
        fixture = build_dense_attention_fixture(
            self,
            case,
            dtype=torch.float32,
            device="cpu",
        )
        server_args = fixture.runner.server_args
        server_args.attention_backend = "sparse_prefill"
        server_args.sparse_policy = policy
        server_args.sparse_ratio = ratio
        server_args.sparse_sink_tokens = 0
        server_args.sparse_recent_tokens = 0
        server_args.selection_unit_size = selection_unit_size
        get_context().set_server_args(server_args)
        self.assertEqual(get_flags().attn.backend, "sparse_prefill")
        fixture.forward_batch.spec_algorithm = fixture.runner.spec_algorithm
        with mock.patch.object(
            sparse_backend_module,
            "_create_triton_backend",
            side_effect=_TestTritonBackend,
        ):
            backend = ATTENTION_BACKENDS["sparse_prefill"](fixture.runner)
        return replace_backend(fixture, backend)

    def test_full_ratio_routes_exact_sparse_triton_kernel(self):
        for policy, selection_unit_size in (
            ("token_h2o", 1),
            ("fixed_chunk", 3),
        ):
            with self.subTest(policy=policy):
                fixture = self._build_fixture(
                    forward_mode=ForwardMode.EXTEND,
                    prefix_lens=(7,),
                    extend_lens=(3,),
                    policy=policy,
                    selection_unit_size=selection_unit_size,
                )
                req_to_token_before = (
                    fixture.runner.req_to_token_pool.req_to_token.clone()
                )
                expected = expected_dense_fixture_output(fixture)
                selection_plans = []
                materialize_selection_plan = (
                    sparse_backend_module.materialize_selection_plan
                )

                def capture_selection_plan(*args, **kwargs):
                    selection_plan = materialize_selection_plan(*args, **kwargs)
                    selection_plans.append(selection_plan)
                    return selection_plan

                with mock.patch.object(
                    sparse_backend_module,
                    "materialize_selection_plan",
                    side_effect=capture_selection_plan,
                ):
                    actual = run_dense_fixture_eager(fixture)

                self.assertEqual(fixture.backend.dense_backend.dense_forward_calls, 0)
                self.assertEqual(
                    fixture.backend.dense_backend.sparse_kv_write_calls, 1
                )
                self.assertEqual(fixture.backend.dense_backend.sparse_kernel_calls, 1)
                self.assertEqual(len(selection_plans), 1)
                torch.testing.assert_close(
                    selection_plans[0].selected_lens,
                    torch.full((3,), 7, dtype=torch.int32),
                )
                torch.testing.assert_close(
                    selection_plans[0].selected_pos,
                    torch.arange(7, dtype=torch.int32).expand(3, -1),
                )
                torch.testing.assert_close(
                    fixture.backend.dense_backend.last_selected_lens,
                    selection_plans[0].selected_lens,
                )

                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(
                    fixture.runner.req_to_token_pool.req_to_token,
                    req_to_token_before,
                )

                _, expected_key, expected_value = (
                    fixture.actual_module.project_qkv(fixture.input_hidden)
                )
                suffix_slots = fixture.forward_batch.out_cache_loc
                key_cache = fixture.runner.token_to_kv_pool.get_key_buffer(0)
                value_cache = fixture.runner.token_to_kv_pool.get_value_buffer(0)
                torch.testing.assert_close(
                    key_cache.index_select(0, suffix_slots),
                    expected_key.view(3, 2, -1),
                )
                torch.testing.assert_close(
                    value_cache.index_select(0, suffix_slots),
                    expected_value.view(3, 2, -1),
                )

    def test_partial_ratio_routes_exact_sparse_triton_kernel(self):
        fixture = self._build_fixture(
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(7,),
            extend_lens=(3,),
            ratio=0.5,
        )
        req_to_token_before = fixture.runner.req_to_token_pool.req_to_token.clone()
        expected_shape = expected_dense_fixture_output(fixture).shape
        _, expected_key, expected_value = fixture.actual_module.project_qkv(
            fixture.input_hidden
        )

        selection_plans = []
        materialize_selection_plan = sparse_backend_module.materialize_selection_plan

        def capture_selection_plan(*args, **kwargs):
            selection_plan = materialize_selection_plan(*args, **kwargs)
            selection_plans.append(selection_plan)
            return selection_plan

        with mock.patch.object(
            sparse_backend_module,
            "materialize_selection_plan",
            side_effect=capture_selection_plan,
        ):
            actual = run_dense_fixture_eager(fixture)

        self.assertEqual(actual.shape, expected_shape)
        self.assertIs(
            fixture.backend.dense_backend.forward_metadata,
            fixture.backend.dense_backend.initialized_forward_metadata,
        )
        self.assertEqual(fixture.backend.dense_backend.dense_forward_calls, 0)
        self.assertEqual(fixture.backend.dense_backend.sparse_kv_write_calls, 1)
        self.assertEqual(fixture.backend.dense_backend.sparse_kernel_calls, 1)
        self.assertEqual(
            fixture.backend.dense_backend.last_sparse_kernel_kwargs["page_size"], 4
        )
        self.assertEqual(len(selection_plans), 1)
        torch.testing.assert_close(
            fixture.backend.dense_backend.last_selected_lens,
            selection_plans[0].selected_lens,
        )
        torch.testing.assert_close(
            fixture.runner.req_to_token_pool.req_to_token,
            req_to_token_before,
        )
        suffix_slots = fixture.forward_batch.out_cache_loc
        key_cache = fixture.runner.token_to_kv_pool.get_key_buffer(0)
        value_cache = fixture.runner.token_to_kv_pool.get_value_buffer(0)
        torch.testing.assert_close(
            key_cache.index_select(0, suffix_slots),
            expected_key.view(3, 2, -1),
        )
        torch.testing.assert_close(
            value_cache.index_select(0, suffix_slots),
            expected_value.view(3, 2, -1),
        )
        query, _, _ = fixture.actual_module.project_qkv(fixture.input_hidden)
        sparse_view = sparse_backend_module.build_sparse_kv_view(
            selection_plans[0],
            fixture.backend.dense_backend.forward_metadata.kv_indices,
        )
        sparse_call = sparse_backend_module.build_sparse_extend_call(
            selection_plans[0], sparse_view, 3
        )
        sparse_output = sparse_backend_module.sparse_extend_sdpa(
            query.view(3, 4, -1),
            key_cache,
            value_cache,
            suffix_slots,
            sparse_call,
            scaling=fixture.actual_module.attn.scaling,
        )
        expected = fixture.actual_module.o_proj(sparse_output.flatten(1))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_prefix_miss_explicitly_routes_dense(self):
        fixture = self._build_fixture(
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(0,),
            extend_lens=(3,),
        )
        expected = expected_dense_fixture_output(fixture)
        _, expected_key, expected_value = fixture.actual_module.project_qkv(
            fixture.input_hidden
        )

        with (
            mock.patch.object(
                sparse_backend_module, "compute_causal_token_scores"
            ) as probe,
            mock.patch.object(
                sparse_backend_module, "materialize_selection_plan"
            ) as selector,
        ):
            actual = run_dense_fixture_eager(fixture)

        probe.assert_not_called()
        selector.assert_not_called()
        self.assertEqual(fixture.backend.dense_backend.dense_forward_calls, 1)
        self.assertEqual(fixture.backend.dense_backend.sparse_kv_write_calls, 0)
        self.assertEqual(fixture.backend.dense_backend.sparse_kernel_calls, 0)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        prefix_slots = fixture.forward_batch.out_cache_loc
        key_cache = fixture.runner.token_to_kv_pool.get_key_buffer(0)
        value_cache = fixture.runner.token_to_kv_pool.get_value_buffer(0)
        torch.testing.assert_close(
            key_cache.index_select(0, prefix_slots),
            expected_key.view(3, 2, -1),
        )
        torch.testing.assert_close(
            value_cache.index_select(0, prefix_slots),
            expected_value.view(3, 2, -1),
        )

    def test_decode_is_dispatched_by_hybrid_backend(self):
        fixture = self._build_fixture(
            forward_mode=ForwardMode.DECODE,
            prefix_lens=(7,),
        )
        fixture = replace_backend(
            fixture,
            HybridAttnBackend(
                fixture.runner,
                prefill_backend=fixture.backend,
                decode_backend=fixture.backend.dense_backend,
            ),
        )

        expected = expected_dense_fixture_output(fixture)
        actual = run_dense_fixture_eager(fixture)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_non_single_request_extend_fails_fast(self):
        fixture = self._build_fixture(
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(5, 7),
            extend_lens=(2, 1),
            ratio=0.5,
        )

        with self.assertRaisesRegex(AssertionError, "one request"):
            run_dense_fixture_eager(fixture)


if __name__ == "__main__":
    unittest.main()
