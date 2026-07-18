import unittest

import torch

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.sparse_prefill_backend import SparsePrefillBackend
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
        backend = ATTENTION_BACKENDS["sparse_prefill"](fixture.runner)
        return replace_backend(fixture, backend)

    def test_single_request_prefix_hit_matches_dense_at_full_ratio(self):
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
                actual = run_dense_fixture_eager(fixture)

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

    def test_sparse_ratio_uses_the_same_extend_path_and_canonical_cache(self):
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

        actual = run_dense_fixture_eager(fixture)

        self.assertEqual(actual.shape, expected_shape)
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

    def test_zero_prefix_uses_dense_and_writes_prefix_kv(self):
        fixture = self._build_fixture(
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(0,),
            extend_lens=(3,),
        )
        expected = expected_dense_fixture_output(fixture)
        _, expected_key, expected_value = fixture.actual_module.project_qkv(
            fixture.input_hidden
        )

        actual = run_dense_fixture_eager(fixture)

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

    def test_decode_keeps_torch_native_implementation(self):
        self.assertIs(
            SparsePrefillBackend.forward_decode,
            TorchNativeAttnBackend.forward_decode,
        )
        fixture = self._build_fixture(
            forward_mode=ForwardMode.DECODE,
            prefix_lens=(7,),
        )

        expected = expected_dense_fixture_output(fixture)
        actual = run_dense_fixture_eager(fixture)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_non_single_request_extend_fails_fast(self):
        fixture = self._build_fixture(
            forward_mode=ForwardMode.EXTEND,
            prefix_lens=(5, 7),
            extend_lens=(2, 1),
        )

        with self.assertRaisesRegex(AssertionError, "one request"):
            run_dense_fixture_eager(fixture)


if __name__ == "__main__":
    unittest.main()
