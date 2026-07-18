import math
import unittest

import torch

from sglang.srt.layers.attention.sparse_prefill_backend import (
    build_sparse_extend_call,
    build_sparse_kv_view,
    compute_causal_token_scores,
    materialize_selection_plan,
    sparse_extend_sdpa,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _unpadded_rows(values: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    return [
        values[row, : int(lengths[row])].tolist() for row in range(values.shape[0])
    ]


def _manual_causal_token_scores(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    suffix_key: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    suffix_len, num_query_heads, _ = query.shape
    prefix_len, num_kv_heads, _ = prefix_key.shape
    query_heads_per_kv_head = num_query_heads // num_kv_heads
    token_scores = torch.empty((suffix_len, prefix_len), dtype=torch.float32)
    cumulative_score = torch.zeros(prefix_len, dtype=torch.float32)

    for query_index in range(suffix_len):
        current_mass = torch.zeros(prefix_len, dtype=torch.float32)
        for query_head in range(num_query_heads):
            kv_head = query_head // query_heads_per_kv_head
            current_query = query[query_index, query_head].float()
            current_prefix_logits = torch.mv(
                prefix_key[:, kv_head].float(), current_query
            ).mul(scaling)
            current_suffix_logits = torch.mv(
                suffix_key[: query_index + 1, kv_head].float(), current_query
            ).mul(scaling)
            probability = torch.softmax(
                torch.cat((current_prefix_logits, current_suffix_logits)), dim=0
            )
            current_mass.add_(probability[:prefix_len])
        cumulative_score = cumulative_score + current_mass
        token_scores[query_index] = cumulative_score

    return token_scores


def _manual_dense_attention(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
) -> torch.Tensor:
    suffix_len, num_query_heads, head_dim = query.shape
    num_kv_heads = prefix_key.shape[1]
    query_heads_per_kv_head = num_query_heads // num_kv_heads
    scaling = head_dim**-0.5
    output = torch.empty(
        (suffix_len, num_query_heads, prefix_value.shape[-1]), dtype=torch.float32
    )

    for query_index in range(suffix_len):
        for query_head in range(num_query_heads):
            kv_head = query_head // query_heads_per_kv_head
            current_query = query[query_index, query_head].float()
            prefix_logits = torch.mv(
                prefix_key[:, kv_head].float(), current_query
            ).mul(scaling)
            suffix_logits = torch.mv(
                suffix_key[: query_index + 1, kv_head].float(), current_query
            ).mul(scaling)
            value = torch.cat(
                (
                    prefix_value[:, kv_head].float(),
                    suffix_value[: query_index + 1, kv_head].float(),
                )
            )
            probability = torch.softmax(
                torch.cat((prefix_logits, suffix_logits)), dim=0
            )
            output[query_index, query_head] = probability @ value

    return output


class TestCausalProbeOracle(CustomTestCase):
    def test_probe_uses_one_causal_softmax_domain_with_gqa(self):
        query = torch.tensor(
            [
                [[1.0, 0.5], [0.5, -1.0], [1.5, 0.0], [-0.5, 1.0]],
                [[0.0, 1.0], [1.0, 1.0], [-1.0, 0.5], [0.5, 0.5]],
                [[1.0, -0.5], [-0.5, -1.0], [0.5, 1.5], [1.0, 1.0]],
            ]
        )
        prefix_key = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 1.0], [1.0, -0.5]],
                [[-1.0, 0.5], [0.5, 0.5]],
            ]
        )
        suffix_key = torch.tensor(
            [
                [[0.25, 0.75], [0.5, -1.0]],
                [[-0.5, 1.0], [1.0, 0.25]],
                [[2.0, -1.0], [-0.75, 0.5]],
            ]
        )
        scaling = 0.7

        actual_scores = compute_causal_token_scores(
            query, prefix_key, suffix_key, scaling=scaling
        )
        expected_scores = _manual_causal_token_scores(
            query, prefix_key, suffix_key, scaling
        )

        torch.testing.assert_close(actual_scores, expected_scores)
        self.assertLess(actual_scores[0].sum().item(), query.shape[1])

    def test_future_suffix_does_not_change_earlier_selection_or_output(self):
        torch.manual_seed(17)
        query = torch.randn(4, 4, 3)
        prefix_key = torch.randn(6, 2, 3)
        prefix_value = torch.randn(6, 2, 2)
        suffix_key = torch.randn(4, 2, 3)
        suffix_value = torch.randn(4, 2, 2)
        prefix_kv_slots = torch.arange(6)
        suffix_kv_slots = torch.arange(6, 10)

        changed_query = query.clone()
        changed_suffix_key = suffix_key.clone()
        changed_suffix_value = suffix_value.clone()
        changed_query[2:] = torch.tensor(25.0) * torch.randn_like(changed_query[2:])
        changed_suffix_key[2:] = torch.tensor(25.0) * torch.randn_like(
            changed_suffix_key[2:]
        )
        changed_suffix_value[2:] = torch.tensor(25.0) * torch.randn_like(
            changed_suffix_value[2:]
        )

        token_scores = compute_causal_token_scores(query, prefix_key, suffix_key)
        changed_scores = compute_causal_token_scores(
            changed_query, prefix_key, changed_suffix_key
        )
        plan = materialize_selection_plan(
            token_scores,
            prefix_kv_slots,
            policy="token_h2o",
            ratio=0.5,
            sink_tokens=0,
            recent_tokens=0,
            selection_unit_size=1,
            cache_page_size=4,
        )
        changed_plan = materialize_selection_plan(
            changed_scores,
            prefix_kv_slots,
            policy="token_h2o",
            ratio=0.5,
            sink_tokens=0,
            recent_tokens=0,
            selection_unit_size=1,
            cache_page_size=4,
        )
        view = build_sparse_kv_view(plan, prefix_kv_slots)
        changed_view = build_sparse_kv_view(changed_plan, prefix_kv_slots)
        call = build_sparse_extend_call(plan, view, query.shape[0])
        changed_call = build_sparse_extend_call(
            changed_plan, changed_view, changed_query.shape[0]
        )
        output = sparse_extend_sdpa(
            query,
            torch.cat((prefix_key, suffix_key)),
            torch.cat((prefix_value, suffix_value)),
            suffix_kv_slots,
            call,
        )
        changed_output = sparse_extend_sdpa(
            changed_query,
            torch.cat((prefix_key, changed_suffix_key)),
            torch.cat((prefix_value, changed_suffix_value)),
            suffix_kv_slots,
            changed_call,
        )

        torch.testing.assert_close(token_scores[:2], changed_scores[:2])
        torch.testing.assert_close(plan.selected_pos[:2], changed_plan.selected_pos[:2])
        torch.testing.assert_close(output[:2], changed_output[:2])


class TestSparseSelectionOracle(CustomTestCase):
    def test_token_h2o_ties_sink_and_recent_window(self):
        token_scores = torch.ones(3, 6)
        plan = materialize_selection_plan(
            token_scores,
            torch.arange(6),
            policy="token_h2o",
            ratio=0.5,
            sink_tokens=1,
            recent_tokens=3,
            selection_unit_size=1,
            cache_page_size=4,
        )

        self.assertEqual(
            _unpadded_rows(plan.selected_unit_ids, plan.selected_unit_lens),
            [[0, 4, 5], [0, 1, 5], [0, 1, 2]],
        )
        self.assertEqual(
            _unpadded_rows(plan.selected_pos, plan.selected_lens),
            [[0, 4, 5], [0, 1, 5], [0, 1, 2]],
        )

    def test_fixed_chunk_uses_prefix_boundaries_and_partial_tail(self):
        token_scores = torch.tensor([[3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 10.0]])
        plan = materialize_selection_plan(
            token_scores,
            torch.arange(7),
            policy="fixed_chunk",
            ratio=0.5,
            sink_tokens=0,
            recent_tokens=0,
            selection_unit_size=3,
            cache_page_size=4,
        )

        self.assertEqual(
            _unpadded_rows(plan.selected_unit_ids, plan.selected_unit_lens), [[0, 2]]
        )
        self.assertEqual(
            _unpadded_rows(plan.selected_pos, plan.selected_lens), [[0, 1, 2, 6]]
        )
        self.assertEqual(plan.selected_token_count.tolist(), [4])

    def test_fixed_chunk_mandatory_tokens_expand_to_complete_chunks(self):
        token_scores = torch.zeros(2, 8)
        prefix_kv_slots = torch.arange(8)
        plan = materialize_selection_plan(
            token_scores,
            prefix_kv_slots,
            policy="fixed_chunk",
            ratio=0.125,
            sink_tokens=1,
            recent_tokens=2,
            selection_unit_size=3,
            cache_page_size=4,
        )

        self.assertEqual(
            _unpadded_rows(plan.selected_unit_ids, plan.selected_unit_lens),
            [[0, 2], [0]],
        )
        self.assertEqual(
            _unpadded_rows(plan.selected_pos, plan.selected_lens),
            [[0, 1, 2, 6, 7], [0, 1, 2]],
        )
        self.assertEqual(plan.selected_token_count.tolist(), [5, 3])
        view = build_sparse_kv_view(plan, prefix_kv_slots)
        call = build_sparse_extend_call(plan, view, 2)
        torch.testing.assert_close(
            call.combined_mask,
            torch.tensor(
                [
                    [True, True, True, True, True, True, False],
                    [True, True, True, False, False, True, True],
                ]
            ),
        )

    def test_fixed_chunk_smaller_prefix_is_one_indivisible_tail_chunk(self):
        plan = materialize_selection_plan(
            torch.tensor([[2.0, 1.0]]),
            torch.arange(2),
            policy="fixed_chunk",
            ratio=0.5,
            sink_tokens=0,
            recent_tokens=0,
            selection_unit_size=4,
            cache_page_size=2,
        )

        self.assertEqual(plan.selected_unit_lens.tolist(), [1])
        self.assertEqual(
            _unpadded_rows(plan.selected_pos, plan.selected_lens), [[0, 1]]
        )

    def test_fixed_chunk_size_one_matches_token_h2o(self):
        token_scores = torch.tensor(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0],
                [1.0, 3.0, 5.0, 2.0, 4.0],
                [2.0, 2.0, 1.0, 4.0, 3.0],
            ]
        )
        common = dict(
            token_scores=token_scores,
            prefix_kv_slots=torch.tensor([7, 8, 15, 20, 31]),
            ratio=0.4,
            sink_tokens=1,
            recent_tokens=2,
            selection_unit_size=1,
            cache_page_size=4,
        )
        token_plan = materialize_selection_plan(policy="token_h2o", **common)
        chunk_plan = materialize_selection_plan(policy="fixed_chunk", **common)

        for field_name in (
            "selected_unit_ids",
            "selected_unit_lens",
            "selected_pos",
            "selected_lens",
            "union_pos",
            "selected_to_union",
            "page_union",
            "selected_token_count",
        ):
            torch.testing.assert_close(
                getattr(token_plan, field_name), getattr(chunk_plan, field_name)
            )

    def test_logical_chunks_are_independent_of_physical_page_layout(self):
        token_scores = torch.tensor([[3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 10.0]])
        first_slots = torch.tensor([5, 6, 7, 8, 9, 10, 11])
        second_slots = torch.tensor([40, 1, 18, 7, 33, 12, 29])
        expected_positions = torch.tensor([[0, 1, 2, 6]], dtype=torch.int32)

        reference_plan = None
        for page_size in (2, 3, 4):
            with self.subTest(page_size=page_size):
                plan = materialize_selection_plan(
                    token_scores,
                    first_slots,
                    policy="fixed_chunk",
                    ratio=0.5,
                    sink_tokens=0,
                    recent_tokens=0,
                    selection_unit_size=3,
                    cache_page_size=page_size,
                )
                self.assertEqual(
                    _unpadded_rows(plan.selected_pos, plan.selected_lens),
                    [expected_positions[0].tolist()],
                )
                expected_pages = torch.unique(
                    torch.div(
                        first_slots[expected_positions[0]],
                        page_size,
                        rounding_mode="floor",
                    ),
                    sorted=True,
                ).to(torch.int32)
                torch.testing.assert_close(plan.page_union, expected_pages)
                if reference_plan is None:
                    reference_plan = plan
                else:
                    torch.testing.assert_close(
                        plan.selected_unit_ids, reference_plan.selected_unit_ids
                    )
                    torch.testing.assert_close(
                        plan.selected_pos, reference_plan.selected_pos
                    )

        relocated_plan = materialize_selection_plan(
            token_scores,
            second_slots,
            policy="fixed_chunk",
            ratio=0.5,
            sink_tokens=0,
            recent_tokens=0,
            selection_unit_size=3,
            cache_page_size=4,
        )
        torch.testing.assert_close(
            relocated_plan.selected_unit_ids, reference_plan.selected_unit_ids
        )
        torch.testing.assert_close(
            relocated_plan.selected_pos, reference_plan.selected_pos
        )
        self.assertEqual(relocated_plan.page_union.tolist(), [0, 4, 7, 10])

    def test_sparse_extend_call_preserves_per_query_mask_within_one_page(self):
        token_scores = torch.tensor(
            [[5.0, 4.0, 1.0, 0.0], [0.0, 1.0, 5.0, 4.0]]
        )
        prefix_kv_slots = torch.tensor([8, 9, 10, 11])
        plan = materialize_selection_plan(
            token_scores,
            prefix_kv_slots,
            policy="token_h2o",
            ratio=0.5,
            sink_tokens=0,
            recent_tokens=0,
            selection_unit_size=1,
            cache_page_size=8,
        )
        view = build_sparse_kv_view(plan, prefix_kv_slots)
        call = build_sparse_extend_call(plan, view, 2)

        torch.testing.assert_close(
            plan.union_pos, torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        )
        torch.testing.assert_close(view.union_kv_slots, prefix_kv_slots)
        torch.testing.assert_close(
            call.combined_mask,
            torch.tensor(
                [
                    [True, True, False, False, True, False],
                    [False, False, True, True, True, True],
                ]
            ),
        )
        self.assertEqual(plan.page_union.tolist(), [1])


class TestReducedDomainAttentionOracle(CustomTestCase):
    def test_full_selection_matches_dense_reference_for_both_policies(self):
        torch.manual_seed(23)
        query = torch.randn(3, 4, 3)
        prefix_key = torch.randn(5, 2, 3)
        prefix_value = torch.randn(5, 2, 4)
        suffix_key = torch.randn(3, 2, 3)
        suffix_value = torch.randn(3, 2, 4)
        token_scores = compute_causal_token_scores(query, prefix_key, suffix_key)
        expected = _manual_dense_attention(
            query, prefix_key, prefix_value, suffix_key, suffix_value
        )
        prefix_kv_slots = torch.arange(5)
        suffix_kv_slots = torch.arange(5, 8)
        key_cache = torch.cat((prefix_key, suffix_key))
        value_cache = torch.cat((prefix_value, suffix_value))

        for policy, selection_unit_size in (("token_h2o", 1), ("fixed_chunk", 2)):
            with self.subTest(policy=policy):
                plan = materialize_selection_plan(
                    token_scores,
                    prefix_kv_slots,
                    policy=policy,
                    ratio=1.0,
                    sink_tokens=0,
                    recent_tokens=0,
                    selection_unit_size=selection_unit_size,
                    cache_page_size=4,
                )
                view = build_sparse_kv_view(plan, prefix_kv_slots)
                call = build_sparse_extend_call(plan, view, query.shape[0])
                actual = sparse_extend_sdpa(
                    query,
                    key_cache,
                    value_cache,
                    suffix_kv_slots,
                    call,
                )

                torch.testing.assert_close(
                    actual, expected, atol=1e-6, rtol=1e-6
                )
                self.assertEqual(
                    _unpadded_rows(plan.selected_pos, plan.selected_lens),
                    [list(range(5))] * query.shape[0],
                )

    def test_unselected_prefix_is_removed_from_softmax_denominator(self):
        query = torch.ones(1, 1, 1)
        prefix_key = torch.tensor([[[math.log(2.0)]], [[math.log(100.0)]]])
        prefix_value = torch.tensor([[[2.0]], [[100.0]]])
        suffix_key = torch.zeros(1, 1, 1)
        suffix_value = torch.tensor([[[4.0]]])
        prefix_kv_slots = torch.tensor([0, 1])
        suffix_kv_slots = torch.tensor([2])
        plan = materialize_selection_plan(
            torch.tensor([[1.0, 0.0]]),
            prefix_kv_slots,
            policy="token_h2o",
            ratio=0.5,
            sink_tokens=0,
            recent_tokens=0,
            selection_unit_size=1,
            cache_page_size=8,
        )
        view = build_sparse_kv_view(plan, prefix_kv_slots)
        call = build_sparse_extend_call(plan, view, 1)

        actual = sparse_extend_sdpa(
            query,
            torch.cat((prefix_key, suffix_key)),
            torch.cat((prefix_value, suffix_value)),
            suffix_kv_slots,
            call,
            scaling=1.0,
        )
        reduced_probability = torch.softmax(
            torch.tensor([math.log(2.0), 0.0]), dim=0
        )
        expected = reduced_probability @ torch.tensor([2.0, 4.0])
        zero_filled = torch.softmax(
            torch.tensor([math.log(2.0), math.log(100.0), 0.0]), dim=0
        ) @ torch.tensor([2.0, 0.0, 4.0])

        torch.testing.assert_close(actual[0, 0, 0], expected)
        self.assertGreater(abs(actual[0, 0, 0].item() - zero_filled.item()), 1.0)
        self.assertEqual(plan.page_union.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
