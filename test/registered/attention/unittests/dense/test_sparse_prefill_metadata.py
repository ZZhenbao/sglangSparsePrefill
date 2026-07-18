import argparse
import dataclasses
import unittest

import torch

from sglang.srt.layers.attention.sparse_prefill_backend import (
    SparseExtendCall,
    SparseKVView,
    SparseSelectionPlan,
    build_sparse_extend_call,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSparsePrefillConfig(CustomTestCase):
    def test_cli_exposes_independent_p1_parameters(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed_args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "sparse_prefill",
                "--sparse-policy",
                "fixed_chunk",
                "--sparse-ratio",
                "0.25",
                "--sparse-sink-tokens",
                "4",
                "--sparse-recent-tokens",
                "32",
                "--selection-unit-size",
                "7",
                "--page-size",
                "16",
            ]
        )
        args = ServerArgs.from_cli_args(parsed_args)

        self.assertEqual(args.attention_backend, "sparse_prefill")
        self.assertEqual(args.sparse_policy, "fixed_chunk")
        self.assertEqual(args.sparse_ratio, 0.25)
        self.assertEqual(args.sparse_sink_tokens, 4)
        self.assertEqual(args.sparse_recent_tokens, 32)
        self.assertEqual(args.selection_unit_size, 7)
        self.assertEqual(args.page_size, 16)

    def test_sparse_parameters_have_no_implicit_defaults(self):
        args = ServerArgs(model_path="dummy")

        self.assertIsNone(args.sparse_policy)
        self.assertIsNone(args.sparse_ratio)
        self.assertIsNone(args.sparse_sink_tokens)
        self.assertIsNone(args.sparse_recent_tokens)
        self.assertIsNone(args.selection_unit_size)

    def test_selection_unit_size_is_not_derived_from_page_size(self):
        args = ServerArgs(
            model_path="dummy",
            sparse_policy="fixed_chunk",
            selection_unit_size=7,
            page_size=16,
        )

        self.assertEqual(args.selection_unit_size, 7)
        self.assertEqual(args.page_size, 16)


class TestSparsePrefillMetadata(CustomTestCase):
    def test_sidecars_keep_logical_and_physical_metadata_separate(self):
        selected_unit_ids = torch.tensor([[0, 2], [0, 3]], dtype=torch.int32)
        selected_unit_lens = torch.tensor([2, 2], dtype=torch.int32)
        selected_pos = torch.tensor([[0, 4], [0, 6]], dtype=torch.int32)
        selected_lens = torch.tensor([2, 2], dtype=torch.int32)
        union_pos = torch.tensor([0, 4, 6], dtype=torch.int32)
        selected_to_union = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32)
        page_union = torch.tensor([0, 1], dtype=torch.int32)
        selected_token_count = torch.tensor([2, 2], dtype=torch.int32)
        union_kv_slots = torch.tensor([17, 23, 41], dtype=torch.int64)

        plan = SparseSelectionPlan(
            policy="token_h2o",
            selection_unit_size=1,
            selected_unit_ids=selected_unit_ids,
            selected_unit_lens=selected_unit_lens,
            selected_pos=selected_pos,
            selected_lens=selected_lens,
            union_pos=union_pos,
            selected_to_union=selected_to_union,
            page_union=page_union,
            selected_token_count=selected_token_count,
            loaded_page_count=2,
        )
        view = SparseKVView(union_kv_slots=union_kv_slots)
        call = build_sparse_extend_call(plan, view, 2)

        self.assertTrue(dataclasses.is_dataclass(plan))
        self.assertTrue(dataclasses.is_dataclass(view))
        self.assertTrue(dataclasses.is_dataclass(call))
        for field in (
            plan.selected_unit_ids,
            plan.selected_unit_lens,
            plan.selected_pos,
            plan.selected_lens,
            plan.union_pos,
            plan.selected_to_union,
            plan.page_union,
            plan.selected_token_count,
        ):
            self.assertEqual(field.dtype, torch.int32)
        self.assertEqual(view.union_kv_slots.dtype, torch.int64)
        self.assertEqual(call.combined_mask.dtype, torch.bool)
        torch.testing.assert_close(
            call.combined_mask,
            torch.tensor(
                [
                    [True, True, False, True, False],
                    [True, False, True, True, True],
                ]
            ),
        )
        self.assertIs(plan.union_pos, union_pos)
        self.assertIs(plan.page_union, page_union)
        self.assertIs(view.union_kv_slots, union_kv_slots)
        self.assertIs(call.union_kv_slots, union_kv_slots)
        self.assertIsInstance(call, SparseExtendCall)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.loaded_page_count = 3

    def test_sparse_sidecars_are_not_canonical_forward_batch_fields(self):
        canonical_fields = {field.name for field in dataclasses.fields(ForwardBatch)}

        self.assertNotIn("sparse_selection_plan", canonical_fields)
        self.assertNotIn("sparse_kv_view", canonical_fields)


if __name__ == "__main__":
    unittest.main()
