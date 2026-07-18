from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

SparsePolicy = Literal["token_h2o", "fixed_chunk"]


@dataclass(frozen=True)
class SparseSelectionPlan:
    """Request-local, layer-local logical selection metadata.

    Padded entries in ``selected_unit_ids``, ``selected_pos``, and
    ``selected_to_union`` use ``-1``. All index and length tensors are int32.
    ``page_union`` contains physical cache-page ids; it does not replace the
    canonical SGLang page table.
    """

    policy: SparsePolicy
    selection_unit_size: int
    selected_unit_ids: torch.Tensor  # [T, max_selected_units]
    selected_unit_lens: torch.Tensor  # [T]
    selected_pos: torch.Tensor  # [T, selected-token capacity]
    selected_lens: torch.Tensor  # [T]
    union_pos: torch.Tensor  # [U], sorted logical prefix positions
    selected_to_union: torch.Tensor  # [T, selected-token capacity]
    page_union: torch.Tensor  # [num_loaded_pages], sorted physical page ids
    selected_token_count: torch.Tensor  # [T], actual token count per query
    loaded_page_count: int  # scalar, equal to page_union.numel()


@dataclass(frozen=True)
class SparseKVView:
    """Physical prefix slots selected by the current layer."""

    union_kv_slots: torch.Tensor  # int64 [U]


@dataclass(frozen=True)
class SparseExtendCall:
    union_kv_slots: torch.Tensor  # int64 [U]
    combined_mask: torch.Tensor  # bool [T, U + T]


def _expand_kv_heads(kv: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_kv_heads = kv.shape[1]
    return kv.repeat_interleave(num_query_heads // num_kv_heads, dim=1)


def _attention_scale(query: torch.Tensor, scaling: float | None) -> float:
    return query.shape[-1] ** -0.5 if scaling is None else scaling


def _query_key_logits(
    query: torch.Tensor,
    key: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Compute FP32 logits in [query_head, query_token, key_token] order."""

    return torch.einsum(
        "thd,shd->hts",
        query.float(),
        key.float(),
    ).mul_(scaling)


def compute_causal_token_scores(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    suffix_key: torch.Tensor,
    *,
    scaling: float | None = None,
) -> torch.Tensor:
    """Run the full-prefix causal probe.

    Args:
        query: RoPE-applied suffix query, shaped [T, Hq, D].
        prefix_key: RoPE-applied cached-prefix key, shaped [P, Hkv, D].
        suffix_key: RoPE-applied suffix key, shaped [T, Hkv, D].
        scaling: QK scale. The default is ``1 / sqrt(D)``.

    Returns:
        Causal cumulative prefix-token scores [T, P]. Each query head has its
        own softmax over the complete prefix and legal suffix keys before
        probability mass is summed over heads.
    """

    num_query_heads = query.shape[1]
    prefix_key = _expand_kv_heads(prefix_key, num_query_heads)
    suffix_key = _expand_kv_heads(suffix_key, num_query_heads)
    scale = _attention_scale(query, scaling)

    prefix_logits = _query_key_logits(query, prefix_key, scale)
    suffix_logits = _query_key_logits(query, suffix_key, scale)

    suffix_len = query.shape[0]
    prefix_len = prefix_key.shape[0]
    suffix_mask = torch.ones(
        (suffix_len, suffix_len),
        dtype=torch.bool,
        device=query.device,
    ).tril_()
    attention_mask = torch.cat(
        (
            torch.ones(
                (suffix_len, prefix_len),
                dtype=torch.bool,
                device=query.device,
            ),
            suffix_mask,
        ),
        dim=1,
    )
    probe_logits = torch.cat((prefix_logits, suffix_logits), dim=-1)
    probe_prob = torch.softmax(
        probe_logits.masked_fill(~attention_mask.unsqueeze(0), -torch.inf),
        dim=-1,
    )
    return probe_prob[:, :, :prefix_len].sum(dim=0).cumsum(dim=0)


def _mandatory_token_ids(
    prefix_len: int,
    query_index: int,
    sink_tokens: int,
    recent_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    sink_end = min(prefix_len, sink_tokens)
    recent_prefix_len = min(prefix_len, max(0, recent_tokens - query_index - 1))
    mandatory = torch.cat(
        (
            torch.arange(sink_end, device=device),
            torch.arange(
                prefix_len - recent_prefix_len,
                prefix_len,
                device=device,
            ),
        )
    )
    return torch.unique(mandatory, sorted=True)


def _rank_by_score(score: torch.Tensor, unit_ids: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(
        score.index_select(0, unit_ids),
        descending=True,
        stable=True,
    )
    return unit_ids.index_select(0, order)


def _pad_index_rows(rows: list[torch.Tensor]) -> torch.Tensor:
    width = max(row.numel() for row in rows)
    padded = torch.full(
        (len(rows), width),
        -1,
        dtype=torch.int32,
        device=rows[0].device,
    )
    for row_index, row in enumerate(rows):
        padded[row_index, : row.numel()] = row.to(torch.int32)
    return padded


def _select_token_units(
    token_scores: torch.Tensor,
    target_token_count: int,
    sink_tokens: int,
    recent_tokens: int,
) -> list[torch.Tensor]:
    suffix_len, prefix_len = token_scores.shape
    all_token_ids = torch.arange(prefix_len, device=token_scores.device)
    selected_rows = []

    for query_index in range(suffix_len):
        if target_token_count == prefix_len:
            selected_rows.append(all_token_ids)
            continue

        mandatory_ids = _mandatory_token_ids(
            prefix_len,
            query_index,
            sink_tokens,
            recent_tokens,
            token_scores.device,
        )
        candidate_mask = torch.ones(
            prefix_len,
            dtype=torch.bool,
            device=token_scores.device,
        )
        candidate_mask[mandatory_ids] = False
        candidate_ids = all_token_ids[candidate_mask]
        ranked_ids = _rank_by_score(token_scores[query_index], candidate_ids)
        remaining = max(0, target_token_count - mandatory_ids.numel())
        selected_ids = torch.cat((mandatory_ids, ranked_ids[:remaining])).sort().values
        selected_rows.append(selected_ids)
    return selected_rows


def _select_chunk_units(
    token_scores: torch.Tensor,
    target_token_count: int,
    sink_tokens: int,
    recent_tokens: int,
    selection_unit_size: int,
) -> list[torch.Tensor]:
    suffix_len, prefix_len = token_scores.shape
    num_units = math.ceil(prefix_len / selection_unit_size)
    all_unit_ids = torch.arange(num_units, device=token_scores.device)
    unit_sizes = torch.clamp(
        prefix_len - all_unit_ids * selection_unit_size,
        max=selection_unit_size,
    )
    selected_rows = []

    for query_index in range(suffix_len):
        if target_token_count == prefix_len:
            selected_rows.append(all_unit_ids)
            continue

        mandatory_token_ids = _mandatory_token_ids(
            prefix_len,
            query_index,
            sink_tokens,
            recent_tokens,
            token_scores.device,
        )
        mandatory_unit_ids = torch.div(
            mandatory_token_ids,
            selection_unit_size,
            rounding_mode="floor",
        ).unique(sorted=True)
        candidate_mask = torch.ones(
            num_units,
            dtype=torch.bool,
            device=token_scores.device,
        )
        candidate_mask[mandatory_unit_ids] = False
        candidate_ids = all_unit_ids[candidate_mask]
        unit_scores = torch.stack(
            [
                token_scores[
                    query_index,
                    unit_id
                    * selection_unit_size : min(
                        (unit_id + 1) * selection_unit_size,
                        prefix_len,
                    ),
                ].sum()
                for unit_id in range(num_units)
            ]
        )
        ranked_ids = _rank_by_score(unit_scores, candidate_ids)

        mandatory_token_count = int(
            unit_sizes.index_select(0, mandatory_unit_ids).sum().item()
        )
        remaining = max(0, target_token_count - mandatory_token_count)
        if remaining > 0:
            ranked_token_count = unit_sizes.index_select(0, ranked_ids).cumsum(0)
            additional_unit_count = (
                int((ranked_token_count < remaining).sum().item()) + 1
            )
            ranked_ids = ranked_ids[:additional_unit_count]
        else:
            ranked_ids = ranked_ids[:0]
        selected_rows.append(
            torch.cat((mandatory_unit_ids, ranked_ids)).sort().values
        )
    return selected_rows


def map_selected_units(
    *,
    policy: SparsePolicy,
    selection_unit_size: int,
    selected_unit_ids: torch.Tensor,
    selected_unit_lens: torch.Tensor,
    prefix_len: int,
    prefix_kv_slots: torch.Tensor,
    cache_page_size: int,
) -> SparseSelectionPlan:
    """Expand logical units and map their token union to physical cache pages."""

    device = selected_unit_ids.device
    token_ids = torch.arange(prefix_len, device=device)
    unit_offsets = torch.arange(selection_unit_size, device=device)
    selected_pos = (
        selected_unit_ids.to(torch.int64).unsqueeze(-1) * selection_unit_size
        + unit_offsets
    ).flatten(1)
    selected_pos = selected_pos[:, : min(prefix_len, selected_pos.shape[1])]
    selected = (selected_pos >= 0) & (selected_pos < prefix_len)
    selected_pos = selected_pos.masked_fill(~selected, -1).to(torch.int32)
    selected_lens = selected.sum(dim=1, dtype=torch.int32)

    union_counts = torch.zeros(prefix_len, dtype=torch.int32, device=device)
    union_counts.scatter_add_(
        0,
        selected_pos.clamp_min(0).flatten().to(torch.int64),
        selected.flatten().to(torch.int32),
    )
    union_mask = union_counts > 0
    union_pos = token_ids[union_mask].to(torch.int32)
    prefix_to_union = union_mask.cumsum(dim=0, dtype=torch.int32) - 1
    selected_to_union = prefix_to_union.index_select(
        0, selected_pos.clamp_min(0).flatten().to(torch.int64)
    ).view_as(selected_pos)
    selected_to_union.masked_fill_(~selected, -1)

    union_kv_slots = prefix_kv_slots.index_select(0, union_pos.to(torch.int64))
    page_union = torch.unique(
        torch.div(union_kv_slots, cache_page_size, rounding_mode="floor"),
        sorted=True,
    ).to(torch.int32)
    return SparseSelectionPlan(
        policy=policy,
        selection_unit_size=selection_unit_size,
        selected_unit_ids=selected_unit_ids.to(torch.int32),
        selected_unit_lens=selected_unit_lens.to(torch.int32),
        selected_pos=selected_pos,
        selected_lens=selected_lens,
        union_pos=union_pos,
        selected_to_union=selected_to_union,
        page_union=page_union,
        selected_token_count=selected_lens.clone(),
        loaded_page_count=page_union.numel(),
    )


def materialize_selection_plan(
    token_scores: torch.Tensor,
    prefix_kv_slots: torch.Tensor,
    *,
    policy: SparsePolicy,
    ratio: float,
    sink_tokens: int,
    recent_tokens: int,
    selection_unit_size: int,
    cache_page_size: int,
) -> SparseSelectionPlan:
    """Select per-query logical units and materialize the stable P1 contract."""

    _, prefix_len = token_scores.shape
    target_token_count = math.ceil(ratio * prefix_len)
    if policy == "token_h2o":
        selected_rows = _select_token_units(
            token_scores,
            target_token_count,
            sink_tokens,
            recent_tokens,
        )
    else:
        selected_rows = _select_chunk_units(
            token_scores,
            target_token_count,
            sink_tokens,
            recent_tokens,
            selection_unit_size,
        )

    selected_unit_ids = _pad_index_rows(selected_rows)
    selected_unit_lens = torch.tensor(
        [row.numel() for row in selected_rows],
        dtype=torch.int32,
        device=token_scores.device,
    )
    return map_selected_units(
        policy=policy,
        selection_unit_size=selection_unit_size,
        selected_unit_ids=selected_unit_ids,
        selected_unit_lens=selected_unit_lens,
        prefix_len=prefix_len,
        prefix_kv_slots=prefix_kv_slots,
        cache_page_size=cache_page_size,
    )


def build_sparse_kv_view(
    selection_plan: SparseSelectionPlan,
    prefix_kv_slots: torch.Tensor,
) -> SparseKVView:
    """Map the logical prefix-token union to physical KV slots."""

    union_kv_slots = prefix_kv_slots.index_select(
        0, selection_plan.union_pos.to(torch.int64)
    ).to(torch.int64)
    return SparseKVView(union_kv_slots=union_kv_slots)


def build_sparse_extend_call(
    selection_plan: SparseSelectionPlan,
    sparse_kv_view: SparseKVView,
    suffix_len: int,
) -> SparseExtendCall:
    selected_to_union = selection_plan.selected_to_union.to(torch.int64)
    selected = selected_to_union >= 0
    prefix_mask_counts = torch.zeros(
        (suffix_len, sparse_kv_view.union_kv_slots.numel()),
        dtype=torch.int32,
        device=selected_to_union.device,
    )
    prefix_mask_counts.scatter_add_(
        1,
        selected_to_union.clamp_min(0),
        selected.to(torch.int32),
    )
    prefix_mask = prefix_mask_counts > 0
    suffix_mask = torch.ones(
        (suffix_len, suffix_len),
        dtype=torch.bool,
        device=selected_to_union.device,
    ).tril_()
    return SparseExtendCall(
        union_kv_slots=sparse_kv_view.union_kv_slots,
        combined_mask=torch.cat((prefix_mask, suffix_mask), dim=1),
    )


def sparse_extend_sdpa(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    suffix_kv_slots: torch.Tensor,
    sparse_call: SparseExtendCall,
    *,
    scaling: float | None = None,
) -> torch.Tensor:
    kv_slots = torch.cat((sparse_call.union_kv_slots, suffix_kv_slots))
    key = key_cache.index_select(0, kv_slots)
    value = value_cache.index_select(0, kv_slots)
    output = scaled_dot_product_attention(
        query.movedim(0, 1).unsqueeze(0),
        key.movedim(0, 1).unsqueeze(0),
        value.movedim(0, 1).unsqueeze(0),
        attn_mask=sparse_call.combined_mask[None, None],
        is_causal=False,
        enable_gqa=query.shape[1] != key.shape[1],
        scale=scaling,
    )
    return output.squeeze(0).movedim(0, 1)


class SparsePrefillBackend(TorchNativeAttnBackend):
    """Use dense attention for cache misses and sparse attention for prefix hits."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        server_args = model_runner.server_args

        self.sparse_policy = server_args.sparse_policy
        self.sparse_ratio = server_args.sparse_ratio
        self.sparse_sink_tokens = server_args.sparse_sink_tokens
        self.sparse_recent_tokens = server_args.sparse_recent_tokens
        self.selection_unit_size = server_args.selection_unit_size
        self.cache_page_size = self.token_to_kv_pool.page_size

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        assert forward_batch.batch_size == 1, (
            "SparsePrefillBackend only supports one request"
        )
        assert save_kv_cache

        prefix_len = int(forward_batch.extend_prefix_lens[0].item())
        suffix_len = int(forward_batch.extend_seq_lens[0].item())
        seq_len = int(forward_batch.seq_lens[0].item())

        if prefix_len == 0:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        req_pool_idx = int(forward_batch.req_pool_indices[0].item())
        kv_slots = self.req_to_token_pool.req_to_token[
            req_pool_idx, :seq_len
        ].to(torch.int64)
        prefix_kv_slots = kv_slots[:prefix_len]
        suffix_kv_slots = kv_slots[prefix_len:]
        assert torch.equal(
            suffix_kv_slots,
            forward_batch.out_cache_loc.to(torch.int64),
        )

        key_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_cache = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        self.token_to_kv_pool.set_kv_buffer(
            layer,
            KVWriteLoc(forward_batch.out_cache_loc, self.swa_out_cache_loc),
            k,
            v,
        )

        query = q.view(suffix_len, layer.tp_q_head_num, layer.qk_head_dim)
        prefix_key = key_cache.index_select(0, prefix_kv_slots)
        suffix_key = key_cache.index_select(0, suffix_kv_slots)

        token_scores = compute_causal_token_scores(
            query,
            prefix_key,
            suffix_key,
            scaling=layer.scaling,
        )
        selection_plan = materialize_selection_plan(
            token_scores,
            prefix_kv_slots,
            policy=self.sparse_policy,
            ratio=self.sparse_ratio,
            sink_tokens=self.sparse_sink_tokens,
            recent_tokens=self.sparse_recent_tokens,
            selection_unit_size=self.selection_unit_size,
            cache_page_size=self.cache_page_size,
        )
        sparse_kv_view = build_sparse_kv_view(
            selection_plan,
            prefix_kv_slots,
        )
        sparse_call = build_sparse_extend_call(
            selection_plan,
            sparse_kv_view,
            suffix_len,
        )
        output = sparse_extend_sdpa(
            query,
            key_cache,
            value_cache,
            suffix_kv_slots,
            sparse_call,
            scaling=layer.scaling,
        )
        return output.reshape(
            suffix_len,
            layer.tp_q_head_num * layer.v_head_dim,
        )
