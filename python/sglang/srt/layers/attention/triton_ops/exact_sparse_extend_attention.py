"""Exact per-query sparse extend attention for Qwen-style MHA/GQA.

The selector supplies physical prefix slots only. This kernel independently
recomputes QK, online softmax, and AV over those slots plus the causal suffix.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _extract_kv_strides,
)


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _exact_sparse_extend_kernel(
    Q,
    K_Extend,
    V_Extend,
    O,
    K_Buffer,
    V_Buffer,
    Selected_KV_Slots,
    Selected_Lens,
    sm_scale,
    k_scale,
    v_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_selected_t,
    stride_selected_k,
    stride_selected_lens,
    stride_buf_ks,
    stride_buf_kh,
    stride_buf_vs,
    stride_buf_vh,
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    KV_GROUP_SIZE: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    logit_cap: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    query_idx = tl.program_id(0)
    kv_head = tl.program_id(1)

    offsets_h = tl.arange(0, BLOCK_H)
    offsets_d = tl.arange(0, BLOCK_D)
    offsets_dv = tl.arange(0, BLOCK_DV)
    offsets_n = tl.arange(0, BLOCK_N)
    head_mask = offsets_h < KV_GROUP_SIZE
    d_mask = offsets_d < QK_HEAD_DIM
    dv_mask = offsets_dv < V_HEAD_DIM

    query_heads = kv_head * KV_GROUP_SIZE + offsets_h
    q = tl.load(
        Q
        + query_idx * stride_qt
        + query_heads[:, None] * stride_qh
        + offsets_d[None, :] * stride_qd,
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    denominator = tl.zeros((BLOCK_H,), dtype=tl.float32)
    max_logit = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    selected_len = tl.load(
        Selected_Lens + query_idx * stride_selected_lens
    )

    for start_n in range(0, selected_len, BLOCK_N):
        token_offsets = start_n + offsets_n
        token_mask = token_offsets < selected_len
        slots = tl.load(
            Selected_KV_Slots
            + query_idx * stride_selected_t
            + token_offsets * stride_selected_k,
            mask=token_mask,
            other=0,
        )

        if PAGE_SIZE == 1:
            k_offsets = (
                slots[None, :] * stride_buf_ks
                + kv_head * stride_buf_kh
                + offsets_d[:, None]
            )
        else:
            page_ids = slots // PAGE_SIZE
            page_offsets = slots % PAGE_SIZE
            k_offsets = (
                page_ids[None, :] * stride_buf_kpage
                + page_offsets[None, :] * stride_buf_ktok
                + kv_head * stride_buf_kh
                + offsets_d[:, None]
            )
        key = tl.load(
            K_Buffer + k_offsets,
            mask=d_mask[:, None] & token_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q.to(key.dtype), key) * (sm_scale * k_scale)
        if logit_cap > 0:
            qk = logit_cap * _tanh(qk / logit_cap)
        qk = tl.where(head_mask[:, None] & token_mask[None, :], qk, -float("inf"))

        next_max = tl.maximum(max_logit, tl.max(qk, axis=1))
        rescale = tl.exp(max_logit - next_max)
        probability = tl.exp(qk - next_max[:, None])
        denominator = denominator * rescale + tl.sum(probability, axis=1)

        if PAGE_SIZE == 1:
            v_offsets = (
                slots[:, None] * stride_buf_vs
                + kv_head * stride_buf_vh
                + offsets_dv[None, :]
            )
        else:
            v_offsets = (
                page_ids[:, None] * stride_buf_vpage
                + page_offsets[:, None] * stride_buf_vtok
                + kv_head * stride_buf_vh
                + offsets_dv[None, :]
            )
        value = tl.load(
            V_Buffer + v_offsets,
            mask=token_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        acc = (
            acc * rescale[:, None]
            + tl.dot(probability.to(value.dtype), value) * v_scale
        )
        max_logit = next_max

    suffix_len = query_idx + 1
    for start_n in range(0, suffix_len, BLOCK_N):
        token_offsets = start_n + offsets_n
        token_mask = token_offsets < suffix_len
        key = tl.load(
            K_Extend
            + token_offsets[None, :] * stride_kt
            + kv_head * stride_kh
            + offsets_d[:, None] * stride_kd,
            mask=d_mask[:, None] & token_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q.to(key.dtype), key) * sm_scale
        if logit_cap > 0:
            qk = logit_cap * _tanh(qk / logit_cap)
        qk = tl.where(head_mask[:, None] & token_mask[None, :], qk, -float("inf"))

        next_max = tl.maximum(max_logit, tl.max(qk, axis=1))
        rescale = tl.exp(max_logit - next_max)
        probability = tl.exp(qk - next_max[:, None])
        denominator = denominator * rescale + tl.sum(probability, axis=1)
        value = tl.load(
            V_Extend
            + token_offsets[:, None] * stride_vt
            + kv_head * stride_vh
            + offsets_dv[None, :] * stride_vd,
            mask=token_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        acc = acc * rescale[:, None] + tl.dot(probability.to(value.dtype), value)
        max_logit = next_max

    output_offsets = (
        query_idx * stride_ot
        + query_heads[:, None] * stride_oh
        + offsets_dv[None, :] * stride_od
    )
    tl.store(
        O + output_offsets,
        acc / denominator[:, None],
        mask=head_mask[:, None] & dv_mask[None, :],
    )


def exact_sparse_extend_attention_fwd(
    q: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    output: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    selected_kv_slots: torch.Tensor,
    selected_lens: torch.Tensor,
    k_scale: float,
    v_scale: float,
    sm_scale: float | None = None,
    logit_cap: float = 0.0,
    page_size: int = 1,
) -> None:
    qk_head_dim = q.shape[-1]
    v_head_dim = v_extend.shape[-1]
    kv_group_size = q.shape[1] // k_extend.shape[1]
    block_h = max(16, triton.next_power_of_2(kv_group_size))
    block_d = triton.next_power_of_2(qk_head_dim)
    block_dv = triton.next_power_of_2(v_head_dim)
    block_n = 64
    sm_scale = sm_scale or qk_head_dim**-0.5

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = (
        _extract_kv_strides(k_buffer, page_size)
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = (
        _extract_kv_strides(v_buffer, page_size)
    )
    grid = (q.shape[0], k_extend.shape[1])
    _exact_sparse_extend_kernel[grid](
        q,
        k_extend,
        v_extend,
        output,
        k_buffer,
        v_buffer,
        selected_kv_slots,
        selected_lens,
        sm_scale,
        k_scale,
        v_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_extend.stride(0),
        k_extend.stride(1),
        k_extend.stride(2),
        v_extend.stride(0),
        v_extend.stride(1),
        v_extend.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        selected_kv_slots.stride(0),
        selected_kv_slots.stride(1),
        selected_lens.stride(0),
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        KV_GROUP_SIZE=kv_group_size,
        QK_HEAD_DIM=qk_head_dim,
        V_HEAD_DIM=v_head_dim,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_DV=block_dv,
        BLOCK_N=block_n,
        logit_cap=logit_cap,
        PAGE_SIZE=page_size,
        num_warps=4,
        num_stages=2,
    )
