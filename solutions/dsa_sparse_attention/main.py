"""ROCm/MI300 dsa_sparse_attention solution.

DeepSeek sparse MLA attention. Per query token: gather the topk KV tokens given by absolute
indices `sparse_indices` (= page_idx*page_size + offset, -1 = padding), compute MLA logits
    logits = q_nope @ ckv.T + q_pe @ kpe.T   (scaled by sm_scale)
softmax, then output = softmax @ ckv.

The v1 solution wrapped AITER's `unified_attention_sparse_mla`, which requires a single
`[num_pages, page_size, 1, 576]` key/value cache. Building that cache with
`torch.cat([ckv, kpe], -1).contiguous()` over the **entire** paged cache every call costs ~485us
(O(num_pages), ~600MB), dwarfing the ~210us attention kernel and capping the small-token shapes at
~1x (profiled with rocprofv3). This solution instead runs a local Triton sparse-MLA decode kernel
that reads `ckv_cache` (lora + value) and `kpe_cache` (rope) as **separate** pointers, so the
combined cache is never materialized. The kernel body mirrors AITER's
`_kernel_unified_attention_sparse_mla_2d` (same masking / online-softmax math); only the key/value
loads are split across the two source tensors. fp32 accumulation, bf16 output. No cross-call
caching: every call reads the caches as given (same measured work as the reference).

Set DSA_USE_AITER=1 to fall back to the original AITER full-cache path (kept for debugging).
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_mla_decode_kernel(
    output_ptr,        # [num_tokens, num_query_heads, KV_LORA_RANK]
    query_ptr,         # [num_tokens, num_query_heads, KV_LORA_RANK + ROPE_RANK]
    lora_cache_ptr,    # [num_pages, page_size, KV_LORA_RANK]   (ckv: key-lora AND value)
    rope_cache_ptr,    # [num_pages, page_size, ROPE_RANK]      (kpe: key-rope)
    topk_indices_ptr,  # [num_tokens, topk]
    seq_lens_ptr,      # [num_seqs]
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,          # page_size
    stride_l_cache_0: tl.int64,        # lora (ckv) strides
    stride_l_cache_1: tl.int64,
    stride_l_cache_2: tl.constexpr,
    stride_r_cache_0: tl.int64,        # rope (kpe) strides
    stride_r_cache_1: tl.int64,
    stride_r_cache_2: tl.constexpr,
    topk_count: tl.constexpr,
    query_start_len_ptr,               # [num_seqs+1]
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    ROPE_RANK: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    ALL_DECODE: tl.constexpr = True,
):
    BLOCK_Q: tl.constexpr = 1
    q_block_global_idx = tl.program_id(0)
    q_ind = q_block_global_idx // (num_query_heads // BLOCK_M)
    head_ind = q_block_global_idx % (num_query_heads // BLOCK_M)

    # ALL_DECODE: one length-1 sequence per query token, so seq_idx == q_ind.
    cur_batch_in_all_start_index = q_ind
    if q_ind >= num_seqs:
        return

    offs_m = tl.arange(0, BLOCK_M) + head_ind * BLOCK_M
    offs_lora = tl.arange(0, KV_LORA_RANK)
    offs_rope_q = tl.arange(KV_LORA_RANK, KV_LORA_RANK + ROPE_RANK)  # rope cols in q
    offs_rope_k = tl.arange(0, ROPE_RANK)                            # rope cols in kpe cache

    query_pos = offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = offs_m % num_queries_per_kv
    query_mask_1 = query_offset_1 < num_query_heads

    q_rope_offset = (query_offset_0[:, None] * query_stride_0
                     + query_offset_1[:, None] * query_stride_1 + offs_rope_q[None, :])
    Q_rope = tl.load(query_ptr + q_rope_offset, mask=query_mask_1[:, None], other=0.0,
                     cache_modifier=".cg")
    q_lora_offset = (query_offset_0[:, None] * query_stride_0
                     + query_offset_1[:, None] * query_stride_1 + offs_lora[None, :])
    Q_lora = tl.load(query_ptr + q_lora_offset, mask=query_mask_1[:, None], other=0.0,
                     cache_modifier=".cg")

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], dtype=tl.float32)

    num_tiles = (topk_count + TILE_SIZE - 1) // TILE_SIZE
    for t in range(0, num_tiles):
        tile_start = t * TILE_SIZE
        offs_t = tl.arange(0, TILE_SIZE)
        valid_t = (tile_start + offs_t) < topk_count

        topk_row_ptr = topk_indices_ptr + q_ind * topk_count
        topk_pos = tl.load(topk_row_ptr + tile_start + offs_t, mask=valid_t, other=0)
        valid_t = valid_t & (topk_pos != -1)

        physical_block_idx = topk_pos // BLOCK_SIZE
        slot = topk_pos % BLOCK_SIZE

        S = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32)
        # K_rope from kpe cache: (ROPE_RANK, TILE_SIZE)
        k_rope_ptrs = (rope_cache_ptr
                       + physical_block_idx[None, :] * stride_r_cache_0
                       + offs_rope_k[:, None] * stride_r_cache_2
                       + slot[None, :] * stride_r_cache_1)
        K_rope = tl.load(k_rope_ptrs, mask=valid_t[None, :], other=0.0, cache_modifier=".cg")
        S += scale * tl.dot(Q_rope, K_rope)
        # K_lora from ckv cache: (KV_LORA_RANK, TILE_SIZE)
        k_lora_ptrs = (lora_cache_ptr
                       + physical_block_idx[None, :] * stride_l_cache_0
                       + offs_lora[:, None] * stride_l_cache_2
                       + slot[None, :] * stride_l_cache_1)
        K_lora = tl.load(k_lora_ptrs, mask=valid_t[None, :], other=0.0, cache_modifier=".cg")
        S += scale * tl.dot(Q_lora, K_lora)

        S = tl.where(query_mask_1[:, None] & valid_t[None, :], S, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        # V from ckv cache (value == lora): (TILE_SIZE, KV_LORA_RANK)
        v_lora_ptrs = (lora_cache_ptr
                       + physical_block_idx[:, None] * stride_l_cache_0
                       + slot[:, None] * stride_l_cache_1
                       + offs_lora[None, :] * stride_l_cache_2)
        V_lora = tl.load(v_lora_ptrs, mask=valid_t[:, None], other=0.0, cache_modifier=".cg")
        acc = tl.dot(P.to(V_lora.dtype), V_lora, acc=acc)

    one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L
    out_offs = (query_offset_0[:, None] * output_stride_0
                + query_offset_1[:, None] * output_stride_1 + offs_lora[None, :])
    tl.store(output_ptr + out_offs, acc, mask=query_mask_1[:, None])


def _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    device = q_nope.device

    # q: [T, H, 576] = [lora | rope] (small; the heavy full-cache cat is avoided entirely).
    q = torch.cat([q_nope, q_pe], dim=-1).contiguous()
    ckv = ckv_cache.contiguous()
    kpe = kpe_cache.contiguous()
    sparse_indices = sparse_indices.to(torch.int32).contiguous()

    cu_seqlens_q = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
    seqused_k = (sparse_indices != -1).sum(dim=1).to(torch.int32)

    output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv),
                         dtype=torch.bfloat16, device=device)

    BLOCK_M = 16
    if num_qo_heads % BLOCK_M != 0:
        BLOCK_M = num_qo_heads
    TILE_SIZE = page_size
    topk_count = sparse_indices.shape[1]
    grid = (num_tokens * (num_qo_heads // BLOCK_M),)

    if num_tokens > 0:
        _sparse_mla_decode_kernel[grid](
            output, q, ckv, kpe, sparse_indices, seqused_k, sm_scale,
            num_query_heads=num_qo_heads, num_queries_per_kv=num_qo_heads,
            query_stride_0=q.stride(0), query_stride_1=q.stride(1),
            output_stride_0=output.stride(0), output_stride_1=output.stride(1),
            BLOCK_SIZE=page_size,
            stride_l_cache_0=ckv.stride(0), stride_l_cache_1=ckv.stride(1),
            stride_l_cache_2=ckv.stride(2),
            stride_r_cache_0=kpe.stride(0), stride_r_cache_1=kpe.stride(1),
            stride_r_cache_2=kpe.stride(2),
            topk_count=topk_count, query_start_len_ptr=cu_seqlens_q, num_seqs=num_tokens,
            BLOCK_M=BLOCK_M, ROPE_RANK=head_dim_kpe, KV_LORA_RANK=head_dim_ckv,
            TILE_SIZE=TILE_SIZE, ALL_DECODE=True,
            num_warps=4, num_stages=1,
        )
    return (output,)


def _run_aiter(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Original AITER full-cache path, kept as a debug fallback (DSA_USE_AITER=1)."""
    from aiter.ops.triton.attention.unified_attention_sparse_mla import (
        unified_attention_sparse_mla,
    )
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    device = q_nope.device
    q = torch.cat([q_nope, q_pe], dim=-1).contiguous()
    kv = torch.cat([ckv_cache, kpe_cache], dim=-1).unsqueeze(2).contiguous()
    sparse_indices = sparse_indices.to(torch.int32)
    cu_seqlens_q = torch.arange(num_tokens + 1, dtype=torch.int32, device=device)
    seqused_k = (sparse_indices != -1).sum(dim=1).to(torch.int32)
    max_seqlen_k = int(seqused_k.max().item()) if num_tokens > 0 else 0
    block_table = torch.zeros((num_tokens, 1), dtype=torch.int32, device=device)
    output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv),
                         dtype=torch.bfloat16, device=device)
    unified_attention_sparse_mla(q, kv, output, cu_seqlens_q, 1, seqused_k, max_seqlen_k,
                                 sm_scale, sparse_indices, block_table, head_dim_ckv)
    return (output,)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    else:
        sm_scale = float(sm_scale)
    if os.environ.get("DSA_USE_AITER") == "1":
        return _run_aiter(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
    return _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)
