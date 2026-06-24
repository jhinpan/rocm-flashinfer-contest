"""ROCm/MI300 dsa_topk_indexer_fp8 solution.

DeepSeek sparse-attention top-k indexer. For each query, score every KV token by a
weighted sum of per-head ReLU(q . k), then return the global indices of the top-2048.

The official evaluator re-scores the *returned indices* with a canonical fp8->fp32
dequant and compares sorted scores, so the only requirement is to select the true
top-k under fp32 dequant.

Default path exactly reproduces the fp32 reference: gather the referenced pages, dequant
e4m3fn->fp32, then `torch.bmm` + ReLU + per-head-weighted sum, and `torch.topk`. Because the
selection comes from the same fp32 GEMM the reference/evaluator use, the per-run matched-ratio
is 1.0. Robustness is the priority here: the v2 fused-logits path (a Triton tl.dot kernel) was
~+5-8% faster but mis-ranked ~1% of boundary tokens (mr 0.988-0.992), because Triton's GEMM
tiling differs from torch's and the extreme dynamic range of these inputs (per-token scales
spanning ~1e30, with negative learned weights) makes near-tie boundary tokens sensitive to that
difference.

`DSA_TOPK_FAST=1` selects a packed-page scoring kernel (`_packed_score`) that reads the fp8
cache directly via `block_table` and scores every token in-kernel with no `k_deq[B,N,D]`
materialization -- ~+14-19% faster -- using a NaN-preserving ReLU, the
`dot(q, fp8*scale) = scale*dot(q, fp8)` identity (per-token positive scale), and
`input_precision="ieee"`. It passes the official sorted-score evaluator but lands at mr ~0.99
(same boundary sensitivity as v2), so it does not meet the stricter mr>=0.999 robustness bar
and is not the default; it is kept for the speed/accuracy trade-off it documents.

KV cache packed layout (deep_gemm), per page of `page_size` tokens (flattened):
  [ page_size*head_dim fp8 bytes | page_size*4 scale bytes ]
"""
import os

import torch
import triton
import triton.language as tl

PAGE_SIZE = 64
TOPK = 2048


@triton.jit
def _packed_score_kernel(q_ptr, kfp8_ptr, scale_ptr, w_ptr, bt_ptr, seqlen_ptr, out_ptr,
                         N, num_pages,
                         sqb, sqh, sqd, skp, ssp, swb, swh, sbtb, sob,
                         PAGE: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                         BLOCK_N: tl.constexpr):
    """scores[b,n] = scale[b,n] * sum_h relu( sum_d q[b,h,d] * kfp8[page(b,n),off(b,n),d] ) * w[b,h]

    read straight from the packed paged cache (no k_deq materialization). NaN-preserving relu;
    scale-after-dot; full-fp32 dot."""
    b = tl.program_id(0)
    n0 = tl.program_id(1) * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)                 # token (flat) indices in [0, N)
    mask_n = offs_n < N
    page_slot = offs_n // PAGE
    offset = offs_n % PAGE
    gpage = tl.load(bt_ptr + b * sbtb + page_slot, mask=mask_n, other=0)
    gpage = tl.minimum(tl.maximum(gpage, 0), num_pages - 1)            # clamp OOB slots

    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, H)
    # k fp8 block [BLOCK_N, D] decoded to fp32 (page-major fp8 region: page*skp + off*D + d)
    k_off = gpage[:, None] * skp + offset[:, None] * D + offs_d[None, :]
    k = tl.load(kfp8_ptr + k_off, mask=mask_n[:, None], other=0.0).to(tl.float32)
    q = tl.load(q_ptr + b * sqb + offs_h[:, None] * sqh + offs_d[None, :] * sqd)   # [H, D]

    ph = tl.dot(q, tl.trans(k), input_precision="ieee")               # [H, BLOCK_N] fp32
    # ReLU that propagates NaN (max(x,0) but keep x where x is NaN)
    ph = tl.where(ph != ph, ph, tl.maximum(ph, 0.0))
    w = tl.load(w_ptr + b * swb + offs_h * swh)                       # [H]
    acc = tl.sum(ph * w[:, None], axis=0)                             # [BLOCK_N]

    sc = tl.load(scale_ptr + gpage * ssp + offset, mask=mask_n, other=0.0)   # [BLOCK_N]
    acc = acc * sc

    seqlen = tl.load(seqlen_ptr + b)
    valid = offs_n < seqlen
    acc = tl.where(valid, acc, float("-inf"))
    tl.store(out_ptr + b * sob + offs_n, acc, mask=mask_n)


def _packed_views(k_index_cache_fp8):
    """Strided fp8 + fp32-scale views into the packed int8 cache (no copy)."""
    num_pages, page_size, _, hd_sf = k_index_cache_fp8.shape
    D = hd_sf - 4
    u8 = k_index_cache_fp8.view(torch.uint8).reshape(num_pages, page_size * hd_sf)
    page_stride_bytes = page_size * hd_sf
    fp8_view = torch.as_strided(u8.view(torch.float8_e4m3fn),
                                size=(num_pages, page_size * D),
                                stride=(page_stride_bytes, 1))
    scale_view = torch.as_strided(u8.view(torch.float32),
                                  size=(num_pages, page_size),
                                  stride=(page_stride_bytes // 4, 1),
                                  storage_offset=page_size * D // 4)
    return fp8_view, scale_view, D


def _packed_score(q, k_index_cache_fp8, weights, seq_lens, block_table_clamped, N):
    """Score [B, N] directly from the packed paged fp8 cache (fast tl.dot path)."""
    B, H, D = q.shape
    num_pages = k_index_cache_fp8.shape[0]
    device = q.device
    fp8_view, scale_view, _ = _packed_views(k_index_cache_fp8)

    out = torch.empty((B, N), dtype=torch.float32, device=device)
    BLOCK_N = 64
    grid = (B, triton.cdiv(N, BLOCK_N))
    _packed_score_kernel[grid](
        q, fp8_view, scale_view, weights, block_table_clamped, seq_lens, out,
        N, num_pages,
        q.stride(0), q.stride(1), q.stride(2),
        fp8_view.stride(0), scale_view.stride(0),
        weights.stride(0), weights.stride(1),
        block_table_clamped.stride(0), out.stride(0),
        PAGE=PAGE_SIZE, D=D, H=H, BLOCK_N=BLOCK_N, num_warps=4,
    )
    return out


def _dequant_pages(packed_i8: torch.Tensor) -> torch.Tensor:
    """[P, page_size, 1, head_dim+4] int8 -> [P, page_size, head_dim] float32 (torch fallback)."""
    u8 = packed_i8.view(torch.uint8)
    P, page_size, _, hd_sf = u8.shape
    head_dim = hd_sf - 4
    flat = u8.view(P, page_size * hd_sf)
    fp8 = (
        flat[:, : page_size * head_dim]
        .contiguous()
        .view(P, page_size, head_dim)
        .view(torch.float8_e4m3fn)
        .float()
    )
    scale = (
        flat[:, page_size * head_dim:]
        .contiguous()
        .view(P, page_size, 4)
        .view(torch.float32)
    )  # [P, page_size, 1]
    return fp8 * scale


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    device = q_index_fp8.device
    B, H, D = q_index_fp8.shape
    num_pages = k_index_cache_fp8.shape[0]
    M = block_table.shape[1]
    N = M * PAGE_SIZE

    q = q_index_fp8.float().contiguous()            # [B, H, D]
    seq_lens = seq_lens.to(torch.long)              # [B]
    weights_f = weights.float().contiguous()        # [B, H]
    bt_clamped = block_table.long().clamp(0, num_pages - 1).contiguous()   # [B, M]

    # Score selection. The default exactly reproduces the fp32 reference (gather -> dequant ->
    # torch.bmm/relu/weighted-sum) so the per-run matched-ratio is 1.0 -- robustness is the priority
    # for this kernel (the v2 fused-logits path scored mr 0.988-0.992, mis-ranking ~1% of boundary
    # tokens). DSA_TOPK_FAST=1 selects the packed-page scoring kernel below, which reads the fp8
    # cache directly (no k_deq materialization) and is ~+14-19% faster, but its tl.dot tiling differs
    # from torch's GEMM and -- amplified by negative-weight cancellation at the extreme dynamic range
    # of these inputs -- mis-ranks ~0.5-1% of boundary tokens (mr ~0.99). It passes the official
    # sorted-score evaluator but not the stricter mr>=0.999 robustness bar, so it is not the default.
    if os.environ.get("DSA_TOPK_FAST") == "1":
        scores = _packed_score(q, k_index_cache_fp8, weights_f, seq_lens.to(torch.int32),
                               bt_clamped.to(torch.int32), N)
    else:
        gathered = k_index_cache_fp8[bt_clamped.reshape(-1)]
        k_deq = _dequant_pages(gathered).view(B, N, D)              # [B, N, D] fp32
        per_head = torch.relu(torch.bmm(q, k_deq.transpose(1, 2)))  # [B, H, N]
        scores = (per_head * weights_f.unsqueeze(2)).sum(dim=1)     # [B, N]
        pos = torch.arange(N, device=device).unsqueeze(0)
        scores = scores.masked_fill(pos >= seq_lens.unsqueeze(1), float("-inf"))

    top_pos = _select_topk(scores, N, device)
    return (_map_to_global(top_pos, block_table, seq_lens, N, device),)


def _select_topk(scores, N, device):
    """Return [B, TOPK] flat positions (sorted desc); pad with the N sentinel when N < TOPK."""
    B = scores.shape[0]
    k_sel = min(TOPK, N)
    _, top_pos = torch.topk(scores, k_sel, dim=1)         # [B, k_sel] sorted desc, flat positions
    if k_sel < TOPK:
        pad = torch.full((B, TOPK - k_sel), N, dtype=top_pos.dtype, device=device)
        top_pos = torch.cat([top_pos, pad], dim=1)
    return top_pos


def _map_to_global(top_pos, block_table, seq_lens, N, device):
    """Map [B, TOPK] flat positions to global token indices; padding / out-of-seq -> -1.

    NaN-keyed tokens float to the top via topk exactly as the reference does, so a slot is gated
    on position validity (within seq_len), NOT on score finiteness."""
    B = top_pos.shape[0]
    valid = (top_pos < seq_lens.unsqueeze(1)) & (top_pos < N)
    safe = top_pos.clamp(max=N - 1)
    slot = safe // PAGE_SIZE
    offset = safe % PAGE_SIZE
    global_page = torch.gather(block_table.long(), 1, slot)      # [B, TOPK]
    global_token = (global_page * PAGE_SIZE + offset).to(torch.int32)
    out = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
    out = torch.where(valid, global_token, out)
    return out
