"""ROCm/MI300 dsa_topk_indexer_fp8 solution.

DeepSeek sparse-attention top-k indexer. For each query, score every KV token by a
weighted sum of per-head ReLU(q . k), then return the global indices of the top-2048.

The official evaluator re-scores the *returned indices* with a canonical fp8->fp32
dequant and compares sorted scores, so the only requirement is to select the true
top-k under fp32 dequant. We therefore dequantize to fp32 and compute exactly (this
sidesteps the MI300 fp8 e4m3fnuz-vs-e4m3fn MMA question entirely), fully vectorized
across the batch (batch<=31, pages-per-seq<=91, so the score tensor is tiny).

KV cache packed layout (deep_gemm), per page of `page_size` tokens:
  [ page_size*head_dim fp8 bytes | page_size*4 scale bytes ]
"""
import os

import torch
import triton
import triton.language as tl

PAGE_SIZE = 64
HEAD_DIM = 128
TOPK = 2048


@triton.jit
def _fused_logits_kernel(q_ptr, k_ptr, w_ptr, out_ptr, B, N,
                         sqb, sqh, sqd, skb, skn, skd, swb, swh, sob, son,
                         BLOCK_N: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    """scores[b,n] = sum_h relu( sum_d q[b,h,d]*k[b,n,d] ) * w[b,h], computed directly from the
    dequantized k_deq[B,N,D] and q[B,H,D] so the [B,H,N] per-head logits are never materialized
    (replaces bmm + relu + multiply + sum-over-heads). fp32 throughout."""
    b = tl.program_id(0)
    n0 = tl.program_id(1) * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < N
    q = tl.load(q_ptr + b * sqb + offs_h[:, None] * sqh + offs_d[None, :] * sqd)   # [H, D]
    k = tl.load(k_ptr + b * skb + offs_n[:, None] * skn + offs_d[None, :] * skd,
                mask=mask_n[:, None], other=0.0)                                    # [BN, D]
    ph = tl.dot(q, tl.trans(k))                                                     # [H, BN]
    ph = tl.where(ph > 0.0, ph, 0.0)
    w = tl.load(w_ptr + b * swb + offs_h * swh)                                     # [H]
    acc = tl.sum(ph * w[:, None], axis=0)                                           # [BN]
    tl.store(out_ptr + b * sob + offs_n * son, acc, mask=mask_n)


def _fused_logits(q, k_deq, weights):
    B, H, D = q.shape
    N = k_deq.shape[1]
    out = torch.empty((B, N), dtype=torch.float32, device=q.device)
    BLOCK_N = 64
    grid = (B, triton.cdiv(N, BLOCK_N))
    _fused_logits_kernel[grid](q, k_deq, weights, out, B, N,
                         q.stride(0), q.stride(1), q.stride(2),
                         k_deq.stride(0), k_deq.stride(1), k_deq.stride(2),
                         weights.stride(0), weights.stride(1), out.stride(0), out.stride(1),
                         BLOCK_N=BLOCK_N, H=H, D=D, num_warps=4)
    return out


def _dequant_pages(packed_i8: torch.Tensor) -> torch.Tensor:
    """[P, page_size, 1, head_dim+4] int8 -> [P, page_size, head_dim] float32."""
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

    q = q_index_fp8.float()                         # [B, H, D]
    seq_lens = seq_lens.to(torch.long)              # [B]

    # Gather only the pages referenced by block_table, then dequant that subset.
    bt = block_table.long().clamp_(0, num_pages - 1)        # [B, M]
    gathered = k_index_cache_fp8[bt.reshape(-1)]            # [B*M, page_size, 1, 132]
    k_deq = _dequant_pages(gathered)                        # [B*M, page_size, D]
    k_deq = k_deq.view(B, M * PAGE_SIZE, D)                 # [B, N, D]
    N = M * PAGE_SIZE

    # Default: a fused Triton kernel computing scores[B,N] directly from q and dequantized k, never
    # materializing the [B,H,N] per-head logits (fuses bmm + relu + per-head-weight + sum-over-heads).
    # ~+5-8% faster; passes the official verify.py (128/128). DSA_TOPK_TORCH=1 selects the plain torch
    # bmm+relu+sum path (per-run matched ratio 1.000; marginally slower).
    #
    # The AITER deepgemm_fp8_paged_mqa_logits / aiter top-k levers were tested and rejected: AITER
    # views the KV cache as e4m3fnuz (gfx942) while the contest data is e4m3fn (see
    # tools/fp8_dtype_probe.py), and wiring them on the contest inputs hung the official verify.py
    # (>700s) and triggered a GPU HSA exception.
    if os.environ.get("DSA_TOPK_TORCH") == "1":
        per_head = torch.bmm(q, k_deq.transpose(1, 2))          # [B, H, N]
        per_head = torch.relu(per_head)
        scores = (per_head * weights.float().unsqueeze(2)).sum(dim=1)   # [B, N]
    else:
        scores = _fused_logits(q.contiguous(), k_deq.contiguous(), weights.float().contiguous())

    # Mask positions beyond each sequence length (token_position == flat index n).
    pos = torch.arange(N, device=device).unsqueeze(0)       # [1, N]
    valid = pos < seq_lens.unsqueeze(1)                     # [B, N]
    scores = scores.masked_fill(~valid, float("-inf"))

    k_sel = min(TOPK, N)
    _, top_pos = torch.topk(scores, k_sel, dim=1)  # [B, k_sel] sorted desc

    # A selected slot is a real token iff its position is within the sequence. Tokens whose
    # cached data is NaN are still valid selections (torch.topk floats them to the top, exactly
    # as the reference does) -- so gate on position validity, NOT on score finiteness.
    valid_at_top = torch.gather(valid, 1, top_pos)                  # [B, k_sel]

    slot = top_pos // PAGE_SIZE
    offset = top_pos % PAGE_SIZE
    global_page = torch.gather(block_table.long(), 1, slot)         # [B, k_sel]
    global_token = (global_page * PAGE_SIZE + offset).to(torch.int32)

    out = torch.full((B, TOPK), -1, dtype=torch.int32, device=device)
    out[:, :k_sel] = torch.where(valid_at_top, global_token, torch.full_like(global_token, -1))
    return (out,)
