"""ROCm/MI300 moe_fp8_block_scale solution (python/torch).

DeepSeek-V3 FP8 block-scale MoE with no-aux routing. The contest mandates CUDA for the
NVIDIA submission, but the CUDA/tvm-ffi builder needs nvcc (absent on ROCm), so this is a
portable language=python port that reproduces the reference math.

Speed comes from (a) only dequantizing the *active* local experts (not all 32 upfront, which
the reference does at ~5.6 GB of fp32 weights) and (b) bf16 dequant + bf16 matmul on MI300's
fast bf16 MMA. The official tolerance is loose (atol=1, rtol=0.3, matched_ratio=0.9), so bf16
accumulation is comfortably within bounds.

Routing (no-aux): sigmoid(logits); group 256 experts into 8 groups; per group sum of top-2
(with bias); keep top-4 groups; global top-8 by (s+bias); weights from unbiased s, normalized,
scaled by routed_scaling_factor. SwiGLU gate = second half: C = silu(G1[:, I:]) * G1[:, :I].
"""
import os

import torch
import triton
import triton.language as tl

H = 7168
I = 2048
BLOCK = 128
E_GLOBAL = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
COMPUTE_DTYPE = torch.bfloat16


@triton.jit
def _blockscale_gemm_kernel(
    a_ptr,        # [M, K] bf16
    w_ptr,        # [N, K] fp8 (e4m3fn)
    s_ptr,        # [N/128, K/128] fp32 block scales
    c_ptr,        # [M, N] bf16  (= a @ dequant(w).T)
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_sn, stride_sk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C = A @ dequant(W).T with per-128x128 block-scale dequant of W done IN-TILE (fp32 accumulate),
    so the full dequantized bf16 weight is never materialized. BLOCK_N and BLOCK_K are 128 so each
    [128,128] W tile maps to exactly one scale element. K and N are multiples of 128 (no K/N mask)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_blk = pid_n  # BLOCK_N == 128
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
                    mask=m_mask[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :] * stride_wk).to(tl.float32)
        sc = tl.load(s_ptr + n_blk * stride_sn + (k0 // BLOCK_K) * stride_sk)
        w = w * sc                                  # dequant this [128,128] block
        acc += tl.dot(a, tl.trans(w))               # [BM,BK] @ [BK,BN]
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.bfloat16), mask=m_mask[:, None])


def _blockscale_gemm(a, w, scale):
    """a:[M,K] bf16, w:[N,K] fp8, scale:[N/128,K/128] fp32 -> [M,N] bf16 = a @ dequant(w).T."""
    M, K = a.shape
    N = w.shape[0]
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // 128)
    _blockscale_gemm_kernel[grid](
        a, w, scale, c, M, N, K,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        scale.stride(0), scale.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=1,
    )
    return c


def _dequant_block(w, scale):
    """Block-scale dequant of a [R, C] fp8/int8 weight by a [R/128, C/128] fp32 scale, returning
    bf16. Uses block-broadcast instead of `scale.repeat_interleave(128,0).repeat_interleave(128,1)`:
    the expanded [R, C] scale is never materialized, removing the dominant per-expert elementwise /
    memory-traffic cost found in profiling. Numerically identical (the same per-128x128-block scale
    is applied to each element)."""
    R, C = w.shape
    wf = w.to(torch.float32).view(R // BLOCK, BLOCK, C // BLOCK, BLOCK)
    return (wf * scale.to(torch.float32)[:, None, :, None]).view(R, C).to(COMPUTE_DTYPE)


def run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    device = hidden_states.device
    T = routing_logits.shape[0]
    E_local = gemm1_weights.shape[0]
    nhb = H // BLOCK  # 56
    if isinstance(local_expert_offset, torch.Tensor):
        local_start = int(local_expert_offset.item())
    else:
        local_start = int(local_expert_offset)
    if isinstance(routed_scaling_factor, torch.Tensor):
        routed_scaling_factor = float(routed_scaling_factor.item())

    # ---- 1) dequant hidden states -> bf16 [T, H] ----
    A_scale = hidden_states_scale.to(torch.float32).permute(1, 0).contiguous()  # [T, H/128]
    A_scale_exp = A_scale.unsqueeze(-1).expand(T, nhb, BLOCK).reshape(T, H)
    A = (hidden_states.to(torch.float32) * A_scale_exp).to(COMPUTE_DTYPE)        # [T, H]

    # ---- 2) no-aux routing (fp32) ----
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = torch.sigmoid(logits)                                  # [T, E]
    s_with_bias = s + bias

    group_size = E_GLOBAL // N_GROUP                           # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2 = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False).values
    group_scores = top2.sum(dim=2)                            # [T, 8]
    group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False).indices
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_GLOBAL)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False).indices  # [T, 8]

    Mmask = torch.zeros_like(s).scatter_(1, topk_idx, 1.0)
    weights = s * Mmask
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) * routed_scaling_factor  # [T, E]

    # ---- 3) local expert compute ----
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    nib = I // BLOCK                       # 16
    ngb = (2 * I) // BLOCK                 # 32

    # Default path: block-broadcast dequant (_dequant_block) + rocBLAS bf16 matmul. This is the
    # fastest *portable* path measured (~+10-17% vs baseline-v1).
    #
    # MOE_USE_FUSED=1 selects an experimental Triton block-scale GEMM (_blockscale_gemm) that dequants
    # fp8 weights in-tile so the full bf16 weight is never materialized. It is numerically correct
    # (verify.py 19/19, fp32 accumulate) but measured SLOWER than the default everywhere
    # (e.g. seq55 14.1ms vs 10.5ms; seq14107 41ms vs 19ms): a hand-written portable Triton GEMM
    # cannot match rocBLAS/Tensile's tuned matmul for these shapes, so avoiding weight
    # materialization does not pay off. Kept as documented evidence for the ≥20% NO-GO; see
    # results/round4-report.md. (Native fp8 fnuz MMA is separately deprioritized per DEC-4: GEMM is
    # only 3-18% of latency, and gfx942 native fp8 is e4m3fnuz vs the contest's e4m3fn.)
    use_fused = os.environ.get("MOE_USE_FUSED") == "1"

    # which local experts are actually selected
    sel_global = topk_idx                  # [T, 8]
    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue
        sel_mask = (sel_global == ge).any(dim=1)              # [T]
        if not bool(sel_mask.any()):
            continue
        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, token_idx)                   # [Tk, H] bf16

        if use_fused:
            G1 = _blockscale_gemm(A_e, gemm1_weights[le], gemm1_weights_scale[le])  # [Tk, 2I]
        else:
            W13_e = _dequant_block(gemm1_weights[le], gemm1_weights_scale[le])      # [2I, H]
            G1 = A_e.matmul(W13_e.t())
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        C = (torch.nn.functional.silu(X2.float()) * X1.float()).to(COMPUTE_DTYPE)  # [Tk, I]
        if use_fused:
            O = _blockscale_gemm(C, gemm2_weights[le], gemm2_weights_scale[le]).float()  # [Tk, H]
        else:
            W2_e = _dequant_block(gemm2_weights[le], gemm2_weights_scale[le])       # [H, I]
            O = C.matmul(W2_e.t()).float()

        w_tok = weights.index_select(0, token_idx)[:, ge]    # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)
