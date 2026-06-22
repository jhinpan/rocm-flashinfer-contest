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
import torch

H = 7168
I = 2048
BLOCK = 128
E_GLOBAL = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
COMPUTE_DTYPE = torch.bfloat16


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

        # dequant this expert's weights to bf16 (block-broadcast; no repeat_interleave)
        W13_e = _dequant_block(gemm1_weights[le], gemm1_weights_scale[le])   # [2I, H]
        W2_e = _dequant_block(gemm2_weights[le], gemm2_weights_scale[le])    # [H, I]

        G1 = A_e.matmul(W13_e.t())                           # [Tk, 2I]
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        C = (torch.nn.functional.silu(X2.float()) * X1.float()).to(COMPUTE_DTYPE)  # [Tk, I]
        O = C.matmul(W2_e.t()).float()                       # [Tk, H]

        w_tok = weights.index_select(0, token_idx)[:, ge]    # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)
