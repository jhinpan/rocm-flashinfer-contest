"""ROCm/MI300 gdn_prefill solution.

Variable-length Gated Delta Net prefill in k-last state layout. Same delta-rule math as
gdn_decode but over full sequences delimited by `cu_seqlens`.

Uses AITER's `fused_recurrent_gated_delta_rule` in varlen mode (batch B=1, flattened tokens
+ cu_seqlens), which reproduces the per-token recurrence of the reference exactly. The chunk
path (`chunk_gated_delta_rule`) is a Phase-3 optimization.

Layout: g passed in LOG space; recurrent state k-last [N,HV,V,K] <-> AITER [N,HV,K,V].
"""
import math

import torch
import torch.nn.functional as F
from aiter.ops.triton.gated_delta_net import fused_recurrent_gated_delta_rule


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)
    if scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])

    total_seq_len, num_q_heads, K = q.shape
    HV = v.shape[1]
    V = v.shape[-1]
    num_seqs = cu_seqlens.numel() - 1

    # flatten to varlen form: B=1
    q1 = q.unsqueeze(0)            # [1, T, Hq, K]
    k1 = k.unsqueeze(0)            # [1, T, Hk, K]
    v1 = v.unsqueeze(0)            # [1, T, HV, V]

    # gates, fp32, g in LOG space
    x = a.float() + dt_bias.float()                        # [T, HV]
    g_log = (-torch.exp(A_log.float()) * F.softplus(x)).unsqueeze(0)   # [1, T, HV]
    beta = torch.sigmoid(b.float()).unsqueeze(0)          # [1, T, HV]

    # k-last [N, HV, V, K] -> AITER [N, HV, K, V]
    if state is not None:
        init_state = state.transpose(-1, -2).contiguous().float()
    else:
        init_state = torch.zeros(num_seqs, HV, K, V, dtype=torch.float32, device=q.device)

    cu = cu_seqlens.to(torch.long)

    o, final_state = fused_recurrent_gated_delta_rule(
        q=q1,
        k=k1,
        v=v1,
        g=g_log,
        beta=beta,
        scale=scale,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu,
    )

    output = o.squeeze(0).to(torch.bfloat16)              # [T, HV, V]
    new_state = final_state.transpose(-1, -2).contiguous()  # [N, HV, V, K] fp32
    return output, new_state
