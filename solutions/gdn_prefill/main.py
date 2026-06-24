"""ROCm/MI300 gdn_prefill solution.

Variable-length Gated Delta Net prefill in k-last state layout. Same delta-rule math as
gdn_decode but over full sequences delimited by `cu_seqlens`.

Short/medium sequences use AITER's `fused_recurrent_gated_delta_rule` (varlen, B=1), which
reproduces the per-token recurrence of the reference exactly. Long sequences use the chunk-parallel
`chunk_gated_delta_rule`, which processes the sequence in parallel chunks instead of a serial
recurrence — far faster when the sequence is long (the recurrent kernel dominates there), and within
tolerance vs the reference (out diff ~1e-4, state diff <1e-2). The two paths are dispatched by total
sequence length; the recurrent path is the fallback for short sequences and for correctness safety.

Layout: g passed in LOG space; recurrent state k-last [N,HV,V,K] <-> AITER [N,HV,K,V]. The chunk
kernel requires q/k expanded to HV heads (GVA), unlike the recurrent kernel which expands internally.
"""
import math
import os

import torch
import torch.nn.functional as F
from aiter.ops.triton.gated_delta_net import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

# Below this flattened length the recurrent path is faster (chunk has launch/expand overhead);
# at/above it the chunk-parallel path wins (e.g. ~+88% at total_seq_len=8192). Overridable for tuning.
CHUNK_MIN_SEQ_LEN = int(os.environ.get("GDN_PREFILL_CHUNK_MIN", "4096"))


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
    use_chunk = total_seq_len >= CHUNK_MIN_SEQ_LEN and os.environ.get("GDN_PREFILL_RECURRENT") != "1"

    if use_chunk:
        # GVA: the chunk kernel needs q/k at HV heads (the recurrent kernel expands internally).
        if num_q_heads != HV:
            qk_rep = HV // num_q_heads
            qc = q.repeat_interleave(qk_rep, dim=1).unsqueeze(0)
            kc = k.repeat_interleave(qk_rep, dim=1).unsqueeze(0)
        else:
            qc, kc = q1, k1
        o, final_state = chunk_gated_delta_rule(
            q=qc, k=kc, v=v1, g=g_log, beta=beta, scale=scale,
            initial_state=init_state, output_final_state=True,
            use_qk_l2norm_in_kernel=False, cu_seqlens=cu,
        )
    else:
        o, final_state = fused_recurrent_gated_delta_rule(
            q=q1, k=k1, v=v1, g=g_log, beta=beta, scale=scale,
            initial_state=init_state, output_final_state=True,
            use_qk_l2norm_in_kernel=False, cu_seqlens=cu,
        )

    output = o.squeeze(0).to(torch.bfloat16)              # [T, HV, V]
    new_state = final_state.transpose(-1, -2).contiguous()  # [N, HV, V, K] fp32
    return output, new_state
