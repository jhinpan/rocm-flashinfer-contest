"""ROCm/MI300 gdn_decode solution.

Wraps AITER's `fused_recurrent_gated_delta_rule` (Triton, gfx942) to implement the
single-token Gated Delta Net decode in k-last state layout.

Reference math (per head, fp32):
    g    = exp(-exp(A_log) * softplus(a + dt_bias))     # multiplicative gate in (0,1)
    beta = sigmoid(b)
    delta-rule recurrent update; output = scale * q @ state_new

AITER expects the decay `g` in LOG space (it applies exp(g) internally) and the
recurrent state as [N, HV, K, V], whereas the contest uses k-last [B, HV, V, K].
"""
import math

import torch
import torch.nn.functional as F
from aiter.ops.triton.gated_delta_net import fused_recurrent_gated_delta_rule


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    # scale may arrive as python float or 0-d tensor
    if isinstance(scale, torch.Tensor):
        scale = float(scale.item())
    else:
        scale = float(scale)
    if scale == 0.0:
        scale = 1.0 / math.sqrt(q.shape[-1])

    B = q.shape[0]
    HV = v.shape[2]
    K = q.shape[-1]
    V = v.shape[-1]

    # Gates in fp32. g is passed in LOG space: log(reference_g) = -exp(A_log)*softplus(a+dt_bias)
    x = a.float() + dt_bias.float()                       # [B, 1, HV]
    g_log = -torch.exp(A_log.float()) * F.softplus(x)     # [B, 1, HV]
    beta = torch.sigmoid(b.float())                       # [B, 1, HV]

    # k-last [B, HV, V, K] -> AITER [N=B, HV, K, V]
    if state is not None:
        init_state = state.transpose(-1, -2).contiguous().float()
    else:
        init_state = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)

    o, final_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g_log,
        beta=beta,
        scale=scale,
        initial_state=init_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    # o: [B, 1, HV, V] bf16 (matches reference). new_state back to k-last [B, HV, V, K] fp32.
    new_state = final_state.transpose(-1, -2).contiguous()
    return o.to(torch.bfloat16), new_state
