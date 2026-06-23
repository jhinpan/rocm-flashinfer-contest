"""ROCm/MI300 gdn_decode solution.

Single-token Gated Delta Net decode in k-last state layout.

Default path uses AITER's `fused_sigmoid_gating_delta_rule_update`, which computes the sigmoid/
softplus gate AND the recurrent delta-rule update in ONE fused kernel — removing the host-side
gate elementwise ops (exp / softplus / sigmoid) that the previous wrapper launched separately and
that dominated at larger batch. Numerically identical to the recurrent path (max diff 0.0) and
~+23-28% faster. `GDN_DECODE_RECURRENT=1` selects the original
`fused_recurrent_gated_delta_rule` wrapper (host-side gate compute) as a fallback.

Reference math (per head, fp32):
    g    = exp(-exp(A_log) * softplus(a + dt_bias))     # multiplicative gate in (0,1)
    beta = sigmoid(b)
The fused op takes raw A_log/a/dt_bias/b + softplus params and reproduces this internally.
State is k-last [B, HV, V, K]; AITER uses [N, HV, K, V] (transposed at the boundary).
"""
import math
import os

import torch
import torch.nn.functional as F
from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from aiter.ops.triton.gated_delta_net import fused_recurrent_gated_delta_rule


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
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

    # k-last [B, HV, V, K] -> AITER [N=B, HV, K, V]
    if state is not None:
        init_state = state.transpose(-1, -2).contiguous().float()
    else:
        init_state = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)

    if os.environ.get("GDN_DECODE_RECURRENT") == "1":
        # Fallback: host-side gate compute + recurrent kernel.
        x = a.float() + dt_bias.float()
        g_log = -torch.exp(A_log.float()) * F.softplus(x)
        beta = torch.sigmoid(b.float())
        o, final_state = fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g_log, beta=beta, scale=scale,
            initial_state=init_state, output_final_state=True, use_qk_l2norm_in_kernel=False,
        )
        new_state = final_state.transpose(-1, -2).contiguous()
        return o.to(torch.bfloat16), new_state

    # Default: fused gate + recurrent update in one kernel. `init_state` is updated in place.
    idx = torch.arange(B, dtype=torch.int32, device=q.device)
    o = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
        q=q, k=k, v=v, b=b,
        initial_state_source=init_state, initial_state_indices=idx,
        scale=scale, use_qk_l2norm_in_kernel=False,
    )
    # o is [NK=1, ...]; reshape to the reference's [B, 1, HV, V] (the fused op collapses the T dim).
    output = o.reshape(B, 1, HV, V).to(torch.bfloat16)
    new_state = init_state.transpose(-1, -2).contiguous()    # [B, HV, V, K] fp32
    return output, new_state
