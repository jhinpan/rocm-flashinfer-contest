"""ROCm/MI300 gdn_decode solution.

Single-token Gated Delta Net decode in k-last state layout.

Default path is a repo-local Triton kernel (`_run_klast`) that reads the initial state AND writes
the final state in the contest k-last layout [B, HV, V, K] **directly**, with the gate (softplus/
sigmoid), the recurrent delta-rule update, and the output projection fused into ONE dispatch. This
removes BOTH host-side state transpose copies that the AITER-based path needed (the profile showed
those copies at 62-81% of latency). The gate math is copied verbatim from AITER's
`fused_sigmoid_gating_delta_rule_update_kernel`; only the state offsets differ (k stride 1,
v stride K) and the final state is written to a separate buffer (no input mutation). ~+65-78% vs
baseline-v2; verify 54/54.

Fallbacks (for debugging / evidence only):
  - `GDN_DECODE_FUSEDOP=1`: AITER `fused_sigmoid_gating_delta_rule_update` + a transposed-view
    output state (one transpose copy on the way in). `GDN_DECODE_CONTIG_STATE=1` forces a
    contiguous new_state within this path.
  - `GDN_DECODE_RECURRENT=1`: host-side gate compute + `fused_recurrent_gated_delta_rule`.

Reference math (per head, fp32):
    g    = exp(-exp(A_log) * softplus(a + dt_bias))     # multiplicative gate in (0,1)
    beta = sigmoid(b)
State is k-last [B, HV, V, K]; the AITER fallbacks use [N, HV, K, V] (transposed at the boundary).
"""
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from aiter.ops.triton.gated_delta_net import fused_recurrent_gated_delta_rule


@triton.jit
def _klast_decode_kernel(
    A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, o,
    h_in, h_out, HAS_STATE: tl.constexpr, scale,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
):
    """Single-token gated delta-rule decode reading/writing state in CONTEST k-last layout
    [B, HV, V, K] directly (no host-side transpose). Math copied verbatim from AITER's
    fused_sigmoid_gating_delta_rule_update_kernel; only the state offsets are k-last (k stride 1,
    v stride K) and the final state is written to a separate h_out buffer (input not mutated)."""
    i_v, i_nh = tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_q = q + (i_n * H + i_h) * K + o_k
    p_k = k + (i_n * H + i_h) * K + o_k
    p_v = v + (i_n * HV + i_hv) * V + o_v
    p_b = b + i_n * HV + i_hv
    p_o = o + (i_n * HV + i_hv) * V + o_v
    p_A_log = A_log + i_hv
    p_a = a + i_n * HV + i_hv
    p_dt_bias = dt_bias + i_hv
    # k-last state offsets: base + i_hv*V*K + k*1 + v*K
    state_off = i_n * HV * V * K + i_hv * V * K + o_k[:, None] + o_v[None, :] * K

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if HAS_STATE:
        b_h += tl.load(h_in + state_off, mask=mask_h, other=0.0).to(tl.float32)

    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    b_b = tl.load(p_b).to(tl.float32)
    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_a = tl.load(p_a).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(beta_x <= softplus_threshold,
                          (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)), x)
    b_g = -tl.exp(b_A_log) * softplus_x
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    b_q = b_q * scale
    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[:, None], 0)
    b_v *= b_beta
    b_h += b_k[:, None] * b_v[None, :]
    b_o = tl.sum(b_h * b_q[:, None], 0)
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)
    tl.store(h_out + state_off, b_h.to(h_out.dtype.element_ty), mask=mask_h)


def _run_klast(q, k, v, state, A_log, a, dt_bias, b, scale):
    B = q.shape[0]
    H = k.shape[-2]
    HV = v.shape[-2]
    K = q.shape[-1]
    V = v.shape[-1]
    o = torch.empty((B, 1, HV, V), dtype=torch.bfloat16, device=q.device)
    new_state = torch.empty((B, HV, V, K), dtype=torch.float32, device=q.device)
    BK = triton.next_power_of_2(K)
    BV = 64 if V % 64 == 0 else triton.next_power_of_2(V)
    grid = (1, triton.cdiv(V, BV), B * HV)
    has_state = state is not None
    h_in = state if has_state else new_state    # h_in unused when HAS_STATE=False
    _klast_decode_kernel[grid](
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b, o, h_in, new_state, has_state, scale,
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, num_warps=4,
    )
    return o, new_state


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

    if os.environ.get("GDN_DECODE_FUSEDOP") != "1" and os.environ.get("GDN_DECODE_RECURRENT") != "1":
        # Default: custom k-last decode kernel reading/writing state in contest [B,HV,V,K] directly
        # (no host-side input OR output transpose copy; single fused kernel). ~+63-67% vs the
        # AITER fused-op + output-view path; verify 54/54. GDN_DECODE_FUSEDOP=1 selects the AITER
        # fused-op path, GDN_DECODE_RECURRENT=1 the original recurrent wrapper.
        return _run_klast(q, k, v, state, A_log, a, dt_bias, b, scale)

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
    # new_state in contest k-last [B, HV, V, K]: return the transposed VIEW of the (in-place updated)
    # AITER state buffer instead of a `.contiguous()` copy. The fused op already updated init_state;
    # the values are identical and a strided view is correct, so this removes one full state-sized
    # copy (~half the host-side transpose traffic the profile flagged). Set GDN_DECODE_CONTIG_STATE=1
    # to force a contiguous new_state.
    new_state = init_state.transpose(-1, -2)                  # [B, HV, V, K] fp32 (view)
    if os.environ.get("GDN_DECODE_CONTIG_STATE") == "1":
        new_state = new_state.contiguous()
    return output, new_state
