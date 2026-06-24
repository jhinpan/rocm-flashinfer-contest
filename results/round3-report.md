# Round 3 report — moe_fp8 dequant optimization + harness hardening

**GPU:** AMD Instinct MI300X (gfx942) · **Peeled baseline:** `baseline-v1^{}` = 74c0918 ·
**Harness:** `tools/run_benchmarks.py --repeat-runs 3` (mutation-safe paired timing) ·
**Full table:** `results/v2_round3.md`.

## `moe_fp8` → PARTIAL / MARGINAL result (+10–17%, NOT a final verdict)

> This is a marginal result below the ≥20% headline bar, **not** a final AC-3 verdict. The final
> `moe_fp8` verdict (IMPROVEMENT ≥20% / NO-GO / BLOCKED) is pending the fused block-scale MoE path
> (round 4). The block-broadcast `_dequant_block()` change below is shipped as a safe, non-regressing
> partial speedup and the correctness-safe fallback.

| Workload | baseline-v1 ms | candidate ms (min–max) | cand/base | latency reduction | correctness |
|---|---:|---:|---:|---:|:--:|
| seq_len=1 | 2.939 | 2.650 (2.647–2.654) | 1.11× | **+9.8%** | ✅ |
| seq_len=55 | 12.624 | 10.458 (10.455–10.465) | 1.21× | **+17.2%** | ✅ |
| seq_len=14107 | 22.251 | 19.159 (19.157–19.161) | 1.16× | **+13.9%** | ✅ |

(Values from the committed clean-provenance snapshot `results/v2_round3.md`, commit b1406fc,
dirty=no.)

- **Correctness (authoritative):** `verify.py` → **19/19 PASSED** (loose tol atol=1, rtol=0.3,
  mr=0.9), 9.74 ms mean, mean speedup-vs-ref 1.4× → 1.85×.
- **DEC-1 status:** real, low-noise win (tight spread) but **below the ≥20% headline bar**. Shipped
  anyway because it is numerically identical to baseline (zero correctness risk, no regression) and
  reduces a dominant cost — and it does not preclude the deeper lever below. Recorded as a *partial*
  improvement, not a headline one.
- **No regression / no reward hacking:** generalizes across all 19 official workloads and all
  seq_len buckets; the change is a pure dequant-math refactor (no per-workload constants).

## What changed
`solutions/moe_fp8/main.py`: `_dequant_block()` replaces the per-expert
`scale.repeat_interleave(128,0).repeat_interleave(128,1)` weight dequant (which materialized the full
expanded [2I,H] and [H,I] scale tensors) with **block-broadcast** dequant
(`w.view(R/128,128,C/128,128) * scale[:,None,:,None]`). Numerically identical (proven: max diff 0.0),
but removes the large expanded-scale materialization that the rocprofv3 triage showed dominates
(~75% elementwise at seq55).

## Why not ≥20%: next lever (future round)
The remaining cost is still **materializing the dequantized bf16 weights** per active expert before
the GEMM (O(active_experts × weight_size) memory traffic). Reaching ≥20% needs a **fused block-scale
MoE GEMM** that reads fp8 weights + scales and dequants inside the GEMM tile loop (AITER
`moe_op_gemm_a8w8_blockscale` / `fmoe_fp8_blockscale_g1u1`), avoiding the full bf16 weight
materialization. That is higher correctness-risk (layout/SwiGLU-variant matching per GAP_REPORT §2.E)
and is the moe_fp8 follow-up. fp8 native `fnuz` MMA remains deprioritized (GEMM is only 3–18% of time).

## Harness hardening (Codex round-2 blockers, all resolved)
- **Mutation-safe paired timing:** each candidate/baseline timing call runs on a fresh
  `clone_args(inp)` cloned *outside* the timed region; `mutates_inputs()` warns if a candidate writes
  its inputs.
- **`--candidate-ref`:** loads the candidate from a git ref; the unchanged-code self-check
  (`--candidate-ref baseline-v1^{}`) now shows all kernels cvb ≈ 1.00× (the round-1 `gdn_decode` B=16
  0.70× artifact is gone) — see regenerated `results/v2_round0_baseline.md`.
- **`results/amd_mi300.{md,csv}`** reverted to coherent v1 (md/csv agree); v2 evidence lives in the
  per-round `results/v2_round*.md` + reports until task10 finalization.

## Local-verify caveat (finding)
`tools/local_verify.py` is **not reliable for moe_fp8**: the baseline itself "fails" it (mr ≈ 0.88,
borderline vs the 0.9 threshold) while passing the authoritative `verify.py` 19/19. Only `verify.py`
is authoritative for this kernel; local_verify remains fine for the others.
