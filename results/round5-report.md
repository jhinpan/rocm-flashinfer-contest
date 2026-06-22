# Round 5 report — dsa_topk_indexer verdict + moe evidence cleanup

**GPU:** AMD Instinct MI300X (gfx942) · **Baseline:** `baseline-v1^{}` = 74c0918 ·
**Harness:** `tools/run_benchmarks.py --repeat-runs 3` · **Gate:** official `verify.py` (sorted-score).

## Verdict: `dsa_topk_indexer` → NO-GO for ≥20% (baseline retained, no regression)

The v1 solution (vectorized fp8→fp32 dequant + `torch.bmm` logits + `torch.topk`) is already
well-optimized (2.4×–20× vs torch ref, ~0.30–0.33 ms) and has **no removable host-side waste**
analogous to dsa_sparse_attention's full-cache `torch.cat` or moe_fp8's `repeat_interleave`.

### Evidence (cheapest decisive experiment, AC-6)
- **Profile (rocprofv3 kernel-trace):** elementwise ~55% (B=1: 28.7%+26.6%; B=31: 36.7%+20.3%),
  `torch.topk` gather+sort 15–21% at large batch, `bmm` (Tensile GEMM) only 7–9%, reduce ~8%. The
  time is split across ~20 small but **necessary** ops (dequant, ReLU, weighted sum, mask, top-2048,
  index map) — not one dominant removable inefficiency.
- **Attempted fusion:** replaced the post-bmm `relu(per_head)*weights → sum over heads` (3 passes
  over the [B,H,N] logits) with a single fused Triton reduction kernel. Measured **slower**:
  -17.3% / -17.8% / -15.5% (B=1/14/31), and it perturbed tie-break selection (set-match 0.93–0.98).
  A naive Triton reduction does not beat torch's tuned elementwise/reduction kernels — the same
  pattern as the moe_fp8 fused-GEMM-vs-rocBLAS result (`BL-20260622-portable-gemm-cant-beat-rocblas`).
- The remaining big items (`torch.topk` of 2048, `torch.bmm`) are already optimized library calls;
  beating them portably is not feasible within this loop.

### Outcome
Keep the v1 `dsa_topk_indexer` (no change, no regression). Verdict: **NO-GO for ≥20%**. A
production segmented-top-2048 + fused-dequant-logits kernel could in principle help but carries the
rocBLAS/torch-matching + topk-tie-correctness risk demonstrated here and in moe_fp8 — a larger,
higher-risk effort, not a portable quick win.

## moe_fp8 evidence cleanup (Codex round-4)
- `results/round4-report.md` relabeled: `moe_fp8` = **NO-GO for ≥20%** with a marginal +10–17%
  block-broadcast fallback shipped (one verdict per kernel).
- `results/v2_round4_moe.{md,csv}` added (clean provenance, commit 489cd75, dirty=no): the
  `MOE_USE_FUSED=1` path measures **0.94×/0.91×/0.54×** vs baseline (slower) — backs the NO-GO.
- Stripped plan/process terms (`NO-GO`, `DEC-4`, result-file narration) from `solutions/moe_fp8`
  code comments.

## Scoreboard
| Kernel | verdict | result |
|---|---|---|
| `dsa_sparse_attention` | IMPROVEMENT | +70% vs baseline-v1 (verify 23/23) |
| `moe_fp8` | NO-GO ≥20% (marginal fallback) | +10–17% block-broadcast shipped (verify 19/19) |
| `dsa_topk_indexer` | NO-GO ≥20% | already optimized; no portable ≥20% lever |
| `gdn_prefill` | pending | round 6 |
| `gdn_decode` | pending | round 6 |
