# Round 6 report — dsa_topk_indexer verdict (planned levers tested) + moe provenance

**GPU:** AMD Instinct MI300X (gfx942) · **Baseline:** `baseline-v1^{}` = 74c0918 ·
**Harness:** `tools/run_benchmarks.py --repeat-runs 3` (provenance now stamps `sol_env`) ·
**Gate:** official `verify.py`.

## Verdict: `dsa_topk_indexer` → NO-GO for ≥20% (robust torch path shipped; both planned levers tested)

### Lever 1 — AITER `deepgemm_fp8_paged_mqa_logits`: FALSIFIED (fp8 dtype mismatch)
Committed numerics probe: gfx942 `aiter.dtypes.fp8 == torch.float8_e4m3fnuz`, and the op does
`kv_cache_fp8.view(dtypes.fp8)`. The contest KV cache is OCP `e4m3fn`. The same bytes decode ~2×
apart (e.g. `[56,64,60]` → fn `[1.0,2.0,1.5]` vs fnuz `[0.5,1.0,0.75]`), so the AITER op would misread
every KV value → wrong logits → wrong top-k. Consistent with GAP_REPORT §7. Unusable without
modifying AITER (not a portable solution-side option).

### Lever 2 — software-dequant fused-logits Triton kernel (DEC-4 safe path): tested, +5-8% but thinner margin
Committed env-gated candidate `_fused_logits` (`DSA_TOPK_FUSED=1`): computes `scores[B,N]` directly
from `q` and dequantized `k` (no `[B,H,N]` per-head logits; fp32 throughout).

| B | baseline-v1 ms | fused ms (min–max) | cand/base | Δlat | official verify | harness mr |
|---:|---:|---:|---:|---:|:--:|---:|
| 1 | 0.298 | 0.284 (0.282–0.286) | 1.05× | +4.9% | — | 1.000 |
| 14 | 0.314 | 0.295 (0.294–0.296) | 1.07× | +6.2% | — | 0.988 |
| 31 | 0.316 | 0.300 (0.295–0.301) | 1.05× | +4.8% | — | 0.989 |

- Official `verify.py --solution …` with `DSA_TOPK_FUSED=1` → **128/128 PASSED** (0.418 ms mean,
  num_trials tolerance absorbs tie-breaks).
- But the per-run matched ratio drops to ~0.988 (torch path = 1.000); the harness display flags it.
  For a **sub-20%** gain on a strict sorted-score kernel, that thinner correctness margin is not
  worth it (the Ultimate Goal forbids weakening correctness).

### Decision
Ship the **torch path** (default; mr 1.000; unchanged from baseline → no regression). Keep the fused
kernel behind `DSA_TOPK_FUSED=1` as tested, reproducible evidence (`results/v2_round5_topk.*`,
env-stamped). `dsa_topk_indexer` is already well-optimized (2.4–20× vs ref, ~0.3 ms); the remaining
cost (dequant + `bmm` + `torch.topk`) is irreducible portably, and the only ≥20%-class lever (AITER
deepgemm) is fp8-dtype-blocked. **Verdict: NO-GO for ≥20%.**

## moe_fp8 provenance fix (Codex round-5)
`tools/run_benchmarks.py` provenance now records a `sol_env` field (solution-control env vars).
`results/v2_round4_moe.{md,csv}` regenerated with `MOE_USE_FUSED=1` stamped in metadata
(`# sol_env=MOE_USE_FUSED=1`); the fused MoE path measures 0.94×/0.91×/0.54× vs baseline (slower) —
the moe ≥20% NO-GO evidence is now provenance-complete.

## Scoreboard
| Kernel | verdict |
|---|---|
| `dsa_sparse_attention` | IMPROVEMENT +70% (verify 23/23) |
| `moe_fp8` | NO-GO ≥20%; +10–17% block-broadcast fallback shipped (verify 19/19) |
| `dsa_topk_indexer` | NO-GO ≥20% (AITER fnuz-blocked; fused-logits +5-8% but thinner mr; torch shipped, verify 128/128) |
| `gdn_prefill` | pending |
| `gdn_decode` | pending |
