# v3 Round 4 — dsa_topk_indexer verdict + scaffold debt closed

**GPU:** AMD Instinct MI300X (gfx942) · **Base:** `baseline-v2^{}` = 08d4780 (primary, no-regression),
also vs `baseline/v1` = 74c0918 · **Harness:** `tools/run_benchmarks.py --baseline-ref baseline-v2^{}
--repeat-runs 3 --only <kernel>`.

## Mainline: task4 `dsa_topk_indexer` — VERDICT: robustness IMPROVEMENT + perf NO-GO

The plan target (AC-T2) was **>+8% vs baseline-v2 AND per-run mr ≥ 0.999** (DEC-2 hard gate). These
two are **mutually exclusive** for this kernel, established by measurement:

| Path | vs baseline-v2 (B=1/14/31) | vs v1 | mr (harness) | official verify.py |
|---|---|---|---|---|
| **default: exact `torch.bmm`** (gather→dequant→bmm→relu→wsum) | 0.84× / 0.90× / 0.93× (−19/−11/−8%) | 0.88/0.99/0.96× | **1.000** | **128/128** (7.02× ref) |
| `DSA_TOPK_FAST=1`: packed-page Triton scoring | **1.17× / 1.21× / 1.24× (+14/+18/+19%)** | 1.22/1.29/1.30× | 0.990–0.996 | **128/128** (8.18× ref) |
| baseline-v2 (v2 fused `tl.dot`) | 1.00× | — | 0.988 / 0.992 | 128/128 |

**Why mutually exclusive.** The official evaluator and the reference both score with fp32
`torch.bmm`. Only `torch.bmm` reproduces that selection bit-for-bit → mr 1.0. A Triton `tl.dot`
kernel (any precision mode) tiles the reduction differently from torch's GEMM; at these inputs'
extreme dynamic range (per-token scales ~1e30, negative learned weights → catastrophic
cancellation) that ~6e-4 dot difference mis-ranks ~0.5–1% of near-tie boundary tokens (mr ~0.99).
This is intrinsic — confirmed with a micro-test (`ieee` dot vs `torch`: maxrel 6e-4; `tf32`: 1.6) and
by the fact that even the bit-exact torch path momentarily flickered to mr 0.996 once under load.
`torch.bmm` requires the materialized `k_deq[B,N,D]` — the dominant cost the fast kernel removes — so
the exact path can't recover the fast kernel's +14–19%.

Levers genuinely attempted (all measured, not asserted):
- **Packed-page fused scoring** (read fp8 via `block_table` in-kernel, no `k_deq`): +14–19%, mr 0.99.
  NaN-preserving relu (`tl.where(x!=x, x, max(x,0))`) and `input_precision="ieee"` *improved* mr over
  v2 (0.988→0.990, 0.992→0.996) but not to 0.999.
- **Scale-after-dot identity** `dot(q, fp8·scale)=scale·dot(q,fp8)` (per-token positive scale): used.
- **Boundary repair** (fast top-2560 → exact `torch.bmm` re-score → top-2048): regressed to 0.52–0.58×
  because for these workloads N≈2560–2752, so re-scoring 2560 candidates ≈ scoring all N. Dead end.
- **Fused gather+dequant kernel + bmm** (remove the page-gather copy, keep exact bmm): 0.66–0.86×,
  slower than torch's optimized gather. Dead end. Both dead ends removed from the shipped code.

**Decision (per DEC-2).** "prioritize robustness ... otherwise revert to the torch path." The default
is the exact `torch.bmm` path: **mr 1.000, verify.py 128/128**, restoring the robustness the v2 fused
path lost (0.988/0.992). It costs 8–19% vs baseline-v2 (mostly launch overhead at tiny B=1; ~−8% at
B=31). The fast kernel is kept behind `DSA_TOPK_FAST=1` as a documented speed/accuracy option — it is
contest-correct (verify.py 128/128, 8.18× ref) and +14–19%, gated out only by the stricter 0.999 bar.

> **Flagged for the user:** if you prefer the +14–19% speed over the mr≥0.999 robustness bar (the
> fast path still passes the official evaluator 128/128), set `DSA_TOPK_FAST=1` to make it the
> default. The robustness-first default was chosen per DEC-2.

Evidence: `results/v3_round4_dsatopk.{md,csv}` (default, dirty=no); verify logs default 128/128 @7.02×,
fast 128/128 @8.18×.

## Blocking scaffold debt from round 3 — all closed
- **task1**: `results/v3_baseline_v2.*` regenerated from a clean tree, `dirty=no` (commit 625380a).
- **task2**: `results/round0-v3-triage.md` Round-1 stage tables regenerated verbatim from
  `tools/summarize_rocprof_trace.py` over all 7 `/tmp/v3prof` traces, exact schema
  `bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class`, commands + raw paths adjacent
  (commit 6a90062).
- **gdn transcript**: `results/v3_round3_gdndecode_verify.md` — in-process 54/54 (atol=rtol=1e-2,
  mr=1.0), verify.py --fast 2/2 (58.2× ref), inputs not mutated.

## Queued cleanups done
- gdn_decode module docstring + `solution.json` now describe the shipped k-last default; summarizer
  docstring documents `bucket`; unused `GDN_DECODE_KLAST` replaced by the real `GDN_DECODE_FUSEDOP`
  in `sol_env`; `run_benchmarks.py` gained a `--only <dir>` filter (commit 6a90062 / 448fec4).

## task7 — untouched kernels non-regressed (vs baseline-v2)
`gdn_prefill` 0.99/1.00/0.99× (PASS), `dsa_sparse_attention` 1.00/1.00/1.00× (PASS). Code unchanged.

## Status vs v3 plan
- gdn_decode: **IMPROVEMENT** (+65–78% vs baseline-v2, verified). dsa_topk: robustness restored
  (mr→1.0), **perf NO-GO** on >+8%-at-mr≥0.999 (evidence-backed). moe_fp8 (task5/6): next round.
  Finalize (task8, full verify.py all 5 + unified table): after moe_fp8.
