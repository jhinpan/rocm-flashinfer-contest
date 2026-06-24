# dsa_topk_indexer (v3 round 4) — verify transcripts + lever appendix

Evidence packet backing the round-4 AC-T2 verdict (robustness IMPROVEMENT + perf NO-GO). Benchmark
artifacts: `results/v3_round4_dsatopk.{md,csv}` (default exact path) and
`results/v3_round4_dsatopk_fast.{md,csv}` (`DSA_TOPK_FAST=1`). GPU: AMD Instinct MI300X (gfx942).

## Official verify.py — both paths pass 128/128

Default (exact `torch.bmm`, the shipped default):
```
$ python verify.py --solution solutions/dsa_topk_indexer/solution.json --dataset <flashinfer-trace>
solution:   dsa_topk_indexer_rocm_v1
definition: dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
passed:     128/128
latency:    0.443426 ms mean
speedup:    7.0186x mean
```

Fast packed-page scorer (`DSA_TOPK_FAST=1`):
```
$ DSA_TOPK_FAST=1 python verify.py --solution solutions/dsa_topk_indexer/solution.json \
      --dataset <flashinfer-trace>
solution:   dsa_topk_indexer_rocm_v1
definition: dsa_topk_indexer_fp8_h64_d128_topk2048_ps64
passed:     128/128
latency:    0.386356 ms mean
speedup:    8.1830x mean
```

Both are **contest-correct (128/128)**. The fast path is the faster of the two by the official
mean (8.18× vs 7.02× ref). The difference between them is only our stricter per-run `mr ≥ 0.999`
display gate (DEC-2), which the fast path misses (mr ~0.99) and the default meets (mr 1.000).

## Paired benchmark vs baseline-v2 (`--repeat-runs 3`, dirty=no)

| Path | B=1 | B=14 | B=31 | sol_env |
|---|---|---|---|---|
| default exact torch.bmm | 0.84× (−19%) | 0.90× (−11%) | 0.93× (−8%) | (defaults) |
| **`DSA_TOPK_FAST=1`** | **1.11× (+9.9%)** | **1.27× (+21.3%)** | **1.27× (+21.5%)** | `DSA_TOPK_FAST=1` |

vs baseline/v1 the fast path is 1.17× / 1.38× / 1.36×. Harness mr: default 1.000/1.000/1.000; fast
1.000/0.991/0.996 (improved over baseline-v2's own 0.988/0.992, still < 0.999).

## Rejected lever appendix (why ">+8% AND mr≥0.999" is infeasible)

All measured with `python tools/run_benchmarks.py --repeat-runs 3 --baseline-ref baseline-v2^{}
--only dsa_topk_indexer`, MI300X. Latency reduction vs baseline-v2 (negative = slower):

| Lever | B=1 | B=14 | B=31 | mr | Why rejected |
|---|---|---|---|---|---|
| candidate-superset re-score (fast top-2560 → exact `torch.bmm` → top-2048) | −91.2% | −74.8% | −71.7% | (n/a) | topk≈N for these workloads (N≈2560–2752), so re-scoring 2560 candidates ≈ scoring all N — erases the win and adds a gather. |
| fused gather+dequant kernel + `torch.bmm` (remove the page-gather copy, keep exact bmm) | −34.6% | −17.0% | −14.3% | 1.000 | a hand-rolled fused gather+decode is slower than torch's optimized `index_select` gather; the bmm still materializes `k_deq`. |
| packed-page `tl.dot` scorer (`DSA_TOPK_FAST`) | +9.9% | +21.3% | +21.5% | 0.99 | fast + contest-correct, but mr < 0.999 (intrinsic: `tl.dot` tiling ≠ `torch.bmm`). |

Micro-test (`/tmp/dottest.py`): `tl.dot(input_precision="ieee")` vs `torch.matmul` maxrel **6e-4**;
`tf32` maxrel **1.6**; `tf32x3` failed to compile. Confirms `ieee` is true fp32 yet still cannot
bit-match torch's GEMM tiling, which at these inputs' extreme dynamic range (per-token scales ~1e30,
signed weights → cancellation) mis-ranks ~0.5–1% of near-tie boundary tokens.

**Conclusion.** Only `torch.bmm` reaches mr 1.0, and it needs the `k_deq` materialization that is the
dominant cost the fast kernel removes — so >+8% and mr≥0.999 are mutually exclusive here. Per DEC-2
(prioritize robustness), the default is the exact path; the fast path stays behind
`DSA_TOPK_FAST=1`.
