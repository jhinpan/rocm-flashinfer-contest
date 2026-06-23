# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 11:19:15Z` |
| command | `python tools/run_benchmarks.py --out results/v3_round4_dsatopk_fast --repeat-runs 3 --baseline-ref baseline-v2^{} --only dsa_topk_indexer` |
| commit | `83917e0eb7b4c987456839715912d81c56670609` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-18-g83917e0` |
| primary_baseline_ref | `baseline-v2^{}` |
| primary_baseline_commit | `08d4780` |
| baseline_v1_commit | `74c0918` |
| dirty | `no` |
| gpu | `AMD Instinct MI300X` |
| device | `cuda:0` |
| torch | `2.9.1+rocm7.2.0.git7e1940d4` |
| hip | `7.2.26015-fc0010cf6a` |
| triton | `3.6.0+git42270451` |
| aiter | `editable-source (/sgl-workspace/aiter/aiter@7d604afe5)` |
| dataset | `/sgl-workspace/workspace/mlsys2026-flashinfer-contest/data/flashinfer-trace` |
| fib_cache | `n/a` |
| sol_env | `DSA_TOPK_FAST=1` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v2 ms | cand/baseline-v2 | Δ vs baseline-v2 | baseline-v1 ms | cand/v1 | Δ vs v1 | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.740 | 0.264 (0.263–0.264) | 0.292 (0.292–0.293) | 1.11× | +9.9% | 0.308 | 1.17× | +14.5% | **2.81×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.229 | 0.238 (0.238–0.241) | 0.303 (0.302–0.303) | 1.27× | +21.3% | 0.328 | 1.38× | +27.3% | **13.56×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.517 | 0.244 (0.243–0.244) | 0.311 (0.309–0.311) | 1.27× | +21.5% | 0.331 | 1.36× | +26.2% | **26.73×** | ❌ |
