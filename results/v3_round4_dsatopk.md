# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 11:00:28Z` |
| command | `python tools/run_benchmarks.py --out results/v3_round4_dsatopk --repeat-runs 3 --baseline-ref baseline-v2^{} --only dsa_topk_indexer` |
| commit | `448fec4a61e4200a990fa1254911d79aeae7c2a9` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-16-g448fec4` |
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
| sol_env | `(defaults)` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v2 ms | cand/baseline-v2 | Δ vs baseline-v2 | baseline-v1 ms | cand/v1 | Δ vs v1 | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.737 | 0.356 (0.351–0.356) | 0.299 (0.294–0.300) | 0.84× | -19.1% | 0.315 | 0.88× | -13.1% | **2.07×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.253 | 0.342 (0.340–0.343) | 0.307 (0.307–0.311) | 0.90× | -11.3% | 0.338 | 0.99× | -1.2% | **9.51×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.651 | 0.346 (0.344–0.352) | 0.321 (0.319–0.406) | 0.93× | -7.8% | 0.333 | 0.96× | -3.7% | **19.23×** | ✅ |
