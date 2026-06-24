# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 11:26:41Z` |
| command | `python tools/run_benchmarks.py --out results/v3_round5_moe --repeat-runs 3 --baseline-ref baseline-v2^{} --only moe_fp8` |
| commit | `dc49716bba7e2dd95777c002534a1c9e30e92780` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-20-gdc49716` |
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
| `moe_fp8` | {'seq_len': 1} | 11.384 | 2.584 (2.577–2.633) | 2.664 (2.657–2.665) | 1.03× | +3.0% | 2.939 | 1.14× | +12.1% | **4.41×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.632 | 9.313 (9.254–9.333) | 10.387 (10.381–10.391) | 1.12× | +10.3% | 12.691 | 1.36× | +26.6% | **1.68×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.832 | 14.518 (14.502–14.549) | 19.098 (19.097–19.115) | 1.32× | +24.0% | 22.187 | 1.53× | +34.6% | **2.12×** | ✅ |
