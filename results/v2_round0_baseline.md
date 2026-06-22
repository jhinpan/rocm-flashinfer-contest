# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 19:58:41Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round0_baseline` |
| commit | `7a177f90912781697745554a7e10231dba1d1761` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-3-g7a177f9` |
| baseline_commit | `74c0918` |
| dirty | `no` |
| gpu | `AMD Instinct MI300X` |
| device | `cuda:0` |
| torch | `2.9.1+rocm7.2.0.git7e1940d4` |
| hip | `7.2.26015-fc0010cf6a` |
| triton | `3.6.0+git42270451` |
| aiter | `n/a (/sgl-workspace/aiter/aiter@7d604afe5)` |
| dataset | `/sgl-workspace/workspace/mlsys2026-flashinfer-contest/data/flashinfer-trace` |
| fib_cache | `/tmp/fib_cache` |

| Kernel | Workload | reference ms | solution ms | baseline-v1 ms | cand/base | Δlat% | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `gdn…` | {'batch_size': 1} | 1.825 | 0.147 | 0.147 | 1.00× | +0.2% | **12.44×** | ✅ |
| `gdn…` | {'batch_size': 16} | 28.077 | 0.315 | 0.222 | 0.70× | -42.2% | **89.05×** | ✅ |
| `gdn…` | {'batch_size': 64} | 110.931 | 0.186 | 0.182 | 0.98× | -2.4% | **596.33×** | ✅ |
| `gdn…` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.569 | 0.168 | 0.166 | 0.99× | -0.7% | **9.36×** | ✅ |
| `gdn…` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.122 | 0.175 | 0.175 | 1.00× | -0.4% | **183.10×** | ✅ |
| `gdn…` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1831.328 | 3.460 | 3.477 | 1.00× | +0.5% | **529.29×** | ✅ |
| `dsa…` | {'num_tokens': 1, 'num_pages': 8462} | 0.803 | 0.807 | 0.808 | 1.00× | +0.1% | **1.00×** | ✅ |
| `dsa…` | {'num_tokens': 6, 'num_pages': 8462} | 2.435 | 0.835 | 0.834 | 1.00× | -0.1% | **2.91×** | ✅ |
| `dsa…` | {'num_tokens': 8, 'num_pages': 8462} | 3.089 | 0.831 | 0.829 | 1.00× | -0.2% | **3.72×** | ✅ |
| `dsa…` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.737 | 0.313 | 0.314 | 1.00× | +0.1% | **2.35×** | ✅ |
| `dsa…` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.241 | 0.330 | 0.331 | 1.00× | +0.1% | **9.81×** | ✅ |
| `dsa…` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.671 | 0.332 | 0.330 | 0.99× | -0.7% | **20.10×** | ✅ |
| `moe…` | {'seq_len': 1} | 11.409 | 2.990 | 2.977 | 1.00× | -0.4% | **3.82×** | ✅ |
| `moe…` | {'seq_len': 55} | 15.961 | 12.872 | 12.887 | 1.00× | +0.1% | **1.24×** | ✅ |
| `moe…` | {'seq_len': 14107} | 31.161 | 22.337 | 22.324 | 1.00× | -0.1% | **1.40×** | ✅ |
