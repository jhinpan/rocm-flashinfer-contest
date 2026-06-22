# Benchmark results — AMD Instinct MI300X

Speedup = torch-reference latency / solution latency (same reference on every platform).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 19:32:54Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round0_baseline` |
| commit | `d2f7a45c14504196b84af5645b53e2db6c6dbbfa` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-1-gd2f7a45-dirty` |
| baseline_tag | `15463ff` |
| dirty | `yes` |
| gpu | `AMD Instinct MI300X` |
| device | `cuda:0` |
| torch | `2.9.1+rocm7.2.0.git7e1940d4` |
| hip | `7.2.26015-fc0010cf6a` |
| triton | `3.6.0` |
| aiter | `n/a` |
| dataset | `/sgl-workspace/workspace/mlsys2026-flashinfer-contest/data/flashinfer-trace` |
| fib_cache | `/tmp/fib_cache` |

| Kernel | Workload | reference ms | solution ms | speedup | correctness |
|---|---|---:|---:|---:|:--:|
| `gdn…` | {'batch_size': 1} | 1.841 | 0.150 | **12.28×** | ✅ |
| `gdn…` | {'batch_size': 16} | 27.197 | 0.143 | **190.72×** | ✅ |
| `gdn…` | {'batch_size': 64} | 106.990 | 0.186 | **575.14×** | ✅ |
| `gdn…` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.588 | 0.172 | **9.21×** | ✅ |
| `gdn…` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 31.883 | 0.174 | **183.11×** | ✅ |
| `gdn…` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1825.474 | 3.473 | **525.63×** | ✅ |
| `dsa…` | {'num_tokens': 1, 'num_pages': 8462} | 0.797 | 0.805 | **0.99×** | ✅ |
| `dsa…` | {'num_tokens': 6, 'num_pages': 8462} | 2.399 | 0.830 | **2.89×** | ✅ |
| `dsa…` | {'num_tokens': 8, 'num_pages': 8462} | 3.061 | 0.829 | **3.69×** | ✅ |
| `dsa…` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.735 | 0.304 | **2.42×** | ✅ |
| `dsa…` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.189 | 0.322 | **9.91×** | ✅ |
| `dsa…` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.486 | 0.318 | **20.42×** | ✅ |
| `moe…` | {'seq_len': 1} | 11.342 | 2.939 | **3.86×** | ✅ |
| `moe…` | {'seq_len': 55} | 15.668 | 12.659 | **1.24×** | ✅ |
| `moe…` | {'seq_len': 14107} | 30.842 | 22.183 | **1.39×** | ✅ |
