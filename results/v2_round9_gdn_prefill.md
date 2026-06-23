# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 01:19:39Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round9_gdn_prefill --repeat-runs 3` |
| commit | `10d865cb2829c24bed8be1937b61301a25fb5468` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-21-g10d865c` |
| baseline_commit | `74c0918` |
| dirty | `no` |
| gpu | `AMD Instinct MI300X` |
| device | `cuda:0` |
| torch | `2.9.1+rocm7.2.0.git7e1940d4` |
| hip | `7.2.26015-fc0010cf6a` |
| triton | `3.6.0+git42270451` |
| aiter | `editable-source (/sgl-workspace/aiter/aiter@7d604afe5)` |
| dataset | `/sgl-workspace/workspace/mlsys2026-flashinfer-contest/data/flashinfer-trace` |
| fib_cache | `/tmp/fib_cache` |
| sol_env | `(defaults)` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v1 ms | cand/base | Δlat% | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.857 | 0.148 (0.147–0.149) | 0.149 (0.149–0.149) | 1.00× | +0.5% | **12.53×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.618 | 0.147 (0.146–0.147) | 0.147 (0.147–0.148) | 1.00× | +0.0% | **195.19×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 112.055 | 0.180 (0.180–0.186) | 0.181 (0.180–0.182) | 1.00× | +0.1% | **621.18×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.587 | 0.170 (0.169–0.171) | 0.168 (0.168–0.170) | 0.99× | -1.1% | **9.32×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.895 | 0.174 (0.174–0.175) | 0.174 (0.173–0.175) | 1.00× | +0.2% | **188.84×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1908.396 | 0.544 (0.534–0.553) | 3.507 (3.506–3.509) | 6.45× | +84.5% | **3508.36×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.787 | 0.232 (0.231–0.233) | 0.803 (0.802–0.805) | 3.47× | +71.1% | **3.40×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.336 | 0.244 (0.244–0.245) | 0.823 (0.822–0.826) | 3.37× | +70.3% | **9.56×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 2.969 | 0.246 (0.246–0.248) | 0.824 (0.823–0.825) | 3.35× | +70.1% | **12.06×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.723 | 0.295 (0.295–0.296) | 0.291 (0.290–0.292) | 0.99× | -1.2% | **2.45×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.106 | 0.306 (0.305–0.307) | 0.305 (0.305–0.306) | 1.00× | -0.5% | **10.15×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.266 | 0.311 (0.310–0.311) | 0.307 (0.307–0.307) | 0.99× | -1.3% | **20.14×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.264 | 2.545 (2.544–2.551) | 2.844 (2.838–2.844) | 1.12× | +10.5% | **4.43×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.430 | 10.254 (10.251–10.262) | 12.539 (12.532–12.562) | 1.22× | +18.2% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.642 | 19.069 (19.068–19.075) | 22.153 (22.139–22.155) | 1.16× | +13.9% | **1.61×** | ✅ |
