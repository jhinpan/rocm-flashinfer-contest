# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 08:19:26Z` |
| command | `python tools/run_benchmarks.py --out results/amd_mi300 --repeat-runs 3` |
| commit | `e792634c76626d523249b73b463ffaff32d0ff01` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-24-ge792634` |
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
| `gdn_decode` | {'batch_size': 1} | 1.852 | 0.109 (0.107–0.121) | 0.145 (0.144–0.145) | 1.33× | +24.8% | **17.03×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.971 | 0.103 (0.101–0.104) | 0.142 (0.141–0.142) | 1.38× | +27.4% | **272.11×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 110.431 | 0.141 (0.140–0.143) | 0.182 (0.181–0.183) | 1.29× | +22.6% | **784.53×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.553 | 0.167 (0.165–0.168) | 0.165 (0.165–0.166) | 0.99× | -1.2% | **9.30×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.341 | 0.174 (0.174–0.175) | 0.174 (0.173–0.174) | 1.00× | -0.1% | **185.66×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1873.470 | 0.535 (0.530–0.538) | 3.510 (3.509–3.511) | 6.57× | +84.8% | **3504.58×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.792 | 0.233 (0.231–0.233) | 0.805 (0.803–0.808) | 3.46× | +71.1% | **3.40×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.379 | 0.245 (0.245–0.246) | 0.826 (0.826–0.826) | 3.38× | +70.4% | **9.73×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.010 | 0.245 (0.245–0.246) | 0.826 (0.826–0.827) | 3.37× | +70.4% | **12.29×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.726 | 0.303 (0.302–0.305) | 0.301 (0.300–0.302) | 0.99× | -0.5% | **2.40×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.192 | 0.315 (0.314–0.316) | 0.312 (0.310–0.314) | 0.99× | -0.9% | **10.12×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.516 | 0.321 (0.318–0.321) | 0.318 (0.316–0.318) | 0.99× | -0.9% | **20.31×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.305 | 2.604 (2.578–2.642) | 2.864 (2.855–2.867) | 1.10× | +9.1% | **4.34×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.863 | 10.563 (10.542–10.586) | 12.722 (12.721–12.727) | 1.20× | +17.0% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.965 | 19.197 (19.118–19.204) | 22.211 (22.210–22.237) | 1.16× | +13.6% | **1.61×** | ✅ |
