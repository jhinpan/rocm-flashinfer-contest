# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 04:40:33Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round10_all --repeat-runs 3` |
| commit | `c15bd55522bd4c9af73429824195c7455400976b` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-23-gc15bd55` |
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
| `gdn_decode` | {'batch_size': 1} | 1.850 | 0.107 (0.106–0.108) | 0.148 (0.148–0.148) | 1.38× | +27.7% | **17.31×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.136 | 0.104 (0.103–0.104) | 0.144 (0.142–0.145) | 1.39× | +27.9% | **261.74×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 107.262 | 0.141 (0.140–0.143) | 0.183 (0.182–0.184) | 1.30× | +23.0% | **762.89×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.571 | 0.166 (0.164–0.167) | 0.165 (0.164–0.167) | 1.00× | -0.3% | **9.47×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 31.804 | 0.175 (0.174–0.175) | 0.174 (0.173–0.174) | 0.99× | -0.5% | **181.99×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1818.550 | 0.532 (0.521–0.537) | 3.456 (3.449–3.462) | 6.49× | +84.6% | **3416.98×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.796 | 0.232 (0.230–0.233) | 0.805 (0.805–0.806) | 3.47× | +71.2% | **3.43×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.398 | 0.245 (0.244–0.246) | 0.826 (0.824–0.827) | 3.38× | +70.4% | **9.80×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.038 | 0.246 (0.246–0.248) | 0.827 (0.825–0.827) | 3.36× | +70.3% | **12.36×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.735 | 0.292 (0.290–0.292) | 0.306 (0.306–0.307) | 1.05× | +4.6% | **2.52×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.209 | 0.307 (0.306–0.307) | 0.323 (0.321–0.326) | 1.05× | +5.1% | **10.47×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.489 | 0.307 (0.306–0.309) | 0.326 (0.325–0.329) | 1.06× | +5.7% | **21.11×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.305 | 2.591 (2.582–2.596) | 2.872 (2.871–2.877) | 1.11× | +9.8% | **4.36×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.816 | 10.521 (10.514–10.523) | 12.689 (12.669–12.696) | 1.21× | +17.1% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 31.120 | 19.128 (19.102–19.150) | 22.176 (22.174–22.208) | 1.16× | +13.7% | **1.63×** | ✅ |
