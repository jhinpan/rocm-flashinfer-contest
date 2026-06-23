# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 09:51:37Z` |
| command | `python tools/run_benchmarks.py --out results/v3_round2_gdndecode --repeat-runs 3 --baseline-ref baseline-v2^{}` |
| commit | `1813571c867e46281e1a55613b35781cf91b3fce` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-10-g1813571` |
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
| fib_cache | `/tmp/fib_cache` |
| sol_env | `(defaults)` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v2 ms | cand/baseline-v2 | Δ vs baseline-v2 | baseline-v1 ms | cand/v1 | Δ vs v1 | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.858 | 0.100 (0.100–0.102) | 0.109 (0.109–0.109) | 1.09× | +8.1% | 0.148 | 1.48× | +32.4% | **18.53×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.357 | 0.096 (0.096–0.098) | 0.105 (0.105–0.106) | 1.09× | +8.1% | 0.144 | 1.49× | +32.8% | **283.49×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 107.820 | 0.090 (0.089–0.092) | 0.139 (0.139–0.141) | 1.55× | +35.7% | 0.183 | 2.04× | +51.0% | **1204.65×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.573 | 0.170 (0.168–0.171) | 0.170 (0.168–0.170) | 0.99× | -0.5% | 0.168 | 0.98× | -1.6% | **9.23×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 31.804 | 0.174 (0.174–0.175) | 0.175 (0.174–0.175) | 1.00× | +0.1% | 0.174 | 1.00× | -0.1% | **182.36×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1844.805 | 0.540 (0.533–0.543) | 0.541 (0.536–0.544) | 1.00× | +0.3% | 3.511 | 6.51× | +84.6% | **3419.42×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.798 | 0.232 (0.231–0.234) | 0.231 (0.231–0.233) | 1.00× | -0.1% | 0.809 | 3.50× | +71.4% | **3.45×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.415 | 0.245 (0.245–0.246) | 0.245 (0.244–0.247) | 1.00× | +0.1% | 0.830 | 3.39× | +70.5% | **9.86×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.067 | 0.245 (0.245–0.246) | 0.245 (0.245–0.245) | 1.00× | -0.1% | 0.829 | 3.38× | +70.4% | **12.51×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.738 | 0.293 (0.291–0.296) | 0.292 (0.292–0.293) | 1.00× | -0.4% | 0.307 | 1.05× | +4.7% | **2.52×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.215 | 0.309 (0.308–0.311) | 0.309 (0.305–0.309) | 1.00× | -0.1% | 0.324 | 1.05× | +4.7% | **10.41×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.521 | 0.314 (0.313–0.315) | 0.312 (0.310–0.314) | 0.99× | -0.7% | 0.328 | 1.05× | +4.4% | **20.77×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.380 | 2.654 (2.649–2.655) | 2.657 (2.651–2.660) | 1.00× | +0.1% | 2.939 | 1.11× | +9.7% | **4.29×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.667 | 10.450 (10.433–10.463) | 10.447 (10.437–10.447) | 1.00× | -0.0% | 12.650 | 1.21× | +17.4% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.918 | 19.171 (19.160–19.179) | 19.169 (19.157–19.179) | 1.00× | -0.0% | 22.234 | 1.16× | +13.8% | **1.61×** | ✅ |
