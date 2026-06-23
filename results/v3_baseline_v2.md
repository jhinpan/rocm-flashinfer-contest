# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 10:34:07Z` |
| command | `python tools/run_benchmarks.py --out results/v3_baseline_v2 --repeat-runs 3 --baseline-ref baseline-v2^{}` |
| commit | `6a9006255403e721e6eec1b45aa247fa9ce1b6a1` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-14-g6a90062` |
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
| `gdn_decode` | {'batch_size': 1} | 1.832 | 0.037 (0.037–0.040) | 0.108 (0.108–0.109) | 2.91× | +65.7% | 0.145 | 3.91× | +74.4% | **49.30×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 26.805 | 0.034 (0.034–0.035) | 0.105 (0.104–0.105) | 3.07× | +67.4% | 0.141 | 4.12× | +75.7% | **783.83×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 105.477 | 0.030 (0.030–0.033) | 0.139 (0.139–0.140) | 4.67× | +78.6% | 0.181 | 6.06× | +83.5% | **3533.80×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.533 | 0.167 (0.166–0.168) | 0.166 (0.165–0.166) | 0.99× | -0.9% | 0.164 | 0.98× | -1.8% | **9.16×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 31.066 | 0.174 (0.174–0.175) | 0.174 (0.174–0.174) | 1.00× | -0.2% | 0.174 | 1.00× | -0.3% | **178.13×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1813.901 | 0.537 (0.534–0.546) | 0.535 (0.534–0.543) | 1.00× | -0.4% | 3.511 | 6.54× | +84.7% | **3377.21×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.791 | 0.232 (0.231–0.233) | 0.231 (0.231–0.233) | 1.00× | -0.1% | 0.807 | 3.48× | +71.3% | **3.41×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.351 | 0.245 (0.244–0.246) | 0.245 (0.245–0.247) | 1.00× | +0.0% | 0.828 | 3.38× | +70.4% | **9.61×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 2.973 | 0.245 (0.245–0.247) | 0.246 (0.245–0.246) | 1.00× | +0.1% | 0.826 | 3.37× | +70.3% | **12.11×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.728 | 0.287 (0.286–0.288) | 0.287 (0.286–0.288) | 1.00× | +0.1% | 0.300 | 1.05× | +4.4% | **2.54×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.138 | 0.297 (0.295–0.299) | 0.297 (0.294–0.297) | 1.00× | +0.1% | 0.311 | 1.05× | +4.6% | **10.58×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.349 | 0.302 (0.300–0.302) | 0.304 (0.298–0.306) | 1.01× | +0.6% | 0.317 | 1.05× | +4.9% | **21.05×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.288 | 2.560 (2.552–2.561) | 2.553 (2.550–2.558) | 1.00× | -0.3% | 2.843 | 1.11× | +10.0% | **4.41×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.505 | 10.295 (10.295–10.300) | 10.306 (10.298–10.311) | 1.00× | +0.1% | 12.544 | 1.22× | +17.9% | **1.51×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.776 | 19.023 (19.009–19.045) | 19.031 (19.006–19.039) | 1.00× | +0.0% | 22.106 | 1.16× | +13.9% | **1.62×** | ✅ |
