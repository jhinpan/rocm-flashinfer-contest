# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 10:13:53Z` |
| command | `python tools/run_benchmarks.py --out results/v3_baseline_v2 --repeat-runs 3 --baseline-ref baseline-v2^{}` |
| commit | `2e1bed0dd3ffa59a5a5d31d98e8f525ce1deaa34` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-12-g2e1bed0` |
| primary_baseline_ref | `baseline-v2^{}` |
| primary_baseline_commit | `08d4780` |
| baseline_v1_commit | `74c0918` |
| dirty | `yes` |
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
| `gdn_decode` | {'batch_size': 1} | 1.818 | 0.037 (0.036–0.037) | 0.108 (0.106–0.109) | 2.92× | +65.8% | 0.146 | 3.97× | +74.8% | **49.36×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.144 | 0.033 (0.033–0.033) | 0.104 (0.102–0.104) | 3.16× | +68.4% | 0.142 | 4.34× | +77.0% | **858.73×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 110.626 | 0.030 (0.030–0.033) | 0.139 (0.139–0.139) | 4.67× | +78.6% | 0.180 | 6.05× | +83.5% | **3706.38×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.554 | 0.168 (0.166–0.171) | 0.167 (0.166–0.169) | 1.00× | -0.4% | 0.166 | 0.99× | -0.8% | **9.27×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.227 | 0.173 (0.173–0.175) | 0.173 (0.173–0.174) | 1.00× | +0.1% | 0.173 | 1.00× | -0.1% | **186.38×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1881.360 | 0.529 (0.524–0.530) | 0.529 (0.524–0.530) | 1.00× | +0.0% | 3.512 | 6.64× | +84.9% | **3556.43×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.794 | 0.231 (0.231–0.234) | 0.231 (0.230–0.233) | 1.00× | -0.1% | 0.808 | 3.50× | +71.4% | **3.43×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.385 | 0.245 (0.244–0.246) | 0.244 (0.244–0.247) | 1.00× | -0.1% | 0.826 | 3.38× | +70.4% | **9.75×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.012 | 0.246 (0.245–0.247) | 0.245 (0.245–0.246) | 1.00× | -0.1% | 0.827 | 3.37× | +70.3% | **12.26×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.728 | 0.283 (0.282–0.287) | 0.283 (0.281–0.284) | 1.00× | +0.1% | 0.302 | 1.07× | +6.4% | **2.57×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.177 | 0.300 (0.299–0.303) | 0.301 (0.293–0.303) | 1.00× | +0.4% | 0.317 | 1.06× | +5.3% | **10.60×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.552 | 0.309 (0.305–0.376) | 0.312 (0.310–0.464) | 1.01× | +1.1% | 0.319 | 1.03× | +3.2% | **21.22×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.300 | 2.568 (2.566–2.580) | 2.572 (2.565–2.588) | 1.00× | +0.2% | 2.863 | 1.11× | +10.3% | **4.40×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.869 | 10.530 (10.505–10.543) | 10.530 (10.526–10.538) | 1.00× | +0.0% | 12.715 | 1.21× | +17.2% | **1.51×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 31.227 | 19.224 (19.221–19.247) | 19.214 (19.208–19.239) | 1.00× | -0.1% | 22.262 | 1.16× | +13.6% | **1.62×** | ✅ |
