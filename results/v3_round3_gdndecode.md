# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 10:12:41Z` |
| command | `python tools/run_benchmarks.py --out results/v3_round3_gdndecode --repeat-runs 3 --baseline-ref baseline-v2^{}` |
| commit | `2e1bed0dd3ffa59a5a5d31d98e8f525ce1deaa34` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-12-g2e1bed0` |
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
| `gdn_decode` | {'batch_size': 1} | 1.838 | 0.037 (0.036–0.037) | 0.106 (0.104–0.107) | 2.86× | +65.0% | 0.145 | 3.91× | +74.4% | **49.68×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.290 | 0.034 (0.034–0.034) | 0.103 (0.103–0.104) | 3.02× | +66.9% | 0.140 | 4.10× | +75.6% | **829.19×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 111.205 | 0.031 (0.029–0.033) | 0.139 (0.139–0.140) | 4.56× | +78.1% | 0.181 | 5.92× | +83.1% | **3642.59×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.566 | 0.167 (0.164–0.170) | 0.168 (0.167–0.169) | 1.01× | +0.5% | 0.166 | 0.99× | -0.7% | **9.35×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.195 | 0.174 (0.173–0.174) | 0.174 (0.173–0.176) | 1.00× | +0.0% | 0.174 | 1.00× | -0.1% | **185.25×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1878.258 | 0.528 (0.523–0.530) | 0.530 (0.524–0.531) | 1.00× | +0.4% | 3.512 | 6.66× | +85.0% | **3560.27×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.794 | 0.232 (0.231–0.233) | 0.231 (0.231–0.233) | 1.00× | -0.3% | 0.806 | 3.48× | +71.3% | **3.43×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.398 | 0.245 (0.244–0.246) | 0.244 (0.244–0.246) | 1.00× | -0.3% | 0.826 | 3.38× | +70.4% | **9.80×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.043 | 0.246 (0.245–0.247) | 0.246 (0.246–0.246) | 1.00× | +0.1% | 0.827 | 3.36× | +70.3% | **12.39×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.728 | 0.286 (0.286–0.291) | 0.289 (0.286–0.290) | 1.01× | +0.9% | 0.305 | 1.07× | +6.2% | **2.54×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.183 | 0.304 (0.302–0.305) | 0.302 (0.297–0.306) | 0.99× | -0.8% | 0.317 | 1.04× | +4.0% | **10.46×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.576 | 0.313 (0.306–0.314) | 0.313 (0.312–0.314) | 1.00× | +0.2% | 0.323 | 1.03× | +3.1% | **21.03×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.349 | 2.595 (2.587–2.606) | 2.597 (2.585–2.597) | 1.00× | +0.1% | 2.879 | 1.11× | +9.9% | **4.37×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.852 | 10.546 (10.511–10.549) | 10.529 (10.521–10.538) | 1.00× | -0.2% | 12.719 | 1.21× | +17.1% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.938 | 19.203 (19.197–19.234) | 19.268 (19.191–19.281) | 1.00× | +0.3% | 22.275 | 1.16× | +13.8% | **1.61×** | ✅ |
