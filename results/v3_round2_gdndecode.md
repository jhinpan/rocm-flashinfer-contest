# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 09:47:15Z` |
| command | `python tools/run_benchmarks.py --out results/v3_round2_gdndecode --repeat-runs 3 --baseline-ref baseline-v2^{}` |
| commit | `e150f36dcebcc3358a197151967f56ebbba27db6` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-7-ge150f36-dirty` |
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
| `gdn_decode` | {'batch_size': 1} | 1.836 | 0.099 (0.097–0.100) | 0.107 (0.106–0.107) | 1.08× | +7.3% | 0.147 | 1.49× | +32.8% | **18.55×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.079 | 0.094 (0.094–0.096) | 0.103 (0.103–0.103) | 1.09× | +8.0% | 0.144 | 1.52× | +34.3% | **286.69×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 107.031 | 0.086 (0.085–0.089) | 0.139 (0.139–0.140) | 1.61× | +38.0% | 0.184 | 2.13× | +53.1% | **1239.11×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.581 | 0.166 (0.165–0.167) | 0.163 (0.163–0.165) | 0.98× | -1.8% | 0.164 | 0.99× | -1.3% | **9.53×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 31.725 | 0.176 (0.175–0.176) | 0.176 (0.175–0.176) | 1.00× | +0.0% | 0.175 | 1.00× | -0.4% | **180.58×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1841.410 | 0.537 (0.530–0.541) | 0.536 (0.532–0.541) | 1.00× | -0.1% | 3.510 | 6.54× | +84.7% | **3430.21×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.797 | 0.231 (0.231–0.233) | 0.232 (0.231–0.234) | 1.00× | +0.3% | 0.805 | 3.48× | +71.3% | **3.44×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.405 | 0.245 (0.245–0.246) | 0.245 (0.244–0.247) | 1.00× | -0.0% | 0.828 | 3.38× | +70.4% | **9.82×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.066 | 0.245 (0.245–0.247) | 0.245 (0.245–0.246) | 1.00× | +0.0% | 0.825 | 3.36× | +70.3% | **12.50×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.736 | 0.292 (0.290–0.294) | 0.290 (0.287–0.292) | 0.99× | -0.6% | 0.308 | 1.06× | +5.4% | **2.52×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.211 | 0.302 (0.299–0.304) | 0.302 (0.298–0.302) | 1.00× | +0.1% | 0.326 | 1.08× | +7.4% | **10.63×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.536 | 0.308 (0.305–0.308) | 0.307 (0.306–0.308) | 1.00× | -0.2% | 0.325 | 1.06× | +5.4% | **21.25×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.374 | 2.658 (2.655–2.666) | 2.654 (2.653–2.655) | 1.00× | -0.1% | 2.944 | 1.11× | +9.7% | **4.28×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.579 | 10.436 (10.427–10.449) | 10.441 (10.432–10.446) | 1.00× | +0.0% | 12.668 | 1.21× | +17.6% | **1.49×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.946 | 19.185 (19.171–19.197) | 19.190 (19.185–19.194) | 1.00× | +0.0% | 22.252 | 1.16× | +13.8% | **1.61×** | ✅ |
