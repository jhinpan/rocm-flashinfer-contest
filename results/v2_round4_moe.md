# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 21:26:30Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round4_moe --repeat-runs 3` |
| commit | `489cd75c57f9ece83b6569334085591be6276219` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-12-g489cd75` |
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

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v1 ms | cand/base | Δlat% | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.842 | 0.151 (0.150–0.153) | 0.152 (0.149–0.152) | 1.00× | +0.5% | **12.21×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.370 | 0.146 (0.145–0.146) | 0.145 (0.145–0.146) | 1.00× | -0.3% | **194.89×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 110.574 | 0.181 (0.179–0.187) | 0.181 (0.180–0.183) | 1.00× | -0.1% | **611.41×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.654 | 0.172 (0.170–0.240) | 0.180 (0.171–0.200) | 1.05× | +4.7% | **9.63×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.771 | 0.178 (0.178–0.180) | 0.176 (0.176–0.178) | 0.99× | -1.0% | **184.43×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1921.445 | 3.566 (3.565–3.577) | 3.562 (3.561–3.566) | 1.00× | -0.1% | **538.75×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.809 | 0.233 (0.231–0.234) | 0.814 (0.813–0.817) | 3.50× | +71.4% | **3.48×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.419 | 0.245 (0.244–0.247) | 0.834 (0.832–0.836) | 3.41× | +70.7% | **9.89×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.062 | 0.246 (0.246–0.247) | 0.836 (0.836–0.837) | 3.40× | +70.6% | **12.45×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.743 | 0.315 (0.314–0.316) | 0.314 (0.314–0.315) | 1.00× | -0.2% | **2.36×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.219 | 0.331 (0.330–0.333) | 0.331 (0.330–0.334) | 1.00× | -0.1% | **9.72×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.674 | 0.339 (0.338–0.341) | 0.341 (0.340–0.341) | 1.01× | +0.6% | **19.69×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.323 | 3.094 (3.093–3.115) | 2.896 (2.895–3.124) | 0.94× | -6.8% | **3.66×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.890 | 14.105 (14.101–14.114) | 12.802 (12.783–12.816) | 0.91× | -10.2% | **1.13×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.967 | 41.137 (41.104–41.204) | 22.282 (22.269–22.287) | 0.54× | -84.6% | **0.75×** | ✅ |
