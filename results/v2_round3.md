# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 20:46:51Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round3 --repeat-runs 3` |
| commit | `ab0d85d56e967f3c089e852c53b8e97ebf8bf684` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-8-gab0d85d-dirty` |
| baseline_commit | `74c0918` |
| dirty | `yes` |
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
| `gdn_decode` | {'batch_size': 1} | 1.837 | 0.147 (0.147–0.147) | 0.147 (0.147–0.147) | 1.00× | +0.1% | **12.52×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.172 | 0.143 (0.143–0.145) | 0.143 (0.142–0.144) | 1.00× | -0.0% | **189.63×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 106.830 | 0.180 (0.180–0.186) | 0.181 (0.180–0.181) | 1.00× | +0.2% | **591.95×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.569 | 0.167 (0.167–0.168) | 0.166 (0.165–0.168) | 0.99× | -0.9% | **9.37×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.010 | 0.173 (0.173–0.174) | 0.174 (0.173–0.174) | 1.00× | +0.1% | **184.65×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1854.624 | 3.564 (3.559–3.575) | 3.563 (3.555–3.567) | 1.00× | -0.0% | **520.44×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.798 | 0.233 (0.231–0.234) | 0.807 (0.806–0.808) | 3.47× | +71.2% | **3.43×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.408 | 0.244 (0.244–0.245) | 0.827 (0.827–0.830) | 3.38× | +70.4% | **9.85×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.057 | 0.246 (0.245–0.246) | 0.828 (0.826–0.828) | 3.37× | +70.3% | **12.44×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.732 | 0.307 (0.305–0.308) | 0.306 (0.305–0.309) | 1.00× | -0.2% | **2.39×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.217 | 0.325 (0.323–0.326) | 0.326 (0.324–0.328) | 1.00× | +0.2% | **9.89×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.494 | 0.327 (0.326–0.328) | 0.330 (0.328–0.331) | 1.01× | +0.9% | **19.84×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.355 | 2.602 (2.586–2.617) | 2.895 (2.867–2.922) | 1.11× | +10.1% | **4.36×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.895 | 10.591 (10.588–10.608) | 12.695 (12.691–12.711) | 1.20× | +16.6% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.967 | 19.181 (19.164–19.185) | 22.263 (22.241–22.275) | 1.16× | +13.8% | **1.61×** | ✅ |
