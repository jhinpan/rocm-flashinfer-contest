# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 20:26:01Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round2 --repeat-runs 3` |
| commit | `d3864a93a0c1b27a7c68a332ac4fb94a039d45e3` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-6-gd3864a9` |
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
| `gdn_decode` | {'batch_size': 1} | 1.842 | 0.152 (0.152–0.153) | 0.152 (0.152–0.153) | 1.00× | -0.1% | **12.08×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.162 | 0.144 (0.144–0.145) | 0.144 (0.144–0.150) | 1.00× | -0.1% | **195.13×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 110.952 | 0.183 (0.182–0.186) | 0.181 (0.181–0.183) | 0.99× | -1.0% | **606.97×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.579 | 0.172 (0.171–0.175) | 0.172 (0.171–0.174) | 1.00× | +0.0% | **9.19×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 33.039 | 0.176 (0.175–0.178) | 0.177 (0.175–0.178) | 1.00× | +0.3% | **187.55×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1844.736 | 3.468 (3.448–3.471) | 3.471 (3.469–3.472) | 1.00× | +0.1% | **531.86×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.792 | 0.232 (0.230–0.234) | 0.807 (0.807–0.809) | 3.48× | +71.3% | **3.42×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.362 | 0.245 (0.245–0.246) | 0.825 (0.824–0.826) | 3.37× | +70.3% | **9.65×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 2.978 | 0.245 (0.245–0.247) | 0.827 (0.827–0.827) | 3.38× | +70.4% | **12.16×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.733 | 0.305 (0.304–0.306) | 0.305 (0.304–0.306) | 1.00× | -0.0% | **2.41×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.168 | 0.321 (0.320–0.323) | 0.321 (0.320–0.321) | 1.00× | -0.1% | **9.87×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.356 | 0.319 (0.318–0.323) | 0.319 (0.317–0.321) | 1.00× | -0.2% | **19.90×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.329 | 2.855 (2.853–2.860) | 2.845 (2.840–2.856) | 1.00× | -0.3% | **3.97×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.403 | 12.584 (12.572–12.592) | 12.606 (12.596–12.608) | 1.00× | +0.2% | **1.22×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.699 | 22.114 (22.104–22.135) | 22.119 (22.113–22.140) | 1.00× | +0.0% | **1.39×** | ✅ |
