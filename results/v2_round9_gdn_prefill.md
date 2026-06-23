# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 01:18:07Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round9_gdn_prefill --repeat-runs 3` |
| commit | `17829d2148b51bee8009e37cc6f90c9ebae97082` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-19-g17829d2-dirty` |
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
| sol_env | `(defaults)` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v1 ms | cand/base | Δlat% | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.837 | 0.147 (0.147–0.148) | 0.147 (0.147–0.148) | 1.00× | -0.0% | **12.46×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.033 | 0.143 (0.142–0.144) | 0.143 (0.141–0.143) | 1.00× | +0.1% | **189.38×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 106.326 | 0.181 (0.181–0.186) | 0.183 (0.182–0.183) | 1.01× | +0.9% | **587.40×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.555 | 0.166 (0.165–0.168) | 0.165 (0.165–0.165) | 0.99× | -0.8% | **9.36×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 31.499 | 0.173 (0.173–0.173) | 0.173 (0.173–0.173) | 1.00× | -0.0% | **181.99×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1830.056 | 0.534 (0.527–0.538) | 3.515 (3.514–3.515) | 6.58× | +84.8% | **3424.92×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.794 | 0.233 (0.232–0.234) | 0.805 (0.803–0.806) | 3.46× | +71.1% | **3.41×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.393 | 0.245 (0.245–0.247) | 0.827 (0.825–0.827) | 3.37× | +70.3% | **9.75×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.036 | 0.246 (0.246–0.249) | 0.826 (0.825–0.826) | 3.35× | +70.1% | **12.32×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.729 | 0.304 (0.304–0.305) | 0.304 (0.302–0.304) | 1.00× | +0.1% | **2.40×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.198 | 0.322 (0.322–0.323) | 0.319 (0.315–0.320) | 0.99× | -0.9% | **9.92×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.475 | 0.324 (0.323–0.326) | 0.321 (0.321–0.321) | 0.99× | -0.9% | **20.00×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.352 | 2.648 (2.638–2.653) | 2.944 (2.930–2.947) | 1.11× | +10.0% | **4.29×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.638 | 10.430 (10.424–10.448) | 12.678 (12.668–12.679) | 1.22× | +17.7% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.866 | 19.157 (19.157–19.166) | 22.247 (22.238–22.259) | 1.16× | +13.9% | **1.61×** | ✅ |
