# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 22:37:07Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round5_topk --repeat-runs 3` |
| commit | `a452f05bc2a37aaf1654371c8ffdeb8f206bbf1c` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-16-ga452f05-dirty` |
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
| sol_env | `DSA_TOPK_FUSED=1` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v1 ms | cand/base | Δlat% | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.830 | 0.146 (0.144–0.148) | 0.146 (0.144–0.146) | 1.00× | -0.1% | **12.56×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.213 | 0.143 (0.143–0.143) | 0.142 (0.142–0.142) | 1.00× | -0.4% | **190.51×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 106.736 | 0.182 (0.182–0.187) | 0.183 (0.182–0.183) | 1.00× | +0.2% | **585.71×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.571 | 0.165 (0.164–0.165) | 0.165 (0.164–0.165) | 1.00× | +0.1% | **9.54×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 31.601 | 0.173 (0.173–0.174) | 0.173 (0.173–0.173) | 1.00× | +0.0% | **182.59×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1844.653 | 3.573 (3.571–3.581) | 3.573 (3.572–3.574) | 1.00× | -0.0% | **516.29×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.794 | 0.232 (0.230–0.233) | 0.804 (0.803–0.805) | 3.47× | +71.2% | **3.43×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.407 | 0.244 (0.243–0.245) | 0.825 (0.824–0.827) | 3.38× | +70.4% | **9.85×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.048 | 0.245 (0.245–0.246) | 0.826 (0.825–0.827) | 3.37× | +70.4% | **12.45×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.738 | 0.291 (0.290–0.291) | 0.307 (0.305–0.307) | 1.05× | +5.1% | **2.54×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.195 | 0.303 (0.302–0.304) | 0.324 (0.322–0.325) | 1.07× | +6.5% | **10.55×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.493 | 0.308 (0.307–0.310) | 0.328 (0.323–0.329) | 1.06× | +5.9% | **21.05×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.363 | 2.644 (2.640–2.649) | 2.934 (2.927–2.935) | 1.11× | +9.9% | **4.30×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.589 | 10.431 (10.412–10.438) | 12.699 (12.686–12.704) | 1.22× | +17.9% | **1.49×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 31.052 | 19.084 (19.076–19.091) | 22.205 (22.197–22.210) | 1.16× | +14.1% | **1.63×** | ✅ |
