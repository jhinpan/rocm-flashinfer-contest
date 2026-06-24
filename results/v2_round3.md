# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 20:48:50Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round3 --repeat-runs 3` |
| commit | `b1406fc2d241a691b84dd12bf1f3e7adc82981d7` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-9-gb1406fc` |
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
| `gdn_decode` | {'batch_size': 1} | 1.848 | 0.147 (0.147–0.148) | 0.147 (0.146–0.147) | 1.00× | -0.2% | **12.57×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.267 | 0.144 (0.142–0.145) | 0.144 (0.143–0.144) | 1.00× | +0.2% | **189.90×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 106.955 | 0.182 (0.181–0.188) | 0.183 (0.182–0.183) | 1.00× | +0.4% | **587.36×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.566 | 0.166 (0.165–0.166) | 0.167 (0.164–0.168) | 1.01× | +1.0% | **9.46×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 31.712 | 0.174 (0.174–0.175) | 0.174 (0.174–0.174) | 1.00× | -0.2% | **181.84×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1846.718 | 3.564 (3.563–3.571) | 3.563 (3.559–3.563) | 1.00× | -0.0% | **518.13×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.800 | 0.231 (0.230–0.232) | 0.806 (0.805–0.807) | 3.49× | +71.3% | **3.46×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.400 | 0.245 (0.244–0.245) | 0.826 (0.825–0.829) | 3.38× | +70.4% | **9.80×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.053 | 0.245 (0.245–0.248) | 0.827 (0.827–0.828) | 3.37× | +70.3% | **12.44×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.734 | 0.305 (0.303–0.306) | 0.304 (0.304–0.306) | 1.00× | -0.2% | **2.41×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.221 | 0.325 (0.323–0.327) | 0.326 (0.325–0.327) | 1.00× | +0.4% | **9.92×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.468 | 0.327 (0.326–0.329) | 0.326 (0.326–0.330) | 1.00× | -0.1% | **19.81×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.348 | 2.650 (2.647–2.654) | 2.939 (2.926–2.939) | 1.11× | +9.8% | **4.28×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.677 | 10.458 (10.455–10.465) | 12.624 (12.619–12.637) | 1.21× | +17.2% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.966 | 19.159 (19.157–19.161) | 22.251 (22.232–22.277) | 1.16× | +13.9% | **1.62×** | ✅ |
