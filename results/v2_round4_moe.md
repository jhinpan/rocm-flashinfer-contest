# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 22:34:25Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round4_moe --repeat-runs 3` |
| commit | `27a28da2dca58f8e74bf6f4267370f6ac51c717b` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-15-g27a28da` |
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
| sol_env | `MOE_USE_FUSED=1` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v1 ms | cand/base | Δlat% | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.844 | 0.153 (0.151–0.153) | 0.152 (0.150–0.152) | 0.99× | -0.7% | **12.08×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.149 | 0.147 (0.147–0.148) | 0.147 (0.145–0.147) | 1.00× | -0.4% | **184.37×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 107.017 | 0.183 (0.183–0.187) | 0.183 (0.183–0.183) | 1.00× | +0.1% | **584.61×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.557 | 0.170 (0.168–0.172) | 0.170 (0.169–0.171) | 1.00× | +0.1% | **9.17×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.374 | 0.177 (0.175–0.180) | 0.176 (0.174–0.177) | 1.00× | -0.5% | **183.40×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1912.907 | 3.572 (3.570–3.575) | 3.570 (3.568–3.572) | 1.00× | -0.1% | **535.56×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.795 | 0.232 (0.231–0.233) | 0.808 (0.807–0.808) | 3.49× | +71.3% | **3.43×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.387 | 0.245 (0.245–0.247) | 0.828 (0.828–0.831) | 3.38× | +70.4% | **9.73×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.025 | 0.246 (0.246–0.247) | 0.829 (0.828–0.830) | 3.37× | +70.3% | **12.29×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.733 | 0.295 (0.295–0.299) | 0.307 (0.307–0.308) | 1.04× | +3.8% | **2.48×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.179 | 0.308 (0.308–0.486) | 0.325 (0.321–0.336) | 1.05× | +4.9% | **10.31×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.533 | 0.317 (0.309–0.317) | 0.326 (0.324–0.328) | 1.03× | +2.8% | **20.61×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.334 | 3.092 (3.091–3.094) | 2.899 (2.883–3.053) | 0.94× | -6.6% | **3.67×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.881 | 14.068 (14.057–14.118) | 12.762 (12.756–12.764) | 0.91× | -10.2% | **1.13×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 31.020 | 41.120 (41.104–41.223) | 22.218 (22.214–22.258) | 0.54× | -85.1% | **0.75×** | ✅ |
