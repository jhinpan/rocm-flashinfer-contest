# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 11:52:52Z` |
| command | `python tools/run_benchmarks.py --out results/amd_mi300 --repeat-runs 3 --baseline-ref baseline-v2^{}` |
| commit | `4d15a6064094f195ea01f69c7883e16b3212d101` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-24-g4d15a60` |
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
| fib_cache | `n/a` |
| sol_env | `(defaults)` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v2 ms | cand/baseline-v2 | Δ vs baseline-v2 | baseline-v1 ms | cand/v1 | Δ vs v1 | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.826 | 0.038 (0.036–0.038) | 0.105 (0.105–0.107) | 2.78× | +64.0% | 0.145 | 3.81× | +73.8% | **48.11×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.210 | 0.033 (0.033–0.037) | 0.101 (0.101–0.102) | 3.03× | +67.0% | 0.139 | 4.15× | +75.9% | **843.20×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 110.447 | 0.030 (0.029–0.033) | 0.140 (0.140–0.140) | 4.70× | +78.7% | 0.181 | 6.09× | +83.6% | **3717.74×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.542 | 0.168 (0.166–0.169) | 0.165 (0.163–0.166) | 0.98× | -2.1% | 0.164 | 0.98× | -2.2% | **9.17×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.171 | 0.174 (0.174–0.176) | 0.174 (0.173–0.175) | 1.00× | +0.4% | 0.174 | 1.00× | -0.0% | **185.28×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1879.911 | 0.528 (0.527–0.536) | 0.531 (0.530–0.536) | 1.00× | +0.5% | 3.500 | 6.63× | +84.9% | **3558.81×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.792 | 0.231 (0.231–0.234) | 0.232 (0.232–0.235) | 1.00× | +0.1% | 0.807 | 3.49× | +71.3% | **3.42×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.386 | 0.245 (0.244–0.246) | 0.245 (0.244–0.246) | 1.00× | +0.1% | 0.828 | 3.38× | +70.4% | **9.75×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.035 | 0.246 (0.246–0.248) | 0.246 (0.246–0.246) | 1.00× | +0.0% | 0.828 | 3.37× | +70.3% | **12.35×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.734 | 0.336 (0.333–0.337) | 0.285 (0.285–0.288) | 0.85× | -17.8% | 0.300 | 0.89× | -11.8% | **2.19×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.175 | 0.320 (0.320–0.324) | 0.298 (0.294–0.303) | 0.93× | -7.3% | 0.315 | 0.98× | -1.7% | **9.91×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.499 | 0.326 (0.323–0.326) | 0.305 (0.304–0.307) | 0.94× | -6.8% | 0.319 | 0.98× | -2.0% | **19.95×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.313 | 2.526 (2.525–2.542) | 2.591 (2.587–2.600) | 1.03× | +2.5% | 2.872 | 1.14× | +12.1% | **4.48×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.802 | 9.498 (9.455–9.503) | 10.520 (10.505–10.534) | 1.11× | +9.7% | 12.668 | 1.33× | +25.0% | **1.66×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 31.092 | 14.745 (14.744–14.757) | 19.152 (19.145–19.166) | 1.30× | +23.0% | 22.258 | 1.51× | +33.8% | **2.11×** | ✅ |
