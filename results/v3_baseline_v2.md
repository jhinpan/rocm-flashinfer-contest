# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 09:10:37Z` |
| command | `python tools/run_benchmarks.py --out results/v3_baseline_v2 --repeat-runs 3 --baseline-ref baseline-v2^{}` |
| commit | `a979e9fb095e643b04da2e21448ec54a8184e281` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-1-ga979e9f` |
| baseline_commit | `08d4780` |
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

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v1 ms | cand/base | Δlat% | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.819 | 0.106 (0.106–0.109) | 0.106 (0.105–0.108) | 1.00× | +0.4% | **17.18×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 27.948 | 0.102 (0.101–0.103) | 0.102 (0.101–0.102) | 0.99× | -0.6% | **272.95×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 110.047 | 0.140 (0.140–0.145) | 0.140 (0.140–0.141) | 1.00× | -0.2% | **783.93×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.540 | 0.165 (0.164–0.168) | 0.165 (0.165–0.167) | 1.00× | -0.4% | **9.31×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.162 | 0.175 (0.174–0.175) | 0.174 (0.174–0.174) | 1.00× | -0.4% | **184.21×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1867.119 | 0.532 (0.525–0.545) | 0.529 (0.524–0.536) | 1.00× | -0.5% | **3511.94×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.790 | 0.231 (0.231–0.233) | 0.232 (0.231–0.233) | 1.00× | +0.1% | **3.41×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.378 | 0.245 (0.244–0.246) | 0.244 (0.244–0.246) | 1.00× | -0.1% | **9.72×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.009 | 0.246 (0.246–0.248) | 0.246 (0.246–0.246) | 1.00× | +0.0% | **12.22×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.729 | 0.285 (0.284–0.287) | 0.284 (0.280–0.288) | 1.00× | -0.5% | **2.56×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.175 | 0.299 (0.297–0.300) | 0.297 (0.290–0.298) | 0.99× | -0.5% | **10.64×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.517 | 0.305 (0.300–0.306) | 0.304 (0.304–0.307) | 1.00× | -0.5% | **21.34×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.311 | 2.571 (2.569–2.606) | 2.586 (2.584–2.591) | 1.01× | +0.6% | **4.40×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.895 | 10.533 (10.505–10.538) | 10.522 (10.504–10.538) | 1.00× | -0.1% | **1.51×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 31.209 | 19.136 (19.126–19.148) | 19.132 (19.112–19.201) | 1.00× | -0.0% | **1.63×** | ✅ |
