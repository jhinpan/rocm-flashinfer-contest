# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 20:22:10Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round2 --repeat-runs 3` |
| commit | `2e83093a5dcdf8c6588bb89d674dc80c348792dd` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-4-g2e83093-dirty` |
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
| `gdn_decode` | {'batch_size': 1} | 1.846 | 0.151 (0.149–0.152) | 0.151 (0.151–0.151) | 1.00× | -0.1% | **12.24×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.034 | 0.144 (0.144–0.145) | 0.143 (0.143–0.144) | 0.99× | -0.7% | **194.43×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 111.013 | 0.181 (0.180–0.185) | 0.181 (0.181–0.183) | 1.00× | +0.1% | **614.38×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.598 | 0.172 (0.172–0.176) | 0.171 (0.171–0.172) | 0.99× | -0.7% | **9.28×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.872 | 0.178 (0.176–0.179) | 0.176 (0.176–0.179) | 0.99× | -1.2% | **184.25×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1855.690 | 3.471 (3.449–3.472) | 3.472 (3.472–3.472) | 1.00× | +0.0% | **534.62×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.796 | 0.233 (0.231–0.234) | 0.808 (0.807–0.808) | 3.46× | +71.1% | **3.41×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.381 | 0.245 (0.244–0.246) | 0.824 (0.823–0.826) | 3.36× | +70.2% | **9.71×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.010 | 0.246 (0.245–0.248) | 0.825 (0.825–0.826) | 3.36× | +70.2% | **12.24×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.735 | 0.309 (0.307–0.310) | 0.309 (0.306–0.310) | 1.00× | -0.0% | **2.38×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.181 | 0.326 (0.322–0.326) | 0.323 (0.322–0.326) | 0.99× | -0.9% | **9.77×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.447 | 0.325 (0.322–0.328) | 0.325 (0.325–0.326) | 1.00× | -0.1% | **19.83×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.291 | 2.843 (2.839–3.059) | 2.844 (2.836–2.848) | 1.00× | +0.0% | **3.97×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.459 | 12.614 (12.598–12.623) | 12.623 (12.605–12.627) | 1.00× | +0.1% | **1.23×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.869 | 22.136 (22.116–22.149) | 22.151 (22.126–22.159) | 1.00× | +0.1% | **1.39×** | ✅ |
