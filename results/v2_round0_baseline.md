# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency (same reference on every platform).
`cand/base` = baseline-v1 solution latency / candidate latency (>1 = candidate faster). `Δlat%` = latency reduction vs baseline-v1 (positive = faster).

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-22 20:45:41Z` |
| command | `python tools/run_benchmarks.py --out results/v2_round0_baseline --repeat-runs 3 --candidate-ref baseline-v1^{}` |
| commit | `ab0d85d56e967f3c089e852c53b8e97ebf8bf684` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v1-8-gab0d85d` |
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
| `gdn_decode` | {'batch_size': 1} | 1.849 | 0.153 (0.152–0.153) | 0.154 (0.152–0.154) | 1.00× | +0.4% | **12.07×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.285 | 0.148 (0.147–0.148) | 0.149 (0.149–0.149) | 1.01× | +1.0% | **191.43×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 111.264 | 0.181 (0.179–0.185) | 0.180 (0.180–0.180) | 1.00× | -0.2% | **615.90×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.592 | 0.174 (0.174–0.176) | 0.173 (0.172–0.173) | 0.99× | -0.5% | **9.15×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.905 | 0.178 (0.178–0.183) | 0.178 (0.178–0.180) | 1.00× | +0.1% | **184.60×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1860.145 | 3.560 (3.557–3.560) | 3.556 (3.555–3.564) | 1.00× | -0.1% | **522.56×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.797 | 0.819 (0.819–0.820) | 0.819 (0.818–0.821) | 1.00× | +0.0% | **0.97×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.391 | 0.836 (0.836–0.839) | 0.836 (0.836–0.837) | 1.00× | +0.0% | **2.86×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.002 | 0.837 (0.835–0.837) | 0.837 (0.835–0.837) | 1.00× | +0.0% | **3.59×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.746 | 0.313 (0.312–0.314) | 0.312 (0.310–0.312) | 1.00× | -0.5% | **2.38×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.198 | 0.327 (0.323–0.328) | 0.325 (0.325–0.325) | 1.00× | -0.5% | **9.79×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.416 | 0.329 (0.328–0.330) | 0.329 (0.328–0.329) | 1.00× | -0.1% | **19.48×** | ✅ |
| `moe_fp8` | {'seq_len': 1} | 11.254 | 2.854 (2.842–2.926) | 2.851 (2.848–2.856) | 1.00× | -0.1% | **3.94×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.457 | 12.623 (12.620–12.632) | 12.637 (12.636–12.641) | 1.00× | +0.1% | **1.22×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 30.852 | 22.218 (22.206–22.228) | 22.222 (22.214–22.241) | 1.00× | +0.0% | **1.39×** | ✅ |
