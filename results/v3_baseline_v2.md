# Benchmark results — AMD Instinct MI300X

`speedup_vs_ref` = torch-reference latency / solution latency.
`cand/baseline-v2` = baseline-v2 solution latency / candidate latency (>1 = candidate faster); `Δ vs baseline-v2` = latency reduction vs baseline-v2.
`cand/v1` / `Δ vs v1` = same against the contest base `baseline/v1`.

## Provenance

Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are
only valid when these match).

| field | value |
|---|---|
| timestamp_utc | `2026-06-23 09:27:40Z` |
| command | `python tools/run_benchmarks.py --out results/v3_baseline_v2 --repeat-runs 3 --baseline-ref baseline-v2^{}` |
| commit | `da17f23f9be788c9dedea62f6d65b049c7cca6cd` |
| branch | `rlcr/optimize-5-kernels` |
| describe | `baseline-v2-5-gda17f23` |
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
| fib_cache | `/tmp/fib_cache` |
| sol_env | `(defaults)` |

| Kernel | Workload | reference ms | solution ms (min–max) | baseline-v2 ms | cand/baseline-v2 | Δ vs baseline-v2 | baseline-v1 ms | cand/v1 | Δ vs v1 | speedup_vs_ref | correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `gdn_decode` | {'batch_size': 1} | 1.832 | 0.109 (0.109–0.110) | 0.110 (0.109–0.111) | 1.01× | +1.4% | 0.146 | 1.34× | +25.3% | **16.84×** | ✅ |
| `gdn_decode` | {'batch_size': 16} | 28.111 | 0.105 (0.104–0.106) | 0.105 (0.103–0.105) | 1.00× | +0.1% | 0.144 | 1.38× | +27.4% | **268.14×** | ✅ |
| `gdn_decode` | {'batch_size': 64} | 110.911 | 0.140 (0.140–0.142) | 0.140 (0.140–0.141) | 1.00× | +0.1% | 0.183 | 1.31× | +23.5% | **790.99×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 6, 'num_seqs': 1, 'len_cu_seqlens': 2} | 1.555 | 0.168 (0.167–0.170) | 0.169 (0.168–0.170) | 1.01× | +0.6% | 0.167 | 1.00× | -0.3% | **9.28×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 139, 'num_seqs': 3, 'len_cu_seqlens': 4} | 32.259 | 0.175 (0.174–0.175) | 0.175 (0.174–0.175) | 1.00× | +0.1% | 0.175 | 1.00× | -0.1% | **184.55×** | ✅ |
| `gdn_prefill` | {'total_seq_len': 8192, 'num_seqs': 39, 'len_cu_seqlens': 40} | 1857.038 | 0.531 (0.531–0.546) | 0.538 (0.535–0.541) | 1.01× | +1.3% | 3.472 | 6.54× | +84.7% | **3497.19×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 1, 'num_pages': 8462} | 0.791 | 0.231 (0.230–0.234) | 0.231 (0.230–0.234) | 1.00× | -0.2% | 0.803 | 3.47× | +71.2% | **3.42×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 6, 'num_pages': 8462} | 2.384 | 0.244 (0.244–0.246) | 0.244 (0.244–0.246) | 1.00× | -0.1% | 0.827 | 3.39× | +70.5% | **9.77×** | ✅ |
| `dsa_sparse_attention` | {'num_tokens': 8, 'num_pages': 8462} | 3.017 | 0.245 (0.245–0.246) | 0.245 (0.245–0.245) | 1.00× | -0.0% | 0.826 | 3.37× | +70.3% | **12.31×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 1, 'max_num_pages': 1, 'num_pages': 11923} | 0.733 | 0.288 (0.288–0.292) | 0.287 (0.287–0.288) | 1.00× | -0.5% | 0.304 | 1.05× | +5.2% | **2.54×** | ✅ |
| `dsa_topk_indexer` | {'batch_size': 14, 'max_num_pages': 40, 'num_pages': 11923} | 3.180 | 0.305 (0.304–0.305) | 0.302 (0.298–0.308) | 0.99× | -1.2% | 0.316 | 1.03× | +3.3% | **10.42×** | ❌ |
| `dsa_topk_indexer` | {'batch_size': 31, 'max_num_pages': 43, 'num_pages': 11923} | 6.608 | 0.311 (0.302–0.311) | 0.318 (0.312–0.493) | 1.02× | +2.2% | 0.322 | 1.04× | +3.5% | **21.24×** | ❌ |
| `moe_fp8` | {'seq_len': 1} | 11.308 | 2.580 (2.576–2.603) | 2.585 (2.573–2.596) | 1.00× | +0.2% | 2.859 | 1.11× | +9.7% | **4.38×** | ✅ |
| `moe_fp8` | {'seq_len': 55} | 15.819 | 10.546 (10.511–10.556) | 10.518 (10.513–10.534) | 1.00× | -0.3% | 12.693 | 1.20× | +16.9% | **1.50×** | ✅ |
| `moe_fp8` | {'seq_len': 14107} | 31.011 | 19.165 (19.160–19.194) | 19.160 (19.137–19.205) | 1.00× | -0.0% | 22.216 | 1.16× | +13.7% | **1.62×** | ✅ |
