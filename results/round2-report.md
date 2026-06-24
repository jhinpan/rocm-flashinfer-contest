# Round 2 report — dsa_sparse_attention optimization

**GPU:** AMD Instinct MI300X (gfx942) · **Peeled baseline:** `baseline-v1^{}` = 74c0918 ·
**Harness:** `tools/run_benchmarks.py --repeat-runs 3` (paired candidate/baseline, median+spread) ·
**Full table:** `results/v2_round2.md`.

## Verdict: `dsa_sparse_attention` → IMPROVEMENT

| Workload | baseline-v1 ms | candidate ms (min–max) | cand/base | latency reduction | correctness |
|---|---:|---:|---:|---:|:--:|
| num_tokens=1 | 0.808 | 0.233 (0.231–0.234) | **3.46×** | **+71.1%** | ✅ |
| num_tokens=6 | 0.824 | 0.245 (0.244–0.246) | **3.36×** | **+70.2%** | ✅ |
| num_tokens=8 | 0.825 | 0.246 (0.245–0.248) | **3.36×** | **+70.2%** | ✅ |

- **Correctness (authoritative):** `python verify.py --solution solutions/dsa_sparse_attention/solution.json`
  → **23/23 PASSED**, mean speedup-vs-torch-ref 8.06× (was ~2.5×), 0.243 ms mean.
- **≥20% bar (DEC-1):** cleared decisively (~70% reduction); spread is tight and does not overlap
  the win, so this is not measurement noise.
- **No regression:** all other kernels measured ≈1.00× cand/base (`results/v2_round2.md`); the
  `--repeat-runs` paired timing also resolved the round-1 `gdn_decode` B=16 noise artifact
  (0.70× → 0.99×).

## What changed
`solutions/dsa_sparse_attention/main.py` now runs a self-contained Triton sparse-MLA decode kernel
(`_sparse_mla_decode_kernel`, adapted from AITER's `_kernel_unified_attention_sparse_mla_2d`, same
online-softmax/masking math) that reads `ckv_cache` (lora + value) and `kpe_cache` (rope) as
**separate pointers**. The ~600MB per-call `torch.cat([ckv,kpe]).contiguous()` over all 8462 pages
(~485 µs, the profiled bottleneck) is eliminated; only the tiny query cat remains. The original AITER
full-cache path is retained behind `DSA_USE_AITER=1` as a debug fallback.

## Generality / correctness guardrails honored
- No cross-call caching or prepacking — every `run()` reads the caches as given (same measured work
  as `baseline/v1`); the speedup comes from not materializing the combined cache, not from hoisting
  work out of the measured path (DEC-2, Codex trap #3).
- No per-workload constants; the kernel handles any `num_tokens`/`num_pages`/`topk`/`page_size` and
  passed all 23 official workloads (num_tokens 1–8). `language=python`, portable Triton (no nvcc);
  `run()` signature and bf16 `[T,16,512]` output unchanged (AC-4).
- Tolerances unchanged (official `atol=rtol=1e-2`).
