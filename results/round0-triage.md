# Round 0 — Measurement foundation + triage (RLCR optimize-5-kernels)

**Date:** 2026-06-22 · **GPU:** AMD Instinct MI300X (gfx942) · **Commit:** d2f7a45 (working) ·
**Baseline ref:** tag `baseline-v1` (15463ff) · **torch:** 2.9.1+rocm7.2 · **triton:** 3.6.0 ·
**Harness:** flashinfer-bench (HIP-event timing), `tools/run_benchmarks.py` (5 warmup + fixed iters).

Optimization bar (DEC-1): a headline IMPROVEMENT needs **≥20% median `solution_ms` reduction vs
`baseline/v1`** (aim 20–30%+), correctness gate (`verify.py`) intact, no reward hacking (DEC-2).

## Baseline snapshot (provenance-stamped)

Reproduced v1 within noise; full table in `results/v2_round0_baseline.md`.

| Kernel | solution_ms range | speedup vs torch-ref | headroom note |
|---|---|---|---|
| `dsa_sparse_attention` | 0.80–0.83 | 0.99×–3.69× | **largest** — overhead-bound at small `num_tokens` |
| `moe_fp8` | 2.9–22.2 | 3.86×(seq1), 1.24×(seq55), 1.39×(seq14107) | large absolute time; weak at seq55+ |
| `dsa_topk_indexer` | 0.30–0.32 | 2.42×–20.4× | moderate |
| `gdn_prefill` | 0.17–3.47 | 9.2×–526× | near-saturated vs ref; NO-GO check first |
| `gdn_decode` | 0.15–0.19 | 12.3×–575× | near-saturated vs ref; NO-GO check first |

## Round-0 profile (rocprofv3 kernel-trace) — dsa_sparse_attention

Captured at `num_tokens=1` and `num_tokens=8` (raw kept untracked in `/tmp/round0_prof/`).
Per-call cost is dominated by **host-side layout prep, not the attention kernel**:

| component | per-call cost | scales with num_tokens? |
|---|---|---|
| `torch.cat` (`CatArrayBatchedCopy`, 2/call) | ~243µs **each** | **no** (constant) |
| `__amd_rocclr_copyBuffer` (~6/call) | ~54µs each | no (constant) |
| `_kernel_unified_attention_sparse_mla_2d` | 209µs → 226µs | barely |

**Root cause:** `solutions/dsa_sparse_attention/main.py` rebuilds the *entire* paged KV cache every
call: `kv = torch.cat([ckv_cache, kpe_cache], -1).unsqueeze(2).contiguous()` over all
`num_pages=8462` (~600MB) — O(num_pages) work independent of `num_tokens`, even though each token
touches only `topk ≤ 2048` positions. This is why it is 0.99× at `num_tokens=1` (overhead-bound).
Secondary: the AITER kernel hardcodes `BLOCK_M=16, TILE_SIZE=64, num_warps=4, num_stages=1`, grid =
`num_tokens` blocks (1–8 on a 304-CU GPU → underfilled), but the cat overhead dwarfs this.

## Ranked targets (Claude profile + Codex `ask-codex` triage agree)

1. **`dsa_sparse_attention`** — high confidence, ~60–70% latency-cut potential. Remove the
   O(num_pages) host-side cache rebuild; pass `ckv_cache`/`kpe_cache` separately and compose
   `K=[ckv,kpe]` logically inside the load path (V stays `ckv[..., :512]`).
2. **`moe_fp8`** — medium. Large absolute latency, weak at seq55+. Profile seq55/seq14107 to split
   routing/permute vs dequant vs GEMM vs combine *before* touching the fp8 format/MMA path.
3. **`dsa_topk_indexer`** — medium-low. Profile B=1/B=31 for launch/copy/occupancy; fuse adjacent
   pre/post steps if launch-bound.
4. **`gdn_prefill`** — low. Roofline/NO-GO on the longest seq only.
5. **`gdn_decode`** — low. NO-GO microprofile at B=1/B=64; stop unless an obvious host copy/sync exists.

## Direction change vs plan feasibility hint

The plan's feasibility hint for `dsa_sparse_attention` assumed kernel tuning
(`BLOCK_M`/`TILE_SIZE`/`num_warps`/`num_stages`, split-K). Round-0 profiling shows the dominant cost
is the **host-side full-cache `torch.cat`/`.contiguous()` rebuild**, so task3's primary direction
shifts to eliminating that redundant layout rebuild; kernel-config tuning / split-K becomes
secondary. This still serves AC-2 (speedup for dsa) and the Ultimate Goal. Logged in goal-tracker
Plan Evolution.

## Correctness / generality guardrails (carry into every candidate)

- Preserve exact paged-cache indexing, topk semantics, dtype/scaling, output tolerances (`verify.py`),
  and all tested `num_tokens`/page layouts. Do **not** specialize to `num_pages=8462`, fixed token
  counts, or the benchmark distribution (reward hacking — DEC-2).
- Keep V semantics: V is the first 512 channels of the combined KV view.
- fp8 kernels: keep software dequant (`e4m3fn→fp32`) default; native `e4m3fnuz` MMA only if profiling
  proves the fp8 GEMM compute-bound and `verify.py` still passes (DEC-4).
- Judge on absolute `solution_ms` vs `baseline/v1`, not speedup-vs-ref.
