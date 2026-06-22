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

## Round 1 — full measured triage (all 5 kernels, clean profiler)

Profiler driver fixed (inputs built once; no per-iter clone — the spurious `copyBuffer` dispatches
that previously appeared in the dsa trace are gone, confirming they were clone artifacts). Harness
now times `baseline-v1^{}` per workload (`results/v2_round0_baseline.md`, all ratios ≈1.00× since
solutions are unchanged). rocprofv3 `--kernel-trace`, top dispatches by total time:

| Kernel (bucket) | dominant dispatches | bottleneck class | lever |
|---|---|---|---|
| `dsa_sparse_attention` (min/max) | `CatArray` 65/63%, MLA kernel 28/30% | **host-side full-cache `torch.cat`** | remove O(num_pages) KV rebuild |
| `moe_fp8` (seq55) | elementwise 44%+39% (**~16k small dispatches**), GEMM 3–6% | **elementwise / launch-bound** | fuse per-expert dequant/swiglu/scale/combine |
| `moe_fp8` (seq14107) | elementwise 36%+32%, GEMM (Tensile) 18% | elementwise-bound, GEMM grows | fused experts; GEMM tuning secondary |
| `dsa_topk_indexer` (min/max) | elementwise (dequant) 29–55%, topk gather+sort 14–15% @max, logits GEMM 6–9% | dequant-elementwise + topk/sort | fuse logits + segmented top-k |
| `gdn_prefill` (max) | `_fused_recurrent_gated_delta_rule` **96%** | recurrent-kernel-bound | chunk path (parallel over chunks) |
| `gdn_decode` (min/max) | elementwise (gate/transpose) 46/64%, recurrent kernel 25/22% | elementwise gate/transpose, tiny absolute (0.15–0.19ms) | fuse gate+drop transpose — low ROI |

**Refinement to DEC-4 (fp8 MMA):** `moe_fp8` is launch/elementwise-bound; the GEMM is 3–18% of
time, so native `fnuz` fp8 MMA would address a minority of the cost. The dominant win is **fusing
the thousands of small per-expert elementwise launches** (dequant, SwiGLU, scaling, combine). fp8
MMA stays a low-priority, profiling-gated experiment for the large-seq case only.

**Measurement-noise note (DEC-1):** at the ~0.15–0.3ms scale (gdn_decode, dsa_topk), run-to-run
variance was up to ~40% for *identical* code (e.g. baseline-vs-candidate of the unchanged
gdn_decode B=16 showed 0.70×). The ≥20% IMPROVEMENT bar must therefore be confirmed as a stable
median over the harness's fixed iters (and re-run) for these small kernels, not a single sample.

**Revised ranking (Claude profile + Codex round-1 triage; Codex swapped 3↔4):**
1. `dsa_sparse_attention` — high: remove host-side cat (~60% of per-call time).
2. `moe_fp8` — medium: fuse per-expert elementwise launches (not fp8 MMA).
3. `gdn_prefill` — low-medium: chunk path vs the 96%-dominant recurrent kernel.
4. `dsa_topk_indexer` — medium-low: fuse logits + top-k.
5. `gdn_decode` — low: tiny absolute; likely NO-GO unless the gate/transpose fuse is cheap.

**DSA guardrail (Codex trap #3):** the legitimate fix removes the redundant work **within a single
`run()` call** — gather only the referenced pages, or read `ckv`/`kpe` separately in a custom kernel
so the full `[..,576]` cache is never materialized. Do **not** cache/prepack KV across calls or
hoist work outside the measured path; `baseline/v1` rebuilds per call, so cross-call caching would
change the measured work illegitimately (reward hacking).

## Correctness / generality guardrails (carry into every candidate)

- Preserve exact paged-cache indexing, topk semantics, dtype/scaling, output tolerances (`verify.py`),
  and all tested `num_tokens`/page layouts. Do **not** specialize to `num_pages=8462`, fixed token
  counts, or the benchmark distribution (reward hacking — DEC-2).
- Keep V semantics: V is the first 512 channels of the combined KV view.
- fp8 kernels: keep software dequant (`e4m3fn→fp32`) default; native `e4m3fnuz` MMA only if profiling
  proves the fp8 GEMM compute-bound and `verify.py` still passes (DEC-4).
- Judge on absolute `solution_ms` vs `baseline/v1`, not speedup-vs-ref.
