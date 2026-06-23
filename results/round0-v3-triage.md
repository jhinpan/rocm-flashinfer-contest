# v3 Round 0 — scaffold + triage (3 target kernels)

**GPU:** MI300X (gfx942) · **Base:** `baseline-v2^{}` = 08d4780 (no-regression base) · also reported
vs `baseline/v1` = 74c0918 · **Harness:** `tools/run_benchmarks.py --baseline-ref baseline-v2^{}
--repeat-runs 3` (NEW paired mode). Self-check snapshot: `results/v3_baseline_v2.{md,csv}`
(candidate==v2 → all ≈1.00×, clean provenance, dirty=no).

## Removable-cost ranking (from v2 stage profiles + the gen-plan Codex first-pass)

| Kernel | v2 latency | Dominant removable cost | v3 lever | Risk |
|---|---|---|---|---|
| `gdn_decode` | 0.10–0.14 ms | 2 host-side state transposes (k-last↔AITER [K,V]); copies are O(B·HV·K·V) | fresh k-last fused decode kernel (no transpose in/out); solve B=1 occupancy | low–med |
| `dsa_topk_indexer` | 0.28–0.31 ms | ~55% elementwise = fp8 dequant + `k_deq[B,N,D]` materialization; v2 fused-logits tie-break drops mr to ~0.97 | packed-page fused scoring (read fp8 in-kernel, no k_deq), preserve NaN, `scale·dot` identity; restore mr≥0.999; pick best selection primitive | med |
| `moe_fp8` | 2.6 / 10.5 / 19.1 ms | per-expert schedule: `index_select`→materialize bf16 weights→rocBLAS→`index_add_`; GEMM only 3–18% | grouped scheduling / per-seq-len dispatch; fused block-scale only if it beats rocBLAS (parity harness first) | high |

## Notes / guardrails (locked decisions)
- DEC-1: headline IMPROVEMENT = beat baseline-v2 by **>3–5%** above noise, no per-workload
  regression; moe ≥20% vs baseline/v1.
- DEC-2: dsa_topk **mr ≥ 0.999** hard gate (the self-check shows v2 at mr≈0.97 — must fix).
- DEC-3: no warmup/persistent caches (reward hacking).
- DEC-4: in-process sweep + `verify.py --fast` during the loop; **full official verify.py at finalize**.
- Order (lowest-risk first): gdn_decode → dsa_topk_indexer → moe_fp8.
- Per-kernel rounds will capture a fresh rocprofv3 stage profile (round-0 trace per the profiling
  contract) before each candidate; the gen-plan Codex first-pass already sharpened the levers
  (grouped MoE not just fused GEMM; topk selection primitive choice for k≈N; NaN-preserving relu;
  transpose byte-math + B=1 occupancy).

## Round 1 — fresh rocprofv3 stage profiles + ceiling probes

Command per kernel/bucket: `rocprofv3 --kernel-trace --output-format csv -d <dir> -o k -- python
tools/profile_kernel.py --kernel <k> --bucket <min|med|max> --iters <N>`. Raw traces (untracked):
`/tmp/v3prof/<k>_<bucket>/k_kernel_trace.csv`. Top dispatches by total traced time:

| Kernel (bucket) | dominant dispatch | launch | avg µs | % | class |
|---|---|---:|---:|---:|---|
| gdn_decode (B=1) | `elementwise_manual` (state transpose copies) | 110 (2/call) | 6.3 | 59.8 | copy |
| | `fused_sigmoid_gating_delta_rule_update` | 55 | 6.3 | 29.8 | fused-recurrent |
| gdn_decode (B=64) | `elementwise_manual` (state transpose copies) | 110 (2/call) | 64.9 | **81.1** | copy |
| | fused gate kernel | 55 | 25.0 | 15.6 | fused-recurrent |
| dsa_topk (B=1) | `elementwise_manual` (dequant) | 440 | 2.8 | 30.5 | elementwise |
| | `vectorized_elementwise` | 552 | 2.1 | 27.7 | elementwise |
| | `_fused_logits_kernel` | 55 | 4.8 | 6.4 | fused-logits |
| dsa_topk (B=31) | `elementwise_manual` (dequant) | 550 | 10.1 | 36.4 | elementwise |
| | `sbtopk::gatherTopK` + `radixSortKVInPlace` | 110 | — | 24.8 | topk/sort |
| | `_fused_logits_kernel` | 55 | 22.9 | 8.2 | fused-logits |
| moe_fp8 (seq55) | `elementwise_manual` (weight dequant) | 5460 | 33.2 | **54.8** | elementwise |
| | `vectorized_elementwise` | 7253 | 11.3 | 24.8 | elementwise |
| | Tensile GEMM (`Cijk_…`) | 1680 | 14.2 | 7.2 | GEMM |
| moe_fp8 (seq14107) | `elementwise_manual` (weight dequant) | 4600 | 33.3 | 41.8 | elementwise |
| | Tensile GEMM | 1280 | 66.1 | 23.1 | GEMM |
| | `vectorized_elementwise` | 4748 | 16.2 | 21.0 | elementwise |
| moe_fp8 (seq1) | weight_dequant_elementwise | 1045 | 34.1 | 40.2 | elementwise |
| | vectorized_elementwise | 3858 | 6.7 | 29.0 | elementwise |
| | routing_reduce | 2530 | 3.2 | 9.1 | reduce |
| | GEMM | 330 | 14.3 | 5.3 | GEMM |

MoE seq1 (12,448 dispatches via `tools/summarize_rocprof_trace.py
/tmp/v3prof/moe_fp8_min/k_kernel_trace.csv --kernel moe_fp8`): even at seq_len=1 the per-expert
weight dequant (40%) + elementwise (29%) dominate (69%); GEMM only 5.3%. So the grouped/fused
weight-materialization lever applies across all seq lengths (not just launch overhead at small seq).
Stage tables are reproducible via the committed summarizer.

### Ceiling probes (byte-math)
- **gdn_decode**: state `[B,HV,K,V]` fp32, B=64,HV=8,K=V=128 → 33.6 MB/transpose; 2 transposes ×
  (read+write) ≈ 134 MB ≈ ~0.10 ms at ~1.3 TB/s — matches the measured ~0.13 ms elementwise at
  B=64. Removing both transposes (k-last fused kernel) is the recoverable win → target ≥15% (AC-T1).
- **dsa_topk**: `k_deq[B,N,D]` fp32, B=31,N≈2752,D=128 → ~44 MB materialized + read; a packed-page
  fused scoring kernel (read fp8 in-kernel) removes this write+read. topk/sort (~25% at B=31) is the
  other lever (selection-primitive choice; k≈N).
- **moe_fp8**: per-active-expert bf16 weight materialization, ~32 experts × (`W13`+`W2`) × 2 B ≈
  2.8 GB at seq55 — dominates (54.8% elementwise). Grouped/fused block-scale (dequant in-tile, no
  materialization) is the lever; GEMM is 7–23% (only the large-seq GEMM is a secondary target).

Fresh traces agree with the prior ranking. Order stays lowest-risk-first: gdn_decode → dsa_topk →
moe_fp8.

## Untouched kernels (must not regress)
`dsa_sparse_attention` (+70%), `gdn_prefill` (+84.8%) — confirmed ≈1.00× vs baseline-v2 in the
self-check; re-verified non-regressed after every candidate (task7) and at finalize.
