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

## Untouched kernels (must not regress)
`dsa_sparse_attention` (+70%), `gdn_prefill` (+84.8%) — confirmed ≈1.00× vs baseline-v2 in the
self-check; re-verified non-regressed after every candidate (task7) and at finalize.
