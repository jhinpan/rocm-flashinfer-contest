# Round 4 report — moe_fp8 final verdict

**GPU:** AMD Instinct MI300X (gfx942) · **Baseline:** `baseline-v1^{}` = 74c0918 ·
**Harness:** `tools/run_benchmarks.py --repeat-runs 3` · **Gate:** official `verify.py`.

## Verdict: `moe_fp8` → IMPROVEMENT (+10–17%, shipped) + NO-GO for ≥20% via the portable path

**Shipped (default):** block-broadcast dequant (`_dequant_block`) + rocBLAS bf16 matmul.
- `verify.py` **19/19 PASSED**; vs baseline-v1 **+9.8% / +17.2% / +13.9%** (seq 1 / 55 / 14107),
  tight spread (`results/v2_round3.md`). Below the ≥20% headline bar but a real, zero-risk win
  (numerically identical to baseline, no regression).

**≥20% target: NO-GO via the portable path (evidence-backed).**

### What was tried (the cheapest decisive experiments, AC-6)
1. **Ceiling probe.** Timing with weights pre-dequantized to bf16 (rocBLAS GEMM only) gave 3.56ms vs
   10.45ms at seq55 → weight dequant/materialization is ~66% of latency, suggesting a fused
   in-tile-dequant GEMM *could* help.
2. **Implemented the fused kernel.** `_blockscale_gemm` (Triton): reads fp8 `e4m3fn` weights + 128×128
   block scales, dequants **in-tile** (fp32 accumulate, exact — no `fnuz`), never materializing the
   bf16 weight. Verified equivalent to dequant+matmul (max rel 3.4e-3) and **`verify.py` 19/19**.
3. **Measured it — it loses everywhere:**

   | seq_len | fused Triton GEMM | block-broadcast dequant + rocBLAS | winner |
   |---:|---:|---:|---|
   | 1 | 3.17 ms | 2.69 ms | dequant |
   | 55 | 14.13 ms | 10.52 ms | dequant |
   | 14107 | 40.99 ms | 19.16 ms | dequant |

### Why ≥20% is not safely reachable here
- The ceiling probe's "free weights" used **rocBLAS/Tensile** matmul. A hand-written *portable*
  Triton block-scale GEMM cannot match rocBLAS's tuned matmul for these shapes, so although it
  removes weight materialization, the slower GEMM more than cancels the gain (and is far worse at
  large seq, which is GEMM-bound). Reaching ≥20% would require a production-grade autotuned grouped
  block-scale MoE GEMM that *both* fuses dequant *and* matches rocBLAS — out of scope for this loop's
  portable constraints.
- The AITER fused block-scale MoE path (`fused_moe` / `fmoe_fp8_blockscale_g1u1`) is blocked by the
  contest's `e4m3fn` data vs gfx942 native `e4m3fnuz` (value-preserving conversion risk, DEC-4) plus
  unverified block-scale-layout / SwiGLU-half-order / routing-parity matching (GAP_REPORT §2.E) —
  high risk of failing the 19/19 gate.
- Native fp8 `fnuz` MMA is separately deprioritized per **DEC-4**: the GEMM is only 3–18% of latency
  (not compute-bound), so faster fp8 MMA addresses a minority of the cost.

### Outcome
Ship the +17% block-broadcast path (safe, 19/19). Record the ≥20% target as a NO-GO via the portable
path with the above evidence. The experimental fused kernel is retained behind `MOE_USE_FUSED=1` as
reproducible evidence. Revisiting ≥20% would need an autotuned grouped block-scale MoE GEMM — a
larger, separate effort (candidate for human go/no-go under the binding-constraint escalation gate).

No reward hacking: both paths are general (no per-workload constants); the shipped path generalizes
across all 19 official workloads.
