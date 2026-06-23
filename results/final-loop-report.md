# RLCR Final Report — optimize-5-kernels (ROCm/MI300X)

**GPU:** AMD Instinct MI300X (gfx942) · **Branch:** `rlcr/optimize-5-kernels` ·
**Baseline:** `baseline-v1^{}` = 74c0918 (immutable) · **Harness:** `tools/run_benchmarks.py
--repeat-runs 3` (HIP-event timing, candidate-vs-baseline, provenance-stamped) ·
**Final unified table:** `results/amd_mi300.{md,csv}` (commit e792634, dirty=no).

All speedups are **candidate vs `baseline/v1`** on identical workloads/dtype/GPU; correctness is the
official contest gate (full pass counts preserved, official tolerances, never weakened).

## One verdict per kernel

| Kernel | Verdict | Latency reduction vs baseline-v1 | Correctness gate |
|---|---|---|---|
| `dsa_sparse_attention` | **IMPROVEMENT** | **+70–71%** (3.4×) all shapes | verify.py 23/23 |
| `gdn_prefill` | **IMPROVEMENT** | **+84.8%** long-seq (6.6×); short unchanged | 100/100 (in-process; --fast 2/2) |
| `gdn_decode` | **IMPROVEMENT** | **+22.6–27.4%** all batches | 54/54 (in-process; --fast 2/2) |
| `moe_fp8` | **IMPROVEMENT (marginal)** | **+9–17%** (seq55 +17%) | verify.py 19/19 (loose tol) |
| `dsa_topk_indexer` | **NO-GO ≥20%** (robust path shipped) | 0% (torch default; fused +5–8% available, see below) | verify.py 128/128 |

## How each win was obtained
- **dsa_sparse_attention (+70%)** — replaced the per-call `torch.cat` of the entire 8462-page KV
  cache (~485 µs, the profiled bottleneck) with a self-contained Triton sparse-MLA kernel that reads
  `ckv`/`kpe` as separate pointers (no full-cache materialization).
- **gdn_prefill (+84.8% long-seq)** — length-dispatch to chunk-parallel `chunk_gated_delta_rule` for
  `total_seq_len ≥ 4096` (GVA q/k expand), recurrent for short sequences.
- **gdn_decode (+23–28%)** — AITER `fused_sigmoid_gating_delta_rule_update` fuses the gate compute
  into the kernel (removes host-side exp/softplus/sigmoid); numerically identical to the recurrent
  path (maxdiff 0.0).
- **moe_fp8 (+9–17%)** — block-broadcast weight dequant replacing per-expert `repeat_interleave`
  (numerically identical, less memory traffic). A fused block-scale GEMM and AITER fused-MoE were
  tested and lost to rocBLAS (evidence: `results/round4-report.md`, `results/v2_round4_moe.*`); native
  fp8 `fnuz` MMA deprioritized (GEMM only 3–18% of latency; gfx942 is e4m3fnuz vs contest e4m3fn).
- **dsa_topk_indexer (NO-GO ≥20%)** — already well-optimized (2.4–20× vs ref). Both planned levers
  tested: AITER `deepgemm_fp8_paged_mqa_logits` falsified by the committed fnuz/fn probe
  (`tools/fp8_dtype_probe.py`: contest e4m3fn read as e4m3fnuz → 2× decode error) + verify timeout /
  GPU exception; a software fused-logits Triton kernel gives +5–8% and passes official verify.py
  128/128 but lowers the per-run matched-ratio to ~0.98 (harness FAIL display). Shipped the robust
  torch path (mr 1.000); fused available behind `DSA_TOPK_FUSED=1` (a human ship/no-ship decision).

## Correctness gates (official verify.py)
- dsa_sparse_attention 23/23 · moe_fp8 19/19 · dsa_topk_indexer 128/128 — full `verify.py`.
- gdn_decode 54/54 and gdn_prefill 100/100 — via an in-process full sweep at the official tolerances
  (atol=rtol=1e-2 on output + new_state) plus `verify.py --fast` 2/2 each; the full `verify.py` is
  impractically slow for these two (100+ workloads × per-workload subprocess isolation, >30 min,
  independent of the optimizations), so the in-process sweep performs the identical comparison.

## Deliverables
- Per-kernel commits on `rlcr/optimize-5-kernels` with candidate lineage; final unified v2 table
  `results/amd_mi300.{md,csv}`; per-round reports `results/round{2,4,5,7,9}-report.md` and snapshots
  `results/v2_round*.{md,csv}`; profiling driver `tools/profile_kernel.py`; fp8 probe
  `tools/fp8_dtype_probe.py`; harness provenance + baseline-comparison in `tools/run_benchmarks.py`.

## Summary
**4 of 5 kernels improved** (three by 23–85%, one by 9–17%), one evidence-backed NO-GO with the
robust path retained. No correctness regressions; no tolerance weakening; all changes generalize
(no per-workload hard-coding). Pattern: kernels with removable host-side waste (`torch.cat`,
`repeat_interleave`, host-side gate compute) or a genuine algorithmic lever (chunk-parallel prefill)
yielded large wins; already-tight vendor-wrapped kernels (dsa_topk) converged to evidence-backed
NO-GO because portable hand-written kernels cannot beat the tuned rocBLAS/torch primitives.
