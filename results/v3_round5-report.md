# v3 Round 5 — moe_fp8 IMPROVEMENT + dsa_topk evidence packet closed

**GPU:** AMD Instinct MI300X (gfx942) · **Base:** `baseline-v2^{}` = 08d4780 (primary), also vs
`baseline/v1` = 74c0918 · **Harness:** `tools/run_benchmarks.py --baseline-ref baseline-v2^{}
--repeat-runs 3 --only <kernel>`.

## Mainline: task5/task6 `moe_fp8` — VERDICT: IMPROVEMENT

The round-1 profile showed the per-expert **block-scale weight dequant** dominating at every
seq_len (69%/80%/63% at seq 1/55/14107); GEMM was only 5–23%. The torch `_dequant_block` does
`(w.to(float32).view(blocks) * scale[:,None,:,None]).view().to(bf16)` — materializing **two full
fp32 weight temporaries** before the bf16 cast (~19 bytes/elem for a 1-byte fp8 input). That memory
traffic, not the GEMM, was the bottleneck.

**Fix (keeps rocBLAS for the GEMM — BL-20260622 compliant):** a fused Triton dequant kernel
(`_dequant`) — one program per 128×128 weight block = exactly one block-scale element — reads fp8 +
scale and writes bf16 **directly** (~3 bytes/elem, no fp32 intermediate, no scale broadcast). It is
**bit-identical** to `_dequant_block` (parity harness: exact, maxabs 0) and ~5× faster standalone
(0.036 ms vs 0.179 ms per weight). rocBLAS still does the matmul.

| seq_len | candidate ms | baseline-v2 ms | **vs baseline-v2** | vs baseline/v1 | mr | verify |
|---|---|---|---|---|---|---|
| 1 | 2.584 | 2.664 | +3.0% (1.03×) | 1.14× | 0.988 | — |
| 55 | 9.313 | 10.387 | **+10.3% (1.12×)** | **1.36×** | 0.995 | — |
| 14107 | 14.518 | 19.098 | **+24.0% (1.32×)** | **1.53×** | 0.994 | — |

Official **verify.py 19/19** (2.09× mean vs torch ref). vs baseline/v1, seq55 (+36%) and seq14107
(+53%) both exceed the **≥20%-vs-v1** AC-T3 target; seq1 is a small positive (+3%, near the noise
bar — the dequant is a smaller share at one token where launch/routing overhead dominates). The win
grows with seq_len as the GEMM amortizes per-launch overhead. Evidence: `results/v3_round5_moe.*`
(dirty=no). `MOE_DEQUANT_TORCH=1` restores the torch dequant for parity/debug.

**Parity harness** (`tools/moe_parity.py`, task6): dequant bit-exactness (maxabs 0), contiguous-half
SwiGLU order, and full-path output parity vs the reference (4/4 on official inputs) — passes on
synthetic + official inputs. Because the shipped dequant is bit-identical to the reference dequant,
no AITER fused (fnuz-risk) path was needed; the harness remains as the gate for any future fused
attempt.

## Blocking side issue closed: task4 dsa_topk NO-GO evidence packet (Codex round-4 finding 1)
- `results/v3_round4_dsatopk_fast.{md,csv}`: `DSA_TOPK_FAST=1` paired benchmark, dirty=no,
  sol_env recorded → **+9.9/+21.3/+21.5%** vs baseline-v2 (1.17/1.38/1.36× vs v1), mr 1.0/0.991/0.996.
- `results/v3_round4_dsatopk_verify.md`: official **verify.py 128/128 for both** default (7.02×) and
  fast (8.18×); rejected-lever appendix (candidate re-score −72..−91%, fused gather+dequant
  −14..−35%) + `tl.dot` vs `torch.bmm` micro-test (`ieee` maxrel 6e-4) substantiating that
  ">+8% AND mr≥0.999" is infeasible. AC-T2 NO-GO now fully evidence-backed.

## task7 — no regression (vs baseline-v2)
moe is the only kernel changed this round. Re-checked all five: moe verify.py 19/19; gdn_decode
2.75/2.93/4.05×, gdn_prefill 1.01/0.99/0.99×, dsa_sparse 1.00×, dsa_topk 0.85/0.92/0.92× (the
round-4 robustness-first default), all PASS.

## v3 status after round 5 (all 3 target kernels have verdicts)
- **gdn_decode**: IMPROVEMENT — +65–78% vs baseline-v2 (3.9–5.9× vs v1).
- **moe_fp8**: IMPROVEMENT — +10–24% vs baseline-v2 on seq55/14107 (1.36/1.53× vs v1); +3% seq1.
- **dsa_topk_indexer**: robustness IMPROVEMENT (mr→1.0) + perf NO-GO on >+8%-at-mr≥0.999
  (`DSA_TOPK_FAST=1` gives +10–21% at mr~0.99, contest-correct, behind a flag).

## Remaining: task8 finalize (round 6)
Full official `verify.py` for all five kernels (DEC-4), unified v3 `results/amd_mi300.{md,csv}`, and
the final per-loop report with one verdict per kernel.
