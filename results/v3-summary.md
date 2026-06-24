# ROCm/MI300X FlashInfer kernels — RLCR optimization summary (v2 + v3)

This is the human-readable wrap-up of the two autonomous **RLCR** (Reinforcement-Learning-from-
Code-Review) optimization loops that produced the kernels in this repo. It covers *what was
optimized*, *the measured results*, *how each win was found*, and *what we learned about the method
itself*. Per-round artifacts are linked at the end.

> **Tooling:** [`rocm-KDA-pilot`](https://github.com/jhinpan/rocm-KDA-pilot) — a ROCm/MI300
> kernel-optimization harness inspired by [Humanize](https://github.com/PolyArch) and the
> Kernel Design Agent (KDA). GPU: AMD Instinct MI300X (gfx942), ROCm 7.2, Triton 3.6,
> torch 2.9.1+rocm. All solutions stay `language=python` (portable PythonBuilder, no nvcc).

## 1. What we optimized

Five contest kernels, ported from NVIDIA CUDA to ROCm and then optimized **without ever weakening
correctness** (the official `verify.py` pass counts are a hard gate):

| Kernel | Role |
|---|---|
| `gdn_decode` | Gated Delta-Net single-token decode (k-last state layout) |
| `gdn_prefill` | Gated Delta-Net prefill (varlen) |
| `dsa_sparse_attention` | DeepSeek sparse MLA attention over a paged KV cache |
| `dsa_topk_indexer` | DeepSeek sparse-attention fp8 top-2048 indexer |
| `moe_fp8` | DeepSeek-V3 fp8 block-scale MoE (no-aux routing) |

Two loops ran against **immutable** baselines:
- **v2** vs `baseline/v1` (the contest denominator) — landed all five.
- **v3** vs `baseline-v2` (the shipped v2 state) — deepened the three highest-headroom kernels
  (`gdn_decode`, `moe_fp8`, `dsa_topk_indexer`); the other two were left untouched and re-checked for
  non-regression.

## 2. Results

Speedup vs the pure-torch `reference` is the contest's headline metric (`reference_ms / solution_ms`)
and is directly comparable across vendors (same torch code on every platform). The two ratio columns
below are vs the locked baselines. All numbers from the clean unified snapshot
[`results/amd_mi300.md`](amd_mi300.md) (`dirty=no`).

| Kernel | correctness | speedup vs torch ref | vs `baseline/v1` (contest) | v3 gain vs `baseline-v2` |
|---|---|---|---|---|
| `gdn_decode` | ✅ 54/54 | 48× – 3700× (grows w/ batch) | **3.8× – 6.1×** | **+64 – 79%** |
| `gdn_prefill` | ✅ 100/100 | 9× – ~3500× (grows w/ seq) | **+84.8%** (long seq) | ≈0% (untouched) |
| `dsa_sparse_attention` | ✅ 23/23 | 3.4× – 12.3× | **+70%** | ≈0% (untouched) |
| `dsa_topk_indexer` | ✅ 128/128 | 2.2× – 20× | ~1.0× (mr 1.0 default) | robustness fix¹ |
| `moe_fp8` | ✅ 19/19 (loose tol) | 1.5× – 4.5× | **1.14× – 1.51×** | **+3 to +24%** |

All five pass the **full** official `verify.py` at finalize (no `--fast` shortcut):
[`results/v3_final_verify.md`](v3_final_verify.md).

¹ `dsa_topk_indexer` carries a deliberate speed/robustness choice — see §3.

## 3. How each v3 win was found (the levers)

The throughline of RLCR is **profile first, then cut the dominant cost** — every candidate answered a
named profiling question (rocprofv3 stage trace) before any code was written
([`results/round0-v3-triage.md`](round0-v3-triage.md)).

### gdn_decode — IMPROVEMENT (+64–79% vs baseline-v2; 3.8–6.1× vs v1)
The profile showed **62–81%** of time in two host-side state transpose copies: the vendor decode
kernel wanted state as `[N,HV,K,V]` while the contest uses k-last `[B,HV,V,K]`. We wrote a repo-local
Triton kernel that copies the vendor gate/recurrent math verbatim but reads the input state and writes
the final state in the contest k-last layout **directly** (into a separate output buffer, no input
mutation), fusing gate + recurrent update + output into one dispatch. Both transposes vanish.
Evidence: [`v3_round3_gdndecode.md`](v3_round3_gdndecode.md), [`v3_round3_gdndecode_verify.md`](v3_round3_gdndecode_verify.md).

### moe_fp8 — IMPROVEMENT (+3–24% vs baseline-v2; up to 1.51× vs v1)
The profile showed the per-expert block-scale **weight dequant** dominating at *every* sequence
length (**63–80%**; the GEMM was only 5–23%). The torch dequant materialized two full fp32 weight
temporaries before the bf16 cast (~19 bytes of traffic per 1-byte fp8 input). We replaced **only the
dequant** with a Triton kernel that reads fp8 + the 128×128 block scale and writes bf16 directly
(~3 bytes/elem, one program per scale block) — **bit-identical** to the reference dequant, ~5× faster
standalone — and **kept rocBLAS for the GEMM**. The win grows with sequence length as the GEMM
amortizes launch overhead. Routing / SwiGLU / dequant parity is proven against independent references
in a standalone harness. Evidence: [`v3_round5_moe.md`](v3_round5_moe.md), [`v3_round5_moe_verify.md`](v3_round5_moe_verify.md).

### dsa_topk_indexer — robustness IMPROVEMENT + evidence-backed perf NO-GO
This is the honest one. We built the planned packed-page scoring kernel (reads the fp8 cache directly,
no dense gather/materialization, NaN-preserving ReLU, scale-after-dot identity, full-fp32 dot): it is
**+10–21%** and passes the official 128/128 — but its per-run matched-ratio is **~0.99**, below the
self-imposed `mr ≥ 0.999` robustness bar. We proved the two goals are **mutually exclusive** here:
only `torch.bmm` reproduces the reference selection bit-for-bit (mr 1.0), and that needs the dense
materialization the fast kernel removes (a Triton `tl.dot` tiles the reduction differently from
torch's GEMM; at these inputs' extreme dynamic range with signed weights, that tiny difference
mis-ranks ~0.5–1% of near-tie boundary tokens). So we **prioritized robustness**: the default is the
exact path (mr 1.0, 128/128), restoring the matched-ratio that the v2 fused path had eroded; the fast
kernel ships behind `DSA_TOPK_FAST=1` as a documented, contest-correct speed option.
Evidence: [`v3_round4_dsatopk_verify.md`](v3_round4_dsatopk_verify.md) (includes the rejected-lever
appendix and the `tl.dot`-vs-`torch.bmm` micro-test).

### Untouched kernels (non-regressed)
`dsa_sparse_attention` (v2 +70% — removed a per-call full-cache `torch.cat` via a kernel reading the
two paged caches separately) and `gdn_prefill` (v2 +84.8% long-seq — chunk-parallel prefill dispatch)
were left unchanged and confirmed ≈1.0× vs baseline-v2.

## 4. RLCR findings (what generalizes)

Reusable engineering lessons distilled across both loops:

1. **Profile host overhead before tuning the kernel.** Several "kernels" were overhead-bound — the
   biggest wins came from deleting host-side waste (full-cache `torch.cat`, state transposes, fp32
   dequant temporaries), not from micro-tuning launch params.
2. **A portable hand-written GEMM does not beat rocBLAS/Tensile.** A fused in-tile-dequant Triton GEMM
   was numerically correct yet slower end-to-end. The winning shape is *fused dequant feeding
   rocBLAS*, not *fused GEMM*.
3. **For a layout-mismatched vendor kernel, re-implement the math in the target layout** rather than
   transposing around it — removing both boundary copies beat shaving one.
4. **Exact top-k selection gated on bit-level reference parity needs the same GEMM the reference
   uses.** `tl.dot` (even full-fp32 `ieee`) can't bit-match `torch.bmm`; "fast + mr≈1.0" can be
   mutually exclusive, and the right answer is an evidence-backed NO-GO with a flagged speed option.
5. **gfx942 native fp8 is `e4m3fnuz`, not the contest's `e4m3fn`** (≈2× decode error) — keep software
   dequant; never bit-reinterpret to make a vendor fp8 path "work."
6. **An evidence-backed NO-GO is a first-class result.** It prevents reward-hacking (no per-workload
   fitting, no warmup/input-keyed caches) and produces an honest, generalizing verdict.

## 5. How the loop performed (methodology view)

- **7 rounds, within the 12-round budget**, exited *complete*. Convergence was monotonic: early rounds
  closed broad measurement-scaffold gaps, later rounds closed single specific evidence gaps, the last
  two closed exactly one issue each. No circling.
- **Review caught only real issues** (a mislabeling bug, a "clean" snapshot that was still dirty,
  unbacked pass-count claims, a self-comparison test, a silently-softened hard gate) — effectively
  zero false positives.
- **Main avoidable cost:** summaries occasionally claimed results before committing the backing
  artifact, turning a one-round task into two. The fix (now folded into our practice) is a
  claim↔committed-evidence discipline — doubly important because the reviewer could not re-run the
  GPU gates, so committed artifacts were the only verification channel.

## 6. Reproduce

```bash
export FIB_DATASET_PATH=<flashinfer-trace dataset>
# unified table (all 5 kernels, candidate vs baseline-v2 and baseline/v1)
python tools/run_benchmarks.py --out results/amd_mi300 --repeat-runs 3 --baseline-ref 'baseline-v2^{}'
# one kernel only
python tools/run_benchmarks.py --out /tmp/x --repeat-runs 3 --baseline-ref 'baseline-v2^{}' --only moe_fp8
# official correctness
python verify.py --solution solutions/<kernel>/solution.json --dataset $FIB_DATASET_PATH
# moe routing/SwiGLU/dequant parity vs independent references
python tools/moe_parity.py
```

Env switches recorded in result provenance (`sol_env`): `DSA_TOPK_FAST=1` (fast topk path),
`MOE_DEQUANT_TORCH=1` (reference dequant), `GDN_DECODE_FUSEDOP`/`GDN_DECODE_RECURRENT` (gdn fallbacks),
`MOE_USE_FUSED=1` (experimental fused GEMM, slower — kept as evidence).

## 7. Artifact index

- Unified results: [`amd_mi300.md`](amd_mi300.md) · final verdicts: [`v3_final-loop-report.md`](v3_final-loop-report.md) · full correctness: [`v3_final_verify.md`](v3_final_verify.md)
- Round-0 profiling/triage: [`round0-v3-triage.md`](round0-v3-triage.md)
- gdn_decode: [`benchmark`](v3_round3_gdndecode.md), [`verify`](v3_round3_gdndecode_verify.md), [`round report`](v3_round4-report.md)
- dsa_topk: [`default`](v3_round4_dsatopk.md), [`fast`](v3_round4_dsatopk_fast.md), [`verify+appendix`](v3_round4_dsatopk_verify.md), [`round report`](v3_round4-report.md)
- moe_fp8: [`benchmark`](v3_round5_moe.md), [`verify`](v3_round5_moe_verify.md), [`round report`](v3_round5-report.md)
- v2 loop (for history): [`final-loop-report.md`](final-loop-report.md)
