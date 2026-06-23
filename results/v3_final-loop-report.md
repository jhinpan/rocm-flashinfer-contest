# v3 RLCR Loop — Final Report

Second optimization loop over the three highest-headroom ROCm/MI300X contest kernels (`gdn_decode`,
`dsa_topk_indexer`, `moe_fp8`), each pushed beyond its shipped **v2** state without weakening
correctness. Primary comparison base: immutable `baseline-v2` (commit 08d4780); also reported vs the
contest denominator `baseline/v1` (74c0918) and the pure-torch reference. GPU: AMD Instinct MI300X
(gfx942), ROCm 7.2, Triton 3.6, torch 2.9.1+rocm. Tooling: `rocm-KDA-pilot`, inspired by Humanize
and the Kernel Design Agent.

## Correctness (full official gate, DEC-4) — all five kernels pass
`results/v3_final_verify.md`: `moe_fp8` 19/19, `dsa_topk_indexer` 128/128, `dsa_sparse_attention`
23/23 (full `verify.py`); `gdn_decode` 54/54 and `gdn_prefill` 100/100 (in-process full sweep at
official atol=rtol=1e-2 + `verify.py --fast` 2/2; full subprocess `verify.py` is impractically slow
for these two — identical comparison performed in-process). No regression on any kernel.

## Per-kernel verdicts

### gdn_decode — IMPROVEMENT
Repo-local k-last Triton decode kernel that reads **and** writes state in the contest `[B,HV,V,K]`
layout directly (gate + recurrent update + output fused into one dispatch), removing both host-side
state transpose copies that dominated the profile (62–81%). **+65.0 / +66.9 / +78.1% vs baseline-v2**
(B=1/16/64); **3.9–5.9× vs baseline/v1**. 54/54, mr 1.0, inputs not mutated.
Evidence: `results/v3_round3_gdndecode.*`, `results/v3_round3_gdndecode_verify.md`.

### moe_fp8 — IMPROVEMENT
The per-expert block-scale weight dequant dominated the profile (63–80% at every seq_len; GEMM only
5–23%). The torch dequant materialized two full fp32 weight temporaries (~19 bytes/elem for 1-byte
fp8). Replaced **only** the dequant with a fused Triton kernel (fp8 + 128×128 block scale → bf16
direct, ~3 bytes/elem) — bit-identical to the reference dequant, ~5× faster standalone — while
keeping **rocBLAS for the GEMM** (a portable GEMM cannot match it). **+3.0 / +10.3 / +24.0% vs
baseline-v2** (seq 1/55/14107; seq1 marginal); **1.14 / 1.36 / 1.53× vs baseline/v1** (seq55 +36%,
seq14107 +53%, both exceeding the ≥20%-vs-v1 target). 19/19. Routing/SwiGLU/dequant parity proven
against independent references (`tools/moe_parity.py`).
Evidence: `results/v3_round5_moe.*`, `results/v3_round5_moe_verify.md`.

### dsa_topk_indexer — robustness IMPROVEMENT + perf NO-GO
Built the planned packed-page fused scoring kernel (reads fp8 cache via `block_table`, no `k_deq`
materialization, NaN-preserving relu, scale-after-dot, `input_precision="ieee"`): **+9.9 / +21.3 /
+21.5% vs baseline-v2** and official `verify.py 128/128`, but per-run **mr ~0.99** — below the DEC-2
`mr ≥ 0.999` hard gate. Proven infeasible to get both: only `torch.bmm` reproduces the reference
selection bit-for-bit (mr 1.0), and that needs the `k_deq` materialization the fast kernel removes
(`tl.dot` tiling ≠ torch GEMM; `ieee` maxrel 6e-4 amplified by negative-weight cancellation at the
inputs' extreme dynamic range). Per DEC-2 (prioritize robustness, else revert to torch), the
**default is the exact `torch.bmm` path: mr 1.000, 128/128**, restoring the robustness v2's fused
path lost (0.988/0.992). The fast kernel ships behind `DSA_TOPK_FAST=1` as a documented,
contest-correct speed option (+10–21%, 128/128).
Evidence: `results/v3_round4_dsatopk.*`, `results/v3_round4_dsatopk_fast.*`,
`results/v3_round4_dsatopk_verify.md`.

### Untouched this loop (confirmed non-regressed)
- `dsa_sparse_attention`: 23/23, ≈1.00× vs baseline-v2 (the v2 +70%-vs-v1 win retained).
- `gdn_prefill`: 100/100, ≈1.00× vs baseline-v2 (the v2 +84.8%-long-seq win retained).

## Headline v3 results (authoritative unified snapshot `results/amd_mi300.{md,csv}`, dirty=no)
Per-workload ratios from the clean all-kernel run (candidate vs baseline-v2 / vs baseline/v1):

| Kernel | buckets | **vs baseline-v2** | **vs baseline/v1** | correctness |
|---|---|---|---|---|
| gdn_decode | B=1/16/64 | **2.78 / 3.03 / 4.70×** (+64/+67/+79%) | **3.81 / 4.15 / 6.09×** | 54/54 |
| moe_fp8 | seq 1/55/14107 | 1.03 / **1.11 / 1.30×** (+2.5/+9.7/+23.0%) | **1.14 / 1.33 / 1.51×** | 19/19 |
| dsa_topk_indexer (default, mr 1.0) | B=1/14/31 | 0.85 / 0.93 / 0.94× (robustness fix) | 0.89 / 0.98 / 0.98× | 128/128 |
| dsa_sparse_attention (untouched) | tok 1/6/8 | ≈1.00× | 3.49 / 3.38 / 3.37× | 23/23 |
| gdn_prefill (untouched) | seq 6/139/8192 | ≈1.00× | up to 6.63× (long seq) | 100/100 |

`dsa_topk_indexer` with `DSA_TOPK_FAST=1` is +9.9/+21.3/+21.5% vs baseline-v2 (1.17/1.38/1.36× vs v1)
at mr ~0.99 (contest-correct, 128/128) — available behind the flag for speed over the 0.999 bar.
moe_fp8 seq1 (+2.5%) is marginal (near the noise bar), not a headline win.

## Loop accounting
6 rounds (within `--max 12`). Round 3 gdn_decode; round 4 dsa_topk verdict + scaffold debt; round 5
moe_fp8 + dsa_topk evidence; round 6 finalize. All DEC resolutions honored (DEC-1 >3–5% noise bar /
moe ≥20% vs v1; DEC-2 dsa_topk mr≥0.999 hard gate → robustness default; DEC-3 no input-keyed caches;
DEC-4 full official verify at finalize). No reward hacking: shape-family kernels only, full official
sweeps checked, no per-workload constants or warmup caches.
