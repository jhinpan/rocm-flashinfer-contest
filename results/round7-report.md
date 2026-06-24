# Round 7 report — dsa_topk_indexer task5 closed with executed evidence

**GPU:** AMD Instinct MI300X (gfx942) · **Baseline:** `baseline-v1^{}` = 74c0918 · **Gate:** official
`verify.py`. All claims below were executed this round (not report-only).

## Verdict: `dsa_topk_indexer` → NO-GO for ≥20% (robust torch path shipped, verify.py 128/128)

Both planned levers were implemented and run; neither yields a safe ≥20% win.

### Lever 1 — AITER `deepgemm_fp8_paged_mqa_logits` (logits): FALSIFIED
- **Committed probe `tools/fp8_dtype_probe.py`** (reproducible): `aiter.dtypes.fp8 ==
  torch.float8_e4m3fnuz` on gfx942. The contest KV cache (`e4m3fn`) decoded as fnuz vs fn is
  **exactly 2× off**: mean|v| 21.38 (fnuz) vs 42.76 (fn), `max_rel=0.50`, `max_abs=224.0` over
  96.5M finite entries. AITER's op does `kv_cache.view(dtypes.fp8)` → it would misread the KV.
- **Wired & executed:** the env-gated AITER-logits candidate run through official `verify.py` **hung
  >700 s** (vs the default's ~minutes) and a direct single-workload run triggered a **GPU HSA
  hardware exception** (coredump). Non-viable on the contest inputs; not retained in the solution
  (shipping GPU-crashing code is unsafe).

### Lever 2 — AITER `aiter.ops.triton.topk.topk`: non-viable
- Wired & executed: env-gated AITER top-k candidate run through official `verify.py` **hung >700 s**;
  not retained.

### Lever 3 — software-dequant fused-logits Triton kernel (`DSA_TOPK_FUSED=1`): correct but marginal
- Computes `scores[B,N]` directly from q + dequantized k (no `[B,H,N]`). Official `verify.py`
  **128/128**, **+5–8%** vs baseline-v1 (`results/v2_round5_topk.*`, env-stamped). BUT per-run matched
  ratio drops to ~0.988 (torch = 1.000) — a thinner correctness margin not worth a sub-20% gain.
  Kept behind `DSA_TOPK_FUSED=1` as tested evidence; not the shipped default.

### Decision
Ship the **torch path** (default; `verify.py` 128/128, mr 1.000, unchanged → no regression). The
kernel is already well-optimized (2.4–20× vs ref, ~0.3 ms); its cost (dequant + `bmm` + `torch.topk`)
is irreducible portably, and the only ≥20%-class lever (AITER deepgemm) is fp8-dtype-blocked.
**Verdict: NO-GO for ≥20%.**

## Scoreboard
| Kernel | verdict | evidence |
|---|---|---|
| `dsa_sparse_attention` | IMPROVEMENT +70% | verify 23/23, `results/v2_round2.md` |
| `moe_fp8` | NO-GO ≥20%; +10–17% fallback shipped | verify 19/19, `results/round4-report.md`, `v2_round4_moe` |
| `dsa_topk_indexer` | NO-GO ≥20% | verify 128/128, probe + AITER timeout/crash + fused +5-8%/mr0.988 |
| `gdn_prefill` | pending | |
| `gdn_decode` | pending | |
