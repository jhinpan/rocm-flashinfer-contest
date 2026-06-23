# Round 9 report — gdn_prefill chunk path (IMPROVEMENT) [drift recovery]

**GPU:** AMD Instinct MI300X (gfx942) · **Baseline:** `baseline-v1^{}` = 74c0918 ·
**Harness:** `tools/run_benchmarks.py --repeat-runs 3`.

## Verdict: `gdn_prefill` → IMPROVEMENT (+84.8% on long sequences)

| Workload | baseline-v1 ms | candidate ms (min–max) | cand/base | Δlat | path | correctness |
|---|---:|---:|---:|---:|---|:--:|
| total_seq_len=6 | 0.165 | 0.166 (0.165–0.168) | 0.99× | −0.8% | recurrent | ✅ |
| total_seq_len=139 | 0.173 | 0.173 | 1.00× | −0.0% | recurrent | ✅ |
| total_seq_len=8192 | 3.515 | 0.534 (0.527–0.538) | **6.58×** | **+84.8%** | chunk | ✅ |

- **Correctness:** in-process full sweep over all 100 workloads at the official tolerances
  (`atol=rtol=1e-2`, output + new_state) → **100/100 PASS** (chunk path exercised on 18 workloads);
  `verify.py --fast` → 2/2 PASS. (Note: the full `verify.py` is inherently slow for gdn_prefill —
  ~100 workloads × per-workload subprocess isolation > 30 min, independent of this change; the
  in-process sweep performs the identical correctness comparison without that overhead.)
- **≥20% bar:** cleared decisively on the long-seq family (+84.8%, 6.58×, tight spread). Short
  sequences are unchanged (recurrent path) — no regression.

## What changed
`solutions/gdn_prefill/main.py` length-dispatches: `total_seq_len ≥ 4096` → `chunk_gated_delta_rule`
(parallel over chunks; q/k expanded to HV heads for GVA, as the chunk kernel — unlike the recurrent
kernel — does not expand internally); otherwise the original `fused_recurrent_gated_delta_rule`. Same
log-space gate / fp32 beta math; k-last state adapter unchanged. The recurrent path is the fallback
(`GDN_PREFILL_RECURRENT=1` forces it).

## Why threshold 4096
The chunk kernel autotune-compiles per shape; threshold 4096 limits chunk to 3 distinct shapes in the
workload set (vs 14 at 1024) while preserving the long-seq win (the only long benchmark bucket is
8192). Compilation is one-time/cached; benchmark timing (warmed) reflects the +84.8% win.

## Scoreboard
| Kernel | verdict |
|---|---|
| `dsa_sparse_attention` | IMPROVEMENT **+70%** (verify 23/23) |
| `gdn_prefill` | IMPROVEMENT **+84.8%** long-seq (100/100 in-process; --fast 2/2) |
| `moe_fp8` | NO-GO ≥20%; +10–17% fallback shipped (verify 19/19) |
| `dsa_topk_indexer` | NO-GO ≥20% (verify 128/128) |
| `gdn_decode` | pending |
