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

Profiling command per kernel/bucket (raw traces untracked at `/tmp/v3prof/<k>_<bucket>/k_kernel_trace.csv`):

```
rocprofv3 --kernel-trace --output-format csv -d /tmp/v3prof/<k>_<bucket> -o k -- \
    python tools/profile_kernel.py --kernel <k> --bucket <min|med|max> --iters <N>
```

Stage tables below are **regenerated verbatim** from those traces by the committed summarizer (not
hand-edited); per-bucket rows use the tool's exact schema
`bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class`:

```
python tools/summarize_rocprof_trace.py /tmp/v3prof/<k>_<bucket>/k_kernel_trace.csv \
    --kernel <k> --bucket <min|med|max>
```

### gdn_decode — `/tmp/v3prof/gdn_decode_{min,max}/k_kernel_trace.csv`
```
# gdn_decode_min  (227 dispatches, 1153 us total)
bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class
min,state_transpose_copy,elementwise_kernel_manual|CatArray|copyBuffer,117,718.2,6.14,62.3,copy
min,fused_recurrent_gate,fused_sigmoid_gating|fused_recurrent_gated,55,344.1,6.26,29.8,fused-recurrent
min,misc_elementwise,elementwise|reduce_kernel,55,90.8,1.65,7.9,elementwise
# gdn_decode_max  (224 dispatches, 8803 us total)
bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class
max,state_transpose_copy,elementwise_kernel_manual|CatArray|copyBuffer,114,7153.5,62.75,81.3,copy
max,fused_recurrent_gate,fused_sigmoid_gating|fused_recurrent_gated,55,1374.4,24.99,15.6,fused-recurrent
max,misc_elementwise,elementwise|reduce_kernel,55,274.8,5.00,3.1,elementwise
```

### dsa_topk_indexer — `/tmp/v3prof/dsa_topk_indexer_{min,max}/k_kernel_trace.csv`
```
# dsa_topk_indexer_min  (1492 dispatches, 4101 us total)
bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class
min,dequant_score,elementwise|reduce_kernel,1215,3097.0,2.55,75.5,elementwise
min,topk_sort,topk|MergeSort|radixSort|gatherTopK,110,546.9,4.97,13.3,topk
min,logits,_fused_logits|Cijk|bmm|gemm,55,262.0,4.76,6.4,fused-logits
min,other,,112,195.1,1.74,4.8,other
# dsa_topk_indexer_max  (1602 dispatches, 15319 us total)
bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class
max,dequant_score,elementwise|reduce_kernel,1270,9621.9,7.58,62.8,elementwise
max,topk_sort,topk|MergeSort|radixSort|gatherTopK,110,3796.8,34.52,24.8,topk
max,logits,_fused_logits|Cijk|bmm|gemm,55,1258.3,22.88,8.2,fused-logits
max,index_map,index|scatter|gather,55,347.9,6.32,2.3,indexing
max,other,,112,294.0,2.62,1.9,other
```

### moe_fp8 — `/tmp/v3prof/moe_fp8_{min,med,max}/k_kernel_trace.csv`
```
# moe_fp8_min (seq_len=1)  (12448 dispatches, 88637 us total)
bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class
min,weight_dequant_elementwise,elementwise_kernel_manual,1045,35598.7,34.07,40.2,elementwise
min,vectorized_elementwise,vectorized_elementwise,3858,25700.5,6.66,29.0,elementwise
min,other,,4080,12296.2,3.01,13.9,other
min,routing_reduce,reduce_kernel|trampoline|rocprim,2530,8099.2,3.20,9.1,reduce
min,gemm,Cijk|gemm,330,4711.9,14.28,5.3,GEMM
min,scatter_index,indexFunc|index|scatter,605,2230.7,3.69,2.5,indexing
# moe_fp8_med (seq_len=55)  (25602 dispatches, 330652 us total)
bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class
med,weight_dequant_elementwise,elementwise_kernel_manual,5460,181113.5,33.17,54.8,elementwise
med,vectorized_elementwise,vectorized_elementwise,7253,82060.2,11.31,24.8,elementwise
med,gemm,Cijk|gemm,1680,23812.8,14.17,7.2,GEMM
med,routing_reduce,reduce_kernel|trampoline|rocprim,5670,20502.3,3.62,6.2,reduce
med,scatter_index,indexFunc|index|scatter,2590,11930.0,4.61,3.6,indexing
med,other,,2949,11233.4,3.81,3.4,other
# moe_fp8_max (seq_len=14107)  (18477 dispatches, 365996 us total)
bucket,stage,kernel_pattern,launches,total_us,avg_us,pct,class
max,weight_dequant_elementwise,elementwise_kernel_manual,4600,153142.2,33.29,41.8,elementwise
max,gemm,Cijk|gemm,1280,84616.2,66.11,23.1,GEMM
max,vectorized_elementwise,vectorized_elementwise,4748,76995.6,16.22,21.0,elementwise
max,routing_reduce,reduce_kernel|trampoline|rocprim,3880,20815.4,5.36,5.7,reduce
max,other,,3289,19262.0,5.86,5.3,other
max,scatter_index,indexFunc|index|scatter,680,11164.9,16.42,3.1,indexing
```

**Reading the tables.** gdn_decode (now shipped in v3 round 3) was dominated by the two host-side
state transpose copies (62–81%). dsa_topk_indexer is dominated by `dequant_score` (the gather +
`k_deq[B,N,D]` fp8→fp32 materialization: 75.5% at B=1, 62.8% at B=31), then `topk_sort` (13–25%) —
this is the round-4 target. moe_fp8 is dominated at **every** seq_len by per-expert weight
dequant/materialization (`weight_dequant_elementwise` + `vectorized_elementwise` = 69%/80%/63% at
seq 1/55/14107); GEMM is only 5.3%/7.2%/23.1%. So the grouped/fused weight-materialization lever
applies across all seq lengths, not just launch overhead at small seq.

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
