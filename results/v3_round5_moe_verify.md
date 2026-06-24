# moe_fp8 (v3) — official verify.py transcript

Authoritative correctness gate for the shipped moe_fp8 solution (fused Triton block-scale dequant +
rocBLAS GEMM; helpers refactored in round 6 with `run()` behavior unchanged). Benchmark artifact:
`results/v3_round5_moe.*`. GPU: AMD Instinct MI300X (gfx942) · dataset: `flashinfer-trace`.

```
$ python verify.py --solution solutions/moe_fp8/solution.json --dataset <flashinfer-trace>
solution:   moe_fp8_block_scale_rocm_v1
definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048
passed:     19/19
latency:    8.444888 ms mean
speedup:    2.0794x mean
```

**19/19** at the official loose tolerance (atol=1, rtol=0.3, matched_ratio=0.9). The round-6 helper
refactor (`_dequant_hidden` / `_route` / `_swiglu_contiguous`) preserves the exact pass count.

Parity (tools/moe_parity.py, `PARITY OK`): weight dequant bit-exact (maxabs 0); hidden dequant vs
independent expand exact; routing top-k id sets + normalized weights vs an independent recompute
exact; contiguous-half SwiGLU == reference and != interleaved variant; synthetic full-path
triton-dequant == `MOE_DEQUANT_TORCH=1` exact; official full-path 4/4 vs reference.
