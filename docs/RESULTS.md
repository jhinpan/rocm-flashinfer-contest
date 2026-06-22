# ROCm/MI300 port — results

All 5 MLSys2026 FlashInfer contest kernels ported to AMD Instinct MI300X (gfx942, ROCm 7.2,
torch 2.9.1+rocm, Triton 3.6 ROCm, AITER). Each `solution.json` is `language=python` (uses
AITER Triton ops or pure torch), built by the portable PythonBuilder — no nvcc, no CUDA deps.

Validated against the pure-torch `reference` (the contest's correctness oracle AND speedup
denominator) through the official `verify.py` / `flashinfer_bench` scoring path. Speedup =
`reference_latency / solution_latency` on MI300 (the reference is an unoptimized python loop,
so these are denominator-relative, not vs an NVIDIA kernel).

| # | Kernel | Status | Correctness | Speedup vs ref (MI300) | Approach |
|---|--------|--------|-------------|------------------------|----------|
| 1 | gdn_decode | ✅ PASS | 54/54 local; official 2/2 | 12×–574× (grows with batch) | AITER `fused_recurrent_gated_delta_rule` + k-last state transpose, log-space g |
| 2 | dsa_sparse_attention | ✅ PASS | 23/23 local; official 2/2 | ~1.0×–1.4× (kernel unoptimized) | AITER `unified_attention_sparse_mla`, fused [ckv|kpe], absolute topk indices |
| 3 | gdn_prefill | ✅ PASS | small+multi-seq+medium; official 2/2 | up to 272× (67× on --fast) | AITER `fused_recurrent_gated_delta_rule` varlen (B=1 + cu_seqlens) |
| 4 | dsa_topk_indexer_fp8 | ✅ PASS | official on small+largest workloads | 2×–13.8× | vectorized fp32 dequant + scores + topk (sidesteps fnuz/fn) |
| 5 | moe_fp8_block_scale | ✅ PASS | small+seq901+seq14107; official 2/2 | ~1.3×–3.8× | torch routing + per-active-expert bf16 dequant/GEMM/SwiGLU |

## Key facts that made it work
- **Harness**: one 48-line patch to `flashinfer_bench/bench/timing.py` (HIP-event fallback for
  the CUDA `flashinfer.testing.bench_gpu_time_with_cupti` import). Python/Triton builders are
  fully portable; CUDA/tvm-ffi builder (nvcc) is not — so all solutions are `language=python`.
- **Baseline denominator** is the pure-torch `reference`, which runs as-is on ROCm. The NVIDIA
  `flashinfer_wrapper_*` baselines do NOT run on ROCm (CUDA-only ops) and are not needed.
- **fp8 FNUZ vs FN**: avoided entirely. The topk-indexer evaluator re-scores returned indices
  with a canonical fp8→fp32 dequant; MoE tolerance is loose. Both solutions dequant fp8→fp32/bf16
  in software, so MI300's e4m3fnuz MMA dtype never enters the correctness path.
- **moe is "CUDA-mandated"** for the NV submission; this is a portable python equivalent that
  passes the official loose tolerance (atol=1, rtol=0.3, matched_ratio=0.9).

## Layout / math gotchas captured in the solutions
- GDN: AITER wants decay `g` in **log space** (`-exp(A_log)*softplus(a+dt_bias)`), state as
  `[N,HV,K,V]` vs contest k-last `[N,HV,V,K]` (transpose last two dims). GVA handled internally.
- DSA sparse MLA: layout is `[lora(ckv 0:512) | rope(kpe 512:576)]`; each query token is a
  length-1 sequence (ALL_DECODE); kernel masks `-1` padding directly; `block_table` unused for
  indexing (topk indices are absolute).
- topk-indexer: deep_gemm packed KV is per-page `[page_size*128 fp8 | page_size*4 scale]`;
  token_position == flat index; global token = `block_table[b, pos//64]*64 + pos%64`.
- MoE: SwiGLU gate = **second** half (`silu(G1[:, I:]) * G1[:, :I]`); `hidden_states_scale`
  is transposed `[H/128, T]`; weights from **unbiased** sigmoid `s`, normalized, ×routed_scaling.

## Reproduce
```bash
export FIB_DATASET_PATH=/sgl-workspace/workspace/mlsys2026-flashinfer-contest/data/flashinfer-trace
export FIB_CACHE_PATH=/tmp/fib_cache
cd /sgl-workspace/workspace/rocm-port
# fast local check (faithful to official scoring; pass moe tol for moe):
python tools/local_verify.py --def <definition> --sol solutions/<k>/main.py [--atol 1 --rtol 0.3 --mr 0.9]
# official scorer:
cd /sgl-workspace/workspace/mlsys2026-flashinfer-contest
python verify.py --solution /sgl-workspace/workspace/rocm-port/solutions/<k>/solution.json --fast
```

## Next (Phase 3 — optimization, not yet done)
- dsa_sparse_attention: AITER kernel self-described "not optimized" (BLOCK_M=16, stages=1) —
  tune for gfx942; biggest headroom (~1×).
- gdn_decode/prefill: try AITER chunk path for long prefill; fuse gate compute / drop transposes.
- topk-indexer & moe: move to AITER fp8 paged-mqa-logits / fused_experts for native fp8 MMA
  if more speedup is needed (correctness already locked via software dequant).
