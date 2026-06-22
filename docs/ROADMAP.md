# ROADMAP — Porting the MLSys2026 FlashInfer Contest to ROCm / MI300

A phased, concrete execution plan to (0) bring the harness up on ROCm, (1) establish per-kernel ROCm baselines, (2) port the four addressable kernels to Triton/AITER easiest→hardest, and (3) optimize and compare NV-vs-AMD. Owners are named by kernel. Cite file paths/function names throughout.

Principle: the pure-torch `reference` field in each definition JSON is BOTH the correctness oracle AND the speedup denominator on ROCm. Never depend on the NVIDIA baselines (`flashinfer`/`deep_gemm`/`cutlass` deps don't import here). Author all solutions as `language=python` or `language=triton` so the portable PythonBuilder/TritonBuilder handle them; never `language=cuda` (nvcc absent).

---

## Phase 0 — Harness bring-up on ROCm

Goal: a clean, reviewable patch set that makes `import flashinfer_bench` succeed and lets python+triton solutions build, run, and time against the torch reference on MI300.

### Tasks

- **0.1 Turn the bring-up patch into a clean patch set.** File: `flashinfer_bench/bench/timing.py`.
  - Replace the top-level `from flashinfer.testing import bench_gpu_time_with_cupti` (line 13) with a guarded import. Preferred 3-tier resolution inside `time_runnable`: try `flashinfer.testing.bench_gpu_time_with_cupti` → `flashinfer.testing.bench_gpu_time_with_cuda_event` → a local `torch.cuda.Event(enable_timing=True)` HIP-event fallback.
  - In `time_runnable` (around lines 71-82), bind `bound = lambda: fn(*args)` and call `bench_gpu_time_with_cupti(bound, dry_run_iters=warmup, repeat_iters=iters)` — **drop** the fork-incompatible `input_args=tuple(args)` and `cold_l2_cache=True` kwargs; use `l2_flush=True` (fork default) if available. Keep `with torch.cuda.device(device)` and the per-device lock. Return `statistics.median(times)`.
  - Tag changes `FIB-ROCm-PATCH`. (The applied bring-up diff was 1 file, +48/-1; this is the basis.)
- **0.2 (Optional, cleaner errors) Gate CUDA builders on nvcc.** In `compile/builders/tvm_ffi_builder.py` `TVMFFIBuilder.is_available()` add `and shutil.which('nvcc') is not None`. Optionally same for `torch_builder.py`. Makes `language=cuda` solutions fail fast with "no builder" instead of a late nvcc BuildError. Not required for python/triton.
- **0.3 Install.** `pip install -e /tmp/flashinfer-bench --no-deps` (done in bring-up). Verify runtime deps (pydantic, numpy, torch, click/typer) individually; never let pip pull the CUDA `flashinfer-python` wheel.
- **0.4 Establish the torch reference as the ROCm denominator.** Wire a tiny harness that loads a definition via `Definition.model_validate`, exec's `reference`, runs `run()` on `cuda` (HIP), and times it via the patched fallback. Generalize `/tmp/run_gdn_ref.py` into `tools/run_reference.py <definition.json> <workload_idx>`.
- **0.5 Smoke test.** `python -c 'import flashinfer_bench'` (succeeds), then run a trivial python reference-only definition through `Benchmark` with `use_isolated_runner` to confirm the baseline timing + correctness path works end-to-end on MI300.

### Acceptance criteria
- `import flashinfer_bench` succeeds with no `flashinfer` installed.
- `BuilderRegistry` reports PythonBuilder + TritonBuilder available; a trivial `language=python` solution builds and times.
- `tools/run_reference.py` reproduces the gdn_decode reference at ~1.86 ms (B=1,T=1) within noise.
- Patch set is a single coherent diff (timing.py + optional builder gate), `FIB-ROCm-PATCH` tagged, ready for a PR.

**Owner:** harness lead (cross-kernel). **Exit gate for all of Phase 1.**

---

## Phase 1 — Per-kernel ROCm baselines (run the torch reference for all 5)

Goal: confirm every reference `run()` executes on MI300 and record the denominator latency per workload. No solutions yet.

### Tasks (one per kernel; same shape)
For each kernel: load its definition JSON, exec `reference`, build inputs for the **8 representative dev workloads** (UUIDs in `prompts/*/phase1.md`), run `run()` on cuda(HIP), verify shapes/dtypes/finiteness, time via the HIP-event fallback, and log to `benchmark.csv` (denominator column).

- **1.1 gdn_decode** — `/tmp/fib-meta/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json`. Batches 1,1,4,8,16,32,48,64; HV=8,H=4,K=V=128,T=1. (Already validated at B=1.)
- **1.2 gdn_prefill** — `gdn/gdn_prefill_qk4_v8_d128_k_last.json`. Seqlens 6..8192, num_seqs 1..48.
- **1.3 dsa_sparse_attention** — `dsa_paged/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64.json`. num_tokens 1-8, num_pages 8462.
- **1.4 dsa_topk_indexer_fp8** — `.../dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.json`. Confirm fp8 e4m3fn cache views work (verified at harness level).
- **1.5 moe_fp8_block_scale** — `.../moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json`. Seqlens 1..14107. (Denominator only; kernel is CUDA-mandated.)

### Acceptance criteria
- All 5 references run on MI300; outputs finite with correct shapes/dtypes.
- `benchmark.csv` has per-workload reference latency for all 5 kernels (the speedup denominators).
- fp8 references (1.4, 1.5) confirmed to execute with e4m3fn data unchanged.

**Owners:** one per kernel. **Blocked by:** Phase 0.

---

## Phase 2 — Per-kernel Triton/AITER ports (easiest → hardest)

Goal: a correct ROCm solution per addressable kernel, validated against the torch reference at the harness tolerances (atol 1, rtol 0.3, matched-ratio 0.9). Order chosen by difficulty from the per-kernel analysis.

### 2.A gdn_decode — **easy** (do first)
**Building blocks:** AITER `fused_sigmoid_gating_delta_rule_update` (`/sgl-workspace/aiter/aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_sigmoid_gating_recurrent.py`, kernel+launcher, HIP cfg BV=64/warps=4 lines 24-27); public `fused_recurrent_gated_delta_rule` (`.../gated_delta_net/gated_delta_rule.py:36`); usage example `op_tests/test_gated_delta_rule.py:375`.

**Tasks:**
1. Standalone repro importing `fused_sigmoid_gating_delta_rule_update`; run the 8 shapes vs the definition reference at atol1/rtol0.3/0.9.
2. Layout adapter: transpose k-last `[B,8,V,K]`→`[B,8,K,V]` clone; `initial_state_indices=arange(B)`; `softplus_beta=1.0`, `softplus_threshold=20.0`, `use_qk_l2norm_in_kernel=False`, scale passthrough; kernel updates state in-place → transpose back to k-last for `new_state`. None→zeros `[B,8,128,128]`.
3. Package as `language=python` solution; `run(q,k,v,state,A_log,a,dt_bias,b,scale)` mirroring NV `main.py`.
4. `verify.py` on 8 reps then all 54 workloads.
5. If launch/transpose overhead dominates at batch 64, prototype a fresh k-last Triton kernel writing `new_state` directly (no transposes, fused gates).

**Acceptance:** passes all 54 at tolerance; latency vs denominator logged. **Owner:** gdn_decode.

### 2.B dsa_sparse_attention — **easy** (do second)
**Building blocks:** AITER `unified_attention_sparse_mla` wrapper (`/sgl-workspace/aiter/aiter/ops/triton/attention/unified_attention_sparse_mla.py`), kernel `_kernel_unified_attention_sparse_mla_2d` (`_triton_kernels/attention/unified_attention_sparse_mla.py`), calling example `op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py`; lse pattern in `mla.py`/`mla_decode.py` if needed later.

**Tasks:**
1. `run()`: `q=cat([q_nope,q_pe],-1)→[T,16,576]`; `kv=cat([ckv,kpe],-1)` reshaped `[num_pages,64,1,576]`; `cu_seqlens_q=arange(T+1)` (ALL_DECODE, max_seqlen_q=1); `seqused_k=(sparse_indices!=-1).sum(1).int()`; `max_seqlen_k=seqused_k.max()`; `topk_indices=sparse_indices` (absolute flattened); trivial `block_table`; out `[T,16,512]` bf16; return `(out,)` (lse optional per evaluator).
2. Guard all-`-1` token edge case (reference zeroes output / lse=+inf).
3. Validate vs reference on 8 dev workloads; `verify.py --fast` to confirm build+correctness via python/triton builder.

**Acceptance:** correctness passes on dev workloads then full set. **Owner:** dsa_sparse_attention.

### 2.C gdn_prefill — **medium** (do third)
**Building blocks:** AITER `chunk_gated_delta_rule` and `chunk_gated_delta_rule_opt_vk` (`/sgl-workspace/aiter/aiter/ops/triton/gated_delta_net/gated_delta_rule.py`); orchestrators `_triton_kernels/.../prefill/chunk.py`, stages `fused_cumsum_kkt.py`, `fused_solve_tril_recompute.py`+`utils/solve_tril.py`/`wy_representation.py`, `chunk_delta_h.py`, `chunk_o.py`; per-token fallback `fused_recurrent_gated_delta_rule`; sglang wiring `srt/layers/attention/linear/gdn_backend.py` + test kit `gdn_attention.py`.

**Tasks:**
1. `language=python` solution mirroring NV `main.py`: `g = -exp(A_log.float())*softplus(a.float()+dt_bias.float())` (**log-space**), `beta=sigmoid(b.float())`; B=1 unsqueeze for varlen.
2. Prefer `chunk_gated_delta_rule_fwd_opt_vk` (native k-last `[N,H,V,K]`); else transpose initial/final state `(-1,-2)`. Read `chunk_delta_h.py` `_opt_vk` to confirm final_state layout.
3. Confirm GVA: pass 4 q/k + 8 v heads, let AITER expand via Hg (read `chunk_delta_h.py` ~695-746, 1034-1087); do not pre-`repeat_interleave`.
4. Validate vs reference on 8 dev workloads (seqlens 6..8192); check fp32 `new_state` separately (tighter error).
5. If chunk path fails tolerance on new_state, fall back to `fused_recurrent_gated_delta_rule` (per-token, matches reference) as the correctness-safe baseline, then optimize toward chunk.
6. Sanity-check AITER GDN kernels JIT-compile on gfx942 (no tcgen05 assumptions; ensure gfx942 path, not `_is_gfx12_runtime`).

**Acceptance:** passes 8 dev then full 100; new_state within tolerance. **Owner:** gdn_prefill.

### 2.D dsa_topk_indexer_fp8 — **medium / fp8-risk** (do fourth)
**Building blocks:** AITER `deepgemm_fp8_paged_mqa_logits` + `_schedule` (`/sgl-workspace/aiter/aiter/ops/triton/attention/pa_mqa_logits.py`), inner kernels `_deepgemm_fp8_paged_mqa_logits`/`_stage1`/`_ragged_k` (`_triton_kernels/attention/pa_mqa_logits.py`), gluon variants `ops/triton/gluon/pa_mqa_logits.py`; sglang `dsa_indexer.py` (`kv_cache_cast_to_fp8`, `ref_fp8_paged_mqa_logits`), `dsa_topk_backend.py` (`_topk_unfused`, `topk_transform` PAGED); `get_fp8_e4m3_dtype`; `torch.topk`.

**Tasks:**
1. Read AITER inner kernels; confirm KV byte-split for `KVBlockSize=64`/`index_dim=132` matches the contest per-page `[fp8 | scale]` packing; check fp8 pointer dtype and whether it can be set to `float8_e4m3fn`.
2. Minimal repro (batch=1): run reference to get ground-truth `topk_indices`.
3. Prototype stage-1 two ways, diff vs reference logits: (a) AITER `deepgemm_fp8_paged_mqa_logits` with int8 cache as uint8; (b) fresh ~40-line Triton kernel mirroring AITER math with explicit **fp8_e4m3fn→fp32 dequant**. Pick whichever matches the fn-quantized reference.
4. Stage-2 with `torch.topk` first; reproduce page→global transform `block_table[b, idx//64]*64 + idx%64` with -1 padding; validate full `run()` on all 8 dev workloads.
5. **Resolve fnuz/fn definitively** by decoding the contest's actual byte patterns under both dtypes; lock the kernel to the dtype that reproduces the reference (see GAP_REPORT §7).
6. Once correct, if top-k dominates, port a Triton top-2048 (threshold/bitonic) and try the gluon logits variant.

**Acceptance:** full `run()` matches reference on dev workloads; fp8 dtype path locked and documented. **Owner:** dsa_topk_indexer.

### 2.E moe_fp8_block_scale — **medium, CUDA-mandated (optional/last)**
The contest mandates CUDA for this kernel; it is not buildable via the portable path. Pursue only if a ROCm submission is accepted (Open Question 6). If pursued:

**Building blocks:** sglang `topk.py::biased_grouped_topk_impl` (exact DeepSeek routing), `select_experts`/`TopK`; `moe_runner/triton_utils/fused_moe.py::fused_experts` (fp8 w8a8 block-scale), `fused_moe_triton_kernels.py::invoke_fused_moe_kernel`; `quantization/fp8_kernel.py::is_fp8_fnuz`; AITER `moe_op_gemm_a8w8_blockscale.moe_gemm_a8w8_blockscale`, `moe_op_silu_fused`, `moe_align_block_size`, `gemm_afp8wfp8`, `fmoe_fp8_blockscale_g1u1`.

**Tasks:** routing via `biased_grouped_topk_impl` (renormalize=True, routed_scaling_factor on output, +1e-20, **unbiased** s for weights) → assert topk_ids/weights match reference → local-expert remap (global top-8 → `[offset, offset+32)`) → per-expert block-scale GEMM with **fp32 upcast** (e4m3fn→fp32) + **contiguous-half** SwiGLU + weighted scatter. Reconcile transposed `hidden_states_scale [H/128,T]`; avoid AITER's interleaved/add_residual swiglu unless reordered. Validate at atol1/rtol0.3/0.9; then swap inner loop to `fused_experts`.

**Acceptance:** routing matches reference; end-to-end within tolerance (if scored). **Owner:** moe.

### Phase 2 acceptance criteria (all)
- gdn_decode, dsa_sparse_attention, gdn_prefill, dsa_topk_indexer_fp8 each pass `verify.py` at full workload coverage.
- Each solution is `language=python`/`triton`, builds via the portable builder, no CUDA deps.
- fp8 dtype handling documented and locked for the two fp8 kernels.

---

## Phase 3 — Optimization + NV-vs-AMD comparison methodology

Goal: maximize speedup vs the ROCm denominator and produce an apples-to-apples NV-vs-AMD comparison.

### Tasks
- **3.1 Per-kernel tuning.**
  - gdn_decode: profile at batch 64; if transpose/launch overhead dominates, ship the fresh k-last Triton kernel (no transposes, fused gates).
  - dsa_sparse_attention: the AITER kernel is self-described "not optimized" (BLOCK_M=16, TILE_SIZE=64, num_stages=1). It's K-bound (topk=2048, tiny num_tokens ≤8). Tune BLOCK_M/TILE_SIZE/num_warps/num_stages for gfx942; consider split-K over the topk loop.
  - gdn_prefill: tune chunk size / autotune configs on gfx942; verify chunk path beats per-token fallback.
  - dsa_topk_indexer: Triton top-2048 + gluon logits variant if top-k/logits dominate.
- **3.2 Timing rigor.** Use the harness HIP-event path consistently (one CUPTI UserWarning expected). Record min/median/max over ≥50 iters with warmup + L2 flush. Log all to `benchmark.csv`.
- **3.3 NV-vs-AMD methodology.** Compare **speedup ratios** (solution / its-platform torch reference), not raw ms, because the reference denominator differs per platform. Document: same workload params, same tolerance, ratio = `reference_latency / solution_latency` on each platform. Note fp8 path differences (NV native e4m3fn MMA vs MI300 fp32-upcast or fnuz) as a caveat on the two fp8 kernels.
- **3.4 (Optional) ROCm profiling backend.** If kernel-level profiling is needed, add rocprofv3/omniperf behind `shutil.which` in `agents/ncu.py`/`agents/sanitizer.py` (off the scoring path; graceful today).

### Acceptance criteria
- Each ported kernel shows a measured speedup over its ROCm reference denominator, logged in `benchmark.csv`.
- A short NV-vs-AMD comparison table using speedup ratios, with fp8 caveats noted.
- Tuning configs committed alongside solutions.

---

## Prioritized next 5 actions

1. **Land the Phase 0 patch set.** Clean up `flashinfer_bench/bench/timing.py` (guarded import + zero-arg closure + HIP-event fallback, `FIB-ROCm-PATCH`), `pip install -e --no-deps`, confirm `import flashinfer_bench` and a trivial python solution builds+times. (Owner: harness lead.)
2. **Run all 5 torch references on MI300 and record denominators** into `benchmark.csv` via `tools/run_reference.py` for the 8 dev workloads each. (Owners: per kernel.)
3. **Ship gdn_decode** by wrapping AITER `fused_sigmoid_gating_delta_rule_update` with the k-last↔[K,V] state-transpose adapter; pass `verify.py` on all 54 workloads. (Owner: gdn_decode.) — easiest, fastest first win.
4. **Ship dsa_sparse_attention** by wrapping AITER `unified_attention_sparse_mla` (fused q/kv + varlen plumbing + absolute topk indices); pass dev then full workloads. (Owner: dsa_sparse_attention.) — second easy win, native gfx942 kernel exists.
5. **Resolve the fp8 fnuz-vs-fn question** for dsa_topk_indexer: decode the contest byte patterns under both dtypes, decide AITER-op vs fresh-dequant-Triton for stage-1, and lock the dtype path. This unblocks both fp8 kernels and is the single biggest technical risk. (Owner: dsa_topk_indexer.)
