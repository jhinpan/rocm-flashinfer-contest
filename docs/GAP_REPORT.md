# GAP REPORT — Running the MLSys2026 FlashInfer Contest on ROCm / MI300

What's missing (and what already works) to run the MLSys2026 FlashInfer kernel contest on an AMD Instinct MI300X (gfx942, CDNA3) under ROCm 7.2.0, torch 2.9.1+rocm7.2, triton 3.6.0 (ROCm), amd-aiter, and sglang.

---

## 1. Executive summary

**The harness core runs on ROCm today after exactly one small patch.** The pure-torch references — which are simultaneously the *correctness ground truth* and the *speedup denominator* — execute unchanged on the MI300X via torch's HIP backend, and kernel timing works via HIP events. We measured the `gdn_decode` reference at **~1.86 ms median** (smallest workload) end-to-end through the patched harness.

**In plain terms:**

- **Works now:** the `flashinfer_bench` data models (pydantic), the builder registry, the Python and Triton builders, pure-torch reference execution, and HIP-event timing. This is everything needed to (a) load a definition, (b) run its reference, and (c) time/score a `python` or `triton` solution against that reference.
- **One hard blocker, already fixed:** `flashinfer_bench/bench/timing.py` line 13 does a *top-level* `from flashinfer.testing import bench_gpu_time_with_cupti`. `flashinfer` is not installed on this box, so `import flashinfer_bench` itself failed. A 48-line try/except patch with a `torch.cuda.Event` (HIP-event) fallback unblocks the entire package.
- **Cannot run on ROCm:** all five contest *baselines*. They are `language=python, deps=['flashinfer']` (or `deep_gemm`, `cutlass`) wrappers around CUDA/SM90/SM100 ops (`gated_delta_rule_decode_pretranspose`, `trtllm_fp8_block_scale_moe`, `trtllm_batch_decode_with_kv_cache_mla`, `deep_gemm.fp8_paged_mqa_logits`, `top_k_page_table_transform`). None of these ops exist in the ROCm flashinfer fork, and their pip deps don't import on MI300. **Baselines must be reconstructed from AITER/sglang building blocks.**
- **The CUDA/C++ builder path (tvm-ffi + nvcc + `-lcuda -lcublas`) is non-portable.** This rules out `language=cuda` solutions, including the contest-mandated CUDA-only `moe_fp8_block_scale` kernel. The four other kernels are addressable in `python`/`triton`.
- **fp8 FNUZ vs OCP is a real correctness risk** — but only for *solutions* that touch native fp8 MMA, and only for the two fp8 kernels (`dsa_topk_indexer_fp8`, `moe_fp8_block_scale`). The harness data plumbing and references are unaffected (verified: OCP `float8_e4m3fn` tensors create/cast/byte-reinterpret fine on torch 2.9.1+rocm7.2).

**Bottom line:** scoring infrastructure is ready; the work is authoring ROCm-native (`python`/`triton`) solutions on top of AITER and sglang, with the torch reference as both oracle and denominator.

---

## 2. Architecture recap

Three concepts drive everything:

- **Definition** (`/tmp/fib-meta/definitions/**/*.json`): contains a `reference` field = a pure-torch `run()` that is the *mathematical ground truth* AND the *speedup denominator*. It is device-agnostic and runs as-is on ROCm. Correctness is checked by comparing a solution's output against this reference in fp32; the speedup metric is `reference_latency / solution_latency`.
- **Baseline solution** (`/tmp/fib-meta/solutions/baseline/`): `language=python, deps=['flashinfer']` (etc.) that calls one specific flashinfer/deep_gemm op. These are the NVIDIA reference *implementations*, not the denominator. **They do not run on ROCm** because their ops/deps are CUDA-only.
- **Harness builders** (`flashinfer_bench/compile/builders/`): dispatch by solution `language`.
  - `PythonBuilder` — imports a module, no compiler. Portable.
  - `TritonBuilder` (extends PythonBuilder) — triton 3.6 ROCm build present. Portable; can target AITER triton ops.
  - `TileLangBuilder` (extends PythonBuilder) — present, imports, targets ROCm via its own backend. Usable but unverified end-to-end here.
  - `TVMFFIBuilder` — cuda/cpp via `tvm_ffi.cpp.build`, needs `nvcc` + `-lcuda -lcublas`. **Not portable** (nvcc absent).
  - `TorchBuilder` — cuda/cpp via `torch.utils.cpp_extension.load` (would hipify), only for `binding==TORCH`; not used by any contest baseline.

Dispatch is by `can_build()` (language match), so `python`/`triton` solutions never reach the CUDA builders.

---

## 3. Harness gaps (audit)

| Component | File | Issue | Severity | ROCm fix |
|---|---|---|---|---|
| Timing / import chain | `bench/timing.py:13` | Top-level `from flashinfer.testing import bench_gpu_time_with_cupti`; flashinfer not installed → `import flashinfer_bench` fails entirely (chain: `__init__`→bench→benchmark→runner→evaluators.default→timing). | **blocker** | Wrap import in try/except inside `time_runnable`; add HIP-events fallback using `torch.cuda.Event(enable_timing=True)`. Done (see §4). |
| Timing / API mismatch | `bench/timing.py:74-81` | Even with the ROCm fork, its `bench_gpu_time_with_cupti` (fork `testing/utils.py:646`) has no `input_args`/`cold_l2_cache` kwargs and calls `fn()` with zero args → `TypeError` or `fn` called with no args. | **blocker** | Bind args into a zero-arg closure `bound = lambda: fn(*args)`; call `bench_gpu_time_with_cupti(bound, dry_run_iters=warmup, repeat_iters=iters)`; drop `input_args`/`cold_l2_cache`. |
| Builder / CUDA dispatch | `compile/builders/tvm_ffi_builder.py:58-77,286-303` | `is_available()` only checks `tvm_ffi` imports → registered; `_find_cuda_lib_path()` looks for nvcc/libcudart (absent); injects `-lcuda -lcublas`. Any `cuda` solution → BuildError. | major | Not needed for python/triton (can_build False). Optional: gate `is_available()` on `shutil.which('nvcc')` for clearer fast-fail; or port to hipcc (`-L/opt/rocm/lib -lamdhip64 -lhipblas`). |
| Builder / Torch ext | `compile/builders/torch_builder.py:42-54,134-162` | `is_available()` True (torch.cuda via HIP); compiles cuda/cpp via `cpp_extension.load`, only `binding==TORCH`. DependencyManager maps cublas/cudnn/cutlass to NVIDIA pip pkgs (194-198, currently disabled). | minor | No action for python/triton. If cuda-via-torch ever wanted, rely on torch hipify; do not re-enable the NVIDIA-pinned DependencyManager. |
| Agents / NCU profiler | `agents/ncu.py:45-49,90-128,254-258` | Hard-wired to NVIDIA Nsight Compute (`ncu`); absent on MI300. Returns "NCU executable not found". | major (off scoring path) | Non-blocking; not on verify/benchmark path. Optionally add rocprofv3/omniperf backend behind `shutil.which`. |
| Agents / sanitizer | `agents/sanitizer.py:29,143-147` | Hard-wired to NVIDIA `compute-sanitizer`; absent. Returns "not found". | major (off scoring path) | Non-blocking. Optionally wire ROCm `rocgdb`/ASan; else leave graceful error. |
| Agents / solution runner NVTX | `agents/_solution_runner.py:58` | `torch.cuda.nvtx.range(...)` for NCU marking; maps to roctx/no-op on ROCm, harmless but meaningless without ncu. | minor | No change for verify/benchmark. |
| Data / fp8 dtype semantics | `utils.py:54-56`; `bench/utils.py:33-36`; `evaluators/dsa_topk_indexer.py:42,187` | Defs use OCP `float8_e4m3fn`; MI300 native fp8 is FNUZ. Concern was data/reference breakage. **Verified:** create/`.to`/`.view(float8_e4m3fn)` all work on cuda(HIP). Mismatch only bites native fp8 MMA in solutions. | minor (harness) | No harness change. Document for solution authors (see §6). |
| Runners / device naming | `utils.py:106-110`; `isolated_runner.py:223,290`; `persistent_runner.py:121,398,646` | Devices enumerated as `cuda:i`, parsed `int(split(':')[1])`; torch.cuda.* throughout. Maps to HIP; no break. | minor | No change. Confirm `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` if restricting GPUs. |
| Builder is_available ordering | `compile/registry.py:16-22,86-90` | Priority Triton, TileLang, Python, TVMFFI, Torch. Dispatch by `can_build` (language). No selection break; risk only if CUDA builders chosen for cuda/cpp. | minor | No change for python/triton. Optional: deregister TVMFFI/Torch when nvcc absent for clearer errors. |

**Buildable today on ROCm:** `python`, `triton`, `tilelang` (last unverified e2e). **Not buildable:** `cuda`, `cpp`-via-tvm-ffi (incl. the CUDA-mandated `moe_fp8` kernel).

---

## 4. Bring-up result (what actually ran on MI300)

A real bring-up was performed on the MI300X. Summary:

1. `pip install -e /tmp/flashinfer-bench --no-deps` → built/installed `flashinfer_bench-0.0.1.dev1` editable; no native build step.
2. `from flashinfer_bench import TraceSet, Definition` → **failed**: `ModuleNotFoundError: No module named 'flashinfer'` from `bench/timing.py:13`, reached via `__init__`→bench→benchmark→runner→evaluators→default→timing. Confirms the single hard blocker.
3. **Patch applied (one file):** `/tmp/flashinfer-bench/flashinfer_bench/bench/timing.py`. Wrapped the top-level import in try/except; on `ImportError` defined a drop-in `bench_gpu_time_with_cupti` using `torch.cuda.Event(enable_timing=True)` (HIP events on ROCm), with optional cold-L2 flush, matching the kwargs `time_runnable()` actually passes (`fn, dry_run_iters, repeat_iters, input_args, cold_l2_cache, use_cuda_graph`). `git diff`: **1 file changed, 48 insertions(+), 1 deletion(-)**. Tagged in-code `FIB-ROCm-PATCH`. No shim package, no fork build.
4. Re-ran import → **OK** (full `__init__` chain: Benchmark, apply, tracing, data models, compile registry).
5. `BuilderRegistry.get_instance()` → all 5 builders report `is_available()==True`; only `PythonBuilder`/`TritonBuilder` are genuinely portable (TVMFFI/TileLang would fail at nvcc/compile time, not invoked).
6. Loaded `/tmp/fib-meta/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json` via `Definition.model_validate(raw)` (parsed cleanly: `name=gdn_decode_qk4_v8_d128_k_last`, `op_type=gdn`); exec'd `reference` to obtain `run()`.
7. Built device inputs for smallest workload (B=1,T=1,nq=4,nk=4,nv=8,D=128; q/k/v/a/b bf16, state/A_log/dt_bias fp32, scale=1/sqrt(128)); ran `run()` on the MI300X.
8. **Output verified:** `output (1,1,8,128)` bf16, `new_state (1,8,128,128)` fp32, all finite — shape/dtype PASS.
9. **Timed** (50 iters, 10 warmup) via `torch.cuda.Event`: **MEDIAN 1.8585 ms** (min 1.8201, max 2.4217). Through the patched harness fallback: **1.8399 ms** (consistent).

**Measured GDN-decode reference latency: ~1.86 ms** (B=1,T=1). NOTE: the reference is an unoptimized per-(batch,head) Python loop, so this is the *denominator* number, not a kernel number.

**Exact patch to apply (turn into a clean patch set in Phase 0):**
- File: `flashinfer_bench/bench/timing.py`.
- Replace top-level `from flashinfer.testing import bench_gpu_time_with_cupti` with try/except.
- On `ImportError`, define a fallback `bench_gpu_time_with_cupti` (HIP `torch.cuda.Event`); in `time_runnable`, bind `bound = lambda: fn(*args)` and drop the fork-incompatible `input_args=`/`cold_l2_cache=` kwargs.
- Cleaner upstream form: try `flashinfer.testing.bench_gpu_time_with_cupti`, then `bench_gpu_time_with_cuda_event`, then the bare torch fallback. The bare torch fallback alone is sufficient and dependency-free.

Repro artifacts: `/tmp/run_gdn_ref.py` (self-contained); patched file at `/tmp/flashinfer-bench/flashinfer_bench/bench/timing.py` (git diff in that repo).

---

## 5. ROCm capability map

### 5.1 ROCm flashinfer fork (`/tmp/rocm-flashinfer`, branch `amd-integration`)

**Covers (HIP and/or AITER):** single/batch decode attention (paged, fp8 KV E4M3FNUZ), single/batch prefill (paged+ragged), cascade attention, **MLA (AITER-only, no HIP fallback; bf16, page_size=1)**, POD (JIT), RoPE (incl. fused RoPE+fp8 quant+paged-KV append, FNUZ), paged-KV append (incl fp8), RMSNorm/fused_add_rmsnorm, LayerNorm/Gemma RMSNorm, sampling (TopK/TopP/MinP/etc.), logits pipeline, SiLU/GELU+gating, packbits, **BlockSparseAttentionWrapper + VariableBlockSparseAttentionWrapper** (exported in HIP branch), and crucially **`flashinfer.testing.bench_gpu_time_with_cupti` + `bench_gpu_time_with_cuda_event`** (HIP-event fallback) — the harness's timing dependency.

**Does NOT provide:** GDN / gated-delta (no `gdn_decode`, no `gated_delta_rule_decode_pretranspose`); DeepSeek sparse-attention top-k indexer (`top_k_page_table_transform`); fp8 block-scale MoE (`trtllm_fp8_block_scale_moe` is IS_CUDA-only); `trtllm_batch_decode_with_kv_cache_mla` (CUDA-only; `decode_rocm.py` lacks it).

### 5.2 AITER building blocks (`/sgl-workspace/aiter`)

- GDN: `ops.triton.gated_delta_net.fused_recurrent_gated_delta_rule` (decode), `chunk_gated_delta_rule`/`_opt`/`_opt_vk` (prefill); plus `_triton_kernels/.../decode/fused_sigmoid_gating_recurrent.py` (computes gate+beta internally, HIP cfg BV=64/warps=4).
- DSA logits: `ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits` (direct ROCm equivalent of `deep_gemm.fp8_paged_mqa_logits`); `+_schedule` (metadata), gluon variants; `fp8_mqa_logits` (non-paged).
- Sparse MLA: `ops.triton.attention.unified_attention_sparse_mla` (gfx942-validated), `pa_decode_sparse`, `mla_decode(_rope)`.
- Top-k: `ops.triton.topk.topk`/`two_stage_topk`/`one_stage_topk`; `_triton_kernels.topk` (stage1/2, bitonic/argsort).
- MoE: `ops.triton.moe.moe_routing.grouped_topk` (DeepSeek n_group/topk_group), `routing`/`sort_tokens`/`moe_align_block_size`; `moe_op_gemm_a8w8_blockscale.moe_gemm_a8w8_blockscale` + `moe_op_silu_fused`; `ops.moe_op.fmoe_fp8_blockscale_g1u1` (C++/asm, blk 128x128); `topk_softmax`/`topk_sigmoid`/`moe_sum`.
- GEMM: `ops.triton.gemm.basic.gemm_afp8wfp8` (true fp8), `gemm_a8w8_blockscale` (INT8), `gemm_a8w8_blockscale_preshuffle`; `ops.deepgemm`, `ops.gemm_op_a8w8.*`.
- fp8 dtype: `ops.triton.utils.types.get_fp8_e4m3_dtype` → **e4m3fnuz on gfx942**.

### 5.3 sglang building blocks (`/sgl-workspace/sglang`)

- GDN: `srt/layers/attention/fla/` (chunk*, fused_recurrent) — full FLA triton port; `linear/gdn_backend.py` + test kit `gdn_attention.py`.
- DSA indexer: `srt/layers/attention/dsa/dsa_indexer.py` (line 649 imports `deepgemm_fp8_paged_mqa_logits`; `kv_cache_cast_to_fp8` produces the exact 132-byte layout) + `dsa_topk_backend.py` (`_topk_unfused` portable torch.topk; `topk_transform` page→global mapping).
- NSA/MLA: `nsa_backend.py`, `nsa_indexer.py`, `transform_index.py`; `deepseek_v4_backend(_hip_radix).py`; `models/deepseek_common/attention_forward_methods/forward_mla(_fused_rope_rocm).py`.
- MoE: `srt/layers/moe/moe_runner/triton_utils/fused_moe.py` (`fused_experts`, fp8 w8a8 block-scale), `fused_moe_triton_kernels.py`, `topk.py::biased_grouped_topk_impl` (exact DeepSeek routing), `rocm_moe_utils.py`; `quantization/fp8_kernel.py::is_fp8_fnuz`.

### 5.4 Timing

Functional on ROCm out of the box: `bench_gpu_time_with_cupti` tries `from cupti import cupti` (absent here), emits one harmless `UserWarning`, and dispatches to `bench_gpu_time_with_cuda_event`, which uses `torch.cuda.Event(enable_timing=True)` → HIP events. Slightly coarser than CUPTI activity tracing but correct. Our patched fallback reproduces this independently of the fork.

---

## 6. Per-kernel gap table

| Kernel | NV baseline op | ROCm baseline status | Recommended baseline | Port strategy | Difficulty | Key blockers |
|---|---|---|---|---|---|---|
| **gdn_decode** (qk4 v8 d128 k_last) | `flashinfer.gdn_decode.gated_delta_rule_decode_pretranspose` (use_qk_l2norm=False, DPS) | Missing in fork. AITER has direct kernel `fused_sigmoid_gating_delta_rule_update` (computes g+beta internally, HIP cfg, GVA-aware). | torch `reference` as oracle+denominator; wrap AITER `fused_sigmoid_gating_delta_rule_update`. | language=python wrapper: pass A_log/a/dt_bias/b raw, `softplus_beta=1.0`/`threshold=20.0`, l2norm=False, scale passthrough; **state-layout transpose** k-last `[B,H,V,K]`↔AITER `[B,H,K,V]` (K==V==128), in-place state update. | **easy** | State transpose both directions; AITER updates state in-place (feed transposed clone, read back); confirm BV=64 covers V=128; loose tol (atol1/rtol0.3/0.9) makes fp32-accum pass easily. No fp8. |
| **gdn_prefill** (qk4 v8 d128 k_last) | FlashInfer Blackwell CuTe-DSL `gdn_blackwell.gdn.chunk_gated_delta_rule` (tcgen05/TMEM, SM100) | Missing in fork. AITER `chunk_gated_delta_rule` has near-identical signature; `_opt_vk` takes k-last `[N,H,V,K]` natively. sglang FLA chunk pipeline mirrors it. | torch `reference` (denominator) + wrap AITER `chunk_gated_delta_rule` (or `_opt_vk`). | language=python: compute `g` in **log-space**, `beta=sigmoid(b)` in fp32; B=1 unsqueeze for varlen; prefer `_opt_vk` to skip state transpose; squeeze output→bf16. | **medium** | State layout; GVA wiring (4 q/k vs 8 v heads, no manual repeat_interleave); g must be log-space; varlen needs B=1 & cu_seqlens int64; chunked-cumsum drift vs per-token reference (esp. fp32 final_state); JIT on gfx942; forward-only. No fp8. |
| **dsa_sparse_attention** (h16 ckv512 kpe64 topk2048 ps64) | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` | Missing in fork. AITER `unified_attention_sparse_mla` (gfx942-validated, math identical to reference). | torch `reference` + wrap AITER `unified_attention_sparse_mla`. | language=python: `cat([q_nope,q_pe])→[T,16,576]`, `cat([ckv,kpe])→[pages,64,576]` reshaped to `[blks,bs,h_kv=1,d]`; `cu_seqlens_q=arange(T+1)`, `seqused_k=(idx!=-1).sum(1)`; `topk_indices=sparse_indices` (absolute); return `(out,)` (lse optional). | **easy** | No technical blocker (native kernel exists/runs). Layout adaptation (h_kv axis, absolute-index convention); kernel explicitly "not optimized" (perf is the work); guard all-`-1` token edge case. No fp8 (bf16). |
| **dsa_topk_indexer_fp8** (h64 d128 topk2048 ps64) | `deep_gemm.fp8_paged_mqa_logits` + `flashinfer.top_k_page_table_transform` | Both missing. AITER `deepgemm_fp8_paged_mqa_logits` (gfx942-tuned) = direct stage-1 equiv; sglang `dsa_indexer.py` wires it. Stage-2: torch.topk portable (sgl_kernel fast_topk is CUDA-only). | torch `reference` + AITER stage-1 logits + torch.topk stage-2 (then Triton top-k for speed). | language=python(triton): reshape q→`[B,1,64,128]`, view int8 cache as uint8; build context_lens/kv_indices; torch.topk + page→global index transform with -1 padding. Fallback: ~40-line fresh Triton kernel mirroring AITER inner math with explicit fp8→fp32 dequant. | **medium** | **fp8 FNUZ vs FN** (see §7); KV byte-split layout must match AITER's `view(-1, KVBlockSize*index_dim)` for ps=64/index_dim=132; stage-2 torch.topk over large model_len may be slow; tie-break determinism; exact index mapping `block_table[b,idx//64]*64+idx%64`. |
| **moe_fp8_block_scale** (ds routing topk8 ng8 kg4 e32 h7168 i2048) — **contest mandates CUDA-only** | `flashinfer.fused_moe.trtllm_fp8_block_scale_moe` | Missing in fork (IS_CUDA-only). **Not buildable on ROCm** (cuda/tvm-ffi path). AITER/sglang have block-scale MoE pieces. | torch `reference` (denominator). If a ROCm attempt is wanted: sglang `fused_experts` + `biased_grouped_topk_impl`. | language=python/triton only (cuda path non-portable): sglang `biased_grouped_topk_impl` routing → local-expert remap → `fused_experts(use_fp8_w8a8, block_shape=[128,128])` with fp32-upcast; reconcile transposed `hidden_states_scale [H/128,T]` and contiguous-half SwiGLU. | **medium** (but CUDA-mandated → out of scope for ROCm scoring) | No drop-in fused op; **e4m3fn vs e4m3fnuz** (fp32-upcast safe, forgoes peak fp8); layout (transposed scale, SwiGLU split, add_residual); local-expert offset/filter; routed_scaling_factor+1e-20 with **unbiased** s; cuda builder non-portable. |

---

## 7. fp8 / FNUZ correctness risks

**Harness: no risk.** Verified on torch 2.9.1+rocm7.2 that OCP `float8_e4m3fn` tensors can be created, cast (`.to`), and byte-reinterpreted (`.view(torch.float8_e4m3fn)`) on the GPU. Data generators, pure-torch references, and the `dsa_topk_indexer` evaluator all work unchanged. The correctness check compares against the e4m3fn reference values in fp32, which both platforms reproduce.

**Solutions: real risk, only `dsa_topk_indexer_fp8` and `moe_fp8_block_scale`.** MI300/gfx942 native fp8 is **e4m3fnuz** (FNUZ: exponent bias 8, no inf/NaN encodings, max ~224), not NVIDIA OCP **e4m3fn** (bias 7, has inf/NaN, max 448). `aiter.ops.triton.utils.types.get_fp8_e4m3_dtype` returns fnuz on gfx942.

- **The same byte pattern decodes to different real values under fn vs fnuz** (bias differs by 1). A naive bitcast of the contest's fn-quantized bytes to fnuz produces wrong scores/activations.
- **Safe path (recommended, correctness-first):** keep the pointer dtype `float8_e4m3fn` and **dequantize fp8→fp32 in-kernel before `tl.dot`** (exactly what the references do). ROCm triton 3.6 can upcast e4m3fn to fp32. This sidesteps the hardware fp8-MMA dtype question entirely and is exact. The loose tolerances (atol 1, rtol 0.3, matched-ratio 0.9) make this comfortably sufficient.
- **Peak-perf path (only if perf-limited):** convert fn→fp32→fnuz with scale adjustment (value-preserving, not bitcast) to use native MFMA fp8; measure accuracy impact. Range/precision loss possible.
- **Open hardware question:** whether MI300 `tl.dot`/MFMA can natively consume `float8_e4m3fn` operands, or only fnuz. Must be tested (Phase 2). Until resolved, prefer in-kernel dequant.
- **Block-scale layout:** deep_gemm packs per page `[page_size*128 fp8 bytes | page_size*4 scale bytes]`; the `index_dim=132` / `head_dim_with_scale` view is **not per-token-contiguous** — must reverse via flat view as the reference does. `moe`'s `hidden_states_scale` is **transposed** `[H/128, T]`. SwiGLU uses **contiguous halves** `[:I]/[I:]`, whereas AITER/sglang fused swiglu often uses interleaved `::2,1::2` with an `add_residual` variant — must select the matching variant or reorder weights.

The other three kernels (`gdn_decode`, `gdn_prefill`, `dsa_sparse_attention`) are bf16/fp32 only — FNUZ does not apply; ROCm bf16 matches NV bf16 bit layout.

---

## 8. Open questions

1. **MFMA fp8 operand dtype:** can gfx942 `tl.dot` consume `float8_e4m3fn` directly, or must operands be fnuz? Determines whether the fp8 kernels can use native MMA or must dequant in-kernel. (Test in Phase 2.)
2. **AITER `deepgemm_fp8_paged_mqa_logits` KV split for ps=64/index_dim=132:** does `kv_cache.view(-1, KVBlockSize*index_dim)` byte-align with the contest's per-page `[fp8 | scale]` packing when `KVBlockSize=64`? Needs byte-for-byte verification or a repack.
3. **AITER `chunk_gated_delta_rule` GVA contract:** does it derive `Hg/H` from `k.shape[2]` vs `v.shape[2]`, so we pass 4 q/k + 8 v heads without manual `repeat_interleave`? (Read `chunk_delta_h.py` ~695-746, 1034-1087.)
4. **`_opt_vk` final_state layout:** confirm it returns k-last `[N,H,V,K]` (not `[N,H,K,V]`) so no output transpose is needed.
5. **Chunked-path numerical drift:** does AITER's chunk-size-64 cumsum + WY formulation stay within tolerance on the **fp32 new_state** (tighter relative error than the bf16 output)? If not, fall back to `fused_recurrent_gated_delta_rule`.
6. **moe contest scope:** the `moe_fp8_block_scale` kernel is CUDA-mandated and not buildable on ROCm. Is a ROCm submission for it accepted/scored at all, or should effort focus on the other four?
7. **tilelang end-to-end:** `TileLangBuilder` imports and targets ROCm, but no contest kernel has been built/timed through it. Is it worth validating as a third portable path?
8. **GPU visibility:** confirm `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` semantics for multi-GPU isolation in `isolated_runner`/`persistent_runner` (device strings stay `cuda:i`).
