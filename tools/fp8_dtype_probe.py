#!/usr/bin/env python3
"""Reproducible probe for the fp8 dtype mismatch that blocks the AITER logits lever on gfx942.

The contest KV cache is OCP `float8_e4m3fn`. AITER's `deepgemm_fp8_paged_mqa_logits` views the cache
as `aiter.dtypes.fp8`, which on gfx942 is `float8_e4m3fnuz` (different exponent bias). This probe
prints the dtype AITER uses and the decode error when the contest's actual KV bytes are interpreted
as fnuz instead of fn, on one real dsa_topk_indexer workload.

    FIB_DATASET_PATH=... FIB_CACHE_PATH=/tmp/fib_cache python tools/fp8_dtype_probe.py
"""
import os
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFN = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
PAGE_SIZE, HEAD_DIM = 64, 128


def main():
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"commit={commit}  gpu={torch.cuda.get_device_name(0)}")
    try:
        from aiter import dtypes
        from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
        print(f"aiter.dtypes.fp8={dtypes.fp8}  get_fp8_e4m3_dtype()={get_fp8_e4m3_dtype()}")
    except Exception as e:
        print("aiter import failed:", e)

    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.data import TraceSet
    root = Path(os.environ["FIB_DATASET_PATH"])
    ts = TraceSet.from_path(root)
    d = ts.definitions[DEFN]
    w = sorted([x.workload for x in ts.workloads[DEFN]], key=lambda w: w.axes.get("batch_size", 0))[-1]
    safe = load_safetensors(d, w, root) if any(x.type == "safetensors" for x in w.inputs.values()) else {}
    inp = gen_inputs(d, w, device="cuda:0", safe_tensors=safe)
    k_cache = inp[1]                                  # [num_pages, 64, 1, 132] int8
    u8 = k_cache.view(torch.uint8).view(k_cache.shape[0], PAGE_SIZE, 132)
    fp8_bytes = u8[..., :HEAD_DIM].contiguous()       # [P,64,128] fp8 payload (page-split layout in solution)

    as_fn = fp8_bytes.view(torch.float8_e4m3fn).float()
    as_fnuz = fp8_bytes.view(torch.float8_e4m3fnuz).float()
    # The contest KV legitimately contains NaN tokens; compare only finite entries.
    fin = torch.isfinite(as_fn) & torch.isfinite(as_fnuz)
    diff = (as_fn - as_fnuz).abs()[fin]
    rel = (diff / (as_fn.abs()[fin] + 1e-6))
    print(f"finite entries: {int(fin.sum())}/{fin.numel()}")
    print(f"KV decoded as e4m3fn  : mean|v|={as_fn[fin].abs().mean():.4f}")
    print(f"KV decoded as e4m3fnuz: mean|v|={as_fnuz[fin].abs().mean():.4f}")
    print(f"fn-vs-fnuz decode error: max_abs={diff.max():.4f}  mean_abs={diff.mean():.4f}  "
          f"max_rel={rel.max():.4f}")
    print("=> AITER deepgemm_fp8_paged_mqa_logits (views KV as aiter.dtypes.fp8) would misread the "
          "contest e4m3fn KV; logits/top-k wrong. AITER logits lever is NOT usable portably.")


if __name__ == "__main__":
    main()
