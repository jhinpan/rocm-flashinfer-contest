#!/usr/bin/env python3
"""Drive one solution kernel on a single workload so it can be wrapped by rocprofv3.

This builds the inputs for one representative workload (smallest / median / largest bucket
along the kernel's bucketing axis) using the same harness machinery as run_benchmarks.py,
then calls the solution `run()` a fixed number of times. Run it under a profiler, e.g.:

    rocprofv3 --kernel-trace --stats -- \
        python tools/profile_kernel.py --kernel dsa_sparse_attention --bucket max --iters 50

The point is to answer one named profiling question per invocation (which kernel dominates,
its duration, occupancy/stall class), not to produce timing numbers — use run_benchmarks.py
for timing. Keep raw profiler artifacts untracked; commit only summaries.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch

from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.data import TraceSet

ROOT = Path(__file__).resolve().parent.parent

# kernel short-dir -> definition name and bucketing axis (mirrors run_benchmarks.KERNELS)
KERNELS = {
    "gdn_decode": ("gdn_decode_qk4_v8_d128_k_last", "batch_size"),
    "gdn_prefill": ("gdn_prefill_qk4_v8_d128_k_last", "total_seq_len"),
    "dsa_sparse_attention": ("dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64", "num_tokens"),
    "dsa_topk_indexer": ("dsa_topk_indexer_fp8_h64_d128_topk2048_ps64", "batch_size"),
    "moe_fp8": ("moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048", "seq_len"),
}


def load_run(sdir):
    path = ROOT / "solutions" / sdir / "main.py"
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.run


def pick(workloads, axis, which):
    ws = sorted(workloads, key=lambda w: w.axes.get(axis, 0))
    return {"min": ws[0], "med": ws[len(ws) // 2], "max": ws[-1]}[which]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(KERNELS))
    ap.add_argument("--bucket", default="max", choices=["min", "med", "max"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ds = os.environ.get("FIB_DATASET_PATH")
    assert ds, "set FIB_DATASET_PATH"
    root = Path(ds)
    ts = TraceSet.from_path(root)
    defn, axis = KERNELS[args.kernel]
    definition = ts.definitions[defn]
    run = load_run(args.kernel)
    wls = [w.workload for w in ts.workloads[defn]]
    w = pick(wls, axis, args.bucket)
    safe = (load_safetensors(definition, w, root)
            if any(d.type == "safetensors" for d in w.inputs.values()) else {})
    inp = gen_inputs(definition, w, device=args.device, safe_tensors=safe)
    print(f"profiling {args.kernel} bucket={args.bucket} axes={dict(w.axes)} iters={args.iters}")

    with torch.no_grad():
        for _ in range(args.warmup):
            run(*[x.clone() if torch.is_tensor(x) else x for x in inp])
        torch.cuda.synchronize(args.device)
        for _ in range(args.iters):
            run(*[x.clone() if torch.is_tensor(x) else x for x in inp])
        torch.cuda.synchronize(args.device)
    print("done")


if __name__ == "__main__":
    main()
