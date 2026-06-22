#!/usr/bin/env python3
"""Faithful local verifier for ROCm/MI300 ports of the MLSys2026 FlashInfer contest.

Mirrors flashinfer_bench's official scoring path (TraceSet -> gen_inputs/load_safetensors
-> reference -> compute_error_stats -> time_runnable) but runs a *solution module* directly
against the torch `reference`, so we can iterate without packing solution.json every time.

Usage:
  FIB_DATASET_PATH=.../flashinfer-trace \
  python tools/local_verify.py \
      --def gdn_decode_qk4_v8_d128_k_last \
      --sol solutions/gdn_decode/main.py \
      [--uuids U1 U2 ...] [--limit N] \
      [--atol 1e-2 --rtol 1e-2 --mr 1.0] [--no-time]

Exit code 0 iff all selected workloads pass correctness.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

from flashinfer_bench.bench.config import ResolvedEvalConfig
from flashinfer_bench.bench.utils import compute_error_stats, gen_inputs, load_safetensors
from flashinfer_bench.bench.timing import time_runnable
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import TraceSet


def load_solution_run(sol_path: str):
    spec = importlib.util.spec_from_file_location("sol_under_test", sol_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise RuntimeError(f"{sol_path} has no run() function")
    return mod.run


def as_list(result):
    if isinstance(result, (tuple, list)):
        return list(result)
    return [result]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="definition", required=True)
    ap.add_argument("--sol", required=True)
    ap.add_argument("--uuids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--mr", type=float, default=None, help="required_matched_ratio (None=1.0=strict)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    ds = os.environ.get("FIB_DATASET_PATH")
    if not ds:
        print("FIB_DATASET_PATH not set", file=sys.stderr)
        sys.exit(2)
    root = Path(ds)
    ts = TraceSet.from_path(root)

    if args.definition not in ts.definitions:
        print(f"definition {args.definition} not found. have: {sorted(ts.definitions)}", file=sys.stderr)
        sys.exit(2)
    definition = ts.definitions[args.definition]
    out_names = list(definition.outputs.keys())

    workloads = [w.workload for w in ts.workloads[args.definition]]
    if args.uuids:
        wanted = set(args.uuids)
        workloads = [w for w in workloads if w.uuid in wanted]
    if args.limit is not None:
        workloads = workloads[: args.limit]
    if not workloads:
        print("no workloads selected", file=sys.stderr)
        sys.exit(2)

    cfg = ResolvedEvalConfig(
        warmup_runs=args.warmup, iterations=args.iters, num_trials=1,
        rtol=args.rtol, atol=args.atol, required_matched_ratio=args.mr,
    )

    ref_runnable = BuilderRegistry.get_instance().build_reference(definition)
    run = load_solution_run(args.sol)
    device = args.device

    n_pass = 0
    print(f"== {args.definition} | {len(workloads)} workloads | atol={args.atol} rtol={args.rtol} mr={args.mr or 1.0} ==")
    for w in workloads:
        safe = (
            load_safetensors(definition, w, root)
            if any(d.type == "safetensors" for d in w.inputs.values())
            else {}
        )
        inp = gen_inputs(definition, w, device=device, safe_tensors=safe)
        with torch.no_grad():
            ref_out = as_list(ref_runnable(*inp))
        torch.cuda.synchronize(device)
        try:
            with torch.no_grad():
                sol_out = as_list(run(*[x.clone() if torch.is_tensor(x) else x for x in inp]))
            torch.cuda.synchronize(device)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {w.uuid}  axes={w.axes}  -> SOLUTION RAISED: {e}")
            continue

        ok = True
        details = []
        # flexible: compare only as many outputs as the solution returns (mirrors DSA evaluator)
        n_cmp = min(len(sol_out), len(out_names))
        for i in range(n_cmp):
            name = out_names[i]
            r = ref_out[i]
            s = sol_out[i]
            if not torch.is_tensor(s):
                s = torch.as_tensor(s, device=device)
            if s.shape != r.shape:
                ok = False
                details.append(f"{name}:SHAPE {tuple(s.shape)}!={tuple(r.shape)}")
                continue
            max_abs, max_rel, exceeds, mr = compute_error_stats(s, r, cfg)
            if exceeds:
                ok = False
            details.append(f"{name}:mr={mr:.4f} maxabs={max_abs:.3e} maxrel={max_rel:.3e}{' FAIL' if exceeds else ''}")

        speed = ""
        if not args.no_time:
            try:
                ref_ms = time_runnable(ref_runnable, inp, args.warmup, args.iters, device)
                bound_args = inp
                sol_ms = time_runnable(run, bound_args, args.warmup, args.iters, device)
                speed = f" | ref={ref_ms:.3f}ms sol={sol_ms:.3f}ms speedup={ref_ms/sol_ms:.2f}x"
            except Exception as e:
                speed = f" | timing-err:{e}"

        n_pass += int(ok)
        print(f"  {w.uuid}  axes={w.axes}  -> {'PASS' if ok else 'FAIL'}  [{' ; '.join(details)}]{speed}")

    print(f"== {n_pass}/{len(workloads)} passed ==")
    sys.exit(0 if n_pass == len(workloads) else 1)


if __name__ == "__main__":
    main()
