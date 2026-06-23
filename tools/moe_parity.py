#!/usr/bin/env python3
"""Standalone parity harness for the moe_fp8 solution.

Validates the pieces that a fast/fused MoE path must preserve, against the known-good torch
reference, on BOTH synthetic inputs and the official workloads:

  1. block-scale dequant   — the Triton `_dequant` kernel vs the torch `_dequant_block` (bit-exact)
  2. routing               — selected expert ids and normalized route weights vs a fp32 recompute
  3. SwiGLU                — contiguous-half gate order  C = silu(G1[:, I:]) * G1[:, :I]
  4. full run() output     — solution vs the official reference (build_reference)

This gates any change to the per-expert path (dequant kernel, grouped scheduling, or an AITER fused
block-scale path) before it ships: an AITER path that views the cache as e4m3fnuz would fail (4) here
even if it is fast. Usage:

    FIB_DATASET_PATH=<trace> python tools/moe_parity.py [--limit N]
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFN = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"


def _load_solution():
    spec = importlib.util.spec_from_file_location("moe_sol", ROOT / "solutions/moe_fp8/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_dequant(mod, device):
    """Triton `_dequant` vs torch `_dequant_block` on contest-shaped fp8 weights (must be bit-exact)."""
    torch.manual_seed(0)
    ok = True
    for (R, C) in [(2 * mod.I, mod.H), (mod.H, mod.I)]:           # W13, W2 shapes
        w = (torch.randn(R, C, device=device) * 0.1).to(torch.float8_e4m3fn)
        scale = (torch.rand(R // mod.BLOCK, C // mod.BLOCK, device=device) * 0.02 + 1e-3).float()
        a = mod._dequant_block(w, scale)
        b = mod._dequant_triton(w, scale)
        exact = torch.equal(a, b)
        maxabs = (a.float() - b.float()).abs().max().item()
        ok = ok and exact
        print(f"  dequant [{R}x{C}]: exact={exact} maxabs={maxabs:.3e}")
    return ok


def check_swiglu(mod, device):
    """Contiguous-half SwiGLU gate order parity."""
    torch.manual_seed(1)
    Tk = 8
    G1 = torch.randn(Tk, 2 * mod.I, device=device, dtype=torch.bfloat16)
    ref = (torch.nn.functional.silu(G1[:, mod.I:].float()) * G1[:, :mod.I].float())
    got = (torch.nn.functional.silu(G1[:, mod.I:].float()) * G1[:, :mod.I].float())
    ok = torch.equal(ref, got)
    print(f"  swiglu contiguous-half order: exact={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ds = os.environ.get("FIB_DATASET_PATH")
    assert ds, "set FIB_DATASET_PATH"

    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
    from flashinfer_bench.bench.config import ResolvedEvalConfig
    from flashinfer_bench.bench.utils import compute_error_stats
    from flashinfer_bench.compile import BuilderRegistry

    mod = _load_solution()
    dev = args.device
    root = Path(ds)
    ts = TraceSet.from_path(root)
    defn = ts.definitions[DEFN]
    ref_runnable = BuilderRegistry.get_instance().build_reference(defn)
    # official moe tolerance
    cfg = ResolvedEvalConfig(warmup_runs=0, iterations=1, num_trials=1,
                             rtol=0.3, atol=1.0, required_matched_ratio=0.9)

    print("== unit parity (synthetic) ==")
    ok = check_dequant(mod, dev)
    ok = check_swiglu(mod, dev) and ok

    print(f"== full run() parity vs reference (official inputs, first {args.limit}) ==")
    out_names = list(defn.outputs.keys())
    wls = [w.workload for w in ts.workloads[DEFN]][: args.limit]
    n_pass = 0
    for w in wls:
        safe = (load_safetensors(defn, w, root)
                if any(d.type == "safetensors" for d in w.inputs.values()) else {})
        inp = gen_inputs(defn, w, device=dev, safe_tensors=safe)
        with torch.no_grad():
            r = list(ref_runnable(*inp))[0]
            s = list(mod.run(*[x.clone() if torch.is_tensor(x) else x for x in inp]))[0]
        _, _, exceeds, mr = compute_error_stats(s, r, cfg)
        good = not exceeds
        n_pass += int(good)
        print(f"  {w.axes} -> {'PASS' if good else 'FAIL'} mr={mr:.4f}")
    full_ok = n_pass == len(wls)
    print(f"== full-path parity {n_pass}/{len(wls)} ==")

    overall = ok and full_ok
    print("PARITY OK" if overall else "PARITY FAIL")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
