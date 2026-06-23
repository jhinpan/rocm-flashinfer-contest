#!/usr/bin/env python3
"""Standalone parity harness for the moe_fp8 solution.

Gates any change to the moe_fp8 per-expert path (dequant kernel, routing, SwiGLU, or a future
grouped/AITER fused block-scale path) by checking each piece against an **independent** reference,
on synthetic inputs (with controlled logits forcing local + non-local experts) and official inputs:

  1. block-scale weight dequant  — Triton `_dequant` vs torch `_dequant_block`  (bit-exact)
  2. hidden-state dequant         — `_dequant_hidden` vs an independent [H/128,T]->[T,H] expand
  3. routing                      — `_route` ids (as sets) + weights vs an independent recompute
  4. SwiGLU                       — `_swiglu_contiguous` == contiguous-half ref AND != interleaved
  5. synthetic full-path          — default Triton-dequant run() vs MOE_DEQUANT_TORCH=1 run()
  6. official full-path           — run() vs the reference (build_reference) on official workloads

An AITER fused path that views the cache as e4m3fnuz, or any routing/SwiGLU regression, fails here.

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


# ----- independent references (deliberately different code from solutions/moe_fp8/main.py) -----

def ref_route(logits, bias, scale, E=256, ngroup=8, topk_group=4, top_k=8):
    """Independent no-aux routing: per-group top-2 sum via sort, top-4 groups, global top-8."""
    T = logits.shape[0]
    s = torch.sigmoid(logits.float())
    sb = s + bias.float().reshape(-1)
    gsz = E // ngroup
    sbg = sb.view(T, ngroup, gsz)
    # per-group sum of the two largest (sort desc, take first two)
    g2 = sbg.sort(dim=2, descending=True).values[:, :, :2].sum(dim=2)        # [T, ngroup]
    keep_groups = g2.sort(dim=1, descending=True).indices[:, :topk_group]    # [T, topk_group]
    gmask = torch.zeros(T, ngroup, device=logits.device).scatter_(1, keep_groups, 1.0)
    emask = gmask.repeat_interleave(gsz, dim=1)                              # [T, E]
    sb_masked = sb.masked_fill(emask == 0, torch.finfo(torch.float32).min)
    idx = sb_masked.sort(dim=1, descending=True).indices[:, :top_k]          # [T, top_k]
    sel = torch.zeros_like(s).scatter_(1, idx, 1.0)
    w = s * sel
    w = (w / (w.sum(dim=1, keepdim=True) + 1e-20)) * scale
    return idx, w


def ref_dequant_hidden(hidden, hscale, H=7168, blk=128):
    sc = hscale.float().t().contiguous()                                    # [T, H/128]
    sc_full = sc.repeat_interleave(blk, dim=1)                              # [T, H]
    return (hidden.float() * sc_full).to(torch.bfloat16)


def _ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    return cond


def check_units(mod, device):
    torch.manual_seed(0)
    ok = True
    # 1) weight dequant bit-exactness
    for (R, C) in [(2 * mod.I, mod.H), (mod.H, mod.I)]:
        w = (torch.randn(R, C, device=device) * 0.1).to(torch.float8_e4m3fn)
        scale = (torch.rand(R // mod.BLOCK, C // mod.BLOCK, device=device) * 0.02 + 1e-3).float()
        a, b = mod._dequant_block(w, scale), mod._dequant_triton(w, scale)
        ok = _ok(f"weight dequant [{R}x{C}] bit-exact", torch.equal(a, b),
                 f"maxabs={(a.float()-b.float()).abs().max().item():.3e}") and ok

    # 2) hidden-state dequant vs independent expand
    T = 5
    hs = (torch.randn(T, mod.H, device=device) * 0.1).to(torch.float8_e4m3fn)
    hsc = (torch.rand(mod.H // mod.BLOCK, T, device=device) * 0.02 + 1e-3).float()
    a = mod._dequant_hidden(hs, hsc)
    b = ref_dequant_hidden(hs, hsc, H=mod.H, blk=mod.BLOCK)
    ok = _ok("hidden dequant vs independent expand", torch.equal(a, b),
             f"maxabs={(a.float()-b.float()).abs().max().item():.3e}") and ok

    # 3) routing: controlled logits that force specific (local + non-local) experts to the top
    T = 6
    logits = (torch.randn(T, mod.E_GLOBAL, device=device) * 0.5)
    # force experts {0(local-ish),5,40,99,130,200,255,17} high for token 0 to span groups
    for e in [0, 5, 40, 99, 130, 200, 255, 17]:
        logits[0, e] += 8.0
    bias = (torch.randn(mod.E_GLOBAL, device=device) * 0.1)
    scale = 2.5
    idx_s, w_s = mod._route(logits, bias, scale)
    idx_r, w_r = ref_route(logits, bias, scale, E=mod.E_GLOBAL, ngroup=mod.N_GROUP,
                           topk_group=mod.TOPK_GROUP, top_k=mod.TOP_K)
    same_sets = all(set(idx_s[t].tolist()) == set(idx_r[t].tolist()) for t in range(T))
    ok = _ok("routing top-k id sets vs independent", same_sets) and ok
    ok = _ok("routing normalized weights vs independent",
             torch.allclose(w_s, w_r, atol=1e-5, rtol=1e-4),
             f"maxabs={(w_s-w_r).abs().max().item():.3e}") and ok

    # 4) SwiGLU contiguous-half order (and assert it differs from the interleaved variant)
    G1 = torch.randn(4, 2 * mod.I, device=device, dtype=torch.bfloat16)
    c = mod._swiglu_contiguous(G1)
    ref_c = (torch.nn.functional.silu(G1[:, mod.I:].float()) * G1[:, :mod.I].float()).to(torch.bfloat16)
    interleaved = (torch.nn.functional.silu(G1[:, 1::2].float()) * G1[:, ::2].float()).to(torch.bfloat16)
    ok = _ok("swiglu == contiguous-half ref", torch.equal(c, ref_c)) and ok
    ok = _ok("swiglu != interleaved variant (negative check)",
             not torch.equal(c, interleaved)) and ok
    return ok


def check_synthetic_fullpath(mod, device):
    """Default Triton-dequant run() vs MOE_DEQUANT_TORCH=1 run() on the same contest-shaped inputs."""
    torch.manual_seed(7)
    T, E_local = 12, 8
    H, Inter, blk = mod.H, mod.I, mod.BLOCK
    logits = torch.randn(T, mod.E_GLOBAL, device=device)
    bias = torch.randn(mod.E_GLOBAL, device=device) * 0.1
    hs = (torch.randn(T, H, device=device) * 0.1).to(torch.float8_e4m3fn)
    hsc = (torch.rand(H // blk, T, device=device) * 0.02 + 1e-3).float()
    g1 = (torch.randn(E_local, 2 * Inter, H, device=device) * 0.1).to(torch.float8_e4m3fn)
    g1s = (torch.rand(E_local, (2 * Inter) // blk, H // blk, device=device) * 0.02 + 1e-3).float()
    g2 = (torch.randn(E_local, H, Inter, device=device) * 0.1).to(torch.float8_e4m3fn)
    g2s = (torch.rand(E_local, H // blk, Inter // blk, device=device) * 0.02 + 1e-3).float()
    args = (logits, bias, hs, hsc, g1, g1s, g2, g2s, 0, 2.5)
    os.environ.pop("MOE_DEQUANT_TORCH", None)
    out_triton = mod.run(*args)
    os.environ["MOE_DEQUANT_TORCH"] = "1"
    out_torch = mod.run(*args)
    os.environ.pop("MOE_DEQUANT_TORCH", None)
    diff = (out_triton.float() - out_torch.float()).abs().max().item()
    return _ok("synthetic full-path: triton-dequant == torch-dequant", torch.equal(out_triton, out_torch),
               f"maxabs={diff:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ds = os.environ.get("FIB_DATASET_PATH")
    assert ds, "set FIB_DATASET_PATH"

    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors, compute_error_stats
    from flashinfer_bench.bench.config import ResolvedEvalConfig
    from flashinfer_bench.compile import BuilderRegistry

    mod = _load_solution()
    dev = args.device
    root = Path(ds)
    ts = TraceSet.from_path(root)
    defn = ts.definitions[DEFN]
    ref_runnable = BuilderRegistry.get_instance().build_reference(defn)
    cfg = ResolvedEvalConfig(warmup_runs=0, iterations=1, num_trials=1,
                             rtol=0.3, atol=1.0, required_matched_ratio=0.9)

    print("== unit parity vs independent references (synthetic) ==")
    ok = check_units(mod, dev)
    print("== synthetic full-path parity (triton vs torch dequant) ==")
    ok = check_synthetic_fullpath(mod, dev) and ok

    print(f"== official full-path parity vs reference (first {args.limit}) ==")
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
        print(f"  {'PASS' if good else 'FAIL'}  {w.axes} mr={mr:.4f}")
    full_ok = n_pass == len(wls)
    print(f"== official full-path {n_pass}/{len(wls)} ==")

    overall = ok and full_ok
    print("PARITY OK" if overall else "PARITY FAIL")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
