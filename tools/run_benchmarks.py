#!/usr/bin/env python3
"""Benchmark all ROCm/MI300 solutions against the torch reference via the flashinfer-bench harness.

Produces an apples-to-apples table whose headline metric is the platform-portable
    speedup = reference_latency / solution_latency
measured against the *identical* pure-torch `reference` in each definition. Because the
reference is the same code on every platform, these speedup ratios are directly comparable
between NVIDIA and AMD even though absolute ms differ.

Run on AMD (MI300):    python tools/run_benchmarks.py --out results/amd_mi300
Run on NVIDIA (B200):  python tools/run_benchmarks.py --out results/nvidia_b200 --baseline
  (--baseline also times the contest's NV flashinfer_wrapper baseline solutions, which only
   build on CUDA; on ROCm they are skipped automatically.)

Requires FIB_DATASET_PATH to point at the flashinfer-trace dataset.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from flashinfer_bench.bench.config import ResolvedEvalConfig
from flashinfer_bench.bench.utils import compute_error_stats, gen_inputs, load_safetensors
from flashinfer_bench.bench.timing import time_runnable
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import TraceSet

ROOT = Path(__file__).resolve().parent.parent


def _pkg_version(name):
    try:
        return __import__(name).__version__
    except Exception:
        return "n/a"


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def collect_provenance(device):
    """Reproducibility metadata stamped into every results file so a candidate-vs-baseline
    comparison is apples-to-apples (exact command, commit, baseline ref, GPU, library versions)."""
    commit = _git("rev-parse", "HEAD")
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "command": "python " + " ".join(sys.argv),
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "baseline_tag": _git("rev-parse", "--short", "baseline-v1"),
        "dirty": "yes" if _git("status", "--porcelain") else "no",
        "gpu": torch.cuda.get_device_name(0),
        "device": device,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None) or "n/a",
        "triton": _pkg_version("triton"),
        "aiter": _pkg_version("aiter"),
        "dataset": os.environ.get("FIB_DATASET_PATH", "n/a"),
        "fib_cache": os.environ.get("FIB_CACHE_PATH", "n/a"),
    }


def provenance_md(p):
    lines = ["## Provenance", "",
             "Reproducibility metadata for this run (candidate-vs-`baseline/v1` comparisons are",
             "only valid when these match).", "",
             "| field | value |", "|---|---|"]
    for k in ("timestamp_utc", "command", "commit", "branch", "describe", "baseline_tag",
              "dirty", "gpu", "device", "torch", "hip", "triton", "aiter", "dataset", "fib_cache"):
        lines.append(f"| {k} | `{p[k]}` |")
    return "\n".join(lines) + "\n\n"

# (definition, solution_dir, n_buckets, tol(atol,rtol,mr), n_outputs_to_compare or 'topk', iters)
KERNELS = [
    ("gdn_decode_qk4_v8_d128_k_last", "gdn_decode",
     "batch_size", (1e-2, 1e-2, None), 2, 30),
    ("gdn_prefill_qk4_v8_d128_k_last", "gdn_prefill",
     "total_seq_len", (1e-2, 1e-2, None), 2, 15),
    ("dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64", "dsa_sparse_attention",
     "num_tokens", (1e-2, 1e-2, None), 1, 30),
    ("dsa_topk_indexer_fp8_h64_d128_topk2048_ps64", "dsa_topk_indexer",
     "batch_size", (1e-2, 1e-2, None), "topk", 30),
    ("moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048", "moe_fp8",
     "seq_len", (1.0, 0.3, 0.9), 1, 15),
]


def load_run(path):
    spec = importlib.util.spec_from_file_location("m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.run


def as_list(r):
    return list(r) if isinstance(r, (tuple, list)) else [r]


# Use the official evaluator's scoring helpers so correctness exactly matches verify.py.
from flashinfer_bench.bench.evaluators.dsa_topk_indexer import (
    _compute_scores_at_indices,
    _compute_sorted_score_error_stats,
    _dequant_all_pages,
    _validate_indices,
)


def topk_correct(inp, ref_out, sol_out, cfg):
    """Mirror DsaTopkIndexerEvaluator: validate indices, re-score both sets with the canonical
    dequant, compare sorted scores."""
    q_fp8, k_cache_fp8, weights, seq_lens, block_table = inp
    ref_idx, sol_idx = ref_out[0], sol_out[0]
    oor, dup = _validate_indices(sol_idx, k_cache_fp8.shape[0], 64, seq_lens, block_table)
    if bool(oor.any()) or bool(dup.any()):
        return False, 0.0
    k_all = _dequant_all_pages(k_cache_fp8)
    combined = torch.cat([ref_idx, sol_idx], dim=1)
    all_scores = _compute_scores_at_indices(combined, k_all, q_fp8, weights, 64)
    rs, ss = all_scores.split(2048, dim=1)
    _, _, _, mr = _compute_sorted_score_error_stats(
        ss.sort(dim=1, descending=True).values, rs.sort(dim=1, descending=True).values, cfg
    )
    # The reference's torch.topk tie-breaking is non-deterministic on GPU; at exact score ties a
    # handful of the 2048 boundary positions can differ between two runs even for a correct
    # solution. The authoritative verify.py (num_trials=3) accepts these; allow a tiny slack here
    # so the display does not flicker. Authoritative correctness = verify.py (128/128).
    return mr >= 0.999, mr


def pick_buckets(workloads, axis, n=3):
    ws = sorted(workloads, key=lambda w: w.axes.get(axis, 0))
    if len(ws) <= n:
        return ws
    idx = [0, len(ws) // 2, len(ws) - 1]
    return [ws[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baseline", action="store_true", help="also benchmark NV flashinfer_wrapper baselines")
    args = ap.parse_args()

    ds = os.environ.get("FIB_DATASET_PATH")
    assert ds, "set FIB_DATASET_PATH"
    root = Path(ds)
    ts = TraceSet.from_path(root)
    dev = args.device
    plat = torch.cuda.get_device_name(0)
    prov = collect_provenance(dev)
    print("provenance:", {k: prov[k] for k in ("commit", "branch", "baseline_tag", "gpu", "dirty")})

    rows = []
    for defn, sdir, axis, (atol, rtol, mr), ncmp, iters in KERNELS:
        if defn not in ts.definitions:
            print(f"skip {defn} (not in dataset)"); continue
        definition = ts.definitions[defn]
        cfg = ResolvedEvalConfig(warmup_runs=5, iterations=iters, num_trials=1,
                                 rtol=rtol, atol=atol, required_matched_ratio=mr)
        ref_runnable = BuilderRegistry.get_instance().build_reference(definition)
        run = load_run(ROOT / "solutions" / sdir / "main.py")
        wls = [w.workload for w in ts.workloads[defn]]
        for w in pick_buckets(wls, axis, 3):
            safe = (load_safetensors(definition, w, root)
                    if any(d.type == "safetensors" for d in w.inputs.values()) else {})
            inp = gen_inputs(definition, w, device=dev, safe_tensors=safe)
            with torch.no_grad():
                ref_out = as_list(ref_runnable(*inp))
            torch.cuda.synchronize(dev)
            with torch.no_grad():
                sol_out = as_list(run(*[x.clone() if torch.is_tensor(x) else x for x in inp]))
            torch.cuda.synchronize(dev)

            if ncmp == "topk":
                ok, mr_v = topk_correct(inp, ref_out, sol_out, cfg)
            else:
                ok = True; mr_v = 1.0
                for i in range(min(ncmp, len(sol_out))):
                    s = sol_out[i]
                    _, _, ex, m = compute_error_stats(s, ref_out[i], cfg)
                    mr_v = min(mr_v, m); ok = ok and (not ex)

            ref_ms = time_runnable(ref_runnable, inp, 5, iters, dev)
            sol_ms = time_runnable(run, inp, 5, iters, dev)
            rows.append(dict(kernel=defn, axes=dict(w.axes), ref_ms=ref_ms, sol_ms=sol_ms,
                             speedup=ref_ms / sol_ms, ok=ok, mr=mr_v))
            print(f"{sdir:22s} {str(dict(w.axes)):42s} ref={ref_ms:9.3f} sol={sol_ms:8.3f} "
                  f"speedup={ref_ms/sol_ms:7.2f}x {'PASS' if ok else 'FAIL'}(mr={mr_v:.3f})")

    out = Path(ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_suffix(".csv")
    with open(csv, "w") as f:
        f.write("# " + "; ".join(f"{k}={prov[k]}" for k in
                ("timestamp_utc", "commit", "branch", "baseline_tag", "gpu", "torch", "triton",
                 "aiter", "dirty")) + "\n")
        f.write("kernel,axes,ref_ms,sol_ms,speedup,correctness,matched_ratio\n")
        for r in rows:
            f.write(f"{r['kernel']},\"{r['axes']}\",{r['ref_ms']:.4f},{r['sol_ms']:.4f},"
                    f"{r['speedup']:.3f},{'PASS' if r['ok'] else 'FAIL'},{r['mr']:.4f}\n")
    md = out.with_suffix(".md")
    with open(md, "w") as f:
        f.write(f"# Benchmark results — {plat}\n\n")
        f.write("Speedup = torch-reference latency / solution latency (same reference on every platform).\n\n")
        f.write(provenance_md(prov))
        f.write("| Kernel | Workload | reference ms | solution ms | speedup | correctness |\n")
        f.write("|---|---|---:|---:|---:|:--:|\n")
        for r in rows:
            f.write(f"| `{r['kernel'].split('_')[0]}…` | {r['axes']} | {r['ref_ms']:.3f} | "
                    f"{r['sol_ms']:.3f} | **{r['speedup']:.2f}×** | {'✅' if r['ok'] else '❌'} |\n")
    print(f"\nwrote {md} and {csv}")


if __name__ == "__main__":
    main()
