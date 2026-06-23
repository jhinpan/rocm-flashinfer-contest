#!/usr/bin/env python3
"""Summarize a rocprofv3 kernel-trace CSV into a per-stage table.

Reads a `*_kernel_trace.csv` produced by `rocprofv3 --kernel-trace --output-format csv` and emits
`stage,kernel_pattern,launches,total_us,avg_us,pct,class` so the triage stage tables are reproducible
from the raw traces (not hand-edited).

    python tools/summarize_rocprof_trace.py /tmp/v3prof/moe_fp8_min/k_kernel_trace.csv --kernel moe_fp8

Stage attribution is by kernel-name substring; unmatched dispatches fall into "other".
"""
import argparse
import csv as _csv
from collections import defaultdict

# (stage, class, [name substrings]) — first match wins, in order.
STAGE_RULES = {
    "gdn_decode": [
        ("state_transpose_copy", "copy", ["elementwise_kernel_manual", "CatArray", "copyBuffer"]),
        ("fused_recurrent_gate", "fused-recurrent", ["fused_sigmoid_gating", "fused_recurrent_gated"]),
        ("misc_elementwise", "elementwise", ["elementwise", "reduce_kernel"]),
    ],
    "dsa_topk_indexer": [
        ("topk_sort", "topk", ["topk", "MergeSort", "radixSort", "gatherTopK"]),
        ("logits", "fused-logits", ["_fused_logits", "Cijk", "bmm", "gemm"]),
        ("dequant_score", "elementwise", ["elementwise", "reduce_kernel"]),
        ("index_map", "indexing", ["index", "scatter", "gather"]),
    ],
    "moe_fp8": [
        ("gemm", "GEMM", ["Cijk", "gemm"]),
        ("weight_dequant_elementwise", "elementwise", ["elementwise_kernel_manual"]),
        ("vectorized_elementwise", "elementwise", ["vectorized_elementwise"]),
        ("routing_reduce", "reduce", ["reduce_kernel", "trampoline", "rocprim"]),
        ("scatter_index", "indexing", ["indexFunc", "index", "scatter"]),
    ],
}


def classify(name, rules):
    for stage, cls, subs in rules:
        if any(s in name for s in subs):
            return stage, cls
    return "other", "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_csv")
    ap.add_argument("--kernel", required=True, choices=list(STAGE_RULES))
    args = ap.parse_args()
    rules = STAGE_RULES[args.kernel]
    rows = list(_csv.DictReader(open(args.trace_csv)))
    agg = defaultdict(lambda: [0, 0.0])   # stage -> [launches, total_us]
    cls_of = {}
    tot = 0.0
    for r in rows:
        dur = (float(r["End_Timestamp"]) - float(r["Start_Timestamp"])) / 1000.0  # us
        stage, cls = classify(r["Kernel_Name"], rules)
        agg[stage][0] += 1
        agg[stage][1] += dur
        cls_of[stage] = cls
        tot += dur
    print(f"# {args.trace_csv}  ({len(rows)} dispatches, {tot:.0f} us total)")
    print("stage,class,launches,total_us,avg_us,pct")
    for stage, (n, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
        print(f"{stage},{cls_of[stage]},{n},{t:.1f},{t/n:.2f},{100*t/tot:.1f}")


if __name__ == "__main__":
    main()
