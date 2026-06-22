#!/usr/bin/env python3
"""Pack a solution directory into a flashinfer-bench solution.json.

Usage:
  python tools/pack_solution.py --def <definition_name> --name <sol_name> \
      --dir solutions/<k> --entry main.py::run [--lang python] [--deps aiter ...] \
      --out solutions/<k>/solution.json
All files in --dir (except *.json) are embedded as sources.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="definition", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--entry", default="main.py::run")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--deps", nargs="*", default=[])
    ap.add_argument("--dps", action="store_true", help="destination_passing_style (default False)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.dir)
    sources = []
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix != ".json":
            sources.append({"path": f.name, "content": f.read_text()})

    sol = {
        "name": args.name,
        "definition": args.definition,
        "author": "rocm-port",
        "spec": {
            "language": args.lang,
            "target_hardware": ["AMD MI300X"],
            "entry_point": args.entry,
            "dependencies": args.deps,
            "destination_passing_style": bool(args.dps),
        },
        "sources": sources,
        "description": f"ROCm/MI300 port for {args.definition}",
    }
    Path(args.out).write_text(json.dumps(sol, indent=2))
    print(f"wrote {args.out} with {len(sources)} source file(s)")


if __name__ == "__main__":
    main()
