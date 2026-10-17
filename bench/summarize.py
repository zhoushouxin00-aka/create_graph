"""
Tables from bench/results/matrix: for every model and size, each method's
ms/step, peak memory and largest batch, and what create_graph gains over each
baseline: speedup (baseline ms / create_graph ms), memory saved
(1 - create_graph peak / baseline peak) and max-batch ratio.

    python bench/summarize.py --dir bench/results/matrix > bench/results/summary.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

# create_graph method -> the baselines it is compared with, per model
PAIRS = {
    "allscaip": {
        "cg": ["cg_eager", "explicit_compiled", "2f2b"],
        "cg_budget": ["explicit_budget", "2f2b_budget"],
        "cg_ckpt": ["explicit_ckpt", "2f2b_ckpt"],
    },
    "pet": {"cg": ["native_eager", "cg_eager"], "cg_budget": ["cg_eager"], "cg_ckpt": ["cg_eager"]},
    "nequip": {
        "cg": ["nequip_eager", "nequip_compile", "cg_eager"],
        "cg_budget": ["nequip_compile"],
        "cg_ckpt": ["nequip_compile"],
    },
    "uma": {"cg": ["cg_eager"], "cg_budget": ["cg_eager"], "cg_ckpt": ["cg_eager"]},
}


def load(directory: str) -> dict:
    """{(shard, model, size, atoms, budget dir): {method: result}}; methods are
    only compared within a shard (same batches)."""
    runs: dict = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
        with open(path) as fh:
            r = json.load(fh)
        if r.get("status") != "ok":
            r.setdefault("ms_per_step", float("nan"))
        parts = os.path.relpath(os.path.dirname(path), directory).split(os.sep)
        shard = parts[0] if parts[0] != "." else ""
        sweep = parts[1] if len(parts) > 1 else ""
        runs[(shard, r["model"], r["size"], r["atoms"], sweep)][r["method"]] = r
    return runs


def fmt(x, spec=".1f"):
    return "-" if x is None or x != x else format(x, spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="bench/results/matrix")
    args = parser.parse_args()
    runs = load(args.dir)
    for (shard, model, size, atoms, sweep), methods in sorted(runs.items()):
        where = f" ({shard}{', ' + sweep if sweep else ''})" if shard else ""
        print(f"\n### {model} {size}, {atoms} atoms{where}\n")
        print("| method | ms/step | peak GiB | max atoms | atoms/s at max |")
        print("|---|---|---|---|---|")
        for method, r in methods.items():
            status = "" if r.get("status") == "ok" else f" ({r.get('status')})"
            print(
                f"| {method}{status} | {fmt(r.get('ms_per_step'))} | {fmt(r.get('peak_gib'), '.2f')} "
                f"| {fmt(r.get('max_atoms'), 'd') if r.get('max_atoms') else '-'} "
                f"| {fmt(r.get('atoms_per_s_at_max'), '.0f')} |"
            )
        rows = []
        for ours, baselines in PAIRS.get(model, {}).items():
            a = methods.get(ours)
            if a is None or a.get("status") != "ok":
                continue
            for base in baselines:
                b = methods.get(base)
                if b is None or b.get("status") != "ok":
                    continue
                speed = b["ms_per_step"] / a["ms_per_step"]
                saved = 1 - a["peak_gib"] / b["peak_gib"]
                batch = (
                    a["max_atoms"] / b["max_atoms"]
                    if a.get("max_atoms") and b.get("max_atoms")
                    else None
                )
                rows.append((ours, base, speed, saved, batch))
        if rows:
            print("\n| create_graph | vs | speedup | memory saved | max batch |")
            print("|---|---|---|---|---|")
            for ours, base, speed, saved, batch in rows:
                print(
                    f"| {ours} | {base} | {speed:.2f}x | {100 * saved:+.0f}% "
                    f"| {fmt(batch, '.2f') + 'x' if batch else '-'} |"
                )


if __name__ == "__main__":
    main()
