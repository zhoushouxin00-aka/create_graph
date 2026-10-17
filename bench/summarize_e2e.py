"""
Summary of bench/fairchem_e2e.sh: atoms/s averaged over steps 10-29 (the
trainer's own log, print_every=1), peak GPU memory (nvidia-smi), and for the
2-GPU runs whether both ranks hold the same gradient checksums and how far the
generated mode is from create_graph.

    python bench/summarize_e2e.py --dir bench/results/e2e
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re

LOG_DICT = re.compile(r"(\{'train/loss'.*\})")


def throughput(stdout: str, first: int = 10, last: int = 29):
    rates, losses = {}, {}
    with open(stdout, errors="replace") as fh:
        for line in fh:
            m = LOG_DICT.search(line)
            if not m:
                continue
            try:
                d = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                continue
            step = int(d["train/step"])
            rates[step] = d["train/atoms_per_second(approx)"]
            losses[step] = d["train/loss"]
    window = [rates[s] for s in range(first, last + 1) if s in rates]
    return (sum(window) / len(window) if window else None), len(rates), losses


def peak_gib(memory_csv: str):
    best = 0
    if not os.path.exists(memory_csv):
        return None
    with open(memory_csv) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[1].strip().isdigit():
                best = max(best, int(parts[1]))
    return best / 1024 if best else None


def checksums(directory: str) -> dict:
    """{(rank, step): {name: grad abs-mean}} from debug_checksums_save_path."""
    out = {}
    for path in glob.glob(os.path.join(directory, "*.txt")):
        m = re.search(r"ddp\d+\.(\d+)_gp\d+\.\d+_step(\d+)\.txt", os.path.basename(path))
        if not m:
            continue
        grads = {}
        with open(path) as fh:
            for line in fh:
                kind, *rest = line.rstrip("\n").split(",")
                if kind == "Grad":
                    grads[rest[1]] = float(rest[2])
                elif kind == "Loss":
                    grads["__loss__"] = float(rest[0])
        out[(int(m.group(1)), int(m.group(2)))] = grads
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="bench/results/e2e")
    args = parser.parse_args()

    print("| run | atoms/s (steps 10-29) | peak GiB (nvidia-smi) | steps logged |")
    print("|---|---|---|---|")
    for run in sorted(glob.glob(os.path.join(args.dir, "*_1*"))):
        name = os.path.basename(run)
        rate, n, _ = throughput(os.path.join(run, "stdout.log"))
        peak = peak_gib(os.path.join(run, "memory.csv"))
        print(
            f"| {name} | {rate:.0f} | {peak:.1f} | {n} |"
            if rate is not None and peak is not None
            else f"| {name} | - | {peak} | {n} |"
        )

    sums = {
        mode: checksums(os.path.join(args.dir, f"ddp2_{mode}", "checksums"))
        for mode in ("generated", "cons")
    }
    print("\n2 GPUs, 4 steps:\n")
    for mode, table in sums.items():
        steps = sorted({s for _, s in table})
        same = all(
            {k: v for k, v in table.get((0, s), {}).items() if k != "__loss__"}
            == {k: v for k, v in table.get((1, s), {}).items() if k != "__loss__"}
            for s in steps
        )
        n = len(table.get((0, steps[0]), {})) - 1 if steps else 0
        print(f"- {mode}: steps {steps}, {n} parameter gradients, ranks identical: {same}")
    # The key-projection biases of softmax attention have zero gradient in exact
    # arithmetic (a constant shift of every logit in a row): they hold rounding
    # noise and are left out. The rest: the norm-relative difference of the
    # per-parameter abs-means, and the worst parameter above 1e-4 of the largest.
    a, b = sums["generated"], sums["cons"]
    for key in sorted(set(a) & set(b)):
        names = [
            k
            for k in a[key]
            if k != "__loss__" and k in b[key] and not k.endswith("attn_in_proj_k.bias")
        ]
        zero = len([k for k in a[key] if k.endswith("attn_in_proj_k.bias")])
        diff = sum((a[key][k] - b[key][k]) ** 2 for k in names) ** 0.5
        norm = sum(b[key][k] ** 2 for k in names) ** 0.5
        scale = max(abs(b[key][k]) for k in names)
        large = [k for k in names if abs(b[key][k]) > 1e-4 * scale]
        worst = max(abs(a[key][k] - b[key][k]) / abs(b[key][k]) for k in large)
        loss = abs(a[key]["__loss__"] - b[key]["__loss__"]) / abs(b[key]["__loss__"])
        print(
            f"- rank {key[0]} step {key[1]}: generated vs create_graph: loss {loss:.1e}, "
            f"gradient abs-means {diff / norm:.1e} (norm) over {len(names)} parameters, "
            f"worst {worst:.1e} over the {len(large)} above 1e-4 of the largest "
            f"({zero} key biases with zero gradient left out)"
        )


if __name__ == "__main__":
    main()
