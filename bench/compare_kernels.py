"""
Where two methods spend their GPU time: per-step kernel time by category and
the kernels that differ most, from the nsys summaries of bench/profile_pair.sh.

    python bench/compare_kernels.py <profile_dir> <name_a> <name_b> [--steps 3]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

CATEGORIES = [
    ("gemm", r"gemm|cutlass|nvjet|sm\d+_xmma|cublas|matmul|splitK"),
    ("attention", r"flash|fmha|attention|sdpa"),
    ("triton reduction", r"^triton_(red|per)_"),
    ("triton pointwise", r"^triton_(poi|tem)_"),
    ("triton other", r"^triton_"),
    ("scatter/gather/index", r"scatter|gather|index|segment"),
    ("reduce (aten)", r"reduce|Reduce|sum_kernel|norm"),
    ("elementwise (aten)", r"elementwise|vectorized|unrolled"),
    ("copy/fill", r"copy|fill|memset|cat"),
]


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if re.search(pattern, name):
            return label
    return "other"


def load(path: str) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            ns = float(row["Total Time (ns)"])
            calls = int(row["Instances"])
            out[row["Name"]] = (ns, calls)
    return out


def step_ms(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r.get("Range", "").startswith(":step") or r.get("Range", "").startswith("step")
        ]
    if not rows:
        return None
    return (
        sum(float(r["Total Time (ns)"]) for r in rows)
        / sum(int(r["Instances"]) for r in rows)
        / 1e6
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    ka = load(os.path.join(args.dir, f"{args.a}_cuda_gpu_kern_sum.csv"))
    kb = load(os.path.join(args.dir, f"{args.b}_cuda_gpu_kern_sum.csv"))
    n = args.steps
    print(f"per step, {args.a} vs {args.b}")
    for name in (args.a, args.b):
        ms = step_ms(os.path.join(args.dir, f"{name}_nvtx_sum.csv"))
        kern = load(os.path.join(args.dir, f"{name}_cuda_gpu_kern_sum.csv"))
        total = sum(t for t, _ in kern.values()) / n / 1e6
        calls = sum(c for _, c in kern.values()) / n
        print(f"  {name}: step {ms:.1f} ms (NVTX), kernels {total:.1f} ms in {calls:.0f} launches")
    print("\n| category | a ms | b ms | b - a |")
    print("|---|---|---|---|")
    cat_a, cat_b = defaultdict(float), defaultdict(float)
    for name, (t, _) in ka.items():
        cat_a[category(name)] += t / n / 1e6
    for name, (t, _) in kb.items():
        cat_b[category(name)] += t / n / 1e6
    for label in sorted(set(cat_a) | set(cat_b), key=lambda k: -max(cat_a[k], cat_b[k])):
        print(
            f"| {label} | {cat_a[label]:.1f} | {cat_b[label]:.1f} | {cat_b[label] - cat_a[label]:+.1f} |"
        )
    print(f"\ntop kernels of {args.b} (ms/step, launches/step) and the same name in {args.a}:")
    for name, (t, c) in sorted(kb.items(), key=lambda kv: -kv[1][0])[: args.top]:
        ta, ca = ka.get(name, (0.0, 0))
        print(
            f"  {t / n / 1e6:7.2f} ({c / n:5.0f})  vs {ta / n / 1e6:7.2f} ({ca / n:5.0f})  {name[:110]}"
        )
    print(f"\ntop kernels of {args.a}:")
    for name, (t, c) in sorted(ka.items(), key=lambda kv: -kv[1][0])[: args.top]:
        tb, cb = kb.get(name, (0.0, 0))
        print(
            f"  {t / n / 1e6:7.2f} ({c / n:5.0f})  vs {tb / n / 1e6:7.2f} ({cb / n:5.0f})  {name[:110]}"
        )


if __name__ == "__main__":
    main()
