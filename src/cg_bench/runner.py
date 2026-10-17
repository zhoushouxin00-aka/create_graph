"""
One conservative-force training step (forward, forces, loss, backward) per
process: ms/step and peak memory at a fixed batch, optionally the largest
batch (atoms) that fits and the throughput there.

    python -m cg_bench.runner --model pet --size default --method cg \
        --atoms 2048 --max-atoms --json bench/results/matrix/pet_default_cg.json

Methods:
  every model   cg_eager (eager create_graph: autograd.grad(create_graph=True)
                through the model's own code), cg / cg_budget / cg_ckpt
                (create_graph: region, + memory budget, + per-block checkpoint)
  pet           native_eager (metatrain's PET forward on System objects)
  nequip        nequip_eager / nequip_compile (nequip's force output, its
                eager and compile_mode="compile" paths)
  allscaip      explicit_compiled / explicit_budget / explicit_ckpt (needs the
                feature/explicit_create_graph worktree first on PYTHONPATH),
                2f2b / 2f2b_budget / 2f2b_ckpt (needs a feature/2f2b checkout)

Peak memory is measured above the steady state (parameters and inputs are the
same for every method); gradients are freed before each step. CUDA_VISIBLE_DEVICES
should name one idle GPU: the GPU's other processes are recorded in the JSON.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import subprocess
import time
from collections.abc import Callable

import torch

import create_graph as cg
from cg_bench import train

GIB = 2**30


@dataclasses.dataclass
class Workload:
    make: Callable[[int, int], object]  # (atoms, seed) -> batch
    step: Callable[[object], None]
    atoms_of: Callable[[object], int]
    params: Callable[[], list]


def _config(method: str, budget: float, blocks: str) -> cg.CompileConfig | None:
    return {
        "cg": cg.CompileConfig(),
        "cg_budget": cg.CompileConfig(memory_budget=budget),
        "cg_ckpt": cg.CompileConfig(checkpoint=blocks),
    }.get(method)


def pet_workload(size: str, method: str, budget: float) -> Workload:
    from cg_bench import data as cgdata
    from cg_bench.pet_model import PETEnergy, batch_of, build_pet, native_energy_and_forces

    frames = cgdata.load_omol()
    refs = cgdata.linear_references(frames)
    pet = build_pet(cgdata.elements_of(frames), size)
    config = _config(method, budget, r"pet\.backend\.gnn_layers\.\d+")
    model = PETEnergy(pet, config)

    def make(atoms, seed):
        packed = cgdata.Packer(frames, atoms, seed)[0]
        return batch_of(pet, packed), train.targets_of(packed, refs)

    def step(batch):
        data, t = batch
        if method == "native_eager":
            energy, forces = native_energy_and_forces(pet, data["systems"])
            train.conservative_loss(energy, forces, t).backward()
        else:
            train.step(model, data, t, exact=method == "cg_eager")

    return Workload(make, step, lambda b: int(b[1].natoms.sum()), lambda: list(pet.parameters()))


def nequip_workload(size: str, method: str, budget: float) -> Workload:
    from cg_bench import data as cgdata
    from cg_bench.nequip_model import batch_of, build_nequip, energy_region, type_names_of

    frames = cgdata.load_omol()
    refs = cgdata.linear_references(frames)
    names = type_names_of(cgdata.elements_of(frames))
    if method.startswith("nequip_"):
        mode = method.removeprefix("nequip_")
        model = build_nequip(names, size, do_derivatives=True, compile_mode=mode)
    else:
        model = build_nequip(names, size)
        config = _config(method, budget, r"model\.func\.layer\d+_convnet")
        region = energy_region(model, config) if config is not None else model

    def make(atoms, seed):
        packed = cgdata.Packer(frames, atoms, seed)[0]
        return batch_of(packed, names), train.targets_of(packed, refs, dtype=torch.float64)

    def step(batch):
        data, t = batch
        if method.startswith("nequip_"):
            out = model(dict(data))
            train.conservative_loss(out["total_energy"], out["forces"], t).backward()
        else:
            train.step(region, dict(data), t, outputs="total_energy", exact=method == "cg_eager")

    return Workload(make, step, lambda b: int(b[1].natoms.sum()), lambda: list(model.parameters()))


def fairchem_workload(name: str, size: str, method: str, budget: float) -> Workload:
    from cg_bench import fairchem_models as fm

    model, tasks, dataset, collate = fm.build(name, size)
    model = model.cuda().train()
    batches = fm.Batches(model, tasks, dataset, collate, graph=name == "uma")
    systems: list[int] = []
    energy_cls = fm.AllScAIPEnergy if name == "allscaip" else fm.UMAEnergy
    blocks = fm.ALLSCAIP_BLOCKS if name == "allscaip" else r"backbone\.blocks\.\d+"
    region = energy_cls(model, _config(method, budget, blocks))
    if method.startswith("explicit"):
        fm.explicit_setup(
            model,
            budget=budget if method == "explicit_budget" else 1.0,
            checkpoint=method == "explicit_ckpt",
        )
    if method.startswith("2f2b"):
        fm.two_f_two_b_setup(
            model,
            budget=budget if method == "2f2b_budget" else 1.0,
            checkpoint=method == "2f2b_ckpt",
        )

    def make(atoms, seed):
        data = batches(atoms, seed)
        if name == "uma":
            # UMA's code specializes on the number of systems: one for every batch
            if not systems:
                systems.append(int(data.natoms.numel()))
            want = systems[0]
            tries = 1
            while int(data.natoms.numel()) != want:
                data = batches(atoms, 1_000_000 + 1000 * seed + tries)
                tries += 1
        return data, batches.targets(data)

    def step(batch):
        data, t = batch
        if method.startswith("explicit"):
            energy, forces = fm.explicit_energy_and_forces(model, data)
            train.conservative_loss(energy, forces, t).backward()
        elif method.startswith("2f2b"):
            fm.two_f_two_b_step(model, tasks, data)
        else:
            train.step(region, data, t, exact=method == "cg_eager")

    return Workload(make, step, lambda b: int(b[0].natoms.sum()), lambda: list(model.parameters()))


def build_workload(model: str, size: str, method: str, budget: float) -> Workload:
    if model == "pet":
        return pet_workload(size, method, budget)
    if model == "nequip":
        return nequip_workload(size, method, budget)
    return fairchem_workload(model, size, method, budget)


def _clear(params):
    for p in params:
        p.grad = None
    gc.collect()
    torch.cuda.empty_cache()


def measure(w: Workload, atoms: int, warmup: int, steps: int, seed0: int = 0) -> dict:
    """ms/step and peak memory on fresh batches (one per step, so dynamic shapes
    are exercised). The warm-up steps compile; their batches shrink by 5 atoms a
    step, so the compiles a second size triggers (Dynamo's duck sizing, a
    block's scalar argument becoming dynamic) happen before timing."""
    params = w.params()
    times, peaks, sizes = [], [], []
    first = None
    for i in range(warmup):
        batch = w.make(atoms - 5 * i, seed0 + i)
        _clear(params)
        t0 = time.perf_counter()
        w.step(batch)
        torch.cuda.synchronize()
        if first is None:
            first = time.perf_counter() - t0
        del batch
    for i in range(warmup, warmup + steps):
        batch = w.make(atoms, seed0 + i)
        _clear(params)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        w.step(batch)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if first is None:
            first = dt
        times.append(dt)
        peaks.append(torch.cuda.max_memory_allocated() - base)
        sizes.append(w.atoms_of(batch))
        del batch
    _clear(params)
    return {
        "ms_per_step": 1e3 * sum(times) / len(times),
        "peak_gib": max(peaks) / GIB,
        "mean_atoms": sum(sizes) / len(sizes),
        "atoms_per_s": sum(sizes) / sum(times),
        "first_step_s": first,
    }


def profile_steps(w: Workload, atoms: int, steps: int) -> None:
    """Steady-state steps for ``nsys profile --capture-range=cudaProfilerApi``,
    one NVTX range each."""
    params = w.params()
    batches = [w.make(atoms, 30_000 + i) for i in range(steps)]
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for i, batch in enumerate(batches):
        _clear(params)
        torch.cuda.nvtx.range_push(f"step {i}")
        w.step(batch)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    _clear(params)


def probe(w: Workload, atoms: int) -> dict | None:
    """Two steps on two batches of about ``atoms`` atoms, or None if out of memory."""
    try:
        return measure(w, atoms, warmup=0, steps=2, seed0=10_000 + atoms)
    except torch.OutOfMemoryError:
        _clear(w.params())
        return None


def max_atoms_search(w: Workload, start: int, limit: int) -> dict:
    """The largest batch that fits, to 5%: grow by 1.5x until a step runs out
    of memory, then bisect. The throughput there is that of the probe that fit."""
    best = probe(w, start)
    if best is None:
        return {"max_atoms": None}
    lo, hi, n = start, None, start
    while hi is None:
        nxt = int(n * 1.5)
        if nxt > limit:
            break
        found = probe(w, nxt)
        if found is not None:
            lo, n, best = nxt, nxt, found
        else:
            hi = nxt
    while hi is not None and hi / lo > 1.05:
        mid = int((lo * hi) ** 0.5)
        found = probe(w, mid)
        if found is not None:
            lo, best = mid, found
        else:
            hi = mid
    return {
        "max_atoms": lo,
        "mean_atoms_at_max": best["mean_atoms"],
        "atoms_per_s_at_max": best["atoms_per_s"],
        "peak_gib_at_max": best["peak_gib"],
    }


def _smi(*query: str) -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    return [[f.strip() for f in line.split(",")] for line in out.strip().splitlines()]


def gpu_neighbors() -> str:
    """Other processes on this process's GPU (timings are clean when empty)."""
    try:
        index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        uuids = {i: u for i, u in _smi("--query-gpu=index,uuid")}
        uuid = uuids.get(index, index)
        apps = _smi("--query-compute-apps=gpu_uuid,pid,used_memory")
    except Exception as error:  # noqa: BLE001
        return f"unknown ({error})"
    others = [f"{pid} {mem} MiB" for u, pid, mem in apps if u == uuid and pid != str(os.getpid())]
    return "; ".join(others)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=["pet", "nequip", "allscaip", "uma"])
    parser.add_argument("--size", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--atoms", type=int, default=1024)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-atoms", action="store_true")
    parser.add_argument("--max-limit", type=int, default=1 << 17)
    parser.add_argument("--tf32", action="store_true", help="TF32 matmuls, as in training")
    parser.add_argument(
        "--profile-steps", type=int, default=0, help="extra steps inside cudaProfilerStart/Stop"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.tf32
    result = {
        "model": args.model,
        "size": args.size,
        "method": args.method,
        "atoms": args.atoms,
        "budget": args.budget,
        "tf32": args.tf32,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_neighbors_before": gpu_neighbors(),
    }
    try:
        w = build_workload(args.model, args.size, args.method, args.budget)
        result.update(measure(w, args.atoms, args.warmup, args.steps))
        if args.profile_steps:
            profile_steps(w, args.atoms, args.profile_steps)
        if args.max_atoms:
            result.update(max_atoms_search(w, args.atoms, args.max_limit))
        result["status"] = "ok"
    except torch.OutOfMemoryError:
        result["status"] = "OOM"
    result["gpu_neighbors_after"] = gpu_neighbors()
    print(json.dumps(result), flush=True)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
