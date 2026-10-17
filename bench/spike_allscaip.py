"""
Go/no-go spike, before the package exists: one AllScAIP md_NeNo (85M)
conservative training step on the same real OMol batches, computed four ways.

  cg_eager          E from the eager network, F = -dE/dpos with create_graph=True
  explicit_*        the hand-written adjoint of fairchem feature/explicit_create_graph
                    (compiled, compiled + memory budget, per-block checkpoint)
  gen_compiled      network + energy head as ONE torch.compile graph that also
                    calls torch.autograd.grad(create_graph=True) on the
                    position-dependent features (Dynamo trace_autograd_ops);
                    the eager geometry is chained to pos with autograd
  gen_budget        the same with activation_memory_budget
  gen_ckpt          the network runs eagerly, every GraphAttentionBlock is a
                    BlockFn: compiled block forward without saving, compiled
                    forward+VJP recomputed by hand in the parameter backward

Run with fairchem's venv and the explicit worktree first on the path:

    FAIRCHEM_ROOT=../fairchem-explicit PYTHONPATH=../fairchem-explicit/src \
        ../fairchem/.venv/bin/python bench/spike_allscaip.py --json out.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import json
import logging
import time

import torch
import torch.nn.functional as F
from dual_bench.fairchem_adapters import BatchMaker, _unwrap_head, build, fairchem_loss

MLIP = "/home/tione/notebook/home/zhoushouxin/mlip"
PATH_OVERRIDES = [
    f"cluster.omol_all_path={MLIP}/datasets/omol25",
    f"cluster.omol_4M_train_path={MLIP}/datasets/omol25/train_4M",
    f"cluster.data_root_dir={MLIP}/datasets/omol25",
]
MODES = (
    "cg_eager",
    "explicit_compiled",
    "explicit_budget",
    "explicit_ckpt",
    "gen_compiled",
    "gen_budget",
    "gen_ckpt",
)
GIB = 2**30
INDUCTOR_OPTIONS = {"triton.mix_order_reduction": False}


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------


def prep(backbone, batch, pos):
    batch["pos"] = pos
    batch["atomic_numbers"] = batch["atomic_numbers"].long()
    batch["atomic_numbers_full"] = batch["atomic_numbers"]
    batch["batch_full"] = batch["batch"]
    with torch.autocast(device_type="cuda", enabled=False):
        return backbone.data_preprocess(batch)


def diff_names(x) -> tuple[str, ...]:
    return tuple(
        f.name
        for f in dataclasses.fields(x)
        if isinstance(getattr(x, f.name), torch.Tensor) and getattr(x, f.name).requires_grad
    )


def rms_norm_composite(input, normalized_shape, weight=None, eps=None):
    dims = tuple(range(-len(normalized_shape), 0))
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    out = input * torch.rsqrt(input.pow(2).mean(dims, keepdim=True) + eps)
    return out if weight is None else out * weight


@contextlib.contextmanager
def gen_patches(budget: float | None):
    original = F.rms_norm
    F.rms_norm = rms_norm_composite
    try:
        with torch._dynamo.config.patch(trace_autograd_ops=True, optimize_ddp=False):
            if budget is None:
                yield
            else:
                with torch._functorch.config.patch(activation_memory_budget=budget):
                    yield
    finally:
        F.rms_norm = original


def chain_to_pos(feats, grads, pos):
    pairs = [(f, g) for f, g in zip(feats, grads, strict=True) if g is not None]
    (g,) = torch.autograd.grad([f for f, _ in pairs], pos, [g for _, g in pairs], create_graph=True)
    return -g


# ---------------------------------------------------------------------------
# classic create_graph
# ---------------------------------------------------------------------------


def ef_cg_eager(backbone, head, batch):
    pos = batch.pos.detach().requires_grad_(True)
    x = prep(backbone, batch, pos)
    energy = head.compiled_forward(backbone.compiled_forward(x)).view(-1)
    (g,) = torch.autograd.grad(energy.sum(), pos, create_graph=True)
    return energy, -g


# ---------------------------------------------------------------------------
# hand-written adjoint (explicit branch)
# ---------------------------------------------------------------------------


def set_explicit(backbone, *, compile: bool, budget: float, ckpt: bool):
    cfg = backbone.global_cfg
    cfg.explicit_force = True
    cfg.explicit_force_compile = compile
    cfg.explicit_force_memory_budget = budget
    cfg.explicit_force_checkpointing = ckpt
    backbone.explicit_network_fn = None


def ef_explicit(backbone, head, batch):
    from fairchem.core.models.allscaip.utils.explicit_force import (
        explicit_energy_and_forces,
    )

    pos = batch.pos.detach().requires_grad_(True)
    x = prep(backbone, batch, pos)
    energy, forces = explicit_energy_and_forces(backbone, head, batch, x)
    return energy.view(-1), forces


# ---------------------------------------------------------------------------
# generated: one region with in-graph autograd.grad
# ---------------------------------------------------------------------------


def region(leaves, stripped, names, backbone, head):
    x = dataclasses.replace(stripped, **dict(zip(names, leaves, strict=True)))
    energy = head.compiled_forward(backbone.compiled_forward(x)).view(-1)
    grads = torch.autograd.grad(energy.sum(), leaves, create_graph=True, allow_unused=True)
    return energy, grads


class Gen:
    def __init__(self, budget: float | None):
        self.budget = budget
        self.fn = torch.compile(region, dynamic=True, fullgraph=True, options=INDUCTOR_OPTIONS)

    def __call__(self, backbone, head, batch):
        pos = batch.pos.detach().requires_grad_(True)
        x = prep(backbone, batch, pos)
        names = diff_names(x)
        feats = [getattr(x, n) for n in names]
        leaves = tuple(f.detach().requires_grad_(True) for f in feats)
        stripped = dataclasses.replace(x, **{n: None for n in names})
        with gen_patches(self.budget):
            energy, grads = self.fn(leaves, stripped, names, backbone, head)
        return energy, chain_to_pos(feats, grads, pos)


# ---------------------------------------------------------------------------
# generated, per-block checkpoint: BlockFn / VjpFn
# ---------------------------------------------------------------------------

FORCES_PASS = [False]


@contextlib.contextmanager
def forces_pass():
    FORCES_PASS.append(True)
    try:
        yield
    finally:
        FORCES_PASS.pop()


def _block_forward(block, base, names, x, fields):
    data = dataclasses.replace(base, **dict(zip(names, fields, strict=True)))
    return type(block).forward(block, data, x)


def _block_vjp(block, base, names, x, fields, g, create):
    with torch.enable_grad():
        data = dataclasses.replace(base, **dict(zip(names, fields, strict=True)))
        out = type(block).forward(block, data, x)
        return torch.autograd.grad(out, [x, *fields], g, create_graph=create, allow_unused=True)


class BlockRunner:
    def __init__(self, budget: float | None):
        self.budget = budget
        self.fwd = torch.compile(
            _block_forward, dynamic=True, fullgraph=True, options=INDUCTOR_OPTIONS
        )
        self.vjp = torch.compile(_block_vjp, dynamic=True, fullgraph=True, options=INDUCTOR_OPTIONS)

    def call(self, fn, *args):
        with gen_patches(self.budget):
            return fn(*args)


def _leaf(t, requires_grad=None):
    return t.detach().requires_grad_(t.requires_grad if requires_grad is None else requires_grad)


def _fill(grads, like):
    return [torch.zeros_like(t) if g is None else g for g, t in zip(grads, like, strict=True)]


class VjpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, block, base, names, nf, x, *rest):
        fields, g, params = rest[:nf], rest[nf], rest[nf + 1 :]
        ctx.runner, ctx.block, ctx.base, ctx.names, ctx.nf = runner, block, base, names, nf
        ctx.save_for_backward(x, *fields, g)
        ctx.n_params = len(params)
        out = runner.call(
            runner.vjp, block, base, names, _leaf(x, True),
            [_leaf(f, True) for f in fields], g.detach(), False,
        )  # fmt: skip
        return tuple(_fill(out, [x, *fields]))

    @staticmethod
    def backward(ctx, *ggs):
        x, *rest = ctx.saved_tensors
        fields, g = rest[: ctx.nf], rest[ctx.nf]
        params = [p for p in ctx.block.parameters() if p.requires_grad]
        with torch.enable_grad():
            xl, gl = _leaf(x, True), _leaf(g, True)
            fl = [_leaf(f, True) for f in fields]
            outs = ctx.runner.call(ctx.runner.vjp, ctx.block, ctx.base, ctx.names, xl, fl, gl, True)
            pairs = [
                (o, gg) for o, gg in zip(outs, ggs, strict=True) if o is not None and gg is not None
            ]
            inputs = [xl, *fl, gl, *params]
            grads = torch.autograd.grad(
                [o for o, _ in pairs], inputs, [gg for _, gg in pairs], allow_unused=True
            )
        return (None, None, None, None, None, *grads)


class BlockFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, block, base, names, nf, x, *rest):
        fields, params = rest[:nf], rest[nf:]
        ctx.runner, ctx.block, ctx.base, ctx.names, ctx.nf = runner, block, base, names, nf
        ctx.save_for_backward(x, *fields)
        ctx.n_params = len(params)
        return runner.call(runner.fwd, block, base, names, x.detach(), [f.detach() for f in fields])

    @staticmethod
    def backward(ctx, g):
        x, *fields = ctx.saved_tensors
        params = [p for p in ctx.block.parameters() if p.requires_grad]
        if FORCES_PASS[-1]:
            out = VjpFn.apply(
                ctx.runner, ctx.block, ctx.base, ctx.names, ctx.nf, x, *fields, g, *params
            )
            return (None, None, None, None, None, *out, *([None] * len(params)))
        with torch.enable_grad():
            xl = _leaf(x)
            fl = [_leaf(f) for f in fields]
            out = ctx.runner.call(ctx.runner.fwd, ctx.block, ctx.base, ctx.names, xl, fl)
            inputs = [xl, *fl, *params]
            wanted = [t for t in inputs if t.requires_grad]
            found = iter(torch.autograd.grad(out, wanted, g, allow_unused=True))
            grads = [next(found) if t.requires_grad else None for t in inputs]
        return (None, None, None, None, None, *grads)


class GenCkpt:
    def __init__(self, budget: float | None):
        self.runner = BlockRunner(budget)

    @contextlib.contextmanager
    def blocks(self, backbone):
        runner = self.runner

        def wrap(block):
            def forward(data, neighbor_reps, layer_idx=0):
                names = diff_names(data)
                fields = [getattr(data, n) for n in names]
                base = dataclasses.replace(data, **{n: None for n in names})
                params = [p for p in block.parameters() if p.requires_grad]
                return BlockFn.apply(
                    runner, block, base, names, len(fields), neighbor_reps, *fields, *params
                )

            return forward

        blocks = list(backbone.transformer_blocks)
        for block in blocks:
            block.forward = wrap(block)
        try:
            yield
        finally:
            for block in blocks:
                del block.forward

    def __call__(self, backbone, head, batch):
        pos = batch.pos.detach().requires_grad_(True)
        x = prep(backbone, batch, pos)
        names = diff_names(x)
        feats = [getattr(x, n) for n in names]
        leaves = tuple(f.detach().requires_grad_(True) for f in feats)
        xl = dataclasses.replace(x, **dict(zip(names, leaves, strict=True)))
        with self.blocks(backbone):
            energy = head.compiled_forward(backbone.compiled_forward(xl)).view(-1)
        with forces_pass():
            grads = torch.autograd.grad(energy.sum(), leaves, create_graph=True, allow_unused=True)
        return energy, chain_to_pos(feats, grads, pos)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def make_mode(mode: str, backbone, budget: float):
    if mode == "cg_eager":
        return ef_cg_eager
    if mode.startswith("explicit"):
        set_explicit(
            backbone,
            compile=True,
            budget=budget if mode == "explicit_budget" else 1.0,
            ckpt=mode == "explicit_ckpt",
        )
        return ef_explicit
    backbone.global_cfg.explicit_force = False
    if mode == "gen_compiled":
        return Gen(None)
    if mode == "gen_budget":
        return Gen(budget)
    if mode == "gen_ckpt":
        return GenCkpt(None)
    raise ValueError(mode)


def unique_graphs() -> int:
    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


def step(ef, model, maker, batch, coefs):
    backbone, head = model.backbone, _unwrap_head(model)
    energy, forces = ef(backbone, head, batch)
    losses = fairchem_loss(
        energy,
        forces,
        batch["dual_energy_target"],
        batch["dual_forces_target"],
        batch.natoms.float(),
        batch["dual_free_mask"],
        *coefs,
    )
    loss = sum(losses.values())
    loss.backward()
    return energy.detach(), forces.detach(), loss.detach()


def grads_of(model):
    return [
        (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        for p in model.parameters()
        if p.requires_grad
    ]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def measure(mode, model, maker, args, coefs, tf32):
    torch.compiler.reset()
    ef = make_mode(mode, model.backbone, args.budget)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    seeds = list(range(args.warmup + args.steps))
    times, peaks, atoms = [], [], []
    start_graphs = unique_graphs()
    compile_s = None
    try:
        for i, seed in enumerate(seeds):
            batch = maker(args.atoms, seed)
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step(ef, model, maker, batch, coefs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if i == 0:
                compile_s = dt
            if i >= args.warmup:
                times.append(dt)
                peaks.append(torch.cuda.max_memory_allocated() - base)
                atoms.append(int(batch.natoms.sum()))
            del batch
    except torch.OutOfMemoryError:
        return {"mode": mode, "status": "OOM"}
    finally:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "mode": mode,
        "status": "ok",
        "ms_per_step": 1e3 * sum(times) / len(times),
        "peak_gib": max(peaks) / GIB,
        "mean_atoms": sum(atoms) / len(atoms),
        "first_step_s": compile_s,
        "graphs_compiled": unique_graphs() - start_graphs,
    }


def check(modes, model, maker, args, coefs):
    """fp32, TF32 off: every mode against cg_eager on the same batch."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {}
    ref = None
    for mode in ("cg_eager", *[m for m in modes if m != "cg_eager"]):
        torch.compiler.reset()
        ef = make_mode(mode, model.backbone, args.budget)
        model.zero_grad(set_to_none=True)
        energy, forces, loss = step(ef, model, maker, maker(args.check_atoms, 1234), coefs)
        grads = grads_of(model)
        if ref is None:
            ref = (energy, forces, loss, grads)
            continue
        flat = torch.cat([g.flatten() for g in grads])
        flat_ref = torch.cat([g.flatten() for g in ref[3]])
        # tensors whose reference gradient is ~0 would dominate a plain relative
        # error: floor each tensor's scale at 1e-4 of the whole gradient's norm
        floor = 1e-4 * flat_ref.norm()
        results[mode] = {
            "energy": rel(energy, ref[0]),
            "forces": rel(forces, ref[1]),
            "loss": rel(loss, ref[2]),
            "param_grads": rel(flat, flat_ref),
            "worst_param_tensor": max(
                ((g - r).norm() / torch.maximum(r.norm(), floor)).item()
                for g, r in zip(grads, ref[3], strict=True)
            ),
        }
        print(json.dumps({"check": mode, **results[mode]}), flush=True)
    model.zero_grad(set_to_none=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--atoms", type=int, default=1024)
    parser.add_argument("--check-atoms", type=int, default=400)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    cfg, model, tasks, dataset, collate = build("allscaip", PATH_OVERRIDES)
    model = model.cuda().train()
    maker = BatchMaker(dataset, collate, tasks, "cuda")
    coefs = (
        float(maker.energy_task.loss_fn.coefficient),
        float(maker.force_task.loss_fn.coefficient),
    )
    tf32 = bool(cfg.runner.train_eval_unit.get("tf32", False))
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  tf32: {tf32}", flush=True)

    out = {"atoms": args.atoms, "budget": args.budget, "tf32": tf32}
    if not args.skip_check:
        out["check"] = check(args.modes, model, maker, args, coefs)
    if not args.skip_timing:
        out["timing"] = []
        for mode in args.modes:
            result = measure(mode, model, maker, args, coefs, tf32)
            out["timing"].append(result)
            print(json.dumps(result), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
