"""
Acceptance checks for models and fused operators, against eager create_graph.

Errors are relative norms. For parameter gradients two numbers are reported:
the error of all gradients concatenated, and the worst single tensor, whose
scale is floored at 1e-4 of the whole gradient's norm (tensors whose reference
gradient is ~0 would otherwise dominate with pure rounding noise).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from create_graph.api import create_graph
from create_graph.region import select_output


def rel_err(got: Tensor, want: Tensor) -> float:
    got, want = got.detach().double(), want.detach().double()
    return ((got - want).norm() / want.norm().clamp_min(1e-300)).item()


def grad_errors(got: Sequence[Tensor], want: Sequence[Tensor]) -> dict[str, float]:
    flat_got = torch.cat([g.detach().double().flatten() for g in got])
    flat_want = torch.cat([g.detach().double().flatten() for g in want])
    floor = 1e-4 * flat_want.norm()
    worst = max(
        (
            (g.detach().double() - w.detach().double()).norm()
            / torch.maximum(w.double().norm(), floor)
        ).item()
        for g, w in zip(got, want, strict=True)
    )
    return {"all": rel_err(flat_got, flat_want), "worst_tensor": worst}


def param_grads(model: nn.Module) -> list[Tensor]:
    return [
        torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
        for p in model.parameters()
        if p.requires_grad
    ]


def coupled_loss(seed: int = 0) -> Callable:
    """A loss that couples the output and its gradients, linearly and quadratically,
    with fixed random weights (the same for every call with the same shapes)."""
    cache: dict = {}

    def weights(key, like: Tensor) -> Tensor:
        if key not in cache:
            gen = torch.Generator().manual_seed(seed + len(cache))
            cache[key] = torch.randn(like.shape, generator=gen, dtype=torch.float64)
        return cache[key].to(like)

    def loss(target: Tensor, grads: dict[str, Tensor]) -> Tensor:
        total = (weights("out", target) * target).sum() + 0.5 * target.square().sum()
        for name, g in grads.items():
            total = total + (weights(name, g) * g).sum() + 0.5 * g.square().sum()
        return total

    return loss


def run_step(model, data, *, outputs, inputs, loss: Callable, exact: bool):
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    out, grads = create_graph(model, data, outputs=outputs, inputs=inputs, train=True, exact=exact)
    target = select_output(out, outputs)
    value = loss(target, grads)
    value.backward()
    result = {
        "out": target.detach(),
        "grads": {k: g.detach() for k, g in grads.items()},
        "loss": value.detach(),
        "params": param_grads(model),
    }
    for p in params:
        p.grad = None
    return result


def step_errors(got: dict, want: dict) -> dict[str, float]:
    """Relative errors between two run_step results."""
    errors = {"out": rel_err(got["out"], want["out"]), "loss": rel_err(got["loss"], want["loss"])}
    for name in got["grads"]:
        errors[f"grad_{name}"] = rel_err(got["grads"][name], want["grads"][name])
    params = grad_errors(got["params"], want["params"])
    errors["params"] = params["all"]
    errors["params_worst_tensor"] = params["worst_tensor"]
    return errors


def check_create_graph(
    model: nn.Module,
    data,
    *,
    outputs="energy",
    inputs=("pos",),
    loss: Callable | None = None,
) -> dict[str, float]:
    """
    Relative errors of create_graph against ``exact=True`` (eager create_graph)
    on the same parameters and data: the output, each input gradient, the loss
    and the parameter gradients of ``loss(output, grads)``.
    """
    loss = loss or coupled_loss()
    got = run_step(model, data, outputs=outputs, inputs=inputs, loss=loss, exact=False)
    want = run_step(model, data, outputs=outputs, inputs=inputs, loss=loss, exact=True)
    return step_errors(got, want)


def check_float32(
    model32: nn.Module,
    data32,
    model64: nn.Module,
    data64,
    *,
    outputs="energy",
    inputs=("pos",),
    loss: Callable | None = None,
) -> dict[str, dict[str, float]]:
    """
    create_graph and eager create_graph, both in float32, against eager
    create_graph in float64 on the same parameters (``model64`` holds
    ``model32``'s parameters in float64). A compiled graph rounds differently
    from eager kernels; the criterion is that create_graph is as close to the
    exact result as eager create_graph is.

    Returns ``{"create_graph": errors, "eager": errors}``.
    """
    loss = loss or coupled_loss()
    want = run_step(model64, data64, outputs=outputs, inputs=inputs, loss=loss, exact=True)
    result = {}
    for name, exact in (("create_graph", False), ("eager", True)):
        got = run_step(model32, data32, outputs=outputs, inputs=inputs, loss=loss, exact=exact)
        result[name] = step_errors(got, want)
    return result


def as_close_as_eager(result: dict, floor: dict[str, float], factor: float = 4.0) -> dict:
    """The quantities where create_graph's float32 error exceeds both ``floor``
    and ``factor`` times eager create_graph's (empty when it passes)."""
    return {
        k: (v, result["eager"][k])
        for k, v in result["create_graph"].items()
        if k in floor and v > max(floor[k], factor * result["eager"][k])
    }


def check_op(op, reference: Callable, args: Sequence, *, seed: int = 20260917) -> dict[str, float]:
    """
    forward1, backward1 and backward2 of a ``cg.custom_op`` against a
    differentiable ``reference`` with the same signature: the outputs; the input
    gradients for random cotangents of the differentiable outputs (backward1);
    and the gradients of a random linear function of those input gradients
    w.r.t. the inputs and the cotangents (backward2). Use float64 inputs drawn
    from a different random stream than ``seed`` (equal streams can make the
    random weights equal to the inputs and some gradients vanish).
    """
    gen = torch.Generator().manual_seed(seed)

    def leaves():
        return [
            a.detach().clone().requires_grad_(a.is_floating_point()) if isinstance(a, Tensor) else a
            for a in args
        ]

    def outputs_of(fn, xs):
        out = fn(*xs)
        return out if isinstance(out, tuple | list) else (out,)

    xs_ref, xs_op = leaves(), leaves()
    out_ref, out_op = outputs_of(reference, xs_ref), outputs_of(op, xs_op)
    errors = {}
    for name, a, b in zip(op.outputs, out_op, out_ref, strict=True):
        errors[f"forward1.{name}"] = rel_err(a, b)
    diff = [i for i, name in enumerate(op.outputs) if name not in op.nondiff]
    inputs_ref = [x for x in xs_ref if isinstance(x, Tensor) and x.requires_grad]
    inputs_op = [x for x in xs_op if isinstance(x, Tensor) and x.requires_grad]
    cots = [
        torch.randn(out_ref[i].shape, generator=gen, dtype=torch.float64).to(out_ref[i])
        for i in diff
    ]
    cot_ref = [c.clone().requires_grad_(True) for c in cots]
    cot_op = [c.clone().requires_grad_(True) for c in cots]
    g_ref = torch.autograd.grad([out_ref[i] for i in diff], inputs_ref, cot_ref, create_graph=True)
    g_op = torch.autograd.grad([out_op[i] for i in diff], inputs_op, cot_op, create_graph=True)
    names = [n for n, a in zip(op.params, args, strict=True) if isinstance(a, Tensor)]
    for name, a, b in zip(names, g_op, g_ref, strict=True):
        errors[f"backward1.{name}"] = rel_err(a, b)
    ws = [torch.randn(g.shape, generator=gen, dtype=torch.float64).to(g) for g in g_ref]
    s_ref = sum((w * g).sum() for w, g in zip(ws, g_ref, strict=True))
    s_op = sum((w * g).sum() for w, g in zip(ws, g_op, strict=True))
    h_ref = torch.autograd.grad(s_ref, [*inputs_ref, *cot_ref], allow_unused=True)
    h_op = torch.autograd.grad(s_op, [*inputs_op, *cot_op], allow_unused=True)
    targets = [*names, *[f"grad_{op.outputs[i]}" for i in diff]]
    for name, a, b in zip(targets, h_op, h_ref, strict=True):
        if b is None and a is None:
            continue
        a = torch.zeros_like(b) if a is None else a
        b = torch.zeros_like(a) if b is None else b
        errors[f"backward2.{name}"] = rel_err(a, b)
    return errors
