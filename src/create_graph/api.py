"""
The entry point:

    out, grads = cg.create_graph(model, data, outputs="energy", inputs=("pos",))
    loss = loss_fn(out, -grads["pos"])
    loss.backward()

``data.<input>`` (or ``data[input]``) is replaced by a fresh leaf, ``model(data)``
runs once, and ``grads[input]`` is the gradient of ``sum(out[outputs])``
w.r.t. it, differentiable w.r.t. the parameters (create_graph semantics).
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from create_graph.compile_config import (
    CompileConfig,
    reset_compiled,
    supports_in_graph_grad,
    warn_eager_fallback,
)
from create_graph.errors import ContractError
from create_graph.region import (
    STATES,
    Region,
    RegionModule,
    State,
    reset_regions,
    select_output,
)

_autograd_grad = torch.autograd.grad


def get_input(data, key: str) -> Tensor:
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


@contextlib.contextmanager
def swapped_inputs(data, values: dict[str, Tensor]):
    originals = {key: get_input(data, key) for key in values}
    for key, value in values.items():
        if isinstance(data, dict):
            data[key] = value
        else:
            setattr(data, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            if isinstance(data, dict):
                data[key] = value
            else:
                setattr(data, key, value)


def _forbidden_grad(grad):
    raise ContractError(
        "create_graph: the model computed a gradient w.r.t. a differentiated input during "
        "create_graph (torch.autograd.grad or backward). The model must only compute the "
        "energy here (switch off the force computation of its heads); create_graph computes "
        "the gradients."
    )


@contextlib.contextmanager
def _no_input_gradients(leaves: dict[str, Tensor]):
    """While the model runs, nothing may differentiate w.r.t. the inputs: the
    regions differentiate detached copies, the chain runs after the model."""
    handles = [leaf.register_hook(_forbidden_grad) for leaf in leaves.values()]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _same_entries(target: Tensor, selected: Tensor) -> bool:
    """A view (reshape, squeeze, full slice, ...) with the same elements."""
    if target is selected:
        return True
    return (
        target.numel() == selected.numel()
        and target.data_ptr() == selected.data_ptr()
        and target.untyped_storage().data_ptr() == selected.untyped_storage().data_ptr()
    )


def _reaches_inputs_besides(target: Tensor, selected: Tensor, state: State) -> bool:
    """Whether target depends on the differentiated inputs other than through selected."""
    inputs = state.leaf_index()
    stop = selected.grad_fn
    stack = [target.grad_fn] if target.grad_fn is not None else []
    seen = set()
    while stack:
        node = stack.pop()
        if node is None or node is stop or node in seen:
            continue
        seen.add(node)
        variable = getattr(node, "variable", None)
        if variable is not None and id(variable) in inputs:
            return True
        stack.extend(c for c, _ in node.next_functions)
    return False


def _check_output(target: Tensor, state: State) -> None:
    """
    The region's VJP is that of sum(selected): the model output must have the
    same entries (views, reshapes), or differ from selected by terms that do
    not depend on the differentiated inputs and add selected with coefficient 1.
    """
    selected = state.region.selected
    if _same_entries(target, selected):
        return
    message = (
        f"create_graph: the differentiated model output is not the output of region "
        f"{state.region.name} (or a view with the same entries)"
    )
    if not (target.requires_grad and selected.requires_grad):
        raise ContractError(message + ".")
    (weight,) = _autograd_grad(target.sum(), selected, retain_graph=True, allow_unused=True)
    if weight is None:
        raise ContractError(message + ", and it is not computed from it.")
    if _reaches_inputs_besides(target, selected, state):
        raise ContractError(
            message + ": it also depends on the differentiated inputs through code after the "
            "region. Move that code into the region."
        )
    if not bool(torch.all(weight == 1)):
        raise ContractError(
            message + ": code after the region scales or drops entries of the region's output. "
            "Move that code into the region, or let the region return what the model returns."
        )


_NO_REGION_WARNED: set[int] = set()


def create_graph(
    model: Callable,
    data,
    *,
    outputs: str | Callable | None = "energy",
    inputs: str | Sequence[str] = ("pos",),
    train: bool | None = None,
    exact: bool = False,
):
    """
    ``(model(data), grads)``: the model output and ``grads[name]``, the gradient
    of ``sum(out[outputs])`` w.r.t. ``data.<name>`` (``data[name]`` for dicts)
    for every name in ``inputs``.

    The region (``cg.compile``) that produces the differentiated output computes
    it together with its VJP in one compiled graph; the eager code before it is
    chained with autograd. In training (``train=True``, default when the model
    is in training mode and grad is enabled) the grads are differentiable w.r.t.
    the parameters, so ``loss(out, grads).backward()`` is the parameter gradient
    of the conservative-force loss. ``train=False`` gives first-order grads and
    detached outputs (validation, MD).

    ``outputs``: key or attribute of the output (or of the region's output), a
    callable selecting the tensor, or None for a tensor output. ``exact=True``
    runs everything eagerly with ``torch.autograd.grad(create_graph=True)``: the
    reference for tests. Under DDP pass the DDP-wrapped model.
    """
    names = (inputs,) if isinstance(inputs, str) else tuple(inputs)
    if train is None:
        train = torch.is_grad_enabled() and bool(getattr(model, "training", True))
    if not exact and not supports_in_graph_grad():
        warn_eager_fallback("create_graph")
        exact = True
    leaves = {name: get_input(data, name).detach().requires_grad_(True) for name in names}
    state = State(leaves=leaves, train=train, exact=exact, outputs=outputs)
    STATES.append(state)
    try:
        with swapped_inputs(data, leaves), torch.enable_grad(), _no_input_gradients(leaves):
            out = model(data)
    except RuntimeError as error:
        # the region's output is computed from detached copies of the inputs, so a
        # model that differentiates it w.r.t. an input gets autograd's "not used"
        if "not have been used in the graph" in str(error):
            try:
                _forbidden_grad(None)
            except ContractError as contract:
                raise contract from error
        raise
    finally:
        STATES.pop()
    target = select_output(out, outputs)
    leaf_list = list(leaves.values())
    if state.exact or state.region is None:
        if not exact and id(model) not in _NO_REGION_WARNED:
            _NO_REGION_WARNED.add(id(model))
            warnings.warn(
                "create_graph: the model ran no cg.compile region that depends on "
                f"{', '.join(names)}, so the gradients come from eager create_graph. Mark the "
                "part of the model that computes the energy with cg.compile.",
                stacklevel=2,
            )
        found = _autograd_grad(target.sum(), leaf_list, create_graph=train, allow_unused=True)
        if not train:
            out = _detached(out)
    else:
        _check_output(target, state)
        found = state.region.input_grads(leaf_list, train)
    grads = {
        name: torch.zeros_like(leaf) if g is None else g
        for (name, leaf), g in zip(leaves.items(), found, strict=True)
    }
    return out, grads


def _detached(out):
    from create_graph.pytree import tree_map_tensors

    return tree_map_tensors(lambda t: t.detach(), out)


def compile(
    fn: Callable | nn.Module | None = None,
    *,
    config: CompileConfig | None = None,
    outputs: str | Callable | None = None,
):
    """
    Mark ``fn`` (a function, method or nn.Module) as a create_graph region; a
    drop-in for ``torch.compile`` (also as a decorator).

    Outside ``cg.create_graph`` it is ``torch.compile(fn)``. Inside, if its inputs
    depend on the differentiated inputs, it runs ``fn`` and the VJP of its
    differentiated output (``outputs``, default: the create_graph call's) as one
    graph compiled as ``config`` says.
    """
    if fn is None:
        return lambda f: compile(f, config=config, outputs=outputs)
    if isinstance(fn, nn.Module):
        return RegionModule(fn, config, outputs)
    return Region(fn, config, outputs)


def reset() -> None:
    """Forget compiled regions (and Dynamo's caches); the next call compiles again."""
    reset_regions()
    reset_compiled()
