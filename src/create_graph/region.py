"""
``cg.compile`` regions and what they do inside ``cg.create_graph``.

Outside create_graph a region is ``torch.compile(fn)``. Inside, the region
that depends on the differentiated inputs runs ``fn`` and
``torch.autograd.grad(sum(selected output), region inputs, create_graph=True)``
as ONE compiled graph: Dynamo traces the autograd.grad call, AOTAutograd turns
autograd's backward formulas into ordinary ATen ops of the forward graph
(backward1) and derives the backward of the whole graph (backward2). The
gradients w.r.t. the region inputs are chained to the differentiated inputs
through the eager code before the region with autograd (create_graph.api).

Region inputs, by what their autograd graph reaches:

- only the differentiated inputs (geometry): enter as detached leaves, so the
  parameter backward does not run through the geometry a second time;
- parameters too (eager code mixing both): enter as they are; their
  cotangents are chained as well and parameter gradients keep flowing;
- parameters only: enter as they are and are listed in the in-graph
  autograd.grad (Dynamo refuses to traverse grad_fns from outside the graph),
  their gradients are dropped (and dead-code eliminated).
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import types
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from create_graph.checkpoint import BlockRunner, checkpointed_blocks, forces_pass
from create_graph.compile_config import (
    DEFAULT_CONFIG,
    CompileConfig,
    compile_context,
    is_python_function_node,
    torch_compile,
)
from create_graph.errors import ContractError
from create_graph.pytree import flatten, is_float, tree_map_tensors, unflatten

_autograd_grad = torch.autograd.grad

# leaf kinds an autograd graph reaches
INPUT, OTHER = 1, 2


# ---------------------------------------------------------------------------
# output selection
# ---------------------------------------------------------------------------


def select_output(out, outputs):
    """The differentiated tensor in ``out``: ``outputs`` is a key / attribute name,
    a callable, or None (``out`` is the tensor). A tensor ``out`` is taken as is."""
    if isinstance(out, Tensor) or outputs is None:
        return out
    if callable(outputs):
        return outputs(out)
    if isinstance(out, dict):
        return out[outputs]
    return getattr(out, outputs)


# ---------------------------------------------------------------------------
# state of one create_graph call
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RegionResult:
    """What the energy region produced: its selected output and, per
    differentiated region input, the gradient to chain."""

    name: str
    selected: Tensor
    # ("chain", region input, grad): chained through the eager code before the region
    # ("direct", index of the differentiated input, grad): the input itself
    items: list[tuple[str, Any, Tensor | None]]

    def input_grads(self, leaves: list[Tensor], train: bool) -> list[Tensor | None]:
        grads: list[Tensor | None] = [None] * len(leaves)
        chain = [(x, g) for kind, x, g in self.items if kind == "chain" and g is not None]
        if chain:
            grads = list(
                _autograd_grad(
                    [x for x, _ in chain],
                    leaves,
                    [g for _, g in chain],
                    create_graph=train,
                    allow_unused=True,
                )
            )
        for kind, index, g in self.items:
            if kind == "direct" and g is not None:
                grads[index] = g if grads[index] is None else grads[index] + g
        return grads


@dataclasses.dataclass
class State:
    leaves: dict[str, Tensor]
    train: bool
    exact: bool
    outputs: Any
    memo: dict = dataclasses.field(default_factory=dict)
    region: RegionResult | None = None

    def leaf_index(self) -> dict[int, int]:
        return {id(t): i for i, t in enumerate(self.leaves.values())}


STATES: list[State] = []


def active_state() -> State | None:
    return STATES[-1] if STATES else None


def _node_kinds(node, state: State) -> int:
    memo, inputs = state.memo, state.leaf_index()
    if node in memo:
        return memo[node]
    stack = [(node, False)]
    while stack:
        current, done = stack.pop()
        if current in memo:
            continue
        children = [c for c, _ in current.next_functions if c is not None]
        if not done:
            stack.append((current, True))
            stack.extend((c, False) for c in children if c not in memo)
            continue
        kinds = 0
        variable = getattr(current, "variable", None)
        if variable is not None:
            kinds = INPUT if id(variable) in inputs else OTHER
        for child in children:
            kinds |= memo[child]
        memo[current] = kinds
    return memo[node]


def tensor_kinds(t: Tensor, state: State) -> int:
    if not (is_float(t) and t.requires_grad):
        return 0
    if t.grad_fn is None:
        return INPUT if id(t) in state.leaf_index() else OTHER
    return _node_kinds(t.grad_fn, state)


# ---------------------------------------------------------------------------
# compiled functions, cached per region
# ---------------------------------------------------------------------------

_CACHE: dict[tuple, Any] = {}
_CLONES = itertools.count()


def clone_function(
    fn: types.FunctionType, kind: str, constants: dict[str, Any] | None = None
) -> types.FunctionType:
    """A copy of ``fn`` with its own code object, so Dynamo caches (and counts
    recompiles of) every region separately. ``constants`` are extra globals of
    the copy: Dynamo specializes on values read from globals."""
    name = f"_create_graph_{kind}_{next(_CLONES)}"
    code = fn.__code__.replace(co_name=name, co_qualname=name)
    scope = fn.__globals__ if not constants else {**fn.__globals__, **constants}
    new = types.FunctionType(code, scope, name, fn.__defaults__, fn.__closure__)
    new.__kwdefaults__ = fn.__kwdefaults__
    return new


def fn_key(fn) -> Any:
    if isinstance(fn, types.FunctionType):
        return fn.__code__
    if isinstance(fn, types.MethodType):
        return (id(fn.__self__), fn.__func__.__code__)
    return id(fn)


def _call(fn, args, kwargs):
    return fn(*args, **kwargs)


def _region_body(fn, args, kwargs, wrt, keep, outputs, create):
    with torch.enable_grad():
        out = fn(*args, **kwargs)
        target = select_output(out, outputs)
        grads = _autograd_grad(target.sum(), wrt, create_graph=create, allow_unused=True)
    if not create:
        out = tree_map_tensors(lambda t: t.detach(), out)
    return out, tuple(g for g, k in zip(grads, keep, strict=True) if k)


def _compiled(kind: str, region: Region):
    key = (kind, fn_key(region.fn), region.config)
    entry = _CACHE.get(key)
    if entry is None:
        if kind == "normal":
            entry = torch_compile(clone_function(_call, kind), region.config, fullgraph=False)
        elif kind == "blocks":
            entry = BlockRunner(region.config, clone_function)
        else:
            entry = torch_compile(clone_function(_region_body, kind), region.config)
        _CACHE[key] = entry
    return entry


def reset_regions() -> None:
    _CACHE.clear()
    _CHECKED.clear()


_CHECKED: set = set()


def _check_functions(region: Region, args, kwargs, wrt: list[Tensor]) -> None:
    """
    Once per region and config: run the region eagerly and refuse custom
    torch.autograd.Functions on the path from the differentiated inputs to the
    output. A compiled graph does not differentiate a traced Function's backward
    again, so the VJP through it would be a constant for the parameter backward
    (wrong parameter gradients, no error). Functions of parameters only are
    fine: their backward is not part of the VJP.
    """
    key = (fn_key(region.fn), region.config)
    if region.config.backend == "eager" or key in _CHECKED:
        return
    _CHECKED.add(key)
    with torch.enable_grad():
        out = region.fn(*args, **kwargs)
    region_inputs = {id(t) for t in wrt}
    memo: dict = {}

    def reaches_inputs(node) -> bool:
        stack = [(node, False)]
        while stack:
            current, done = stack.pop()
            if current in memo:
                continue
            children = [c for c, _ in current.next_functions if c is not None]
            if not done:
                stack.append((current, True))
                stack.extend((c, False) for c in children if c not in memo)
                continue
            variable = getattr(current, "variable", None)
            own = variable is not None and id(variable) in region_inputs
            memo[current] = own or any(memo[c] for c in children)
        return memo[node]

    stack = [t.grad_fn for t in flatten(out)[0] if isinstance(t, Tensor) and t.grad_fn is not None]
    visited: set = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        children = [c for c, _ in node.next_functions if c is not None]
        if is_python_function_node(node) and any(reaches_inputs(c) for c in children):
            raise ContractError(
                f"create_graph: region {region.name} applies the torch.autograd.Function "
                f"{type(node).__name__.removesuffix('Backward')} on the path from the "
                "differentiated inputs to its output. A compiled graph does not differentiate "
                "a traced Function's backward again (wrong parameter gradients). Register the "
                "kernel with cg.custom_op (forward1, backward1, backward2) or "
                "torch.library.custom_op + register_autograd, or use "
                "CompileConfig(backend='eager') for this region."
            )
        stack.extend(children)


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------


class Region:
    """``cg.compile(fn)``: ``torch.compile(fn)`` outside create_graph, the
    region's forward + VJP graph inside it."""

    def __init__(self, fn: Callable, config: CompileConfig | None = None, outputs=None):
        self.fn = fn
        self.config = config if config is not None else DEFAULT_CONFIG
        self.outputs = outputs
        functools.update_wrapper(self, fn, updated=())

    @property
    def name(self) -> str:
        return getattr(self.fn, "__qualname__", None) or type(self.fn).__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return Region(types.MethodType(self.fn, obj), self.config, self.outputs)

    def normal(self, *args, **kwargs):
        if self.config.backend == "eager":
            return self.fn(*args, **kwargs)
        compiled = _compiled("normal", self)
        with compile_context(self.config):
            return compiled(self.fn, args, kwargs)

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self.fn(*args, **kwargs)
        state = active_state()
        if state is None:
            return self.normal(*args, **kwargs)
        if state.exact:
            return self.fn(*args, **kwargs)
        return region_call(state, self, args, kwargs)


class RegionModule(nn.Module):
    """``cg.compile(module)``: the whole module is one region."""

    def __init__(self, module: nn.Module, config: CompileConfig | None = None, outputs=None):
        super().__init__()
        self.module = module
        self._region = Region(module.forward, config, outputs)

    def forward(self, *args, **kwargs):
        return self._region(*args, **kwargs)


def _modules_of(region: Region, leaves: list) -> list[nn.Module]:
    roots = []
    owner = getattr(region.fn, "__self__", None)
    if isinstance(owner, nn.Module):
        roots.append(owner)
    roots.extend(x for x in leaves if isinstance(x, nn.Module))
    return roots


def _plain_attributes(state: State, leaves: list) -> list[tuple[object, str, Tensor, int]]:
    """(object, name, tensor, kinds) for float tensor attributes of plain objects
    (fairchem's AtomicData, ...) among the region inputs that require grad."""
    found = []
    for obj in leaves:
        if obj is None or isinstance(obj, Tensor | nn.Module | int | float | bool | str):
            continue
        holder = getattr(obj, "__dict__", None)
        if not isinstance(holder, dict):
            continue
        for name, value in list(holder.items()):
            if isinstance(value, Tensor):
                kinds = tensor_kinds(value, state)
                if kinds:
                    found.append((obj, name, value, kinds))
    return found


def region_call(state: State, region: Region, args, kwargs):
    leaves, spec = flatten((args, kwargs))
    kinds = [tensor_kinds(x, state) if isinstance(x, Tensor) else 0 for x in leaves]
    attributes = _plain_attributes(state, leaves)
    if not any(k & INPUT for k in kinds) and not any(k & INPUT for *_, k in attributes):
        # does not depend on the differentiated inputs: an ordinary compiled call
        return region.normal(*args, **kwargs)
    if state.region is not None:
        raise ContractError(
            f"create_graph: regions {state.region.name} and {region.name} both depend on the "
            "differentiated inputs. Exactly one region may produce the differentiated output: "
            "merge them into one cg.compile region (e.g. backbone network and energy head)."
        )

    inputs = state.leaf_index()
    wrt: list[Tensor] = []
    keep: list[bool] = []
    items: list[tuple[str, Any]] = []
    replaced: dict[int, Tensor] = {}

    def enter(x: Tensor, k: int) -> Tensor:
        if id(x) in replaced:
            return replaced[id(x)]
        if k == INPUT or id(x) in inputs:
            # geometry (or the input itself): a detached leaf, chained afterwards
            new = x.detach().requires_grad_(True)
            wrt.append(new)
            keep.append(True)
            items.append(("direct", inputs[id(x)]) if id(x) in inputs else ("chain", x))
        else:
            new = x
            wrt.append(x)
            keep.append(bool(k & INPUT))
            if k & INPUT:
                items.append(("chain", x))
        replaced[id(x)] = new
        return new

    new_leaves = [enter(x, k) if k else x for x, k in zip(leaves, kinds, strict=True)]
    swaps = [(obj, name, value, enter(value, k)) for obj, name, value, k in attributes]
    new_args, new_kwargs = unflatten(spec, new_leaves)
    outputs = region.outputs if region.outputs is not None else state.outputs

    for obj, name, _, new in swaps:
        setattr(obj, name, new)
    try:
        _check_functions(region, new_args, new_kwargs, wrt)
        if region.config.checkpoint is not None:
            out, grads = _checkpointed(
                state, region, new_args, new_kwargs, new_leaves, wrt, keep, outputs
            )
        else:
            body = _compiled("train" if state.train else "eval", region)
            with compile_context(region.config):
                out, grads = body(
                    region.fn, new_args, new_kwargs, tuple(wrt), tuple(keep), outputs, state.train
                )
    finally:
        for obj, name, original, _ in swaps:
            setattr(obj, name, original)

    selected = select_output(out, outputs)
    if not is_float(selected):
        raise ContractError(
            f"create_graph: region {region.name} must return the differentiated output as a "
            f"float tensor (selected with outputs={outputs!r}), got {type(selected).__name__}."
        )
    state.region = RegionResult(
        name=region.name,
        selected=selected,
        items=[(kind, ref, g) for (kind, ref), g in zip(items, grads, strict=True)],
    )
    return out


def _checkpointed(state, region, args, kwargs, leaves, wrt, keep, outputs):
    runner = _compiled("blocks", region)
    roots = _modules_of(region, leaves)
    with checkpointed_blocks(roots, region.config.checkpoint, runner), torch.enable_grad():
        out = region.fn(*args, **kwargs)
    target = select_output(out, outputs)
    with forces_pass():
        grads = _autograd_grad(target.sum(), wrt, create_graph=state.train, allow_unused=True)
    if not state.train:
        out = tree_map_tensors(lambda t: t.detach(), out)
    return out, tuple(g for g, k in zip(grads, keep, strict=True) if k)
