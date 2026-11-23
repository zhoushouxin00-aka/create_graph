"""
Compiled energy regions that receive the original differentiated inputs.

Inside ``cg.create_graph``, a region evaluates the energy and its input
VJP in one forward graph. Geometry and other differentiable preprocessing
belong inside that region. AOTAutograd derives the parameter backward.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import os
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
    compiled_graph_count,
    is_python_function_node,
    mark_region_inputs,
    observe_step_memory,
    torch_compile,
)
from create_graph.errors import ContractError
from create_graph.pytree import flatten, is_float, tree_map_tensors, unflatten

_autograd_grad = torch.autograd.grad

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
    """Energy and gradients with respect to the named API inputs."""

    name: str
    selected: Tensor
    grads: dict[str, Tensor | None]


@dataclasses.dataclass
class State:
    leaves: dict[str, Tensor]
    train: bool
    exact: bool
    outputs: Any
    region: RegionResult | None = None

    def leaf_index(self) -> dict[int, int]:
        return {id(t): i for i, t in enumerate(self.leaves.values())}


STATES: list[State] = []


def active_state() -> State | None:
    return STATES[-1] if STATES else None


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


def _note_step_memory() -> None:
    """Floor at entry is what the previous step left on the GPU.

    The compiled tape of this step shows up in the peak, not in the floor.
    Rank 0 prints both every 100 steps.
    """
    if not torch.cuda.is_available():
        return
    step = _MEM_STEPS[0]
    _MEM_STEPS[0] = step + 1
    alloc = torch.cuda.memory_allocated()
    peak = max(_MEM_PEAK[0], torch.cuda.max_memory_allocated())
    _MEM_PEAK[0] = peak
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    observe_step_memory(step, rank)
    if step % 100 != 0 or rank != "0":
        return
    print(
        "create_graph memory "
        f"step={step} allocated={alloc / 2**30:.2f}GiB "
        f"peak={peak / 2**30:.2f}GiB "
        f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB "
        f"graphs={compiled_graph_count()}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    _MEM_PEAK[0] = alloc


_MEM_STEPS = [0]
_MEM_PEAK = [0]


def _region_body(fn, args, kwargs, wrt, outputs, create):
    with torch.enable_grad():
        out = fn(*args, **kwargs)
        target = select_output(out, outputs)
        grads = _autograd_grad(target.sum(), wrt, create_graph=create, allow_unused=True)
    if not create:
        out = tree_map_tensors(lambda t: t.detach(), out)
    return out, grads


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
        if self.config.dynamic:
            mark_region_inputs(args, kwargs)
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


def _tensor_attributes(leaves: list) -> list[tuple[object, str, Tensor]]:
    """Tensor fields of plain batch objects that are opaque to our pytree."""
    found = []
    for obj in leaves:
        if obj is None or isinstance(obj, Tensor | nn.Module | int | float | bool | str):
            continue
        holder = getattr(obj, "__dict__", None)
        if isinstance(holder, dict):
            found.extend((obj, name, value) for name, value in holder.items() if is_float(value))
    return found


def region_call(state: State, region: Region, args, kwargs):
    leaves, spec = flatten((args, kwargs))
    attributes = _tensor_attributes(leaves)
    tensors = [x for x in leaves if isinstance(x, Tensor)]
    tensors.extend(value for _, _, value in attributes)
    input_ids = {id(value) for value in state.leaves.values()}
    present = input_ids.intersection(id(value) for value in tensors)
    external_graph = any(value.requires_grad and value.grad_fn is not None for value in tensors)
    if external_graph:
        raise ContractError(
            "create_graph: differentiable preprocessing outside a cg.compile region is not "
            "supported. Pass the original differentiated inputs directly to the region and "
            "move their geometry and other differentiable preprocessing inside it."
        )
    if not present:
        return region.normal(*args, **kwargs)
    if present != input_ids:
        missing = [name for name, value in state.leaves.items() if id(value) not in present]
        raise ContractError(
            "create_graph: the energy region must receive every differentiated input "
            f"directly; missing {missing}. Move the complete energy computation into one "
            "cg.compile region."
        )
    if state.region is not None:
        raise ContractError(
            f"create_graph: regions {state.region.name} and {region.name} both receive the "
            "differentiated inputs. Exactly one region may produce the differentiated output: "
            "merge them into one cg.compile region."
        )

    # The API inputs carry hooks that reject force computation by the model.
    # The region differentiates its own leaves and returns their gradients
    # directly, in API input order; no geometry graph is traversed here.
    replacements = {
        id(value): value.detach().requires_grad_(True) for value in state.leaves.values()
    }
    wrt = tuple(replacements[id(value)] for value in state.leaves.values())
    new_args, new_kwargs = unflatten(spec, [replacements.get(id(x), x) for x in leaves])
    swaps = [
        (obj, name, value, replacements[id(value)])
        for obj, name, value in attributes
        if id(value) in replacements
    ]
    outputs = region.outputs if region.outputs is not None else state.outputs
    for obj, name, _, new in swaps:
        setattr(obj, name, new)
    try:
        _check_functions(region, new_args, new_kwargs, list(wrt))
        _note_step_memory()
        if region.config.dynamic:
            mark_region_inputs(new_args, new_kwargs, wrt)
        if region.config.checkpoint is not None:
            out, grads = _checkpointed(state, region, new_args, new_kwargs, leaves, wrt, outputs)
        else:
            body = _compiled("train" if state.train else "eval", region)
            with compile_context(region.config, joint=True):
                out, grads = body(region.fn, new_args, new_kwargs, wrt, outputs, state.train)
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
        grads=dict(zip(state.leaves, grads, strict=True)),
    )
    return out


def _checkpointed(state, region, args, kwargs, leaves, wrt, outputs):
    runner = _compiled("blocks", region)
    roots = _modules_of(region, leaves)
    with checkpointed_blocks(roots, region.config.checkpoint, runner), torch.enable_grad():
        out = region.fn(*args, **kwargs)
    target = select_output(out, outputs)
    with forces_pass():
        grads = _autograd_grad(target.sum(), wrt, create_graph=state.train, allow_unused=True)
    if not state.train:
        out = tree_map_tensors(lambda t: t.detach(), out)
    return out, grads
