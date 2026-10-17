"""
Per-block checkpointing of a region (``CompileConfig.checkpoint``).

The region runs eagerly and every matching submodule becomes a ``BlockFn``
node of the autograd graph:

- forward: the block's compiled forward without grad; only the block's inputs
  are kept;
- in the force pass (the VJP of the region that create_graph computes) its
  backward returns the input cotangents from a ``VjpFn`` node: the block's
  compiled forward + VJP, run without keeping anything;
- in the parameter backward, ``VjpFn`` recomputes that forward + VJP with grad
  and backpropagates through it (backward2), and ``BlockFn`` recomputes the
  compiled block forward and returns its first-order gradients.

A block's forward runs four times and its VJP twice, but at most one block's
activations are alive at a time. Code between blocks (input embedding, readout)
runs eagerly, its VJP through ordinary autograd.

Python scalars passed directly to a block start as compile-time constants (a
layer index, a flag: one compiled forward and VJP per value, as if the region
were compiled whole). One whose value changes between calls of the same block
(a number of atoms) is dynamic from then on, like Dynamo's automatic dynamic
shapes. Everything inside tensors and containers stays dynamic.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from create_graph.compile_config import CompileConfig, compile_context, torch_compile
from create_graph.errors import ContractError
from create_graph.pytree import flatten, is_float, unflatten

_autograd_grad = torch.autograd.grad

FORCES_PASS: list[bool] = [False]


@contextlib.contextmanager
def forces_pass():
    FORCES_PASS.append(True)
    try:
        yield
    finally:
        FORCES_PASS.pop()


# Each compiled copy of the two functions below has its own value (a global, so
# Dynamo specializes on it): the block's direct scalar arguments.
_STATIC: tuple = ((), ())


def _is_scalar(x) -> bool:
    return isinstance(x, bool | int | float | str)


def _arguments(spec, leaves, static):
    args, kwargs = unflatten(spec, leaves)
    static_args, static_kwargs = static
    args = list(args)
    for i, _, value in static_args:
        args[i] = value
    return args, {**kwargs, **{k: value for k, _, value in static_kwargs}}


def _block_forward(block, spec, leaves):
    args, kwargs = _arguments(spec, leaves, _STATIC)
    return type(block).forward(block, *args, **kwargs)


def _block_vjp(block, spec, leaves, slots, out_slots, g_outs, create):
    with torch.enable_grad():
        args, kwargs = _arguments(spec, leaves, _STATIC)
        out_leaves = flatten(type(block).forward(block, *args, **kwargs))[0]
        pairs = [
            (out_leaves[i], g)
            for i, g in zip(out_slots, g_outs, strict=True)
            if out_leaves[i].requires_grad
        ]
        return _autograd_grad(
            [o for o, _ in pairs],
            [leaves[i] for i in slots],
            [g for _, g in pairs],
            create_graph=create,
            allow_unused=True,
        )


class BlockRunner:
    """Compiled forward and forward + VJP of the blocks of one region, a pair
    per distinct value of the blocks' direct scalar arguments."""

    def __init__(self, config: CompileConfig, clone):
        self.config = config
        self._clone = clone
        self._compiled: dict[tuple, tuple] = {}
        # blocks of different structure, grad modes and the duck-sizing
        # recompile each take a graph of the shared function
        self.recompile_limit = 8
        self._first: dict[tuple, object] = {}  # (block, argument) -> its first scalar value
        self._dynamic: set = set()  # arguments whose scalar changed between calls of a block

    def static_arguments(self, block: nn.Module, args, kwargs) -> tuple:
        """The direct scalar arguments that are compile-time constants."""
        slots = [(("arg", i), a) for i, a in enumerate(args)]
        slots += [(("kw", k), v) for k, v in kwargs.items()]
        static: list = []
        for slot, value in slots:
            if not _is_scalar(value) or slot in self._dynamic:
                continue
            first = self._first.setdefault((id(block), slot), value)
            if type(first) is not type(value) or first != value:
                self._dynamic.add(slot)
                continue
            static.append((slot, type(value), value))
        return (
            tuple((i, t, v) for (kind, i), t, v in static if kind == "arg"),
            tuple((k, t, v) for (kind, k), t, v in static if kind == "kw"),
        )

    def _functions(self, static: tuple):
        pair = self._compiled.get(static)
        if pair is None:
            constants = {"_STATIC": static}
            pair = (
                torch_compile(self._clone(_block_forward, "block_fwd", constants), self.config),
                torch_compile(self._clone(_block_vjp, "block_vjp", constants), self.config),
            )
            self._compiled[static] = pair
        return pair

    def forward(self, call: _BlockCall, inputs: list[Tensor]):
        fwd, _ = self._functions(call.static)
        with compile_context(self.config, self.recompile_limit):
            return fwd(call.block, call.spec, call.with_inputs(inputs))

    def vjp(self, call: _BlockCall, inputs: list[Tensor], g_outs: list[Tensor], create: bool):
        _, vjp = self._functions(call.static)
        with compile_context(self.config, self.recompile_limit):
            return vjp(
                call.block,
                call.spec,
                call.with_inputs(inputs),
                call.slots,
                call.out_slots,
                g_outs,
                create,
            )


class _BlockCall:
    """One call of a checkpointed block: its argument structure and outputs."""

    def __init__(self, runner: BlockRunner, block: nn.Module, args, kwargs):
        self.runner = runner
        self.block = block
        self.static = runner.static_arguments(block, args, kwargs)
        static_args = {i for i, _, _ in self.static[0]}
        static_kwargs = {k for k, _, _ in self.static[1]}
        args = tuple(None if i in static_args else a for i, a in enumerate(args))
        kwargs = {k: None if k in static_kwargs else v for k, v in kwargs.items()}
        self.leaves, self.spec = flatten((args, kwargs))
        self.slots = tuple(i for i, x in enumerate(self.leaves) if is_float(x) and x.requires_grad)
        self.params = [p for p in block.parameters() if p.requires_grad]
        self.out_leaves: list = []
        self.out_spec = None
        self.out_slots: tuple[int, ...] = ()

    @property
    def inputs(self) -> list[Tensor]:
        return [self.leaves[i] for i in self.slots]

    def with_inputs(self, tensors) -> list:
        leaves = list(self.leaves)
        for i, t in zip(self.slots, tensors, strict=True):
            leaves[i] = t
        return leaves

    def keep_output(self, out) -> tuple[Tensor, ...]:
        """The block's float outputs become the node's outputs, except constants it
        passes through (inputs that do not require grad stay as they are)."""
        self.out_leaves, self.out_spec = flatten(out)
        constants = {
            id(x)
            for i, x in enumerate(self.leaves)
            if isinstance(x, Tensor) and i not in set(self.slots)
        }
        self.out_slots = tuple(
            i for i, x in enumerate(self.out_leaves) if is_float(x) and id(x) not in constants
        )
        return tuple(self.out_leaves[i] for i in self.out_slots)

    def selected_outputs(self, out) -> list[Tensor]:
        leaves = flatten(out)[0]
        return [leaves[i] for i in self.out_slots]

    def rebuild_output(self, tensors):
        leaves = list(self.out_leaves)
        for i, t in zip(self.out_slots, tensors, strict=True):
            leaves[i] = t
        self.out_leaves = []  # the node keeps the call alive: drop the forward's outputs
        return unflatten(self.out_spec, leaves)


def _leaf(t: Tensor, requires_grad: bool | None = None) -> Tensor:
    return t.detach().requires_grad_(t.requires_grad if requires_grad is None else requires_grad)


class VjpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, call: _BlockCall, n_out: int, *tensors):
        n = len(call.slots)
        inputs, g_outs = tensors[:n], tensors[n : n + n_out]
        ctx.call, ctx.n_out = call, n_out
        ctx.save_for_backward(*inputs, *g_outs)
        grads = call.runner.vjp(
            call, [_leaf(t, True) for t in inputs], [g.detach() for g in g_outs], False
        )
        return tuple(
            torch.zeros_like(t) if g is None else g for g, t in zip(grads, inputs, strict=True)
        )

    @staticmethod
    def backward(ctx, *ggs):
        call: _BlockCall = ctx.call
        saved = ctx.saved_tensors
        n = len(call.slots)
        inputs, g_outs = saved[:n], saved[n:]
        with torch.enable_grad():
            li = [_leaf(t, True) for t in inputs]
            lg = [_leaf(g, True) for g in g_outs]
            outs = call.runner.vjp(call, li, lg, True)
            pairs = [
                (o, gg)
                for o, gg in zip(outs, ggs, strict=True)
                if o is not None and gg is not None and o.requires_grad
            ]
            wanted = [*li, *lg, *call.params]
            if pairs:
                grads = _autograd_grad(
                    [o for o, _ in pairs], wanted, [gg for _, gg in pairs], allow_unused=True
                )
            else:
                grads = [None] * len(wanted)
        return (None, None, *grads)


class BlockFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, call: _BlockCall, *tensors):
        n = len(call.slots)
        inputs = tensors[:n]
        ctx.call = call
        ctx.save_for_backward(*inputs)
        out = call.runner.forward(call, [t.detach() for t in inputs])
        return call.keep_output(out)

    @staticmethod
    def backward(ctx, *g_outs):
        call: _BlockCall = ctx.call
        inputs = ctx.saved_tensors
        n_params = len(call.params)
        if FORCES_PASS[-1]:
            grads_in = VjpFn.apply(call, len(g_outs), *inputs, *g_outs, *call.params)
            return (None, *grads_in, *([None] * n_params))
        with torch.enable_grad():
            leaves = [_leaf(t) for t in inputs]
            outs = call.selected_outputs(call.runner.forward(call, leaves))
            pairs = [
                (o, g)
                for o, g in zip(outs, g_outs, strict=True)
                if g is not None and o.requires_grad
            ]
            every = [*leaves, *call.params]
            wanted = [t for t in every if t.requires_grad]
            found = iter(
                _autograd_grad(
                    [o for o, _ in pairs], wanted, [g for _, g in pairs], allow_unused=True
                )
                if pairs and wanted
                else [None] * len(wanted)
            )
            grads = [next(found) if t.requires_grad else None for t in every]
        return (None, *grads)


def _wrapped_forward(block: nn.Module, runner: BlockRunner):
    def forward(*args, **kwargs):
        call = _BlockCall(runner, block, args, kwargs)
        outs = BlockFn.apply(call, *call.inputs, *call.params)
        return call.rebuild_output(outs)

    return forward


def matching_blocks(roots: Iterable[nn.Module], pattern: str) -> list[nn.Module]:
    found: list[nn.Module] = []
    seen: set[int] = set()
    for root in roots:
        for name, module in root.named_modules():
            if id(module) in seen or not re.fullmatch(pattern, name):
                continue
            # a match inside another match is part of that block
            if any(module in set(outer.modules()) for outer in found):
                continue
            seen.add(id(module))
            found.append(module)
    return found


@contextlib.contextmanager
def checkpointed_blocks(roots: list[nn.Module], pattern: str, runner: BlockRunner):
    blocks = matching_blocks(roots, pattern)
    if not blocks:
        names = ", ".join(type(r).__name__ for r in roots) or "no nn.Module"
        raise ContractError(
            f"create_graph: CompileConfig(checkpoint={pattern!r}) matched no submodule of the "
            f"region's modules ({names}). The pattern is matched with re.fullmatch against "
            "names from named_modules() of the region's self and module arguments."
        )
    runner.recompile_limit = max(runner.recompile_limit, 8 * len(blocks))
    for block in blocks:
        block.forward = _wrapped_forward(block, runner)
    try:
        yield blocks
    finally:
        for block in blocks:
            del block.forward
