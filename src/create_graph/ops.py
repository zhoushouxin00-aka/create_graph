"""
Opaque fused operators on the path from positions to the energy.

create_graph differentiates such an op twice: its VJP is part of the generated
forward graph (backward1), and the parameter backward differentiates that VJP
(backward2). ``cg.custom_op`` registers the three kernels in one class:

    @cg.custom_op("mylib::attn")
    class Attention:
        outputs = ("out", "lse")
        save = ("q", "k", "v", "lse")      # what forward1 keeps for backward1
        nondiff = ("lse",)                 # outputs that are statistics

        @staticmethod
        def forward1(q: Tensor, k: Tensor, v: Tensor, scale: float) -> tuple[Tensor, Tensor]:
            ...
        @staticmethod
        def backward1(q, k, v, lse, grad_out, scale):
            ...  # -> gradients of q, k, v (one per tensor input of forward1)
        @staticmethod
        def backward2(q, k, v, lse, grad_out, scale, gg_q, gg_k, gg_v):
            ...  # -> gradients of backward1's tensor inputs except the statistics:
                 #    q, k, v, grad_out
        @staticmethod
        def fake(q, k, v, scale):
            ...  # output shapes of forward1

backward1 takes the saved tensors, the cotangents of the differentiable outputs
(``grad_<name>``) and the non-tensor arguments of forward1; backward2 takes
backward1's arguments followed by the cotangents of backward1's outputs. A
statistic (``lse``) gets no gradient: backward2 must account for its dependence
on the inputs analytically. Under the hood these are three
``torch.library.custom_op`` operators and two ``register_autograd`` calls, so
plain torch.library registrations work just as well; ``cg.testing.check_op``
validates all three kernels against a differentiable reference.
"""

from __future__ import annotations

import inspect
import typing

import torch
from torch import Tensor

from create_graph.errors import ContractError

_SCHEMA_TYPES = {float: "float", int: "SymInt", bool: "bool", str: "str"}


def _schema_type(hint, name: str) -> str:
    if hint in _SCHEMA_TYPES:
        return _SCHEMA_TYPES[hint]
    raise ContractError(
        f"cg.custom_op: argument {name!r} of forward1 has type {hint!r}; non-tensor "
        f"arguments must be one of {[t.__name__ for t in _SCHEMA_TYPES]}"
    )


def _as_tuple(out) -> tuple:
    return out if isinstance(out, tuple | list) else (out,)


def _fresh(outs: tuple, args: tuple) -> tuple:
    """Gradients as the generated fakes describe them: new, contiguous tensors
    (custom ops must not return their inputs)."""
    ids = {id(a) for a in args if isinstance(a, Tensor)}
    return tuple(
        (o.clone() if id(o) in ids else o).contiguous() if isinstance(o, Tensor) else o
        for o in outs
    )


def _empty(like: Tensor) -> Tensor:
    return torch.empty(like.shape, dtype=like.dtype, device=like.device)


def _returns(n: int) -> str:
    return "Tensor" if n == 1 else "(" + ", ".join(["Tensor"] * n) + ")"


class FusedOp:
    """The three registered operators of a ``cg.custom_op`` class; calling it calls forward1."""

    def __init__(self, name: str, spec: type):
        forward1 = spec.forward1
        hints = typing.get_type_hints(forward1)
        params = list(inspect.signature(forward1).parameters)
        self.name = name
        self.params = params
        self.tensor_args = [p for p in params if hints.get(p) is Tensor]
        self.other_args = [p for p in params if p not in self.tensor_args]
        self.outputs = tuple(spec.outputs)
        self.save = tuple(spec.save)
        self.nondiff = tuple(getattr(spec, "nondiff", ()))
        self.diff_outputs = [o for o in self.outputs if o not in self.nondiff]
        known = set(params) | set(self.outputs)
        for n in (*self.save, *self.nondiff):
            if n not in known:
                raise ContractError(f"cg.custom_op {name}: {n!r} is not an argument or output")

        self.op = torch.library.custom_op(name, mutates_args=())(forward1)
        if hasattr(spec, "fake"):
            self.op.register_fake(spec.fake)

        other_schema = [f"{_schema_type(hints[p], p)} {p}" for p in self.other_args]
        grad_names = [f"grad_{o}" for o in self.diff_outputs]
        self.bwd1_tensors = [*self.save, *grad_names]
        self.bwd1_params = [*self.bwd1_tensors, *self.other_args]
        bwd1_schema = (
            "("
            + ", ".join([f"Tensor {n}" for n in self.bwd1_tensors] + other_schema)
            + f") -> {_returns(len(self.tensor_args))}"
        )
        self.bwd1_diff = [n for n in self.bwd1_tensors if n not in self.nondiff]
        gg_names = [f"gg_{p}" for p in self.tensor_args]
        bwd2_schema = (
            "("
            + ", ".join(
                [f"Tensor {n}" for n in self.bwd1_tensors]
                + other_schema
                + [f"Tensor {n}" for n in gg_names]
            )
            + f") -> {_returns(len(self.bwd1_diff))}"
        )

        n1, n2 = len(self.tensor_args), len(self.bwd1_diff)

        def backward1_impl(*args):
            outs = _fresh(_as_tuple(spec.backward1(*args)), args)
            return outs if n1 > 1 else outs[0]

        def backward2_impl(*args):
            outs = _fresh(_as_tuple(spec.backward2(*args)), args)
            return outs if n2 > 1 else outs[0]

        self.backward1_op = torch.library.custom_op(
            f"{name}_backward1", backward1_impl, mutates_args=(), schema=bwd1_schema
        )
        self.backward2_op = torch.library.custom_op(
            f"{name}_backward2", backward2_impl, mutates_args=(), schema=bwd2_schema
        )
        self._register_fakes(spec, n1, n2)
        self._register_autograd(n1, n2)

    def _register_fakes(self, spec, n1: int, n2: int) -> None:
        bwd1_tensors = self.bwd1_tensors
        if hasattr(spec, "backward1_fake"):
            self.backward1_op.register_fake(spec.backward1_fake)
        elif all(p in self.save for p in self.tensor_args):
            positions = [bwd1_tensors.index(p) for p in self.tensor_args]

            def backward1_fake(*args):
                outs = tuple(_empty(args[i]) for i in positions)
                return outs if n1 > 1 else outs[0]

            self.backward1_op.register_fake(backward1_fake)
        else:
            raise ContractError(
                f"cg.custom_op {self.name}: backward1 returns a gradient for every tensor input "
                "of forward1, so save them all, or give backward1_fake"
            )
        diff_positions = [bwd1_tensors.index(n) for n in self.bwd1_diff]

        def backward2_fake(*args):
            outs = tuple(_empty(args[i]) for i in diff_positions)
            return outs if n2 > 1 else outs[0]

        self.backward2_op.register_fake(backward2_fake)

    def _register_autograd(self, n1: int, n2: int) -> None:
        op = self

        def setup1(ctx, inputs, output):
            by_name = dict(zip(op.params, inputs, strict=True))
            by_name.update(zip(op.outputs, _as_tuple(output), strict=True))
            ctx.save_for_backward(*[by_name[n] for n in op.save])
            ctx.others = [by_name[p] for p in op.other_args]
            stats = [by_name[n] for n in op.nondiff if n in op.outputs]
            if stats:
                ctx.mark_non_differentiable(*stats)

        def backward1(ctx, *grads):
            diff = [g for o, g in zip(op.outputs, grads, strict=True) if o not in op.nondiff]
            found = iter(_as_tuple(op.backward1_op(*ctx.saved_tensors, *diff, *ctx.others)))
            return tuple(next(found) if p in op.tensor_args else None for p in op.params)

        self.op.register_autograd(backward1, setup_context=setup1)

        def setup2(ctx, inputs, output):
            by_name = dict(zip(op.bwd1_params, inputs, strict=True))
            ctx.save_for_backward(*[by_name[n] for n in op.bwd1_tensors])
            ctx.others = [by_name[p] for p in op.other_args]

        def backward2(ctx, *ggs):
            found = iter(_as_tuple(op.backward2_op(*ctx.saved_tensors, *ctx.others, *ggs)))
            return tuple(next(found) if n in op.bwd1_diff else None for n in op.bwd1_params)

        self.backward1_op.register_autograd(backward2, setup_context=setup2)

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)


def custom_op(name: str):
    """Register a class with ``forward1`` / ``backward1`` / ``backward2`` as a fused
    operator that create_graph can differentiate twice (see the module docstring)."""

    def decorate(spec: type) -> FusedOp:
        return FusedOp(name, spec)

    return decorate
