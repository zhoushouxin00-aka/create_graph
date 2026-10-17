"""
How create_graph regions are compiled, and the only module that touches
private torch APIs (``torch._dynamo``, ``torch._functorch``, ``torch._inductor``
configs). ``tests/test_torch_surface.py`` enforces that: whatever a new torch
minor version changes in those internals is handled here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch._guards
import torch.nn.functional as F
from torch.utils._traceback import format_frame

BACKENDS = ("inductor", "aot_eager", "eager")


@dataclasses.dataclass(frozen=True)
class CompileConfig:
    """
    How a create_graph region is compiled.

    Attributes:
        backend: "inductor" compiles the region's forward + VJP graph with
            Inductor (AOTAutograd derives its backward, the min-cut partitioner
            chooses what to save). "aot_eager" runs AOTAutograd without Inductor.
            "eager" runs the region and its VJP op by op (exact, for debugging).
        dynamic: forwarded to torch.compile; True marks sizes dynamic from the
            first compile, so one graph covers every batch.
        memory_budget: AOTAutograd activation memory budget in (0, 1] for this
            region's graphs; None keeps torch's default (no extra recompute).
        checkpoint: regex matched (``re.fullmatch``) against the module paths
            of the nn.Modules the region receives (its ``self`` and module
            arguments). Matching submodules become checkpointed blocks: compiled
            one by one, only their inputs are kept, and the parameter backward
            recomputes their forward and forward + VJP.
        options: Inductor options for this region, as (key, value) pairs (a dict
            is accepted). Version-specific defaults are added underneath.
        recompile_error: raise when Dynamo hits its recompile limit instead of
            silently running the region eagerly (still exact, only slower).
    """

    backend: str = "inductor"
    dynamic: bool | None = True
    memory_budget: float | None = None
    checkpoint: str | None = None
    options: tuple[tuple[str, Any], ...] = ()
    recompile_error: bool = False

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.memory_budget is not None and not 0.0 < self.memory_budget <= 1.0:
            raise ValueError(f"memory_budget must be in (0, 1], got {self.memory_budget}")
        if isinstance(self.options, dict):
            object.__setattr__(self, "options", tuple(sorted(self.options.items())))

    def replace(self, **changes) -> CompileConfig:
        return dataclasses.replace(self, **changes)


DEFAULT_CONFIG = CompileConfig()


# ---------------------------------------------------------------------------
# feature detection
# ---------------------------------------------------------------------------


def torch_version() -> tuple[int, int]:
    major, minor = torch.__version__.split(".")[:2]
    return int(major), int(minor)


def _probe(x):
    with torch.enable_grad():
        y = (x * x).sum()
        return torch.autograd.grad(y, x, create_graph=True)[0]


@functools.cache
def supports_in_graph_grad() -> bool:
    """
    Whether Dynamo can put ``torch.autograd.grad`` into a graph: the switch
    exists from torch 2.11 on; later versions may trace it by default, which
    the probe detects.
    """
    if hasattr(torch._dynamo.config, "trace_autograd_ops"):
        return True
    try:
        fn = torch.compile(_probe, backend="eager", fullgraph=True)
        fn(torch.ones(2, requires_grad=True))
    except Exception:
        return False
    return True


_FALLBACK_WARNED: set[str] = set()


def warn_eager_fallback(where: str) -> None:
    if where in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(where)
    warnings.warn(
        f"create_graph: torch {torch.__version__} cannot trace torch.autograd.grad into a "
        f"compiled graph (needs torch >= 2.11), so {where} runs as eager create_graph: exact, "
        "but not compiled.",
        stacklevel=3,
    )


def default_inductor_options() -> dict[str, Any]:
    """
    Workarounds for Inductor bugs on dynamic ranges such as (K+1)*N in backward
    graphs, by version:

    - torch 2.12, 2.13: the mix-order reduction fails to split them
      ("CantSplit: s11*s46 + s11 not divisible by s11");
    - torch 2.11: the memory-coalescing analysis of tiling asserts on them
      (tiling_utils.get_pw_red_splits).
    """
    options: dict[str, Any] = {}
    triton = getattr(torch._inductor.config, "triton", None)
    if hasattr(triton, "mix_order_reduction"):
        options["triton.mix_order_reduction"] = False
    if hasattr(triton, "coalesce_tiling_analysis") and torch_version() < (2, 12):
        options["triton.coalesce_tiling_analysis"] = False
    return options


def _sloc_str(self) -> str:
    """torch 2.13's SLoc.__str__: a symbol may have no framework location."""
    loc = self.framework_loc
    floc = loc if isinstance(loc, str) else "" if loc is None else format_frame(loc)
    return f"{self.maybe_user_loc} ({floc})" if self.maybe_user_loc is not None else f"({floc})"


@contextlib.contextmanager
def _guard_comments_without_frames():
    """
    torch 2.11, 2.12: building guards fails with "'NoneType' object has no
    attribute 'filename'" when a symbol has no framework location (NequIP's
    e3nn code), formatting the guard's source comment. Backports the fix.
    """
    sloc = getattr(torch._guards, "SLoc", None)
    if sloc is None or torch_version() >= (2, 13):
        yield
        return
    original = sloc.__str__
    sloc.__str__ = _sloc_str
    try:
        yield
    finally:
        sloc.__str__ = original


# ---------------------------------------------------------------------------
# compiling and the scope of every compiled call
# ---------------------------------------------------------------------------


def torch_compile(fn: Callable, config: CompileConfig, *, fullgraph: bool = True) -> Callable:
    """``fn`` compiled as ``config`` says ("eager": ``fn`` itself)."""
    if config.backend == "eager":
        return fn
    if config.backend == "aot_eager":
        return torch.compile(fn, backend="aot_eager", fullgraph=fullgraph, dynamic=config.dynamic)
    options = default_inductor_options()
    options.update(dict(config.options))
    return torch.compile(fn, fullgraph=fullgraph, dynamic=config.dynamic, options=options or None)


def _rms_norm_composite(input, normalized_shape, weight=None, eps=None):
    dims = tuple(range(-len(normalized_shape), 0))
    if eps is None:
        eps = torch.finfo(input.dtype).eps
    out = input * torch.rsqrt(input.pow(2).mean(dims, keepdim=True) + eps)
    return out if weight is None else out * weight


@contextlib.contextmanager
def symbolic_safe_ops():
    """
    The fused rms_norm kernel's backward formula reads static sizes, which pins
    every batch dimension of a graph that differentiates through it (one
    recompile per batch size). Inside regions nn.RMSNorm calls this composite
    instead; Dynamo guards on the patched global.
    """
    original = F.rms_norm
    F.rms_norm = _rms_norm_composite
    try:
        yield
    finally:
        F.rms_norm = original


@contextlib.contextmanager
def compile_context(config: CompileConfig, recompile_limit: int | None = None):
    """
    Scope of every call into a compiled region (compiles happen lazily on the
    first call, recompiles on later ones):

    - Dynamo traces torch.autograd.grad into the graph;
    - DDPOptimizer is off: it would split the graph at bucket boundaries, and
      the VJP needs the path from the region inputs to the energy in one graph;
    - the region's activation memory budget;
    - symbolic-shape-safe replacements for fused ops;
    - ``recompile_limit``: graphs one compiled function may hold (a block
      function serves blocks of different structure).
    """
    dynamo = torch._dynamo.config
    patches: dict[str, Any] = {"optimize_ddp": False}
    if hasattr(dynamo, "trace_autograd_ops"):
        patches["trace_autograd_ops"] = True
    if config.recompile_error and hasattr(dynamo, "fail_on_recompile_limit_hit"):
        patches["fail_on_recompile_limit_hit"] = True
    if recompile_limit is not None:
        name = "recompile_limit" if hasattr(dynamo, "recompile_limit") else "cache_size_limit"
        patches[name] = max(recompile_limit, getattr(dynamo, name))
    with contextlib.ExitStack() as stack:
        stack.enter_context(dynamo.patch(**patches))
        if config.memory_budget is not None:
            stack.enter_context(
                torch._functorch.config.patch(activation_memory_budget=config.memory_budget)
            )
        stack.enter_context(symbolic_safe_ops())
        stack.enter_context(_guard_comments_without_frames())
        yield


def is_python_function_node(node) -> bool:
    """
    Whether an autograd node comes from a user ``torch.autograd.Function``.
    torch.library custom ops implement their autograd with generated Functions
    too, but Dynamo keeps those opaque: they are not affected (torch 2.13: the
    second derivative through a traced Function's backward drops the path
    through its saved inputs).
    """
    return isinstance(node, torch.autograd.function.BackwardCFunction) and not type(
        node
    ).__name__.startswith("GeneratedBackwardFor")


def reset_compiled() -> None:
    torch.compiler.reset()


def compiled_graph_count() -> int:
    """Graphs Dynamo has compiled in this process (for recompile tests)."""
    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
