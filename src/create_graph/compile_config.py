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
from torch import Tensor, nn
from torch.utils._traceback import format_frame

from create_graph.pytree import flatten

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
        dynamic: forwarded to ``torch.compile``. True also weakly marks every
            input dimension larger than 1 before the compiled call, so an axis
            that changes between batches (a neighbor count) is symbolic from
            the first compile. A dimension the region specializes, such as a
            linear layer's channels, stays static.
        memory_budget: AOTAutograd activation memory budget in (0, 1] for this
            region's graphs. None on the joint forward + VJP graph uses
            ``JOINT_ACTIVATION_BUDGET``. None on a checkpointed region keeps
            torch's default. 1.0 saves the whole tape.
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

# The joint graph inlines dE/dx into the forward. At budget 1.0 the partitioner
# saves every intermediate of that VJP, about 70 GiB on AllScAIP. Budgets below
# 1 did not shrink it: materialized-backward bans and a min-cut that is already
# "smaller than the inputs" both skip the knapsack. The joint compile therefore
# asks for this budget and then, if the saved set is still above
# ``_JOINT_SAVED_CAP_BYTES``, keeps matmuls and custom kernels and recomputes
# only fusible ops. Inputs alone are the last resort.
JOINT_ACTIVATION_BUDGET = 0.3
_JOINT_SAVED_CAP_BYTES = 24 * 1024**3


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
    # A captured CUDA graph keeps its workspace in a private GPU pool for the
    # life of the process. One pool per concrete shape is the compile result
    # that would otherwise stay resident after the step. Replay is not used.
    if hasattr(triton, "cudagraphs"):
        options["triton.cudagraphs"] = False
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


def mark_region_inputs(*trees) -> None:
    """Weakly mark every input size greater than 1.

    A dimension the compiled region specializes (a linear layer's channels)
    stays static. A dimension it only reduces or indexes stays symbolic from
    the first compile. Dims of size 0 or 1, broadcast dims (stride 0), and
    parameters are left alone.
    """
    seen: set[int] = set()
    for tree in trees:
        for leaf in flatten(tree)[0]:
            if not isinstance(leaf, Tensor) or isinstance(leaf, nn.Parameter):
                continue
            if leaf.dim() == 0 or id(leaf) in seen:
                continue
            seen.add(id(leaf))
            for dim, size in enumerate(leaf.shape):
                if size <= 1 or leaf.stride(dim) == 0:
                    continue
                hint_dynamic_dim(leaf, dim)


def hint_dynamic_dim(tensor, dim: int) -> None:
    """Weakly mark one input dimension dynamic.

    A dimension the compiled region then specializes stays static. A dimension
    it only reduces or indexes stays symbolic from the first compile, instead
    of freezing at the first length Dynamo sees.
    """
    torch._dynamo.maybe_mark_dynamic(tensor, dim)


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


_JOINT_COMPILE_LOGGED = False
_PARTITION_INSTALLED = False


def _node_nbytes(part, node) -> int:
    try:
        return int(part._size_of(node))
    except Exception:
        return 0


def _saved_gib(part, nodes) -> float:
    return sum(_node_nbytes(part, node) for node in nodes) / 2**30


def _saved_top(part, nodes, k: int = 6) -> str:
    ranked = sorted(nodes, key=lambda node: _node_nbytes(part, node), reverse=True)[:k]
    bits = []
    for node in ranked:
        val = node.meta.get("val")
        shape = getattr(val, "shape", None)
        dtype = getattr(val, "dtype", None)
        target = getattr(node.target, "__name__", node.target)
        bits.append(f"{target} {shape} {dtype} {_node_nbytes(part, node) / 2**30:.2f}GiB")
    return "; ".join(bits)


def _aggressive_saved(part, joint_graph, node_info):
    """Recompute fusible ops. Matmuls stay saved, and so do custom kernels."""
    options = part.MinCutOptions(
        ban_if_used_far_apart=False,
        ban_if_long_fusible_chains=False,
        ban_if_materialized_backward=False,
        ban_if_not_in_allowlist=False,
        ban_if_reduction=True,
    )
    functorch = torch._functorch.config
    previous = getattr(functorch, "is_non_builtin_to_include", None)
    if previous is not None:
        functorch.is_non_builtin_to_include = True
    try:
        saved, _banned = part.solve_min_cut(joint_graph, node_info, options)
    finally:
        if previous is not None:
            functorch.is_non_builtin_to_include = previous
    return saved


def _install_joint_saved_cap() -> None:
    """Keep the joint graph compiled, and stop it saving a 60 GiB tape.

    Installed once, including for later regions in the process. A budget of 1
    is left to the stock partitioner. A lower budget that still estimates above
    the cap recomputes fusible ops first and, if that estimate is still over
    the cap, saves only the region inputs.
    """
    global _PARTITION_INSTALLED
    if _PARTITION_INSTALLED:
        return
    import torch._functorch.partitioners as part

    original = part.choose_saved_values_set
    if getattr(original, "_cg_cap", False):
        _PARTITION_INSTALLED = True
        return

    def choose(joint_graph, node_info, memory_budget=1):
        saved = original(joint_graph, node_info, memory_budget)
        stock_gib = _saved_gib(part, saved)
        print(
            f"create_graph partition request={memory_budget} "
            f"saved={stock_gib:.2f}GiB inputs={_saved_gib(part, node_info.inputs):.2f}GiB "
            f"n={len(saved)} top=[{_saved_top(part, saved)}]",
            flush=True,
        )
        if memory_budget >= 1 or stock_gib * 2**30 <= _JOINT_SAVED_CAP_BYTES:
            return saved
        try:
            aggressive = _aggressive_saved(part, joint_graph, node_info)
        except Exception as exc:
            aggressive = None
            print(
                f"create_graph partition aggressive failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        if aggressive is not None:
            aggressive_gib = _saved_gib(part, aggressive)
            print(
                f"create_graph partition aggressive saved={aggressive_gib:.2f}GiB "
                f"n={len(aggressive)} top=[{_saved_top(part, aggressive)}]",
                flush=True,
            )
            if aggressive_gib * 2**30 <= _JOINT_SAVED_CAP_BYTES:
                print("create_graph partition choice=aggressive", flush=True)
                return aggressive
        inputs_only = original(joint_graph, node_info, 0)
        print(
            f"create_graph partition choice=inputs "
            f"saved={_saved_gib(part, inputs_only):.2f}GiB",
            flush=True,
        )
        return inputs_only

    choose._cg_cap = True
    part.choose_saved_values_set = choose
    _PARTITION_INSTALLED = True


@contextlib.contextmanager
def compile_context(config: CompileConfig, recompile_limit: int | None = None, *, joint: bool = False):
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
        if joint:
            _install_joint_saved_cap()
        budget = config.memory_budget
        if budget is None and joint:
            budget = JOINT_ACTIVATION_BUDGET
        if budget is not None:
            functorch_patch: dict[str, Any] = {"activation_memory_budget": budget}
            # Inlined VJP ops are tagged materialized-backward and otherwise
            # cannot be dropped, so the budget cannot shrink the tape.
            if joint and hasattr(torch._functorch.config, "ban_recompute_materialized_backward"):
                functorch_patch["ban_recompute_materialized_backward"] = False
            if joint and hasattr(torch._functorch.config, "enable_autograd_cache"):
                functorch_patch["enable_autograd_cache"] = False
            if joint and hasattr(torch._functorch.config, "force_autograd_cache"):
                functorch_patch["force_autograd_cache"] = False
            if joint:
                global _JOINT_COMPILE_LOGGED
                if not _JOINT_COMPILE_LOGGED:
                    _JOINT_COMPILE_LOGGED = True
                    print(
                        f"create_graph joint_compile budget={budget} "
                        "materialized_backward_recompute=True",
                        flush=True,
                    )
            stack.enter_context(torch._functorch.config.patch(**functorch_patch))
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
