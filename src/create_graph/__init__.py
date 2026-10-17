"""
create_graph: generate the gradient of the energy w.r.t. the positions into
the forward graph of conservative-force MLIP training.

    out, grads = cg.create_graph(model, data, outputs="energy", inputs=("pos",))
    loss = loss_fn(out, -grads["pos"])
    loss.backward()

The part of the model that computes the energy is marked with ``cg.compile``
(a drop-in for torch.compile). Inside create_graph it runs as one compiled
graph containing the forward and its VJP (autograd's own backward formulas,
traced), so the force computation is an ordinary first-order graph that
AOTAutograd differentiates, Inductor compiles and activation memory budgets
apply to. ``cg.custom_op`` registers fused kernels (forward1, backward1 and
backward2, the backward of backward1).
"""

from create_graph import testing  # noqa: F401
from create_graph.api import compile, create_graph, reset
from create_graph.compile_config import CompileConfig, supports_in_graph_grad
from create_graph.errors import ContractError
from create_graph.ops import FusedOp, custom_op

__version__ = "0.1.0"

__all__ = [
    "CompileConfig",
    "ContractError",
    "FusedOp",
    "compile",
    "create_graph",
    "custom_op",
    "reset",
    "supports_in_graph_grad",
    "testing",
]
