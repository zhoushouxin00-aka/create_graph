"""
Fused operators registered with cg.custom_op: each kernel against a
differentiable reference (check_op), the op inside a create_graph region
against eager create_graph and against the reference model, registration
errors.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

import create_graph as cg
from create_graph.testing import check_create_graph, check_op
from kernels.edge_attn import EdgeAttention, attention, edge_attention

TOL = 1e-10
DTYPE = torch.float64


def attn_args(seed=0, masked=False):
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(2, 3, 5, 4, generator=gen, dtype=DTYPE) for _ in range(3))
    bias = torch.randn(2, 1, 1, 5, generator=gen, dtype=DTYPE)
    if masked:
        bias[0, 0, 0, 1] = -torch.inf
        bias[1, 0, 0, 4] = -torch.inf
    return q, k, v, bias, 0.5


@pytest.mark.parametrize("masked", [False, True])
def test_edge_attention_kernels_match_reference(masked):
    errors = check_op(EdgeAttention, attention, attn_args(masked=masked))
    bad = {k: v for k, v in errors.items() if v > TOL}
    assert not bad, errors
    assert {"backward2.q", "backward2.bias", "backward2.grad_out"} <= set(errors)


@cg.custom_op("cg_tests::cube")
class Cube:
    outputs = ("y",)
    save = ("x",)

    @staticmethod
    def forward1(x: Tensor) -> Tensor:
        return x**3

    @staticmethod
    def fake(x):
        return torch.empty_like(x)

    @staticmethod
    def backward1(x, grad_y):
        return 3 * x * x * grad_y

    @staticmethod
    def backward2(x, grad_y, gg_x):
        return 6 * x * grad_y * gg_x, 3 * x * x * gg_x


def cube_reference(x):
    return x**3


def test_cube_kernels_match_reference():
    errors = check_op(Cube, cube_reference, (torch.randn(4, 3, dtype=DTYPE),))
    assert max(errors.values()) < TOL, errors


class Pairwise(nn.Module):
    """Every atom attends over all atoms; the bias depends on positions."""

    def __init__(self, op: bool, hidden=8, heads=2):
        super().__init__()
        self.op, self.heads = op, heads
        self.embed = nn.Linear(4, hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.cube_in = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, pos):
        rel = pos[:, None, :] - pos[None, :, :]
        dist2 = (rel * rel).sum(-1)
        h = self.embed(torch.cat([rel, dist2[..., None]], dim=-1)).mean(1)  # (n, hidden)
        n, d = h.shape
        q, k, v = self.qkv(h).reshape(n, 3, self.heads, d // self.heads).permute(1, 2, 0, 3)
        bias = -0.5 * dist2[None, None]  # (1, 1, n, n)
        scale = 1.0 / math.sqrt(d // self.heads)
        if self.op:
            a = edge_attention(q[None], k[None], v[None], bias, scale)[0]
            c = Cube(self.cube_in(h))
        else:
            a = attention(q[None], k[None], v[None], bias, scale)[0][0]
            c = cube_reference(self.cube_in(h))
        a = a.permute(1, 0, 2).reshape(n, d)
        return (self.out(a + 0.1 * c)).sum().reshape(1)


class PairwiseModel(nn.Module):
    def __init__(self, op: bool):
        super().__init__()
        self.net = Pairwise(op)
        self.region = cg.compile(self.net, config=cg.CompileConfig(backend="aot_eager"))

    def forward(self, data):
        return {"energy": self.region(data["pos"])}


def build(op: bool):
    torch.manual_seed(0)
    return PairwiseModel(op).to(DTYPE)


def data(seed=0):
    return {"pos": torch.randn(7, 3, generator=torch.Generator().manual_seed(seed), dtype=DTYPE)}


def test_ops_in_region_match_eager_create_graph():
    errors = check_create_graph(build(op=True), data())
    assert max(errors.values()) < TOL, errors


def test_ops_in_region_match_reference_model():
    loss = cg.testing.coupled_loss()
    common = dict(outputs="energy", inputs=("pos",), loss=loss)
    got = cg.testing.run_step(build(op=True), data(), exact=False, **common)
    want = cg.testing.run_step(build(op=False), data(), exact=True, **common)
    assert cg.testing.rel_err(got["out"], want["out"]) < TOL
    assert cg.testing.rel_err(got["grads"]["pos"], want["grads"]["pos"]) < TOL
    assert cg.testing.grad_errors(got["params"], want["params"])["all"] < TOL


def test_backward1_needs_shapes_of_every_tensor_input():
    with pytest.raises(cg.ContractError, match="save them all"):

        @cg.custom_op("cg_tests::bad")
        class Bad:
            outputs = ("y",)
            save = ("x",)

            @staticmethod
            def forward1(x: Tensor, w: Tensor) -> Tensor:
                return x * w

            @staticmethod
            def backward1(x, grad_y):
                return grad_y, grad_y

            @staticmethod
            def backward2(x, grad_y, gg_x, gg_w):
                return gg_x, gg_x
