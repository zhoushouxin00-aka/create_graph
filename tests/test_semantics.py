"""
Mechanism tests on small functions (CPU, float64): every case is compared with
eager create_graph (``exact=True``): output, input gradients, a loss coupling
both, and the parameter gradients of that loss. They pin down the semantics: eager code before the
region, inputs of every kind, autograd semantics inside the region, outputs,
evaluation mode, the contract errors and per-block checkpointing.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest
import torch
from torch import nn

import create_graph as cg
from create_graph.compile_config import compiled_graph_count
from create_graph.testing import check_create_graph

TOL = 1e-10
DTYPE = torch.float64


def make_data(n=9, e=30, graphs=2, seed=0):
    gen = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (e,), generator=gen)
    dst = (src + torch.randint(1, n, (e,), generator=gen)) % n
    batch = torch.arange(n) * graphs // n
    return {
        "pos": torch.randn(n, 3, generator=gen, dtype=DTYPE),
        "src": src,
        "dst": dst,
        "batch": batch,
        "num_graphs": graphs,
        "z": torch.randint(0, 4, (n,), generator=gen),
    }


def assert_close(errors, tol=TOL):
    bad = {k: v for k, v in errors.items() if v > tol}
    assert not bad, f"errors above {tol}: {bad} (all: {errors})"


def geometry(data):
    vec = data["pos"][data["src"]] - data["pos"][data["dst"]]
    return vec, vec.norm(dim=-1)


class Energy(nn.Module):
    """Edge features -> per-structure energies, the parameterized part."""

    def __init__(self, hidden=16, norm="ln", act="gelu"):
        super().__init__()
        self.embed = nn.Linear(4, hidden)
        self.norm = nn.LayerNorm(hidden) if norm == "ln" else nn.RMSNorm(hidden)
        acts = {"gelu": nn.GELU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), acts[act], nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, 1)

    def forward(self, vec, dist, src, batch, num_graphs):
        h = self.embed(torch.cat([vec, dist[:, None]], dim=-1))
        h = self.norm(h)
        h = h + self.mlp(h)
        h = h * torch.softmax(h, dim=-1)
        e_edge = self.out(h).squeeze(-1)
        e_node = e_edge.new_zeros(batch.shape[0]).index_add(0, src, e_edge)
        return e_node.new_zeros(num_graphs).index_add(0, batch, e_node)


class Model(nn.Module):
    """Eager geometry, then the energy region."""

    def __init__(self, config=None, **kw):
        super().__init__()
        self.energy = Energy(**kw)
        self.region = cg.compile(
            self.energy, config=config or cg.CompileConfig(backend="aot_eager")
        )

    def forward(self, data):
        vec, dist = geometry(data)
        return {"energy": self.region(vec, dist, data["src"], data["batch"], data["num_graphs"])}


def build(cls=Model, **kw):
    torch.manual_seed(0)
    return cls(**kw).to(DTYPE)


@pytest.mark.parametrize("norm", ["ln", "rms"])
@pytest.mark.parametrize("act", ["gelu", "tanh", "silu"])
def test_region_after_eager_geometry(norm, act):
    model = build(norm=norm, act=act)
    assert_close(check_create_graph(model, make_data()))


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_backends(backend):
    model = build(config=cg.CompileConfig(backend=backend))
    assert_close(check_create_graph(model, make_data()))


class WholeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.energy = Energy()

    def forward(self, data):
        vec, dist = geometry(data)
        return self.energy(vec, dist, data["src"], data["batch"], data["num_graphs"])


def test_whole_model_region():
    torch.manual_seed(0)
    model = cg.compile(WholeModel().to(DTYPE), config=cg.CompileConfig(backend="aot_eager"))
    assert_close(check_create_graph(model, make_data(), outputs=None))


class MixedInputs(Model):
    """Eager code combines a parameter with positions (mixed input) and computes a
    parameter-only input (species embedding) before the region."""

    def __init__(self):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor(1.3))
        self.species = nn.Embedding(4, 3)

    def forward(self, data):
        vec, dist = geometry(data)
        vec = vec + self.species(data["z"])[data["src"]]  # parameters only + positions
        dist = dist * self.length_scale  # parameter times positions
        return {"energy": self.region(vec, dist, data["src"], data["batch"], data["num_graphs"])}


def test_mixed_and_parameter_only_inputs():
    model = build(MixedInputs)
    errors = check_create_graph(model, make_data())
    assert_close(errors)
    # the parameters used before the region get their gradients
    out, grads = cg.create_graph(model, make_data())
    (out["energy"].sum() + grads["pos"].square().sum()).backward()
    assert model.length_scale.grad is not None and model.species.weight.grad.abs().sum() > 0


class AutogradSemantics(nn.Module):
    """detach, no_grad, a custom autograd.Function and in-place ops inside the region."""

    class Cube(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x**3

        @staticmethod
        def backward(ctx, g):
            (x,) = ctx.saved_tensors
            return 3 * x * x * g

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.energy = Energy()
        self.w = nn.Linear(3, 3)

    def forward(self, vec, dist, src, batch, num_graphs):
        v = self.w(vec)
        if "detach" in self.features:
            v = v / dist.detach().mean()
        if "no_grad" in self.features:
            with torch.no_grad():
                peak = vec.abs().max()
            v = v / peak
        if "function" in self.features:
            v = self.Cube.apply(v)
        if "geometry_function" in self.features:
            dist = self.Cube.apply(dist)
        if "parameter_function" in self.features:
            v = v * self.Cube.apply(self.w.bias).sum()
        if "inplace" in self.features:
            v = v.clone()
            v[:, 0] += dist
        return self.energy(v, dist, src, batch, num_graphs)


class SemanticsModel(nn.Module):
    def __init__(self, features, backend="aot_eager"):
        super().__init__()
        inner = AutogradSemantics(features)
        self.region = cg.compile(inner, config=cg.CompileConfig(backend=backend))

    def forward(self, data):
        vec, dist = geometry(data)
        return {"energy": self.region(vec, dist, data["src"], data["batch"], data["num_graphs"])}


@pytest.mark.parametrize(
    "features",
    [
        ("detach",),
        ("no_grad",),
        ("inplace",),
        ("parameter_function",),
        ("detach", "no_grad", "inplace", "parameter_function"),
    ],
)
def test_autograd_semantics_inside_region(features):
    assert_close(check_create_graph(build(SemanticsModel, features=features), make_data()))


@pytest.mark.compiled
@pytest.mark.parametrize("feature", ["function", "geometry_function"])
def test_function_on_the_differentiated_path_is_refused(feature):
    """A compiled graph does not differentiate a traced autograd.Function's backward
    again; cg.custom_op is the supported way (test_custom_ops.py)."""
    with pytest.raises(cg.ContractError, match="torch.autograd.Function Cube"):
        cg.create_graph(build(SemanticsModel, features=(feature,)), make_data())


def test_function_on_the_differentiated_path_is_exact_eagerly():
    features = ("function", "geometry_function")
    model = build(SemanticsModel, features=features, backend="eager")
    assert_close(check_create_graph(model, make_data()))


class DropoutModel(Model):
    def __init__(self):
        super().__init__(config=cg.CompileConfig(backend="eager"))
        self.drop = nn.Dropout(0.3)

    def forward(self, data):
        vec, dist = geometry(data)
        return {
            "energy": self.region(
                self.drop(vec), dist, data["src"], data["batch"], data["num_graphs"]
            )
        }


def test_dropout_mask_shared_by_forward_and_vjp():
    model = build(DropoutModel)
    data = make_data()
    loss = cg.testing.coupled_loss()
    torch.manual_seed(5)
    got = cg.testing.run_step(
        model, data, outputs="energy", inputs=("pos",), loss=loss, exact=False
    )
    torch.manual_seed(5)
    want = cg.testing.run_step(
        model, data, outputs="energy", inputs=("pos",), loss=loss, exact=True
    )
    assert cg.testing.rel_err(got["grads"]["pos"], want["grads"]["pos"]) < TOL
    assert cg.testing.grad_errors(got["params"], want["params"])["all"] < TOL


def test_eval_mode_first_order_and_detached():
    model = build()
    data = make_data()
    out, grads = cg.create_graph(model, data, train=False)
    assert not out["energy"].requires_grad and not grads["pos"].requires_grad
    pos = data["pos"].detach().requires_grad_(True)
    vec, dist = geometry({**data, "pos": pos})
    energy = model.energy(vec, dist, data["src"], data["batch"], data["num_graphs"])
    (want,) = torch.autograd.grad(energy.sum(), pos)
    assert cg.testing.rel_err(grads["pos"], want) < TOL
    assert cg.testing.rel_err(out["energy"], energy) < TOL


class TupleOut(Model):
    def forward(self, data):
        vec, dist = geometry(data)
        e = self.region(vec, dist, data["src"], data["batch"], data["num_graphs"])
        return (e, vec.shape[0])


def test_output_selection():
    model = build()
    assert_close(check_create_graph(model, make_data(), outputs=lambda out: out["energy"]))
    assert_close(check_create_graph(build(TupleOut), make_data(), outputs=lambda out: out[0]))


class Field(Model):
    def forward(self, data):
        vec, dist = geometry(data)
        vec = vec + data["field"]
        return {"energy": self.region(vec, dist, data["src"], data["batch"], data["num_graphs"])}


def test_several_inputs():
    data = make_data()
    data["field"] = torch.randn(3, dtype=DTYPE)
    assert_close(check_create_graph(build(Field), data, inputs=("pos", "field")))


@dataclasses.dataclass
class Graph:
    vec: torch.Tensor
    dist: torch.Tensor
    src: torch.Tensor
    batch: torch.Tensor
    num_graphs: int


class Batch:
    """A plain object with tensor attributes (like fairchem's AtomicData)."""

    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, v)


class DataclassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.energy = Energy()
        self.region = cg.compile(self._energy, config=cg.CompileConfig(backend="aot_eager"))

    def _energy(self, g: Graph):
        return self.energy(g.vec, g.dist, g.src, g.batch, g.num_graphs)

    def forward(self, data: Batch):
        vec = data.pos[data.src] - data.pos[data.dst]
        g = Graph(vec, vec.norm(dim=-1), data.src, data.batch, data.num_graphs)
        return {"energy": self.region(g)}


class ObjectRegionModel(nn.Module):
    """The region receives the plain object and computes the geometry itself."""

    def __init__(self):
        super().__init__()
        self.energy = Energy()
        self.region = cg.compile(self._energy, config=cg.CompileConfig(backend="aot_eager"))

    def _energy(self, data: Batch):
        vec = data.pos[data.src] - data.pos[data.dst]
        return self.energy(vec, vec.norm(dim=-1), data.src, data.batch, data.num_graphs)

    def forward(self, data: Batch):
        return {"energy": self.region(data)}


@pytest.mark.parametrize("cls", [DataclassModel, ObjectRegionModel])
def test_structured_inputs(cls):
    assert_close(check_create_graph(build(cls), Batch(make_data())))


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------


class TwoRegions(Model):
    def __init__(self):
        super().__init__()
        self.region2 = cg.compile(Energy(), config=cg.CompileConfig(backend="aot_eager"))

    def forward(self, data):
        vec, dist = geometry(data)
        args = (data["src"], data["batch"], data["num_graphs"])
        return {"energy": self.region(vec, dist, *args) + self.region2(vec, dist, *args)}


class CallsGrad(Model):
    def forward(self, data):
        out = super().forward(data)
        torch.autograd.grad(out["energy"].sum(), data["pos"], create_graph=True)
        return out


class Scaled(Model):
    def forward(self, data):
        return {"energy": 2.0 * super().forward(data)["energy"]}


class ExtraPositionTerm(Model):
    def forward(self, data):
        _, dist = geometry(data)
        return {"energy": super().forward(data)["energy"] + dist.sum()}


class ConstantShift(Model):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor([0.5, -0.25]))

    def forward(self, data):
        return {"energy": super().forward(data)["energy"] + self.shift}


@pytest.mark.compiled
@pytest.mark.parametrize(
    "cls, match",
    [
        (TwoRegions, "both depend on the differentiated inputs"),
        (CallsGrad, "computed a gradient w.r.t. a differentiated input"),
        (Scaled, "scales or drops entries"),
        (ExtraPositionTerm, "also depends on the differentiated inputs"),
    ],
)
def test_contract_errors(cls, match):
    with pytest.raises(cg.ContractError, match=match):
        cg.create_graph(build(cls), make_data())


def test_terms_independent_of_positions_may_be_added():
    assert_close(check_create_graph(build(ConstantShift), make_data()))


class NoRegion(nn.Module):
    def __init__(self):
        super().__init__()
        self.energy = Energy()

    def forward(self, data):
        vec, dist = geometry(data)
        return {"energy": self.energy(vec, dist, data["src"], data["batch"], data["num_graphs"])}


@pytest.mark.compiled
def test_without_region_falls_back_to_eager_create_graph():
    model = build(NoRegion)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        errors = check_create_graph(model, make_data())
    assert_close(errors)
    assert any("ran no cg.compile region" in str(w.message) for w in caught)


def test_old_torch_falls_back_to_eager_create_graph(monkeypatch):
    """Without in-graph autograd.grad (torch < 2.11) regions run as eager
    create_graph: the same numbers as exact=True, one warning."""
    from create_graph import api, compile_config

    monkeypatch.setattr(api, "supports_in_graph_grad", lambda: False)
    monkeypatch.setattr(compile_config, "_FALLBACK_WARNED", set())
    model = build(BlocksModel)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        errors = check_create_graph(model, make_data())
        check_create_graph(model, make_data(seed=1))
    assert all(v == 0 for v in errors.values()), errors
    fallbacks = [w for w in caught if "cannot trace torch.autograd.grad" in str(w.message)]
    assert len(fallbacks) == 1, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# per-block checkpointing
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden)
        )

    def forward(self, h, dist, layer_idx=None, num_edges=None):
        out = h + self.ff(self.norm(h)) * torch.cos(dist)[:, None]
        if num_edges is not None:
            out = out * (1.0 + 1.0 / num_edges)  # a batch size, changes from batch to batch
        if layer_idx is None:
            return out
        # the index read in Python: a profiler label and a branch
        with torch.profiler.record_function(f"block_{layer_idx}"):
            return 0.5 * out if layer_idx % 2 else out


class Blocks(nn.Module):
    def __init__(self, hidden=16, layers=3, indexed=False):
        super().__init__()
        self.embed = nn.Linear(4, hidden)
        self.blocks = nn.ModuleList(Block(hidden) for _ in range(layers))
        self.out = nn.Linear(hidden, 1)
        self.indexed = indexed

    def forward(self, vec, dist, src, batch, num_graphs):
        h = self.embed(torch.cat([vec, dist[:, None]], dim=-1))
        for i, block in enumerate(self.blocks):
            if self.indexed:
                h = block(h, dist, layer_idx=i, num_edges=dist.shape[0])
            else:
                h = block(h, dist)
        e_edge = self.out(h).squeeze(-1)
        e_node = e_edge.new_zeros(batch.shape[0]).index_add(0, src, e_edge)
        return e_node.new_zeros(num_graphs).index_add(0, batch, e_node)


class BlocksModel(nn.Module):
    def __init__(self, pattern=r"blocks\.\d+", backend="aot_eager", indexed=False):
        super().__init__()
        self.net = Blocks(indexed=indexed)
        config = cg.CompileConfig(backend=backend, checkpoint=pattern)
        self.region = cg.compile(self.net, config=config)

    def forward(self, data):
        vec, dist = geometry(data)
        return {"energy": self.region(vec, dist, data["src"], data["batch"], data["num_graphs"])}


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_checkpointed_blocks(backend):
    model = build(BlocksModel, backend=backend)
    assert_close(check_create_graph(model, make_data()))


@pytest.mark.compiled
def test_checkpointed_blocks_with_scalar_arguments():
    """A layer index passed to a block and read in Python is a constant of its
    compiled forward and VJP; a number of edges passed the same way becomes
    dynamic when it changes: compiles per layer, none per batch after the second."""
    model = build(BlocksModel, indexed=True)
    counts = []
    for n, e, g in [(9, 30, 2), (14, 47, 3), (19, 66, 2), (23, 80, 4)]:
        assert_close(check_create_graph(model, make_data(n, e, g, seed=n)))
        counts.append(compiled_graph_count())
    assert counts[-1] == counts[1], counts


def test_checkpointed_blocks_eval_mode():
    model = build(BlocksModel)
    data = make_data()
    _, grads = cg.create_graph(model, data, train=False)
    _, want = cg.create_graph(model, data, train=False, exact=True)
    assert cg.testing.rel_err(grads["pos"], want["pos"]) < TOL


@pytest.mark.compiled
def test_checkpoint_pattern_must_match():
    model = build(BlocksModel, pattern=r"nothing\.here")
    with pytest.raises(cg.ContractError, match="matched no submodule"):
        cg.create_graph(model, make_data())


# ---------------------------------------------------------------------------
# dynamic shapes
# ---------------------------------------------------------------------------


@pytest.mark.compiled
@pytest.mark.parametrize("norm", ["ln", "rms"])
def test_one_compile_covers_every_batch(norm):
    model = build(norm=norm)
    sizes = [(9, 30, 2), (12, 41, 3), (17, 57, 2), (23, 80, 4), (31, 99, 5)]
    counts = []
    for n, e, g in sizes:
        out, grads = cg.create_graph(model, make_data(n, e, g, seed=n))
        (out["energy"].sum() + grads["pos"].square().sum()).backward()
        counts.append(compiled_graph_count())
    # the first two calls may compile twice (Dynamo's duck sizing), never more
    assert counts[-1] - counts[1] == 0, counts
