"""
create_graph under DDP (2 ranks, gloo, CPU, float64). The region is compiled
inside DDP.forward, where Dynamo's DDPOptimizer would split its graph at bucket
boundaries; create_graph compiles it as one graph. The reduced gradients must
equal the average of the per-rank eager create_graph gradients.
"""

from __future__ import annotations

import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import create_graph as cg

pytestmark = [pytest.mark.ddp, pytest.mark.slow]

HIDDEN = 256  # float64: 0.5 MB per layer, several 1 MB buckets


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN) for _ in range(6))
        self.embed, self.out = nn.Linear(4, HIDDEN), nn.Linear(HIDDEN, 1)
        self.region = cg.compile(self._energy, config=cg.CompileConfig(backend="aot_eager"))

    def _energy(self, vec, dist):
        h = torch.tanh(self.embed(torch.cat([vec, dist[:, None]], dim=-1)))
        for layer in self.layers:
            h = h + nn.functional.silu(layer(h))
        return self.out(h).sum().reshape(1)

    def forward(self, data):
        vec = data["pos"][data["src"]] - data["pos"][data["dst"]]
        return {"energy": self.region(vec, vec.norm(dim=-1))}


def _data(rank):
    gen = torch.Generator().manual_seed(100 + rank)
    n = 9
    src = torch.randint(0, n, (24,), generator=gen)
    return {
        "pos": torch.randn(n, 3, generator=gen, dtype=torch.float64),
        "src": src,
        "dst": (src + torch.randint(1, n, (24,), generator=gen)) % n,
    }


def _step(model, data, exact=False):
    out, grads = cg.create_graph(model, data, exact=exact)
    (out["energy"].sum() + grads["pos"].square().sum()).backward()


def _worker(rank, world, port, path, ddp_optimizer):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    if ddp_optimizer:
        # what create_graph would do without its optimize_ddp=False patch
        import create_graph.region as region

        original = region.compile_context

        def with_ddp_optimizer(config):
            stack = original(config)
            patch = torch._dynamo.config.patch(optimize_ddp=True)

            class Both:
                def __enter__(self):
                    stack.__enter__()
                    patch.__enter__()

                def __exit__(self, *exc):
                    patch.__exit__(*exc)
                    return stack.__exit__(*exc)

            return Both()

        region.compile_context = with_ddp_optimizer
    from create_graph.compile_config import compiled_graph_count

    model = DistributedDataParallel(Net(), bucket_cap_mb=1)
    before = compiled_graph_count()
    _step(model, _data(rank))
    grads = [p.grad.clone() for p in model.module.parameters()]
    gathered = [torch.cat([g.flatten() for g in grads]) for _ in range(world)]
    dist.all_gather(gathered, torch.cat([g.flatten() for g in grads]))
    if rank == 0:
        torch.save(
            {
                "grads": grads,
                "graphs": compiled_graph_count() - before,
                "ranks_identical": all(torch.equal(gathered[0], g) for g in gathered[1:]),
            },
            path,
        )
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(ddp_optimizer: bool):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "grads.pt")
        mp.spawn(_worker, args=(2, _free_port(), path, ddp_optimizer), nprocs=2, join=True)
        return torch.load(path)


def _expected_grads():
    torch.manual_seed(0)
    model = Net()
    total = None
    for rank in range(2):
        model.zero_grad()
        _step(model, _data(rank), exact=True)
        grads = [p.grad.clone() for p in model.parameters()]
        total = grads if total is None else [a + b for a, b in zip(total, grads, strict=True)]
    return [g / 2 for g in total]


def _rel(got, want):
    a = torch.cat([g.flatten() for g in got])
    b = torch.cat([g.flatten() for g in want])
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.compiled
def test_ddp_gradients_match_and_region_is_one_graph():
    torch.set_default_dtype(torch.float64)
    try:
        expected = _expected_grads()
        result = _run(ddp_optimizer=False)
    finally:
        torch.set_default_dtype(torch.float32)
    assert result["ranks_identical"]
    assert result["graphs"] == 1, result["graphs"]
    assert _rel(result["grads"], expected) < 1e-10


@pytest.mark.compiled
def test_ddp_optimizer_would_split_the_region():
    torch.set_default_dtype(torch.float64)
    try:
        expected = _expected_grads()
        try:
            result = _run(ddp_optimizer=True)
        except Exception:
            return  # the split graph does not even compile: the patch is needed
    finally:
        torch.set_default_dtype(torch.float32)
    assert result["graphs"] != 1 or _rel(result["grads"], expected) > 1e-6
