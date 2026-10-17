"""
fairchem AllScAIP (four sizes, up to the 85M md_NeNo) and UMA (eSCN-MD S and
M) on real OMol batches, GPU, fairchem's venv:

    PYTHONPATH=src:. ../fairchem/.venv/bin/python -m pytest tests/test_models_fairchem.py

Every case against eager create_graph: float32 as close to the float64 result
as eager create_graph, memory knobs, the EdgeAttention operator in AllScAIP,
one compile across batches, one epoch in lockstep; UMA's Safeacos / Safeatan2
as fused operators.
"""

from __future__ import annotations

import pytest
import torch

import create_graph as cg
from cg_bench import train
from cg_bench.fairchem_models import (
    ALLSCAIP_BLOCKS,
    AllScAIPEnergy,
    Batches,
    SafeAcos,
    SafeAtan2,
    UMAEnergy,
    build,
    float64_copy,
    fused_attention,
)
from create_graph.compile_config import compiled_graph_count
from create_graph.testing import (
    as_close_as_eager,
    check_float32,
    check_op,
    grad_errors,
    rel_err,
    run_step,
)

pytestmark = [pytest.mark.gpu, pytest.mark.fairchem, pytest.mark.slow]

FP32 = {"out": 1e-5, "loss": 1e-5, "grad_pos": 1e-5, "params": 1e-5, "params_worst_tensor": 1e-3}


@pytest.fixture(autouse=True)
def strict_fp32():
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def make(name, size, energy_cls, **config):
    model, tasks, dataset, collate = build(name, size)
    model = model.cuda().train()
    batches = Batches(model, tasks, dataset, collate, graph=name == "uma")
    return model, energy_cls(model, cg.CompileConfig(**config)), batches


def assert_float32(model, region_model, energy_cls, batches, atoms, seed=0):
    model64 = float64_copy(model)
    result = check_float32(
        region_model,
        batches(atoms, seed),
        energy_cls(model64),
        batches(atoms, seed, dtype=torch.float64),
    )
    bad = as_close_as_eager(result, FP32)
    assert not bad, f"create_graph worse than eager in float32: {bad} (all: {result})"


# ---------------------------------------------------------------------------
# AllScAIP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["h128_l2", "h256_l4", "h512_l6", "h640_l10"])
def test_allscaip_float32(size):
    model, region, batches = make("allscaip", size, AllScAIPEnergy)
    assert_float32(model, region, AllScAIPEnergy, batches, 256)


@pytest.mark.parametrize("config", [{"memory_budget": 0.5}, {"checkpoint": ALLSCAIP_BLOCKS}])
def test_allscaip_memory_knobs(config):
    model, region, batches = make("allscaip", "h256_l4", AllScAIPEnergy, **config)
    assert_float32(model, region, AllScAIPEnergy, batches, 256)


def test_allscaip_fused_attention_op():
    model, region, batches = make("allscaip", "h128_l2", AllScAIPEnergy)
    data = batches(256, 0)
    loss = cg.testing.coupled_loss()
    with fused_attention():
        got = run_step(region, data, outputs="energy", inputs=("pos",), loss=loss, exact=False)
    want = run_step(region, data, outputs="energy", inputs=("pos",), loss=loss, exact=True)
    assert rel_err(got["out"], want["out"]) < 1e-5
    assert rel_err(got["grads"]["pos"], want["grads"]["pos"]) < 1e-5
    assert grad_errors(got["params"], want["params"])["all"] < 1e-5


@pytest.mark.compiled
def test_allscaip_one_compile_covers_every_batch():
    model, region, batches = make("allscaip", "h128_l2", AllScAIPEnergy)
    counts = []
    for i, atoms in enumerate([200, 300, 400, 500, 600]):
        data = batches(atoms, i)
        train.step(region, data, batches.targets(data))
        counts.append(compiled_graph_count())
    assert counts[-1] - counts[1] == 0, counts


def test_allscaip_epoch_lockstep():
    model, region, batches = make("allscaip", "h128_l2", AllScAIPEnergy)
    epoch = []
    for i in range(24):
        data = batches(512, 1000 + i)
        epoch.append((data, batches.targets(data)))
    worst, steps = train.lockstep_epoch(region, model.parameters(), epoch)
    assert steps == 24
    renamed = {
        "out": worst["energy"],
        "grad_pos": worst["forces"],
        "loss": worst["loss"],
        "params": worst["params"],
        "params_worst_tensor": worst["params_worst_tensor"],
    }
    bad = {k: v for k, v in renamed.items() if v > FP32[k]}
    assert not bad, renamed


# ---------------------------------------------------------------------------
# UMA (eSCN-MD)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["s", "m"])
def test_uma_float32(size):
    model, region, batches = make("uma", size, UMAEnergy)
    assert_float32(model, region, UMAEnergy, batches, 200)


@pytest.mark.compiled
def test_uma_one_compile_for_a_fixed_number_of_systems():
    """UMA's code specializes on the number of systems per batch; with that
    fixed, one compile covers every atom and edge count."""
    model, region, batches = make("uma", "s", UMAEnergy)
    counts, systems = [], None
    for seed in range(40):
        data = batches(160, seed)
        if systems is None:
            systems = int(data.natoms.numel())
        if int(data.natoms.numel()) != systems:
            continue
        train.step(region, data, batches.targets(data))
        counts.append(compiled_graph_count())
        if len(counts) == 4:
            break
    assert len(counts) == 4 and counts[-1] - counts[1] == 0, counts


def test_uma_safe_rotation_ops_match_fairchem():
    """Safeacos / Safeatan2 (the Euler-angle Wigner path) as cg.custom_op operators:
    every kernel against fairchem's autograd.Functions, differentiated twice."""
    from fairchem.core.models.uma.common.rotation import Safeacos, Safeatan2

    gen = torch.Generator().manual_seed(7)
    x = torch.rand(64, generator=gen, dtype=torch.float64) * 2 - 1
    y, z = (torch.randn(64, generator=gen, dtype=torch.float64) for _ in range(2))
    errors = check_op(SafeAcos, Safeacos.apply, (x,))
    errors.update(
        {f"atan2.{k}": v for k, v in check_op(SafeAtan2, Safeatan2.apply, (y, z)).items()}
    )
    assert max(errors.values()) < 1e-10, errors


@pytest.mark.compiled
def test_uma_euler_path_functions_are_refused():
    """With the Euler-angle Wigner path the region would differentiate fairchem's
    Safeacos / Safeatan2 Functions twice: create_graph refuses them."""
    model, tasks, dataset, collate = build("uma", "s")
    from cg_bench import fairchem_models

    model.backbone.use_quaternion_wigner = False
    model = model.cuda().train()
    region = UMAEnergy(model, cg.CompileConfig())
    data = Batches(model, tasks, dataset, collate, graph=True)(120, 0)
    with pytest.raises(cg.ContractError, match="Safeacos|Safeatan2"):
        cg.create_graph(region, data)
    with fairchem_models.uma_fused_rotation_ops():
        out, grads = cg.create_graph(region, data)
    assert torch.isfinite(grads["pos"]).all()
