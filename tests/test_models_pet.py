"""
Real PET (metatrain 2026.4.1) on OMol25 molecules and periodic metals, GPU.
Every case against eager create_graph on the same parameters: per step (two
sizes, float32 and float64, memory budget, per-block checkpoint, the fused
attention op), one compile across batches, one epoch in lockstep (float32)
and free-running (float64).
"""

from __future__ import annotations

import pytest
import torch

import create_graph as cg
from cg_bench import data as cgdata
from cg_bench import train
from cg_bench.pet_model import PETEnergy, batch_of, build_pet, fused_attention
from create_graph.compile_config import compiled_graph_count
from create_graph.testing import (
    as_close_as_eager,
    check_create_graph,
    check_float32,
    grad_errors,
    rel_err,
    run_step,
)

pytestmark = [pytest.mark.gpu, pytest.mark.pet, pytest.mark.slow]

FP32 = {"out": 1e-5, "loss": 1e-5, "grad_pos": 1e-5, "params": 1e-5, "params_worst_tensor": 1e-3}
FP64 = {"out": 1e-9, "loss": 1e-9, "grad_pos": 1e-9, "params": 1e-9, "params_worst_tensor": 1e-6}
BLOCKS = r"pet\.backend\.gnn_layers\.\d+"


def assert_within(errors, tol):
    bad = {k: v for k, v in errors.items() if k in tol and v > tol[k]}
    assert not bad, f"above tolerance: {bad} (all: {errors})"


@pytest.fixture(scope="module")
def omol():
    frames = cgdata.load_omol()
    return frames, cgdata.linear_references(frames)


@pytest.fixture(autouse=True)
def strict_fp32():
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def make(frames, size="default", dtype=torch.float32, **config):
    pet = build_pet(cgdata.elements_of(frames), size, dtype=dtype)
    return pet, PETEnergy(pet, cg.CompileConfig(**config))


def batch(pet, frames, atoms, seed=0, dtype=torch.float32):
    return batch_of(pet, cgdata.Packer(frames, atoms, seed)[0], dtype=dtype)


def assert_float32(pet, model, frames, elements, size="default"):
    """create_graph in float32 is as close to eager create_graph in float64 (same
    parameters) as eager create_graph in float32 is."""
    pet64 = build_pet(elements, size, dtype=torch.float64)
    pet64.load_state_dict(pet.state_dict())
    result = check_float32(
        model, batch_of(pet, frames), PETEnergy(pet64), batch_of(pet64, frames, dtype=torch.float64)
    )
    bad = as_close_as_eager(result, FP32)
    assert not bad, f"create_graph worse than eager in float32: {bad} (all: {result})"


def packed(frames, atoms, seed=0):
    return cgdata.Packer(frames, atoms, seed)[0]


@pytest.mark.parametrize("size", ["default", "large"])
def test_float32_as_close_as_eager_create_graph(omol, size):
    frames, _ = omol
    pet, model = make(frames, size)
    assert_float32(pet, model, packed(frames, 512), cgdata.elements_of(frames), size)


def test_float64_matches_eager_create_graph(omol):
    frames, _ = omol
    pet, model = make(frames, dtype=torch.float64)
    assert_within(check_create_graph(model, batch(pet, frames, 160, dtype=torch.float64)), FP64)


@pytest.mark.parametrize("config", [{"memory_budget": 0.5}, {"checkpoint": BLOCKS}])
def test_memory_knobs_keep_gradients(omol, config):
    frames, _ = omol
    pet, model = make(frames, **config)
    assert_float32(pet, model, packed(frames, 512), cgdata.elements_of(frames))


def peak_gib(model, data, t):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    train.step(model, data, t)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30


@pytest.mark.compiled
def test_memory_budget_and_checkpoint_lower_the_peak(omol):
    frames, refs = omol
    packed = cgdata.Packer(frames, 1024)[0]
    peaks = {}
    for name, config in {
        "plain": {},
        "budget": {"memory_budget": 0.3},
        "checkpoint": {"checkpoint": BLOCKS},
    }.items():
        cg.reset()
        pet, model = make(frames, "large", **config)
        data, t = batch_of(pet, packed), train.targets_of(packed, refs)
        peak_gib(model, data, t)  # compile
        pet.zero_grad(set_to_none=True)
        peaks[name] = peak_gib(model, data, t)
    assert peaks["budget"] < peaks["plain"] and peaks["checkpoint"] < peaks["plain"], peaks


def test_periodic_metals(omol):
    frames, _ = omol
    pet, model = make(frames)
    assert_float32(pet, model, cgdata.periodic_metals(4), cgdata.elements_of(frames))


def test_fused_attention_op_matches_manual_attention(omol):
    frames, _ = omol
    pet, model = make(frames)
    data = batch(pet, frames, 384)
    loss = cg.testing.coupled_loss()
    with fused_attention():
        got = run_step(model, data, outputs="energy", inputs=("pos",), loss=loss, exact=False)
    want = run_step(model, data, outputs="energy", inputs=("pos",), loss=loss, exact=True)
    assert rel_err(got["out"], want["out"]) < 1e-5
    assert rel_err(got["grads"]["pos"], want["grads"]["pos"]) < 1e-5
    assert grad_errors(got["params"], want["params"])["all"] < 1e-5


@pytest.mark.compiled
def test_one_compile_covers_every_batch(omol):
    frames, refs = omol
    pet, model = make(frames)
    counts = []
    for i, atoms in enumerate([256, 384, 512, 640, 768]):
        packed = cgdata.Packer(frames, atoms, seed=i)[0]
        train.step(model, batch_of(pet, packed), train.targets_of(packed, refs))
        counts.append(compiled_graph_count())
    assert counts[-1] - counts[1] == 0, counts


def epoch_batches(pet, frames, refs, count, atoms, dtype):
    packer = cgdata.Packer(frames[:count], atoms, seed=0)
    return [
        (batch_of(pet, packer[i], dtype=dtype), train.targets_of(packer[i], refs, dtype=dtype))
        for i in range(len(packer))
    ]


def test_epoch_lockstep_float32(omol):
    frames, refs = omol
    pet, model = make(frames)
    batches = epoch_batches(pet, frames, refs, 256, 1024, torch.float32)
    worst, steps = train.lockstep_epoch(model, pet.parameters(), batches)
    assert steps >= 20
    assert_within(
        {"out": worst["energy"], "grad_pos": worst["forces"], "loss": worst["loss"],
         "params": worst["params"], "params_worst_tensor": worst["params_worst_tensor"]},
        FP32,
    )  # fmt: skip


def test_epoch_free_running_float64(omol):
    frames, refs = omol
    pet_ref, _ = make(frames, dtype=torch.float64)
    pet_cg, model_cg = make(frames, dtype=torch.float64)  # same seed: same initial parameters
    assert all(torch.equal(a, b) for a, b in zip(pet_ref.parameters(), pet_cg.parameters()))
    batches = epoch_batches(pet_ref, frames, refs, 96, 512, torch.float64)
    result = train.free_running_epoch(PETEnergy(pet_ref), model_cg, batches)
    assert result["steps"] >= 15
    assert result["loss"] < 1e-8, result
    assert result["params"] < 1e-7, result
