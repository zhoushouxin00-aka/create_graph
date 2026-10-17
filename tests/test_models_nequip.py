"""
Real NequIP (nequip 0.19.1, foundation-model presets) on OMol25 molecules and
periodic metals, GPU. Every case against eager create_graph on the same
parameters: per step (three presets, float32 and float64 models, memory budget,
per-block checkpoint), against nequip's own force output, one compile across
batches, one epoch in lockstep (float32) and free-running (float64).
"""

from __future__ import annotations

import pytest
import torch

import create_graph as cg
from cg_bench import data as cgdata
from cg_bench import train
from cg_bench.nequip_model import batch_of, build_nequip, energy_region, init_nequip, type_names_of
from create_graph.compile_config import compiled_graph_count
from create_graph.testing import (
    as_close_as_eager,
    check_create_graph,
    check_float32,
    param_grads,
    rel_err,
    run_step,
    step_errors,
)

pytestmark = [pytest.mark.gpu, pytest.mark.nequip, pytest.mark.slow]

FP32 = {"out": 1e-5, "loss": 1e-5, "grad_pos": 1e-5, "params": 1e-5, "params_worst_tensor": 1e-3}
FP64 = {"out": 1e-9, "loss": 1e-9, "grad_pos": 1e-9, "params": 1e-9, "params_worst_tensor": 1e-6}
OUT = "total_energy"
BLOCKS = r"model\.func\.layer\d+_convnet"


def assert_within(errors, tol):
    bad = {k: v for k, v in errors.items() if k in tol and v > tol[k]}
    assert not bad, f"above tolerance: {bad} (all: {errors})"


@pytest.fixture(scope="module", autouse=True)
def nequip_state():
    previous = torch.get_default_dtype()
    init_nequip()
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved
    torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def omol():
    frames = cgdata.load_omol()
    return frames, cgdata.linear_references(frames), type_names_of(cgdata.elements_of(frames))


def make(names, preset="S", model_dtype="float32", **config):
    energy = build_nequip(names, preset, model_dtype=model_dtype)
    return energy, energy_region(energy, cg.CompileConfig(**config))


def batch(frames, names, atoms, seed=0):
    return batch_of(cgdata.Packer(frames, atoms, seed)[0], names)


def float64_copy(energy32, names, preset):
    energy64 = build_nequip(names, preset, model_dtype="float64")
    energy64.load_state_dict(energy32.state_dict())
    return energy64


def native_compile_errors(energy32, names, preset, data, energy64, loss):
    """nequip's own compiled training (compile_mode="compile", Inductor) against
    eager create_graph in float64, with the same loss."""
    want = run_step(energy64, dict(data), outputs=OUT, inputs=("pos",), loss=loss, exact=True)
    native = build_nequip(names, preset, do_derivatives=True, compile_mode="compile")
    native.load_state_dict(energy32.state_dict())
    out = native(dict(data))
    value = loss(out[OUT], {"pos": -out["forces"]})
    value.backward()
    got = {
        "out": out[OUT].detach(),
        "grads": {"pos": -out["forces"].detach()},
        "loss": value.detach(),
        "params": param_grads(native),
    }
    return step_errors(got, want)


def assert_float32(model, energy, names, preset, data):
    """create_graph in float32 is as close to the float64 result as eager
    create_graph or nequip's own compiled path is (Inductor's float32 kernels
    lose precision on some architectures, see the README)."""
    loss = cg.testing.coupled_loss()
    energy64 = float64_copy(energy, names, preset)
    result = check_float32(model, data, energy64, data, outputs=OUT, loss=loss)
    native = native_compile_errors(energy, names, preset, data, energy64, loss)
    result["eager"] = {k: max(v, native[k]) for k, v in result["eager"].items()}
    bad = as_close_as_eager(result, FP32)
    assert not bad, (
        f"create_graph worse than eager and nequip's compile: {bad} ({result}, {native})"
    )


@pytest.mark.parametrize("preset", ["S", "M", "L"])
def test_float32_as_close_as_eager_create_graph(omol, preset):
    frames, _, names = omol
    energy, model = make(names, preset)
    assert_float32(model, energy, names, preset, batch(frames, names, 256))


def test_float64_model_matches_eager_create_graph(omol):
    frames, _, names = omol
    _, model = make(names, model_dtype="float64")
    assert_within(check_create_graph(model, batch(frames, names, 160), outputs=OUT), FP64)


@pytest.mark.parametrize("config", [{"memory_budget": 0.5}, {"checkpoint": BLOCKS}])
def test_memory_knobs_keep_gradients(omol, config):
    frames, _, names = omol
    energy, model = make(names, "M", **config)
    assert_float32(model, energy, names, "M", batch(frames, names, 256))


def test_periodic_metals(omol):
    frames, _, names = omol
    energy, model = make(names)
    assert_float32(model, energy, names, "S", batch_of(cgdata.periodic_metals(4), names))


def test_matches_nequip_force_output(omol):
    """create_graph against nequip's own path: ForceStressOutput (autograd.grad
    with create_graph=True inside the model), same parameters and loss."""
    frames, refs, names = omol
    packed = cgdata.Packer(frames, 256, seed=3)[0]
    data, t = batch_of(packed, names), train.targets_of(packed, refs, dtype=torch.float64)
    native = build_nequip(names, "S", do_derivatives=True)
    energy, model = make(names)
    energy.load_state_dict(native.state_dict())
    out = native(dict(data))
    train.conservative_loss(out[OUT], out["forces"], t).backward()
    want = [p.grad.clone() for p in native.parameters()]
    e, f, _ = train.step(model, dict(data), t, outputs=OUT)
    got = [p.grad.clone() for p in energy.parameters()]
    assert rel_err(e.reshape(-1), out[OUT].reshape(-1)) < 1e-5
    assert rel_err(f, out["forces"]) < 1e-5
    assert cg.testing.grad_errors(got, want)["all"] < 1e-5


@pytest.mark.compiled
def test_one_compile_covers_every_batch(omol):
    frames, refs, names = omol
    _, model = make(names)
    counts = []
    for i, atoms in enumerate([128, 256, 384, 512, 640]):
        packed = cgdata.Packer(frames, atoms, seed=i)[0]
        train.step(
            model,
            batch_of(packed, names),
            train.targets_of(packed, refs, dtype=torch.float64),
            outputs=OUT,
        )
        counts.append(compiled_graph_count())
    assert counts[-1] - counts[1] == 0, counts


def epoch_batches(frames, refs, names, count, atoms):
    packer = cgdata.Packer(frames[:count], atoms, seed=0)
    return [
        (batch_of(packer[i], names), train.targets_of(packer[i], refs, dtype=torch.float64))
        for i in range(len(packer))
    ]


def test_epoch_lockstep_float32(omol):
    frames, refs, names = omol
    energy, model = make(names)
    batches = epoch_batches(frames, refs, names, 256, 1024)
    worst, steps = train.lockstep_epoch(model, energy.parameters(), batches, outputs=OUT)
    assert steps >= 20
    assert_within(
        {"out": worst["energy"], "grad_pos": worst["forces"], "loss": worst["loss"],
         "params": worst["params"], "params_worst_tensor": worst["params_worst_tensor"]},
        FP32,
    )  # fmt: skip


def test_epoch_free_running_float64(omol):
    frames, refs, names = omol
    energy_ref, _ = make(names, model_dtype="float64")
    energy_cg, model_cg = make(names, model_dtype="float64")  # same seed: same parameters
    assert all(torch.equal(a, b) for a, b in zip(energy_ref.parameters(), energy_cg.parameters()))
    batches = epoch_batches(frames, refs, names, 128, 512)
    result = train.free_running_epoch(energy_ref, model_cg, batches, outputs=OUT)
    assert result["steps"] >= 15
    assert result["loss"] < 1e-8, result
    assert result["params"] < 1e-7, result
