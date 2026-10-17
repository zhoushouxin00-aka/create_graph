"""
Smoke run of the real models on GPU, before any benchmark:

  pet     metatrain's own path (full PET forward on System objects, forces with
          create_graph=True); the plain-tensor adapter against it; the
          create_graph region against eager create_graph
  nequip  nequip's own force output, eager and compile_mode="compile"; the
          create_graph region against eager create_graph

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench/smoke_native.py --model pet
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import create_graph as cg
from cg_bench import data as cgdata
from create_graph.testing import grad_errors, param_grads, rel_err


def loss_of(energy, forces, e_ref, f_ref, natoms):
    return (((energy - e_ref) / natoms) ** 2).mean() + ((forces - f_ref) ** 2).mean()


def targets(frames, refs, device, dtype):
    e = torch.tensor(
        [cgdata.referenced_energy(f, refs) for f in frames], device=device, dtype=dtype
    )
    f = torch.cat([torch.tensor(a.get_forces(), dtype=dtype) for a in frames]).to(device)
    n = torch.tensor([len(a) for a in frames], device=device, dtype=dtype)
    return e, f, n


def timed(fn, steps):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t0) / steps


def smoke_pet(frames, refs, args):
    from cg_bench.pet_model import PETEnergy, batch_of, build_pet, native_energy_and_forces

    pet = build_pet(cgdata.elements_of(frames), args.size)
    batch = cgdata.Packer(frames, args.atoms)[0]
    data = batch_of(pet, batch)
    e_ref, f_ref, natoms = targets(batch, refs, "cuda", torch.float32)
    report = {
        "model": "pet",
        "size": args.size,
        "atoms": int(natoms.sum()),
        "params": sum(p.numel() for p in pet.parameters()),
    }

    def native_step():
        pet.zero_grad(set_to_none=True)
        energy, forces = native_energy_and_forces(pet, data["systems"])
        loss_of(energy, forces, e_ref, f_ref, natoms).backward()
        return energy.detach(), forces.detach()

    energy_n, forces_n = native_step()
    grads_n = param_grads(pet)
    report["native_eager_ms"] = timed(native_step, args.steps)

    eager = PETEnergy(pet)
    out, grads = cg.create_graph(eager, data, exact=True)
    report["adapter_vs_native"] = {
        "energy": rel_err(out["energy"], energy_n),
        "forces": rel_err(-grads["pos"], forces_n),
    }
    region = PETEnergy(pet, cg.CompileConfig())

    def cg_step():
        pet.zero_grad(set_to_none=True)
        out, grads = cg.create_graph(region, data)
        loss_of(out["energy"], -grads["pos"], e_ref, f_ref, natoms).backward()
        return out["energy"].detach(), -grads["pos"].detach()

    energy_c, forces_c = cg_step()
    report["create_graph_vs_native"] = {
        "energy": rel_err(energy_c, energy_n),
        "forces": rel_err(forces_c, forces_n),
        "params": grad_errors(param_grads(pet), grads_n)["all"],
    }
    report["create_graph_ms"] = timed(cg_step, args.steps)
    return report


def smoke_nequip(frames, refs, args):
    from cg_bench.nequip_model import batch_of, build_nequip, energy_region, type_names_of

    names = type_names_of(cgdata.elements_of(frames))
    batch = cgdata.Packer(frames, args.atoms)[0]
    native = build_nequip(names, args.size, do_derivatives=True)
    data = batch_of(batch, names)
    e_ref, f_ref, natoms = targets(batch, refs, "cuda", torch.float64)
    report = {
        "model": "nequip",
        "preset": args.size,
        "atoms": int(natoms.sum()),
        "params": sum(p.numel() for p in native.parameters()),
    }

    def native_step(model):
        model.zero_grad(set_to_none=True)
        out = model(dict(data))
        energy, forces = out["total_energy"].squeeze(-1), out["forces"]
        loss_of(energy, forces, e_ref, f_ref, natoms).backward()
        return energy.detach(), forces.detach()

    energy_n, forces_n = native_step(native)
    grads_n = param_grads(native)
    report["native_eager_ms"] = timed(lambda: native_step(native), args.steps)

    compiled = build_nequip(names, args.size, do_derivatives=True, compile_mode="compile")
    compiled.load_state_dict(native.state_dict())
    energy_k, forces_k = native_step(compiled)
    report["nequip_compile_vs_eager"] = {
        "energy": rel_err(energy_k, energy_n),
        "forces": rel_err(forces_k, forces_n),
        "params": grad_errors(param_grads(compiled), grads_n)["all"],
    }
    report["nequip_compile_ms"] = timed(lambda: native_step(compiled), args.steps)

    energy_model = build_nequip(names, args.size, do_derivatives=False)
    energy_model.load_state_dict(native.state_dict())
    region = energy_region(energy_model, cg.CompileConfig())

    def cg_step():
        energy_model.zero_grad(set_to_none=True)
        out, grads = cg.create_graph(region, dict(data), outputs="total_energy")
        energy = out["total_energy"].squeeze(-1)
        loss_of(energy, -grads["pos"], e_ref, f_ref, natoms).backward()
        return energy.detach(), -grads["pos"].detach()

    energy_c, forces_c = cg_step()
    report["create_graph_vs_native"] = {
        "energy": rel_err(energy_c, energy_n),
        "forces": rel_err(forces_c, forces_n),
        "params": grad_errors(param_grads(energy_model), grads_n)["all"],
    }
    report["create_graph_ms"] = timed(cg_step, args.steps)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pet", "nequip"], required=True)
    parser.add_argument("--size", default=None)
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    args.size = args.size or ("default" if args.model == "pet" else "S")
    frames = cgdata.load_omol()
    refs = cgdata.linear_references(frames)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = (
        smoke_pet(frames, refs, args) if args.model == "pet" else smoke_nequip(frames, refs, args)
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
