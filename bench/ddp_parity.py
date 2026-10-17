"""
create_graph under DDP (NCCL) on real models, one or several nodes:

    torchrun --nproc_per_node 8 bench/ddp_parity.py --model pet --steps 4
    torchrun --nnodes 2 --node_rank R --master_addr HOST --master_port PORT \
        --nproc_per_node 8 bench/ddp_parity.py --model nequip --steps 4

Every rank packs its own OMol25 batch. Each step runs eager create_graph
(``exact=True``) and create_graph from the same parameters through the same
DDP wrapper; the reduced gradients are compared, every rank must hold
bitwise-identical reduced gradients, then Adam steps with the reference
gradients (lockstep). float32, TF32 off.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import create_graph as cg
from cg_bench import data as cgdata
from cg_bench import train
from create_graph.testing import grad_errors, rel_err


def build(name: str, frames, device):
    elements = cgdata.elements_of(frames)
    if name == "pet":
        from cg_bench.pet_model import PETEnergy, batch_of, build_pet

        pet = build_pet(elements, "default", device=device)
        return (
            PETEnergy(pet, cg.CompileConfig()),
            lambda fs: batch_of(pet, fs, device),
            "energy",
            torch.float32,
        )
    from cg_bench.nequip_model import batch_of, build_nequip, energy_region, type_names_of

    names = type_names_of(elements)
    energy = build_nequip(names, "S", device=device)
    model = energy_region(energy, cg.CompileConfig())
    return model, lambda fs: batch_of(fs, names, device), "total_energy", torch.float64


def fingerprint(flat: torch.Tensor) -> torch.Tensor:
    """Bit-level summary of a float32 tensor: equal on every rank iff (with
    overwhelming probability) the tensors are bitwise equal."""
    bits = flat.contiguous().view(torch.int32).long()
    index = torch.arange(bits.numel(), device=bits.device, dtype=torch.long)
    return torch.stack([bits.sum(), (bits * (index % 1009 + 1)).sum(), (bits ^ index).sum()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pet", "nequip"], required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    frames = cgdata.load_omol()
    refs = cgdata.linear_references(frames)
    model, batch_of, outputs, target_dtype = build(args.model, frames, device)
    ddp = DistributedDataParallel(model, device_ids=[local_rank])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    packer = cgdata.Packer(frames, args.atoms, seed=rank)

    records = []
    for step in range(args.steps):
        packed = packer[step]
        data, t = batch_of(packed), train.targets_of(packed, refs, device, target_dtype)
        times = {}
        for mode, exact in (("create_graph", False), ("eager", True)):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            train.step(ddp, data, t, outputs=outputs, exact=exact)
            torch.cuda.synchronize()
            times[mode] = 1e3 * (time.perf_counter() - t0)
            grads = [p.grad.detach().clone() for p in params]
            if mode == "create_graph":
                generated = grads
        flat = torch.cat([g.flatten() for g in grads]).float()
        prints = [torch.zeros(3, dtype=torch.long, device=device) for _ in range(world)]
        dist.all_gather(prints, fingerprint(flat))
        identical = all(torch.equal(prints[0], p) for p in prints[1:])
        errors = grad_errors(generated, grads)
        record = {
            "step": step,
            "world": world,
            "atoms_rank0": sum(len(f) for f in packed),
            "ranks_bitwise_identical": identical,
            "params_rel_err": errors["all"],
            "params_worst_tensor": errors["worst_tensor"],
            "ms_rank0": times,
        }
        flat_gen = torch.cat([g.flatten() for g in generated])
        record["grad_norm_rel"] = rel_err(flat_gen.norm(), flat.norm())
        records.append(record)
        optimizer.step()  # the reference gradients are in .grad
        if rank == 0:
            print(json.dumps(record), flush=True)
    ok = all(r["ranks_bitwise_identical"] and r["params_rel_err"] < 1e-5 for r in records)
    if rank == 0:
        summary = {
            "model": args.model,
            "world": world,
            "nodes": int(os.environ.get("GROUP_WORLD_SIZE", 1)),
            "ok": ok,
        }
        print(json.dumps(summary), flush=True)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump({"summary": summary, "steps": records}, fh, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
