"""
PET (metatrain) for create_graph.

metatrain's ``PET.forward`` reads ``metatomic.System`` objects into plain
tensors and runs ``PETBackend``: ``preprocess`` (edge vectors from positions,
cutoff factors, neighbor padding with a data-dependent width), then
``calculate_features`` and ``predict``. The preprocessing stays eager (create_graph
chains through it with autograd); features + readout + the per-system sum are
the region. Neighbor lists are computed once per batch, as metatrain's data
pipeline does.

Sizes: metatrain's defaults (d_pet 128, 2 GNN layers x 2 attention layers) and
a PET-MAD-scale model (loaded from a checkpoint when one is given).
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

import create_graph as cg

SIZES = {
    "default": {},
    "large": {
        "d_pet": 256,
        "d_head": 256,
        "d_node": 512,
        "d_feedforward": 512,
        "num_gnn_layers": 3,
    },
}


def build_pet(
    atomic_types, size: str = "default", device="cuda", dtype=torch.float32, checkpoint=None
):
    from metatrain.pet import PET
    from metatrain.utils.architectures import get_default_hypers
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_energy_target_info

    if checkpoint is not None:
        from metatrain.utils.io import load_model

        model = load_model(checkpoint)
    else:
        hypers = dict(get_default_hypers("pet")["model"])
        hypers.update(SIZES[size])
        info = DatasetInfo(
            length_unit="Angstrom",
            atomic_types=list(atomic_types),
            targets={
                "energy": get_energy_target_info(
                    "energy", {"quantity": "energy", "unit": "eV"}, add_position_gradients=True
                )
            },
        )
        torch.manual_seed(0)
        model = PET(hypers, info)
    return model.to(device=device, dtype=dtype).train()


def systems_of(model, frames, device="cuda", dtype=torch.float32):
    from metatomic.torch import System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

    systems = []
    for f in frames:
        system = System(
            types=torch.tensor(f.numbers, dtype=torch.int32),
            positions=torch.tensor(f.positions, dtype=dtype),
            cell=torch.tensor(f.cell[:], dtype=dtype),
            pbc=torch.tensor(f.pbc),
        )
        system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
        systems.append(system.to(device))
    return systems


def batch_of(model, frames, device="cuda", dtype=torch.float32) -> dict:
    """The plain tensors of a batch (neighbor lists included); positions under "pos"."""
    from metatrain.pet.modules.structures import concatenate_structures

    systems = systems_of(model, frames, device, dtype)
    options = model.requested_neighbor_lists()[0]
    positions, centers, neighbors, species, cells, shifts, system_indices, _ = (
        concatenate_structures(systems, options)
    )
    return {
        "pos": positions.detach(),
        "centers": centers,
        "neighbors": neighbors,
        "species": species,
        "cells": cells,
        "cell_shifts": shifts,
        "system_indices": system_indices,
        "num_systems": len(systems),
        "systems": systems,
    }


class PETEnergy(nn.Module):
    """``model(data) -> {"energy": per-system energies}`` on plain tensors. With a
    config, the network is a create_graph region; without, it is eager."""

    def __init__(self, pet, config: cg.CompileConfig | None = None):
        super().__init__()
        self.pet = pet
        self.network = self._network if config is None else cg.compile(self._network, config=config)

    def _network(self, batch_data, cells, system_indices, num_systems: int):
        backend = self.pet.backend
        nodes, edges = backend.calculate_features(batch_data)
        atomic, _, _ = backend.predict(nodes, edges, batch_data, cells, system_indices, ["energy"])
        per_atom = atomic["energy"][0].squeeze(-1)
        return per_atom.new_zeros(num_systems).index_add(0, system_indices, per_atom)

    def forward(self, data):
        backend = self.pet.backend
        batch_data = backend.preprocess(
            data["pos"],
            data["centers"],
            data["neighbors"],
            data["species"],
            data["cells"],
            data["cell_shifts"],
            data["system_indices"],
            self.pet.cutoff_width_adaptive,
        )
        energy = self.network(
            batch_data, data["cells"], data["system_indices"], data["num_systems"]
        )
        return {"energy": energy}


@contextlib.contextmanager
def fused_attention():
    """PET's double-backward-safe attention replaced by the EdgeAttention op
    (forward1 / backward1 / backward2 registered with cg.custom_op)."""
    import metatrain.pet.modules.transformer as transformer

    from kernels.edge_attn import edge_attention

    def attention(q, k, v, attn_mask, temperature: float):
        return edge_attention(q, k, v, attn_mask, 1.0 / (k.size(-1) ** 0.5 * temperature))

    original = transformer.manual_attention
    transformer.manual_attention = attention
    try:
        yield
    finally:
        transformer.manual_attention = original


def native_energy_and_forces(pet, systems, create_graph: bool = True):
    """metatrain's own path: the full PET forward on System objects, forces with
    torch.autograd.grad(create_graph=True) as its trainer does."""
    from metatomic.torch import ModelOutput, System

    positions = [s.positions.detach().clone().requires_grad_(True) for s in systems]
    wrapped = []
    for base, pos in zip(systems, positions, strict=True):
        system = System(types=base.types, positions=pos, cell=base.cell, pbc=base.pbc)
        for options in base.known_neighbor_lists():
            system.add_neighbor_list(options, base.get_neighbor_list(options))
        wrapped.append(system)
    out = pet(wrapped, {"energy": ModelOutput(sample_kind="system")})
    energy = out["energy"].block().values.squeeze(-1)
    grads = torch.autograd.grad(energy.sum(), positions, create_graph=create_graph)
    return energy, -torch.cat(grads)
