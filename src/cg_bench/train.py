"""
Training steps shared by the real-model tests and benchmarks: referenced
energies and forces as targets, the usual conservative loss (per-atom energy
MSE + force MSE), one step through create_graph or eager create_graph.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

import create_graph as cg
from cg_bench import data as cgdata
from create_graph.region import select_output


@dataclasses.dataclass
class Targets:
    energy: Tensor
    forces: Tensor
    natoms: Tensor


def targets_of(frames, refs, device="cuda", dtype=torch.float32) -> Targets:
    energy = torch.tensor([cgdata.referenced_energy(f, refs) for f in frames], dtype=dtype)
    forces = torch.cat([torch.tensor(f.get_forces(), dtype=dtype) for f in frames])
    natoms = torch.tensor([len(f) for f in frames], dtype=dtype)
    return Targets(energy.to(device), forces.to(device), natoms.to(device))


def conservative_loss(energy: Tensor, forces: Tensor, t: Targets, force_weight: float = 10.0):
    energy = energy.reshape(-1).to(t.energy.dtype)
    forces = forces.to(t.forces.dtype)
    e_term = (((energy - t.energy) / t.natoms) ** 2).mean()
    f_term = ((forces - t.forces) ** 2).mean()
    return e_term + force_weight * f_term


def step(model, data, t: Targets, *, outputs="energy", exact=False, backward=True):
    """create_graph (or eager create_graph with exact=True), loss, backward."""
    out, grads = cg.create_graph(model, data, outputs=outputs, exact=exact)
    energy = select_output(out, outputs)
    forces = -grads["pos"]
    loss = conservative_loss(energy, forces, t)
    if backward:
        loss.backward()
    return energy.detach(), forces.detach(), loss.detach()


def _grads(parameters) -> list[Tensor]:
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]


def lockstep_epoch(model, parameters, batches, *, outputs="energy", lr=1e-3):
    """
    One epoch in lockstep: for every batch, create_graph and eager create_graph
    from the same parameters, compared; then Adam steps with the reference
    gradients, so both stay on the same trajectory. Returns the worst error of
    every quantity over the epoch and the number of steps.
    """
    from create_graph.testing import grad_errors, rel_err

    parameters = list(parameters)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    worst: dict[str, float] = {}
    for data, t in batches:
        optimizer.zero_grad(set_to_none=True)
        e1, f1, l1 = step(model, data, t, outputs=outputs)
        g1 = _grads(parameters)
        optimizer.zero_grad(set_to_none=True)
        e0, f0, l0 = step(model, data, t, outputs=outputs, exact=True)
        errors = {"energy": rel_err(e1, e0), "forces": rel_err(f1, f0), "loss": rel_err(l1, l0)}
        params = grad_errors(g1, _grads(parameters))
        errors["params"], errors["params_worst_tensor"] = params["all"], params["worst_tensor"]
        for k, v in errors.items():
            worst[k] = max(worst.get(k, 0.0), v)
        optimizer.step()
    return worst, len(batches)


def free_running_epoch(model_ref, model_cg, batches, *, outputs="energy", lr=1e-3):
    """Two copies trained independently for one epoch, eager create_graph and
    create_graph: per-step losses and the final parameters."""
    from create_graph.testing import rel_err

    opt_ref = torch.optim.Adam(model_ref.parameters(), lr=lr)
    opt_cg = torch.optim.Adam(model_cg.parameters(), lr=lr)
    losses = []
    for data, t in batches:
        opt_ref.zero_grad(set_to_none=True)
        opt_cg.zero_grad(set_to_none=True)
        _, _, l_ref = step(model_ref, data, t, outputs=outputs, exact=True)
        _, _, l_cg = step(model_cg, data, t, outputs=outputs)
        opt_ref.step()
        opt_cg.step()
        losses.append((float(l_ref), float(l_cg)))
    a = torch.cat([p.detach().flatten() for p in model_ref.parameters()])
    b = torch.cat([p.detach().flatten() for p in model_cg.parameters()])
    worst_loss = max(abs(x - y) / max(abs(x), 1e-300) for x, y in losses)
    return {"loss": worst_loss, "params": rel_err(b, a), "steps": len(losses), "losses": losses}
