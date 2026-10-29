"""Smallest training step: energy and dE/dpos from one call.

    python examples/mlp.py

The model computes the energy only. create_graph returns that energy and the
gradient of the energy with respect to the positions. The loss uses both, and
one backward updates the parameters.
"""

from __future__ import annotations

import torch
from torch import nn

import create_graph as cg


class Network(nn.Module):
    """The parameterized part. Its forward return value is the energy."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, pos):
        return self.layers(pos).sum()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = cg.compile(Network())

    def forward(self, batch):
        # Centering stays eager. Only the network is a compiled region.
        centered = batch["pos"] - batch["pos"].mean(dim=0, keepdim=True)
        return {"energy": self.network(centered)}


def energy_loss(energy, target):
    return (energy - target).square()


def force_loss(force):
    return force.square().mean()


def main():
    torch.manual_seed(0)
    model = Model()
    batch = {"pos": torch.randn(8, 3)}
    target = torch.zeros(())

    out, grads = cg.create_graph(model, batch)
    loss = energy_loss(out["energy"], target) + force_loss(-grads["pos"])
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())
    print(f"energy {out['energy'].item():.6f}  loss {loss.item():.6f}")


if __name__ == "__main__":
    main()
