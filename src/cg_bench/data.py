"""
Structures for the real models.

- OMol25: a fixed random subset of the local validation set, exported once to
  extxyz with fairchem's reader (``python -m cg_bench.data export-omol`` in
  fairchem's venv). Every model trains on the same molecules.
- Periodic metals: rattled fcc supercells labelled with ASE's EMT potential,
  production-like neighbor counts under periodic boundary conditions.

Batches are packed to a total number of atoms, like fairchem's max-atoms
sampler, from a seeded random order.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("CG_DATA_DIR", os.path.join(ROOT, "data"))
OMOL_VAL = os.environ.get(
    "OMOL_VAL", "/home/tione/notebook/home/zhoushouxin/mlip/datasets/omol25/val"
)
OMOL_FILE = os.path.join(DATA_DIR, "omol25_val_4096.extxyz")


def export_omol(path: str = OMOL_FILE, count: int = 4096, seed: int = 0) -> str:
    """Write ``count`` random OMol25 validation structures (energy, forces,
    charge, spin) to extxyz. Needs fairchem-core."""
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    from fairchem.core.datasets import AseDBDataset

    dataset = AseDBDataset({"src": OMOL_VAL})
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), count))
    frames = []
    for i in indices:
        atoms = dataset.get_atoms(i)
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        clean = atoms.copy()
        clean.info = {
            "charge": int(atoms.info.get("charge", 0)),
            "spin": int(atoms.info.get("spin", 1)),
            "omol_index": i,
        }
        clean.calc = SinglePointCalculator(clean, energy=energy, forces=forces)
        frames.append(clean)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ase.io.write(path, frames, format="extxyz")
    return path


def load_omol(path: str = OMOL_FILE):
    import ase.io

    return ase.io.read(path, ":")


def periodic_metals(count: int = 64, seed: int = 0, repeat: int = 3):
    """Rattled fcc Cu / Al / Ni supercells with EMT energies and forces."""
    import ase.build
    from ase.calculators.emt import EMT
    from ase.calculators.singlepoint import SinglePointCalculator

    lattice = {"Cu": 3.61, "Al": 4.05, "Ni": 3.52}
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(count):
        element = list(lattice)[i % len(lattice)]
        atoms = ase.build.bulk(element, "fcc", a=lattice[element], cubic=True)
        atoms = atoms.repeat((repeat, repeat, repeat))
        atoms.rattle(stdev=0.08, seed=int(rng.integers(1 << 30)))
        atoms.calc = EMT()
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        frames.append(atoms)
    return frames


def linear_references(frames) -> dict[int, float]:
    """Per-element energies fitted by least squares: energies minus these are
    of the order of eV per structure (the loss scale of real training)."""
    elements = sorted({int(z) for f in frames for z in f.numbers})
    column = {z: i for i, z in enumerate(elements)}
    counts = np.zeros((len(frames), len(elements)))
    for row, f in enumerate(frames):
        for z in f.numbers:
            counts[row, column[int(z)]] += 1
    energies = np.array([f.get_potential_energy() for f in frames])
    coef, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    return {z: float(coef[column[z]]) for z in elements}


def referenced_energy(atoms, references: dict[int, float]) -> float:
    return float(atoms.get_potential_energy()) - sum(references[int(z)] for z in atoms.numbers)


def elements_of(frames) -> list[int]:
    return sorted({int(z) for f in frames for z in f.numbers})


class Packer:
    """Batches of about ``total_atoms`` atoms in a seeded random order; one pass
    over the frames is an epoch."""

    def __init__(self, frames, total_atoms: int, seed: int = 0):
        self.frames, self.total_atoms = frames, total_atoms
        order = list(range(len(frames)))
        random.Random(seed).shuffle(order)
        self.batches: list[list[int]] = []
        current, count = [], 0
        for i in order:
            n = len(frames[i])
            if current and count + n > total_atoms:
                self.batches.append(current)
                current, count = [], 0
            current.append(i)
            count += n
        if current:
            self.batches.append(current)

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int):
        return [self.frames[j] for j in self.batches[i]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export-omol"])
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--path", default=OMOL_FILE)
    args = parser.parse_args()
    print(export_omol(args.path, args.count))


if __name__ == "__main__":
    main()
