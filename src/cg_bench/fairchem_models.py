"""
fairchem models for create_graph (fairchem's venv, create_graph installed with
``uv pip install --no-deps -e``): AllScAIP (non-equivariant) and UMA
(eSCN-MD, equivariant), built from the training yamls, fed with real OMol
batches.

- AllScAIP: the preprocessing (soft-kNN radius graph, SDPA backend switches)
  stays eager; the backbone network and the energy head are ONE region (they
  are two compiled functions in fairchem, but create_graph needs the region to
  produce the energy).
- UMA: the graph is built outside (no gradient), the backbone computes edge
  vectors from positions itself, backbone + energy readout are one region over
  the AtomicData batch. MoLE splits edges by system with data-dependent sizes,
  so experts are off (``num_moe_experts=0``). Its Safeacos / Safeatan2
  torch.autograd.Functions are replaced by cg.custom_op operators.
"""

from __future__ import annotations

import contextlib
import os
import random
import tempfile

import torch
from torch import Tensor, nn

import create_graph as cg

MLIP = "/home/tione/notebook/home/zhoushouxin/mlip"
FAIRCHEM_ROOT = os.environ.get("FAIRCHEM_ROOT", os.path.join(MLIP, "fairchem"))
CONFIGS = {
    "allscaip": "allscaip-scripts/omol_4m_8gpu_atoms1024_cons.yaml",
    "uma": "uma-scripts/uma_s_8gpu_atoms300.yaml",
}
PATHS = [
    f"cluster.omol_all_path={MLIP}/datasets/omol25",
    f"cluster.omol_4M_train_path={MLIP}/datasets/omol25/train_4M",
    f"cluster.data_root_dir={MLIP}/datasets/omol25",
]
ALLSCAIP_SIZES = {
    "h128_l2": ["backbone.hidden_size=128", "backbone.num_layers=2", "backbone.atten_num_heads=4"],
    "h256_l4": ["backbone.hidden_size=256", "backbone.num_layers=4", "backbone.atten_num_heads=8"],
    "h512_l6": ["backbone.hidden_size=512", "backbone.num_layers=6", "backbone.atten_num_heads=8"],
    "h640_l10": [],
}
UMA_SIZES = {"s": ["backbone=K4L2"], "m": ["backbone=K10L4"], "l": ["backbone=K16L6"]}
UMA_BASE = ["cpu_graph=False", "otf_graph=False", "num_moe_experts=0"]
ALLSCAIP_BLOCKS = r"backbone\.transformer_blocks\.\d+"


def build(name: str, size: str):
    import hydra
    from fairchem.core._cli import get_hydra_config_from_yaml

    sizes = ALLSCAIP_SIZES if name == "allscaip" else UMA_SIZES
    extra = PATHS if name == "allscaip" else UMA_BASE
    cfg = get_hydra_config_from_yaml(
        os.path.join(FAIRCHEM_ROOT, CONFIGS[name]),
        [
            "cluster.ranks_per_node=1",
            "cluster.dataloader_workers=0",
            f"cluster.run_dir={tempfile.mkdtemp(prefix='cg_fairchem_')}",
            f"job.run_name=cg_{name}",
            *extra,
            *sizes[size],
        ],
    )
    unit = cfg.runner.train_eval_unit
    model = hydra.utils.instantiate(unit.model)
    tasks = hydra.utils.instantiate(unit.tasks)
    dataset = hydra.utils.instantiate(cfg.runner.train_dataloader.dataset)
    collate = hydra.utils.instantiate(cfg.runner.train_dataloader.collate_fn)
    return model, tasks, dataset, collate


def head_of(model):
    (head,) = model.output_heads.values()
    return getattr(head, "head", head)


class Batches:
    """Real OMol batches of about ``total_atoms`` atoms from a seeded sample of
    the training set, with normalized energy and force targets."""

    def __init__(self, model, tasks, dataset, collate, device="cuda", graph=False):
        self.dataset, self.collate, self.device = dataset, collate, device
        self.backbone = model.backbone if graph else None
        self.energy_task = next(t for t in tasks if t.property == "energy")
        self.force_task = next(t for t in tasks if t.property == "forces")
        for task in (self.energy_task, self.force_task):
            if task.element_references is not None:
                task.element_references.to(device)

    def __call__(self, total_atoms: int, seed: int, dtype=torch.float32):
        rng = random.Random(seed)
        samples, count = [], 0
        while count < total_atoms:
            sample = self.dataset[rng.randrange(len(self.dataset))]
            n = int(sample.natoms.sum())
            if count + n > total_atoms and samples:
                if total_atoms - count < 4:
                    break
                continue
            samples.append(sample)
            count += n
        batch = self.collate(samples).to(self.device)
        _prepare(batch)
        if dtype != torch.float32:
            batch["pos"] = batch["pos"].to(dtype)
            if getattr(batch, "cell", None) is not None:
                batch["cell"] = batch["cell"].to(dtype)
        if self.backbone is not None:
            uma_graph(self.backbone, batch)
        return batch

    def targets(self, batch, dtype=torch.float32):
        from cg_bench.train import Targets

        e_task, f_task = self.energy_task, self.force_task
        with torch.no_grad():
            energy = batch[e_task.name].clone()
            if e_task.element_references is not None:
                energy = e_task.element_references.apply_refs(batch, energy)
            energy = e_task.normalizer.norm(energy).view(-1).to(dtype)
            forces = f_task.normalizer.norm(batch[f_task.name].clone()).view(-1, 3).to(dtype)
        return Targets(energy, forces, batch.natoms.to(dtype))


def uma_graph(backbone, batch) -> None:
    from fairchem.core.graph.compute import generate_graph

    with torch.no_grad():
        graph = generate_graph(
            batch,
            cutoff=backbone.cutoff,
            max_neighbors=backbone.max_neighbors,
            enforce_max_neighbors_strictly=backbone.enforce_max_neighbors_strictly,
            radius_pbc_version=backbone.radius_pbc_version,
            pbc=batch["pbc"],
        )
    batch["edge_index"] = graph["edge_index"]
    batch["cell_offsets"] = graph["cell_offsets"]
    batch["nedges"] = graph["neighbors"]


def _prepare(data) -> None:
    data["atomic_numbers"] = data["atomic_numbers"].long()
    data["atomic_numbers_full"] = data["atomic_numbers"]
    data["batch_full"] = data["batch"]


@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    """AllScAIP's radius graph builds image offsets in the default dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class AllScAIPEnergy(nn.Module):
    """Eager preprocessing, then backbone network + energy head (a region when
    ``config`` is given)."""

    def __init__(self, model, config: cg.CompileConfig | None = None):
        super().__init__()
        self.backbone, self.head = model.backbone, head_of(model)
        self.network = self._network if config is None else cg.compile(self._network, config=config)

    def _network(self, x):
        return self.head.compiled_forward(self.backbone.compiled_forward(x)).view(-1)

    def forward(self, data):
        _prepare(data)
        pos = data["pos"]
        with default_dtype(pos.dtype), torch.autocast(device_type=pos.device.type, enabled=False):
            x = self.backbone.data_preprocess(data)
            return {"energy": self.network(x)}


class UMAEnergy(nn.Module):
    """Backbone + energy readout over the AtomicData batch (a region when
    ``config`` is given)."""

    def __init__(self, model, config: cg.CompileConfig | None = None):
        super().__init__()
        self.backbone, self.head = model.backbone, head_of(model)
        self.network = self._network if config is None else cg.compile(self._network, config=config)

    def _network(self, data):
        emb = self.backbone(data)
        node_energy = self.head.energy_block(emb["node_embedding"][:, 0, :]).view(-1)
        energy = node_energy.new_zeros(data["natoms"].shape[0]).index_add(
            0, data["batch"], node_energy
        )
        if self.head.reduce == "mean":
            energy = energy / data["natoms"]
        return energy

    def forward(self, data):
        return {"energy": self.network(data)}


# ---------------------------------------------------------------------------
# UMA's safe acos / atan2 as fused operators
# ---------------------------------------------------------------------------

EPS = 1e-7


@cg.custom_op("cg_fairchem::safe_acos")
class SafeAcos:
    """acos with fairchem's clamped derivative -1 / max(sqrt(1 - clamp(x)^2), eps)."""

    outputs = ("y",)
    save = ("x",)

    @staticmethod
    def forward1(x: Tensor) -> Tensor:
        return torch.acos(x)

    @staticmethod
    def fake(x):
        return torch.empty_like(x)

    @staticmethod
    def backward1(x, grad_y):
        clamped = x.clamp(-1 + EPS, 1 - EPS)
        return -grad_y / torch.sqrt(1 - clamped.pow(2)).clamp(min=EPS)

    @staticmethod
    def backward2(x, grad_y, gg_x):
        # fairchem's Safeacos saves the clamped x computed inside forward, which
        # autograd treats as a constant: eager training does not differentiate its
        # backward w.r.t. x, so neither does this operator
        denom = torch.sqrt(1 - x.clamp(-1 + EPS, 1 - EPS).pow(2)).clamp(min=EPS)
        return torch.zeros_like(x), -gg_x / denom


@cg.custom_op("cg_fairchem::safe_atan2")
class SafeAtan2:
    """atan2 with fairchem's clamped derivative, denominator max(x^2 + y^2, eps)."""

    outputs = ("angle",)
    save = ("y", "x")

    @staticmethod
    def forward1(y: Tensor, x: Tensor) -> Tensor:
        return torch.atan2(y, x)

    @staticmethod
    def fake(y, x):
        return torch.empty_like(y)

    @staticmethod
    def backward1(y, x, grad_angle):
        denom = (x.pow(2) + y.pow(2)).clamp(min=EPS)
        return x / denom * grad_angle, -y / denom * grad_angle

    @staticmethod
    def backward2(y, x, g, gg_y, gg_x):
        r2 = x.pow(2) + y.pow(2)
        denom = r2.clamp(min=EPS)
        live = (r2 > EPS).to(x.dtype)
        dy, dx = 2 * y * live, 2 * x * live  # derivatives of the clamped denominator
        d2 = denom.pow(2)
        grad_y = gg_y * (-x * g * dy / d2) + gg_x * (-g / denom + y * g * dy / d2)
        grad_x = gg_y * (g / denom - x * g * dx / d2) + gg_x * (y * g * dx / d2)
        grad_g = gg_y * x / denom - gg_x * y / denom
        return grad_y, grad_x, grad_g


class _Apply:
    def __init__(self, op):
        self.apply = op


@contextlib.contextmanager
def uma_fused_rotation_ops():
    """UMA's Safeacos / Safeatan2 autograd.Functions replaced by the operators above."""
    from fairchem.core.models.uma.common import rotation

    saved = rotation.Safeacos, rotation.Safeatan2
    rotation.Safeacos, rotation.Safeatan2 = _Apply(SafeAcos), _Apply(SafeAtan2)
    try:
        yield
    finally:
        rotation.Safeacos, rotation.Safeatan2 = saved


@contextlib.contextmanager
def fused_attention():
    """SDPA calls with an additive float mask (AllScAIP's attention) go through
    the EdgeAttention operator (forward1 / backward1 / backward2)."""
    import torch.nn.functional as F

    from kernels.edge_attn import edge_attention

    original = F.scaled_dot_product_attention

    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if attn_mask is None or attn_mask.dtype == torch.bool or dropout_p or is_causal or kw:
            return original(query, key, value, attn_mask, dropout_p, is_causal, scale, **kw)
        scale = 1.0 / query.size(-1) ** 0.5 if scale is None else scale
        return edge_attention(query, key, value, attn_mask.to(query.dtype), scale)

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def float64_copy(module: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(module).double()


# ---------------------------------------------------------------------------
# baselines from fairchem branches (benchmarks only)
# ---------------------------------------------------------------------------


def explicit_setup(model, *, budget: float = 1.0, checkpoint: bool = False) -> None:
    """The hand-written adjoint of feature/explicit_create_graph (that worktree
    must be first on PYTHONPATH)."""
    cfg = model.backbone.global_cfg
    cfg.explicit_force = True
    cfg.explicit_force_compile = True
    cfg.explicit_force_memory_budget = budget
    cfg.explicit_force_checkpointing = checkpoint
    model.backbone.explicit_network_fn = None


def explicit_energy_and_forces(model, data):
    from fairchem.core.models.allscaip.utils.explicit_force import (
        explicit_energy_and_forces as hand_written,
    )

    data["pos"] = data["pos"].detach().requires_grad_(True)
    _prepare(data)
    with torch.autocast(device_type=data["pos"].device.type, enabled=False):
        x = model.backbone.data_preprocess(data)
    energy, forces = hand_written(model.backbone, head_of(model), data, x)
    return energy.view(-1), forces


def two_f_two_b_setup(model, *, budget: float = 1.0, checkpoint: bool = False) -> None:
    """The hand-written 2F2B of feature/2f2b (a checkout of it, e.g. the
    feature/create_graph worktree, first on PYTHONPATH)."""
    cfg = model.backbone.global_cfg
    cfg.use_dual_compile = True
    cfg.dual_activation_memory_budget = budget
    cfg.dual_activation_checkpointing = checkpoint
    model.backbone.dual_forward_fn = None


def two_f_two_b_step(model, tasks, data) -> Tensor:
    from fairchem.core.units.mlip_unit.conservative_2f2b import two_pass_forward
    from fairchem.core.units.mlip_unit.mlip_unit import compute_loss

    two_pass = two_pass_forward(model, tasks, data, compute_loss)
    two_pass.surrogate.backward()
    return sum(two_pass.loss_dict.values())
