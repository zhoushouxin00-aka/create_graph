"""
NequIP (nequip) for create_graph.

The neighbor list is computed in the data pipeline, the model computes edge
vectors from positions itself, so the energy model (``do_derivatives=False``)
is one region and ``inputs=("pos",)`` is differentiated directly.

Sizes: nequip's foundation-model presets ("S", "M", "L"). Baselines: the same
model with nequip's own force output (``do_derivatives=True``) in its
``compile_mode="eager"`` and ``compile_mode="compile"`` (nequip traces the
force autograd.grad with make_fx and compiles that graph).

nequip keeps data in float64 and sets torch's default dtype accordingly
(``set_global_state``); models are float32 unless asked otherwise.
"""

from __future__ import annotations

import create_graph as cg

PRESETS = ("S", "M", "L")
R_MAX = 5.0


def init_nequip():
    from nequip.utils.global_state import set_global_state

    set_global_state(allow_tf32=False)
    try:
        import e3nn

        e3nn.set_optimization_defaults(jit_script_fx=False)
    except AttributeError:
        pass


def type_names_of(atomic_numbers) -> list[str]:
    import ase.data

    return [ase.data.chemical_symbols[z] for z in atomic_numbers]


def build_nequip(
    type_names,
    preset: str = "S",
    *,
    do_derivatives: bool = False,
    compile_mode: str = "eager",
    model_dtype: str = "float32",
    device="cuda",
    avg_num_neighbors: float = 20.0,
):
    from nequip.model import PresetNequIPGNNModel

    init_nequip()
    model = PresetNequIPGNNModel(
        preset=preset,
        seed=0,
        model_dtype=model_dtype,
        r_max=R_MAX,
        type_names=list(type_names),
        avg_num_neighbors=avg_num_neighbors,
        per_type_energy_scales=1.0,
        per_type_energy_shifts=0.0,
        do_derivatives=do_derivatives,
        compile_mode=compile_mode,
    )
    return model.to(device).train()


def batch_of(frames, type_names, device="cuda") -> dict:
    from nequip.data import AtomicDataDict, compute_neighborlist_, from_ase
    from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper

    mapper = ChemicalSpeciesToAtomTypeMapper(model_type_names=list(type_names))
    datas = []
    for f in frames:
        plain = f.copy()
        plain.calc = None
        data = mapper(from_ase(plain))
        datas.append(compute_neighborlist_(data, R_MAX))
    batch = AtomicDataDict.batched_from_list(datas)
    return {k: v.to(device) for k, v in batch.items()}


def energy_region(energy_model, config: cg.CompileConfig):
    """The whole energy model as one region (outputs="total_energy")."""
    return cg.compile(energy_model, config=config, outputs="total_energy")
