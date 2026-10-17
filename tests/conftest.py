import importlib.util
import os
import warnings

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(ROOT, ".cache", "inductor"))


@pytest.fixture(autouse=True)
def _fresh_compile_state():
    import create_graph as cg

    cg.reset()
    warnings.filterwarnings("ignore", message=".*TensorFloat32.*")
    yield
    cg.reset()


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def pytest_collection_modifyitems(config, items):
    import create_graph as cg

    reasons = {
        "gpu": None if torch.cuda.is_available() else "needs CUDA",
        "pet": None if _has("metatrain") else "needs metatrain",
        "nequip": None if _has("nequip") else "needs nequip",
        "fairchem": None if _has("fairchem.core") else "needs fairchem-core",
        "compiled": None
        if cg.supports_in_graph_grad()
        else f"torch {torch.__version__} has no in-graph autograd.grad (eager fallback)",
    }
    for item in items:
        for marker, reason in reasons.items():
            if reason is not None and marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))
