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


def pytest_collection_modifyitems(config, items):
    import create_graph as cg

    reasons = {
        "gpu": None if torch.cuda.is_available() else "needs CUDA",
        "compiled": None
        if cg.supports_in_graph_grad()
        else f"torch {torch.__version__} has no in-graph autograd.grad (eager fallback)",
    }
    for item in items:
        for marker, reason in reasons.items():
            if reason is not None and marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))
