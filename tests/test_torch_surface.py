"""
Private torch APIs (torch._dynamo, torch._functorch, torch._inductor, torch._C,
Tensor._base, ...) may only be used in compile_config.py: that is the one
place to adapt when a torch minor version changes them.
"""

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "create_graph"
ALLOWED = {"compile_config.py"}


def _private_uses(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("torch."):
            if any(part.startswith("_") for part in node.module.split(".")[1:]):
                found.append(f"{path.name}:{node.lineno} from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(part.startswith("_") for part in alias.name.split(".")[1:]):
                    found.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and not node.attr.startswith("__")
        ):
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            torch_name = isinstance(base, ast.Name) and base.id in ("torch", "F", "nn")
            if torch_name or node.attr in ("_base", "_version", "_cdata"):
                found.append(f"{path.name}:{node.lineno} {ast.unparse(node)}")
    return found


def test_private_torch_apis_only_in_compile_config():
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name not in ALLOWED:
            offenders.extend(_private_uses(path))
    assert not offenders, "private torch APIs outside compile_config.py:\n" + "\n".join(offenders)


def test_compile_config_is_where_they_are():
    assert _private_uses(PACKAGE / "compile_config.py"), "the scan found nothing to check against"
