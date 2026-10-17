"""
Structures of region inputs and outputs: tuples, lists, namedtuples, dicts and
dataclasses are traversed, everything else is a leaf. Specs are hashable (they
key compiled functions) and the same code runs inside Dynamo to rebuild a
structure from its leaves.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor

_LEAF_TYPES = (Tensor, int, float, bool, complex, str, bytes, torch.dtype, torch.device)


def _flatten(obj, leaves: list):
    if obj is None or isinstance(obj, _LEAF_TYPES):
        leaves.append(obj)
        return None
    kind = type(obj)
    if kind is tuple or kind is list:
        return ("seq", kind, tuple(_flatten(x, leaves) for x in obj))
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return ("seq", kind, tuple(_flatten(x, leaves) for x in obj))
    if kind is dict or kind is OrderedDict:
        keys = tuple(obj)
        return ("dict", kind, keys, tuple(_flatten(obj[k], leaves) for k in keys))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = dataclasses.fields(obj)
        if all(f.init for f in fields):
            names = tuple(f.name for f in fields)
            return (
                "dataclass",
                kind,
                names,
                tuple(_flatten(getattr(obj, n), leaves) for n in names),
            )
    leaves.append(obj)
    return None


def flatten(obj) -> tuple[list, Any]:
    leaves: list = []
    spec = _flatten(obj, leaves)
    return leaves, spec


def _unflatten(spec, leaves):
    if spec is None:
        return next(leaves)
    if spec[0] == "seq":
        _, kind, children = spec
        values = [_unflatten(c, leaves) for c in children]
        if kind is tuple or kind is list:
            return kind(values)
        return kind(*values)
    if spec[0] == "dict":
        _, kind, keys, children = spec
        values = [_unflatten(c, leaves) for c in children]
        return kind(zip(keys, values, strict=True))
    _, kind, names, children = spec
    values = [_unflatten(c, leaves) for c in children]
    return kind(**dict(zip(names, values, strict=True)))


def unflatten(spec, leaves) -> Any:
    return _unflatten(spec, iter(leaves))


def is_float(x) -> bool:
    return isinstance(x, Tensor) and (x.is_floating_point() or x.is_complex())


def tree_map_tensors(fn, obj):
    leaves, spec = flatten(obj)
    return unflatten(spec, [fn(x) if isinstance(x, Tensor) else x for x in leaves])
