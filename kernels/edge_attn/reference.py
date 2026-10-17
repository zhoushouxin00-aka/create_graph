"""The same attention as ordinary differentiable PyTorch ops: the oracle."""

from __future__ import annotations

import torch


def attention(q, k, v, bias, scale: float):
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
    lse = torch.logsumexp(logits, dim=-1)
    return torch.matmul(torch.softmax(logits, dim=-1), v), lse
