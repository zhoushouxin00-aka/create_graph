"""
Attention ``out = softmax(q k^T * scale + bias) v`` registered with
``cg.custom_op`` as three kernels:

- forward1 keeps only the row logsumexp ``lse``, not the score matrix;
- backward1 rebuilds the probabilities and returns the gradients of q, k, v
  and the additive bias (summed down to the bias's broadcast shape);
- backward2 is the analytic derivative of backward1 w.r.t. q, k, v, the bias
  and grad_out. ``lse`` is a statistic: its dependence on q, k and the bias is
  inside backward1 and backward2.

q, k, v are (..., S, D); the bias broadcasts to the scores (..., S, S). Rows
whose keys are all -inf (padded atoms) get zero probabilities. The kernels are
plain PyTorch, so the example is about the registration; a Triton version
replaces the bodies, the registration stays.
"""

from __future__ import annotations

import torch
from torch import Tensor

import create_graph as cg


def _scores(q, k, bias, scale: float):
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
    finite = torch.isfinite(logits)
    safe = torch.where(finite, logits, torch.full_like(logits, -torch.inf))
    lse = torch.logsumexp(safe, dim=-1)
    row_ok = finite.any(dim=-1, keepdim=True)
    probs = torch.exp(safe - torch.where(torch.isfinite(lse), lse, 0.0).unsqueeze(-1))
    probs = torch.where(finite & row_ok, probs, torch.zeros_like(probs))
    return probs, lse


def _reduce(grad: Tensor, like: Tensor) -> Tensor:
    """Sum a score-shaped tensor down to the broadcast shape of ``like``."""
    while grad.ndim > like.ndim:
        grad = grad.sum(dim=0)
    for dim, size in enumerate(like.shape):
        if size == 1 and grad.shape[dim] != 1:
            grad = grad.sum(dim=dim, keepdim=True)
    return grad


def _first(q, k, v, bias, grad_out, scale):
    probs, _ = _scores(q, k, bias, scale)
    grad_v = torch.matmul(probs.transpose(-2, -1), grad_out)
    grad_probs = torch.matmul(grad_out, v.transpose(-2, -1))
    grad_logits = probs * (grad_probs - (probs * grad_probs).sum(dim=-1, keepdim=True))
    grad_q = torch.matmul(grad_logits, k) * scale
    grad_k = torch.matmul(grad_logits.transpose(-2, -1), q) * scale
    return grad_q, grad_k, grad_v, _reduce(grad_logits, bias), probs, grad_probs, grad_logits


@cg.custom_op("cg_kernels::edge_attn")
class EdgeAttention:
    outputs = ("out", "lse")
    save = ("q", "k", "v", "bias", "lse")
    nondiff = ("lse",)

    @staticmethod
    def forward1(
        q: Tensor, k: Tensor, v: Tensor, bias: Tensor, scale: float
    ) -> tuple[Tensor, Tensor]:
        probs, lse = _scores(q, k, bias, scale)
        return torch.matmul(probs, v), lse

    @staticmethod
    def fake(q, k, v, bias, scale):
        return q.new_empty((*q.shape[:-1], v.shape[-1])), q.new_empty(q.shape[:-1])

    @staticmethod
    def backward1(q, k, v, bias, lse, grad_out, scale):
        return _first(q, k, v, bias, grad_out, scale)[:4]

    @staticmethod
    def backward2(q, k, v, bias, lse, grad_out, scale, hq, hk, hv, hb):
        _, _, _, _, probs, gp, grad_logits = _first(q, k, v, bias, grad_out, scale)
        # grad_q = scale * grad_logits @ k, grad_k = scale * grad_logits^T @ q,
        # grad_bias = reduce(grad_logits): cotangent of grad_logits
        d_logits = scale * (
            torch.matmul(hq, k.transpose(-2, -1)) + torch.matmul(q, hk.transpose(-2, -1))
        )
        d_logits = d_logits + hb
        grad_q = scale * torch.matmul(grad_logits, hk)
        grad_k = scale * torch.matmul(grad_logits.transpose(-2, -1), hq)
        # grad_logits = probs * (gp - rowsum(probs * gp))
        centered = gp - (probs * gp).sum(dim=-1, keepdim=True)
        d_probs = d_logits * centered
        d_gp = d_logits * probs
        d_rowsum = -(d_logits * probs).sum(dim=-1, keepdim=True)
        d_probs = d_probs + d_rowsum * gp
        d_gp = d_gp + d_rowsum * probs
        # gp = grad_out @ v^T, grad_v = probs^T @ grad_out
        grad_out_cot = torch.matmul(d_gp, v) + torch.matmul(probs, hv)
        grad_v = torch.matmul(d_gp.transpose(-2, -1), grad_out)
        d_probs = d_probs + torch.matmul(grad_out, hv.transpose(-2, -1))
        # probs = softmax(logits)
        d_scores = probs * (d_probs - (probs * d_probs).sum(dim=-1, keepdim=True))
        grad_q = grad_q + torch.matmul(d_scores, k) * scale
        grad_k = grad_k + torch.matmul(d_scores.transpose(-2, -1), q) * scale
        return grad_q, grad_k, grad_v, _reduce(d_scores, bias), grad_out_cot


def edge_attention(q, k, v, bias, scale: float) -> Tensor:
    return EdgeAttention(q, k, v, bias, scale)[0]
