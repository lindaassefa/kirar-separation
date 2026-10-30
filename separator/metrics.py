"""Scale-invariant SDR and correlation for estimated stems."""

from __future__ import annotations

import torch


def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR in dB. estimate and target are shaped [batch, time] or [time]."""
    estimate = estimate.reshape(estimate.shape[0], -1) if estimate.ndim > 1 else estimate.reshape(1, -1)
    target = target.reshape(target.shape[0], -1) if target.ndim > 1 else target.reshape(1, -1)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target * target).sum(dim=-1, keepdim=True).clamp_min(eps)
    projection = dot * target / target_energy
    noise = estimate - projection
    ratio = (projection * projection).sum(dim=-1) / (noise * noise).sum(dim=-1).clamp_min(eps)
    return 10 * torch.log10(ratio.clamp_min(eps))


def correlation(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> float:
    left = left.reshape(-1).float()
    right = right.reshape(-1).float()
    left = left - left.mean()
    right = right - right.mean()
    value = (left * right).sum() / (left.norm() * right.norm()).clamp_min(eps)
    return float(value)
