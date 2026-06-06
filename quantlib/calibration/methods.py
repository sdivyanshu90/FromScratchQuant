"""
Module: quantlib.calibration.methods

The four calibration algorithms that turn collected statistics into a
(scale, zero_point) pair: MinMax, Percentile, KL-Entropy (TensorRT-style),
and MSE grid search.

Mathematical Background:
    MinMax      : asymmetric affine from observed [min, max].
    Percentile  : clip the histogram CDF to [p_low, p_high] then affine-map.
    KL-Entropy  : pick the clip threshold minimizing KL(P_ref || Q_quant).
    MSE         : grid-search the scale minimizing ||x - dequant(quant(x))||².

References:
    Migacz, 2017 — "8-bit Inference with TensorRT" (KL calibration).

Example:
    >>> import torch
    >>> from quantlib.core.observer import MinMaxObserver
    >>> from quantlib.calibration.methods import calibrate_minmax
    >>> obs = MinMaxObserver(); obs.update(torch.tensor([-1.0, 2.0]))
    >>> scale, zp = calibrate_minmax(obs.compute_stats())
    >>> bool(scale > 0)
    True
"""

from __future__ import annotations

from typing import Literal

import torch

from quantlib.core.observer import ObserverStats
from quantlib.quantizers.utils import (
    INT8_MAX,
    INT8_MIN,
    UINT8_MAX,
    compute_scale_asymmetric,
    compute_scale_symmetric,
)


def calibrate_minmax(stats: ObserverStats) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric affine calibration from the observed min/max (baseline).

    Args:
        stats: Collected statistics (only ``running_min``/``running_max`` used).

    Returns:
        tuple[Tensor, Tensor]: ``(scale, zero_point)``.
    """
    return compute_scale_asymmetric(stats.running_min, stats.running_max)


def calibrate_percentile(
    stats: ObserverStats, p: float = 99.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """Percentile calibration: clip the histogram tails before affine mapping.

    Args:
        stats: Statistics carrying ``histogram`` and ``bin_edges``.
        p: Central percentile to keep (e.g. 99.9 clips the outer 0.1%).

    Returns:
        tuple[Tensor, Tensor]: ``(scale, zero_point)``. Falls back to
        :func:`calibrate_minmax` if the histogram is empty/unavailable.
    """
    hist = stats.histogram
    bin_edges = stats.bin_edges
    if hist is None or bin_edges is None:
        return calibrate_minmax(stats)
    total = hist.sum()
    if bool(total == 0):
        return calibrate_minmax(stats)

    p_low = (1.0 - p / 100.0) / 2.0
    p_high = 1.0 - p_low
    cumsum = hist.float().cumsum(0) / total

    low_hits = (cumsum >= p_low).nonzero(as_tuple=True)[0]
    low_idx = int(low_hits[0].item()) if low_hits.numel() > 0 else 0
    high_hits = (cumsum >= p_high).nonzero(as_tuple=True)[0]
    high_idx = int(high_hits[0].item()) if high_hits.numel() > 0 else len(hist) - 1

    x_min = bin_edges[low_idx]
    x_max = bin_edges[high_idx + 1]  # +1: edges are boundaries, not centers
    return compute_scale_asymmetric(x_min, x_max)


def _merge_bins_to_levels(active: torch.Tensor, num_levels: int) -> torch.Tensor:
    """Merge ``active`` histogram bins into ``num_levels`` by summing chunks."""
    length = active.shape[0]
    if length <= num_levels:
        return active.clone()
    chunk_id = (torch.arange(length, device=active.device) * num_levels) // length
    merged = torch.zeros(num_levels, device=active.device)
    merged.scatter_add_(0, chunk_id, active)
    return merged


def _expand_levels_to_bins(quant: torch.Tensor, length: int) -> torch.Tensor:
    """Expand ``num_levels`` merged counts back to ``length`` bins, spread uniformly."""
    num_levels = quant.shape[0]
    if length <= num_levels:
        return quant.clone()
    chunk_id = (torch.arange(length, device=quant.device) * num_levels) // length
    chunk_size = torch.bincount(chunk_id, minlength=num_levels).clamp(min=1)
    return quant[chunk_id] / chunk_size[chunk_id]


def calibrate_entropy(
    stats: ObserverStats,
    num_quantized_levels: int = 256,
    num_candidates: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL-entropy (TensorRT-style) calibration over symmetric clip thresholds.

    Args:
        stats: Statistics carrying ``histogram`` and ``bin_edges``.
        num_quantized_levels: Levels the active range is merged down to.
        num_candidates: Number of clip thresholds tried in ``[absmax/2, absmax]``.

    Returns:
        tuple[Tensor, Tensor]: symmetric ``(scale, zero_point)`` for the clip
        threshold minimizing ``KL(P || Q)``. Falls back to
        :func:`calibrate_minmax` if no histogram is available.
    """
    hist = stats.histogram
    bin_edges = stats.bin_edges
    if hist is None or bin_edges is None:
        return calibrate_minmax(stats)
    hist = hist.float()
    absmax = stats.running_max.abs().clamp(min=stats.running_min.abs())
    candidates = torch.linspace(absmax.item() / 2.0, absmax.item(), num_candidates)

    min_kl = float("inf")
    best_clip = absmax.item()
    for clip_val in candidates.tolist():
        left_mask = bin_edges[1:] <= -clip_val
        right_mask = bin_edges[:-1] >= clip_val
        keep = ~left_mask & ~right_mask
        active = hist[keep]
        if active.numel() == 0 or float(active.sum()) < 1e-8:
            continue
        quant = _merge_bins_to_levels(active, num_quantized_levels)
        expanded = _expand_levels_to_bins(quant, active.shape[0])
        p_dist = active + 1e-8
        p_dist = p_dist / p_dist.sum()
        q_dist = expanded + 1e-8
        q_dist = q_dist / q_dist.sum()
        kl = float((p_dist * (p_dist / q_dist).log()).sum().item())
        if kl < min_kl:
            min_kl = kl
            best_clip = clip_val

    return compute_scale_symmetric(torch.tensor(best_clip))


def calibrate_mse(
    tensor_sample: torch.Tensor,
    n_steps: int = 200,
    scheme: Literal["symmetric", "asymmetric"] = "symmetric",
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE grid-search calibration over a fraction of the absolute maximum.

    Args:
        tensor_sample: A representative sample of the tensor to be quantized.
        n_steps: Number of clip fractions tried in ``[0.5, 1.0] · absmax``.
        scheme: ``"symmetric"`` (int8) or ``"asymmetric"`` (uint8) simulation.

    Returns:
        tuple[Tensor, Tensor]: ``(scale (1,), zero_point (1,))`` minimizing the
        reconstruction MSE.
    """
    absmax = tensor_sample.abs().max().clamp(min=1e-8)
    best_scale = absmax / INT8_MAX
    min_mse = float("inf")

    for alpha in torch.linspace(0.5, 1.0, n_steps).tolist():
        if scheme == "symmetric":
            s = (alpha * absmax / INT8_MAX).clamp(min=1e-8)
            xq = tensor_sample.div(s).round().clamp(INT8_MIN, INT8_MAX)
            xr = xq * s
        else:
            x_min_c = (-alpha * absmax).reshape(())
            x_max_c = (alpha * absmax).reshape(())
            s, zp = compute_scale_asymmetric(x_min_c, x_max_c)
            xq = tensor_sample.div(s).round().add(zp).clamp(0, UINT8_MAX)
            xr = (xq - zp) * s
        mse = float(((tensor_sample - xr) ** 2).mean().item())
        if mse < min_mse:
            min_mse = mse
            best_scale = s

    return best_scale.reshape(1), torch.zeros(1, dtype=torch.int32)
