"""
Module: quantlib.benchmarks.accuracy

Reconstruction-quality metrics for quantization: signal-to-noise ratio (dB),
cosine similarity, and mean squared error.

Mathematical Background:
    SNR_dB  = 10 · log10( E[x²] / E[(x - x_r)²] )
    cosine  = <x, x_r> / (||x|| · ||x_r||)
    MSE     = E[(x - x_r)²]

References:
    Standard signal-processing definitions.

Example:
    >>> import torch
    >>> from quantlib.benchmarks.accuracy import compute_snr_db
    >>> x = torch.randn(1000)
    >>> compute_snr_db(x, x) == float("inf")
    True
"""

from __future__ import annotations

from typing import Final

import torch

_EPS: Final[float] = 1e-12  # guards log10/divide against exact-zero denominators


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Mean squared error between an original tensor and its reconstruction.

    Args:
        original: Reference tensor.
        reconstructed: Approximation of ``original`` (same shape).

    Returns:
        float: ``E[(original - reconstructed)²]``.
    """
    diff = original.float() - reconstructed.float()
    return float((diff * diff).mean().item())


def compute_snr_db(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Signal-to-noise ratio in decibels.

    Args:
        original: Reference tensor (the "signal").
        reconstructed: Approximation; the error is the "noise".

    Returns:
        float: ``10·log10(signal_power / noise_power)``; ``+inf`` for an exact
        reconstruction.

    Example:
        >>> import torch
        >>> x = torch.randn(4096)
        >>> compute_snr_db(x, x)
        inf
    """
    original = original.float()
    reconstructed = reconstructed.float()
    signal_power = (original * original).mean()
    noise_power = ((original - reconstructed) ** 2).mean()
    if float(noise_power.item()) <= _EPS:
        return float("inf")
    return float((10.0 * torch.log10(signal_power / noise_power)).item())


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened).

    Args:
        a: First tensor.
        b: Second tensor (same number of elements as ``a``).

    Returns:
        float: ``<a, b> / (||a|| · ||b||)`` in ``[-1, 1]``; 0.0 if either norm is 0.
    """
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    denom = a_flat.norm() * b_flat.norm()
    if float(denom.item()) <= _EPS:
        return 0.0
    return float((a_flat @ b_flat / denom).item())
