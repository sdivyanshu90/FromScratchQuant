"""
Module: quantlib.calibration.sensitivity

Per-layer sensitivity analysis: measure how much each layer's weights degrade
under a given quantizer, so the most sensitive layers can be kept in FP32.

Mathematical Background:
    For each target layer, reconstruct its weight via quantize->dequantize and
    score the reconstruction (default: SNR in dB). Low score == high sensitivity.

References:
    Dong et al., 2019 — "HAWQ: Hessian Aware Quantization" (sensitivity ranking).

Example:
    >>> import torch
    >>> from torch import nn
    >>> from quantlib.quantizers.int8 import Int8Quantizer
    >>> from quantlib.calibration.sensitivity import layer_sensitivity_analysis
    >>> model = nn.Sequential(nn.Linear(8, 8))
    >>> s = layer_sensitivity_analysis(model, None, Int8Quantizer("symmetric"))
    >>> '0' in s
    True
"""

from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch import nn

from quantlib.benchmarks.accuracy import (
    compute_cosine_similarity,
    compute_mse,
    compute_snr_db,
)
from quantlib.quantizers.base import BaseQuantizer


def layer_sensitivity_analysis(
    model: nn.Module,
    dataloader: Iterable[object] | None,
    quantizer: BaseQuantizer,
    metric: Literal["snr_db", "mse", "cosine"] = "snr_db",
    target_module_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> dict[str, float]:
    """Score every target layer's weight reconstruction under ``quantizer``.

    Args:
        model: Module to analyze.
        dataloader: Reserved for activation-based extensions; unused for the
            weight-reconstruction metrics and may be ``None``.
        quantizer: The quantizer whose impact is measured.
        metric: ``"snr_db"`` (higher = more robust), ``"mse"`` (lower = more
            robust), or ``"cosine"`` (closer to 1 = more robust).
        target_module_types: Layer types to score (default ``(nn.Linear,)``).

    Returns:
        dict[str, float]: ``{layer_name: score}``. For ``"snr_db"``, a low score
        flags a sensitive layer that is a good candidate to keep in FP32.
    """
    del dataloader  # weight-reconstruction metrics do not need calibration data
    scores: dict[str, float] = {}
    for name, module in model.named_modules():
        if not isinstance(module, target_module_types):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        with torch.no_grad():
            recon = quantizer.quantize_dequantize(weight.detach().float())
        if metric == "snr_db":
            scores[name] = compute_snr_db(weight, recon)
        elif metric == "mse":
            scores[name] = compute_mse(weight, recon)
        else:
            scores[name] = compute_cosine_similarity(weight, recon)
    return scores
