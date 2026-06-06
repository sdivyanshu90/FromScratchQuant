"""
Module: quantlib.modules.wrappers

Model-level helpers: :func:`quantize_model` swaps every target layer for its
quantized equivalent, and :func:`get_quantizable_layers` enumerates candidates.

Mathematical Background:
    None — module-graph surgery only. Per-layer math lives in the quantizers.

References:
    None.

Example:
    >>> from torch import nn
    >>> from quantlib.quantizers.int8 import Int8Quantizer
    >>> from quantlib.modules.wrappers import quantize_model
    >>> from quantlib.modules.qlinear import QuantizedLinear
    >>> m = nn.Sequential(nn.Linear(4, 4))
    >>> qm = quantize_model(m, Int8Quantizer("symmetric"))
    >>> isinstance(qm[0], QuantizedLinear)
    True
"""

from __future__ import annotations

import copy
import re

from torch import nn

from quantlib.quantizers.base import BaseQuantizer
from quantlib.modules.qembedding import QuantizedEmbedding
from quantlib.modules.qlinear import QuantizedLinear


def _get_submodule(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted submodule path (e.g. ``"encoder.layer.0"``)."""
    module: nn.Module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def get_quantizable_layers(
    model: nn.Module,
    target_module_types: tuple[type[nn.Module], ...] = (nn.Linear,),
) -> dict[str, nn.Module]:
    """Enumerate target layers eligible for quantization.

    Args:
        model: Module to scan.
        target_module_types: Layer types to collect (default ``(nn.Linear,)``).

    Returns:
        dict[str, nn.Module]: ``{dotted_name: module}`` for every match. Already
        quantized layers are not ``nn.Linear``/``nn.Embedding`` and are skipped.
    """
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, target_module_types)
    }


def _quantize_leaf(module: nn.Module, quantizer: BaseQuantizer) -> nn.Module:
    """Return the quantized replacement for a single target leaf module."""
    if isinstance(module, nn.Linear):
        return QuantizedLinear.from_linear(module, quantizer)
    if isinstance(module, nn.Embedding):
        return QuantizedEmbedding.from_embedding(module, quantizer)
    raise TypeError(f"no quantized equivalent for {type(module).__name__}")


def quantize_model(
    model: nn.Module,
    quantizer: BaseQuantizer,
    target_module_types: tuple[type[nn.Module], ...] = (nn.Linear,),
    exclude_patterns: list[str] | None = None,
    inplace: bool = True,
) -> nn.Module:
    """Replace all target layers with their quantized equivalents.

    Args:
        model: Module to quantize.
        quantizer: Quantizer applied to each target layer's weight.
        target_module_types: Layer types to replace (default ``(nn.Linear,)``).
        exclude_patterns: Regex patterns matched against each dotted layer name;
            a layer is skipped if any pattern matches. ``None`` includes all.
        inplace: If False, operate on a deep copy and leave ``model`` untouched.

    Returns:
        nn.Module: the (possibly copied) model with target layers replaced.

    Note:
        ``named_modules()`` yields full dotted paths, e.g.
        ``"encoder.layer.0.attention.query"``; ``rsplit(".", 1)`` splits into
        ``(parent_path, leaf_attr_name)``.
    """
    if not inplace:
        model = copy.deepcopy(model)

    for name, module in list(model.named_modules()):
        if not isinstance(module, target_module_types):
            continue
        if exclude_patterns and any(re.search(p, name) for p in exclude_patterns):
            continue
        parts = name.rsplit(".", 1)
        parent = model if len(parts) == 1 else _get_submodule(model, parts[0])
        attr = parts[-1]
        setattr(parent, attr, _quantize_leaf(module, quantizer))
    return model
