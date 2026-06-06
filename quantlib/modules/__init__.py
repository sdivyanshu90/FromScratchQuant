"""
Module: quantlib.modules

Quantized ``nn.Module`` replacements and model-graph helpers.

Mathematical Background:
    None — module surgery; per-layer math lives in the quantizers.

References:
    None.

Example:
    >>> from quantlib.modules import QuantizedLinear, quantize_model
    >>> QuantizedLinear.__name__
    'QuantizedLinear'
"""

from __future__ import annotations

from quantlib.modules.qembedding import QuantizedEmbedding
from quantlib.modules.qlinear import QuantizedLinear
from quantlib.modules.wrappers import get_quantizable_layers, quantize_model

__all__ = [
    "QuantizedLinear",
    "QuantizedEmbedding",
    "quantize_model",
    "get_quantizable_layers",
]
