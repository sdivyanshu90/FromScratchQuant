"""
Module: quantlib.serialization

Save / load quantized models (safetensors weights + JSON sidecar).

Mathematical Background:
    None — (de)serialization only.

References:
    safetensors — https://github.com/huggingface/safetensors

Example:
    >>> from quantlib.serialization import save_quantized, load_quantized
    >>> save_quantized.__name__
    'save_quantized'
"""

from __future__ import annotations

from quantlib.serialization.checkpoint import load_quantized, save_quantized

__all__ = ["save_quantized", "load_quantized"]
