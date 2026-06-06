"""
Module: quantlib.quantizers

Quantizer implementations and their shared utilities: INT8 (symmetric/
asymmetric), FP4 (E2M1), NF4 (NormalFloat), plus nibble packing helpers.

Mathematical Background:
    See quantlib.quantizers.int8 and quantlib.quantizers.fp4.

References:
    Jacob et al., 2018; Dettmers et al., 2023.

Example:
    >>> from quantlib.quantizers import Int8Quantizer, NF4Quantizer
    >>> Int8Quantizer("symmetric").scheme
    'symmetric'
"""

from __future__ import annotations

from quantlib.quantizers.base import BaseQuantizer
from quantlib.quantizers.fp4 import (
    FP4_E2M1_VALUES,
    FP4_MAX,
    NF4_QUANTILE_TABLE,
    FP4Quantizer,
    NF4Quantizer,
    double_quantize_scales,
)
from quantlib.quantizers.int8 import Int8Quantizer
from quantlib.quantizers.utils import (
    compute_scale_asymmetric,
    compute_scale_symmetric,
    pack_int4,
    safe_divide,
    unpack_int4,
)

__all__ = [
    "BaseQuantizer",
    "Int8Quantizer",
    "FP4Quantizer",
    "NF4Quantizer",
    "double_quantize_scales",
    "FP4_E2M1_VALUES",
    "FP4_MAX",
    "NF4_QUANTILE_TABLE",
    "pack_int4",
    "unpack_int4",
    "compute_scale_symmetric",
    "compute_scale_asymmetric",
    "safe_divide",
]
