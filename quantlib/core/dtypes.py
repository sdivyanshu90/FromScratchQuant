"""
Module: quantlib.core.dtypes

Enumeration of the quantized storage formats supported by quantlib, together
with their bit-widths and the torch storage dtype each one is packed into.

Mathematical Background:
    INT8  : 8-bit signed two's complement, range [-128, 127].
    UINT8 : 8-bit unsigned, range [0, 255].
    FP4   : 4-bit float E2M1, 16 representable values (see theory.md, Sec. 4).
    NF4   : 4-bit NormalFloat, 16 N(0,1)-quantile values (see theory.md, Sec. 5).

References:
    Jacob et al., 2018 — "Quantization and Training of Neural Networks ...".
    Dettmers et al., 2023 — "QLoRA" (https://arxiv.org/abs/2305.14314).

Example:
    >>> from quantlib.core.dtypes import QuantDtype
    >>> QuantDtype.INT8.bits
    8
    >>> QuantDtype.FP4.bits
    4
    >>> QuantDtype.NF4.storage_dtype
    torch.uint8
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import torch

# Bit-width of each format, keyed by the enum *value* string.
# 4-bit formats are packed two-per-byte at the storage layer (see storage_dtype).
_BITS: Final[dict[str, int]] = {
    "int8": 8,
    "uint8": 8,
    "fp4": 4,
    "nf4": 4,
    "fp8_e4m3": 8,
    "fp8_e5m2": 8,
}


class QuantDtype(Enum):
    """Quantized storage formats supported by quantlib."""

    INT8 = "int8"       # 8-bit signed int,   range [-128, 127]
    UINT8 = "uint8"     # 8-bit unsigned int, range [0, 255]
    FP4 = "fp4"         # 4-bit float E2M1,   16 representable values
    NF4 = "nf4"         # NormalFloat4,        16 quantile-spaced values
    FP8_E4M3 = "fp8_e4m3"  # reserved for future use
    FP8_E5M2 = "fp8_e5m2"  # reserved for future use

    @property
    def bits(self) -> int:
        """Number of bits a single element occupies in the *logical* format.

        Returns:
            int: 8 for INT8/UINT8/FP8, 4 for FP4/NF4.
        """
        return _BITS[self.value]

    @property
    def is_integer(self) -> bool:
        """Whether the format encodes integers (vs. a float / quantile codebook).

        Returns:
            bool: True only for INT8 and UINT8.
        """
        return self in (QuantDtype.INT8, QuantDtype.UINT8)

    @property
    def storage_dtype(self) -> torch.dtype:
        """The torch dtype used to *store* this format on disk / in buffers.

        INT8 maps to ``torch.int8``. Everything else (UINT8 and the 4-bit
        formats, which pack two nibbles per byte) maps to ``torch.uint8``.

        Returns:
            torch.dtype: ``torch.int8`` for INT8, else ``torch.uint8``.
        """
        return torch.int8 if self == QuantDtype.INT8 else torch.uint8
