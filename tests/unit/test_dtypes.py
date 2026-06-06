"""Unit tests for quantlib.core.dtypes.QuantDtype."""

from __future__ import annotations

import torch

from quantlib.core.dtypes import QuantDtype


def test_int8_bits() -> None:
    assert QuantDtype.INT8.bits == 8


def test_uint8_bits() -> None:
    assert QuantDtype.UINT8.bits == 8


def test_fp4_bits() -> None:
    assert QuantDtype.FP4.bits == 4


def test_nf4_bits() -> None:
    assert QuantDtype.NF4.bits == 4


def test_is_integer_true_for_int_types() -> None:
    assert QuantDtype.INT8.is_integer
    assert QuantDtype.UINT8.is_integer


def test_is_integer_false_for_float_codebooks() -> None:
    assert not QuantDtype.FP4.is_integer
    assert not QuantDtype.NF4.is_integer


def test_storage_dtype_int8_is_signed() -> None:
    assert QuantDtype.INT8.storage_dtype == torch.int8


def test_storage_dtype_four_bit_packs_into_uint8() -> None:
    assert QuantDtype.FP4.storage_dtype == torch.uint8
    assert QuantDtype.NF4.storage_dtype == torch.uint8
    assert QuantDtype.UINT8.storage_dtype == torch.uint8
