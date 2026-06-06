"""
Module: quantlib.core.exceptions

Typed exception hierarchy for the quantlib library so that callers can catch
quantization-specific failures without swallowing unrelated errors.

Mathematical Background:
    None — error types only.

References:
    None.

Example:
    >>> from quantlib.core.exceptions import QuantizationError
    >>> raise QuantizationError("NaN detected")
    Traceback (most recent call last):
    ...
    quantlib.core.exceptions.QuantizationError: NaN detected
"""

from __future__ import annotations


class QuantizationError(Exception):
    """Base class for every error raised by quantlib.

    Raised directly when a tensor contains NaN/Inf or when a quantization
    operation cannot proceed without producing silently-wrong results.
    """


class CalibrationError(QuantizationError):
    """Raised when calibration statistics are missing, empty, or inconsistent.

    Also used by the (de)serialization layer for malformed/incompatible
    on-disk metadata, per the serialization spec.
    """


class ShapeError(QuantizationError):
    """Raised when tensor / parameter shapes are incompatible.

    Examples:
        * ``granularity="per_group"`` without a ``group_size``.
        * A tensor whose ``numel()`` is not divisible by ``group_size``.
    """


class NumericalError(QuantizationError):
    """Raised when a numerical invariant is violated.

    The canonical case is a non-positive ``scale`` reaching :class:`QuantParams`.
    """
