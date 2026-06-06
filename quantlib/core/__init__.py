"""
Module: quantlib.core

Core types shared across quantlib: dtypes, the immutable :class:`QuantParams`
container, calibration observers, and the typed exception hierarchy.

Mathematical Background:
    None — re-exports only.

References:
    See individual submodules.

Example:
    >>> from quantlib.core import QuantDtype, QuantParams
    >>> QuantDtype.INT8.bits
    8
"""

from __future__ import annotations

from quantlib.core.dtypes import QuantDtype
from quantlib.core.exceptions import (
    CalibrationError,
    NumericalError,
    QuantizationError,
    ShapeError,
)
from quantlib.core.observer import HistogramObserver, MinMaxObserver, ObserverStats
from quantlib.core.qparams import QuantParams

__all__ = [
    "QuantDtype",
    "QuantParams",
    "MinMaxObserver",
    "HistogramObserver",
    "ObserverStats",
    "QuantizationError",
    "CalibrationError",
    "ShapeError",
    "NumericalError",
]
