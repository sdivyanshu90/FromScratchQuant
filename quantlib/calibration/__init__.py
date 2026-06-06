"""
Module: quantlib.calibration

Calibration entry points: the :class:`Calibrator` context manager, the four
calibration methods, and per-layer sensitivity analysis.

Mathematical Background:
    See quantlib.calibration.methods.

References:
    Migacz, 2017 (TensorRT KL calibration).

Example:
    >>> from quantlib.calibration import Calibrator, layer_sensitivity_analysis
    >>> Calibrator.__name__
    'Calibrator'
"""

from __future__ import annotations

from quantlib.calibration.calibrator import Calibrator
from quantlib.calibration.methods import (
    calibrate_entropy,
    calibrate_minmax,
    calibrate_mse,
    calibrate_percentile,
)
from quantlib.calibration.sensitivity import layer_sensitivity_analysis

__all__ = [
    "Calibrator",
    "calibrate_minmax",
    "calibrate_percentile",
    "calibrate_entropy",
    "calibrate_mse",
    "layer_sensitivity_analysis",
]
