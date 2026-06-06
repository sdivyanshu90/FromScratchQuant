"""
Module: quantlib.benchmarks

Accuracy metrics (SNR, cosine, MSE) and memory/latency profiling helpers.

Mathematical Background:
    See quantlib.benchmarks.accuracy.

References:
    Standard signal-processing definitions.

Example:
    >>> from quantlib.benchmarks import compute_snr_db, memory_footprint_bytes
    >>> compute_snr_db.__name__
    'compute_snr_db'
"""

from __future__ import annotations

from quantlib.benchmarks.accuracy import (
    compute_cosine_similarity,
    compute_mse,
    compute_snr_db,
)
from quantlib.benchmarks.profiler import latency_ms, memory_footprint_bytes

__all__ = [
    "compute_snr_db",
    "compute_cosine_similarity",
    "compute_mse",
    "memory_footprint_bytes",
    "latency_ms",
]
