"""
quantlib — a self-contained quantization library (INT8 / FP4 / NF4).

Zero dependency on bitsandbytes, optimum, quanto, or neural-compressor. Provides
symmetric/asymmetric INT8, FP4 (E2M1) and NF4 (NormalFloat) quantizers, four
calibration methods, drop-in quantized modules, (de)serialization, and metrics.

Mathematical Background:
    See quantlib.quantizers.int8 / quantlib.quantizers.fp4 and docs/theory.md.

References:
    Jacob et al., 2018 (INT8); Dettmers et al., 2023 (NF4/QLoRA);
    Migacz, 2017 (TensorRT KL calibration).

Example:
    >>> import torch
    >>> from quantlib import Int8Quantizer
    >>> x_q, p = Int8Quantizer("symmetric").quantize(torch.tensor([-1.0, 0.0, 1.0]))
    >>> x_q.tolist()
    [-127, 0, 127]
"""

from __future__ import annotations

from quantlib._version import __version__
from quantlib.benchmarks.accuracy import (
    compute_cosine_similarity,
    compute_mse,
    compute_snr_db,
)
from quantlib.benchmarks.profiler import latency_ms, memory_footprint_bytes
from quantlib.calibration.calibrator import Calibrator
from quantlib.calibration.sensitivity import layer_sensitivity_analysis
from quantlib.core.dtypes import QuantDtype
from quantlib.core.exceptions import (
    CalibrationError,
    NumericalError,
    QuantizationError,
    ShapeError,
)
from quantlib.core.qparams import QuantParams
from quantlib.modules.qembedding import QuantizedEmbedding
from quantlib.modules.qlinear import QuantizedLinear
from quantlib.modules.wrappers import get_quantizable_layers, quantize_model
from quantlib.quantizers.base import BaseQuantizer
from quantlib.quantizers.fp4 import (
    NF4_QUANTILE_TABLE,
    FP4Quantizer,
    NF4Quantizer,
    double_quantize_scales,
)
from quantlib.quantizers.int8 import Int8Quantizer
from quantlib.quantizers.utils import pack_int4, unpack_int4
from quantlib.serialization.checkpoint import load_quantized, save_quantized

__all__ = [
    "__version__",
    # quantizers
    "BaseQuantizer",
    "Int8Quantizer",
    "FP4Quantizer",
    "NF4Quantizer",
    "double_quantize_scales",
    "NF4_QUANTILE_TABLE",
    "pack_int4",
    "unpack_int4",
    # core types
    "QuantParams",
    "QuantDtype",
    "QuantizationError",
    "CalibrationError",
    "ShapeError",
    "NumericalError",
    # calibration
    "Calibrator",
    "layer_sensitivity_analysis",
    # modules
    "QuantizedLinear",
    "QuantizedEmbedding",
    "quantize_model",
    "get_quantizable_layers",
    # serialization
    "save_quantized",
    "load_quantized",
    # benchmarks
    "compute_snr_db",
    "compute_cosine_similarity",
    "compute_mse",
    "memory_footprint_bytes",
    "latency_ms",
]
