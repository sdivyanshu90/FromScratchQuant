"""
Module: quantlib.quantizers.base

Abstract base class shared by every quantizer. Defines the
compute_params / quantize / dequantize contract, the straight-through
``quantize_dequantize`` used for QAT, and the NaN/Inf input guard.

Mathematical Background:
    Straight-Through Estimator (STE):
        forward:  x_r = dequantize(quantize(x))
        backward: d x_r / d x = 1  (gradient flows unchanged through rounding)
    Implemented as ``x + (x_r - x).detach()``.

References:
    Bengio et al., 2013 — "Estimating or Propagating Gradients ... (STE)".

Example:
    >>> from quantlib.quantizers.base import BaseQuantizer
    >>> issubclass(BaseQuantizer, object)
    True
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quantlib.core.exceptions import QuantizationError
from quantlib.core.qparams import QuantParams


class BaseQuantizer(ABC):
    """Abstract interface implemented by Int8/FP4/NF4 quantizers.

    Subclasses must implement :meth:`compute_params`, :meth:`quantize`, and
    :meth:`dequantize`. The STE-based :meth:`quantize_dequantize` and the
    :meth:`_validate_finite` guard are provided here.
    """

    @abstractmethod
    def compute_params(self, x: torch.Tensor) -> QuantParams:
        """Compute scale + zero_point without modifying ``x`` (pure function)."""
        raise NotImplementedError

    @abstractmethod
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, QuantParams]:
        """Return ``(x_q, params)`` where ``x_q`` is the quantized tensor."""
        raise NotImplementedError

    @abstractmethod
    def dequantize(self, x_q: torch.Tensor, params: QuantParams) -> torch.Tensor:
        """Return the float32 reconstruction; inverse of :meth:`quantize`."""
        raise NotImplementedError

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate quantization noise with a straight-through gradient.

        Args:
            x: Float input tensor.

        Returns:
            torch.Tensor: ``dequantize(quantize(x))`` in the forward pass, with
            an identity gradient w.r.t. ``x`` (suitable for QAT).
        """
        x_q, params = self.quantize(x)
        x_r = self.dequantize(x_q, params)
        return x + (x_r - x).detach()

    @staticmethod
    def _raise_nonfinite(x: torch.Tensor) -> None:
        """Locate and report exact NaN/Inf positions, then raise (slow path).

        Args:
            x: Tensor known to contain a non-finite value.

        Raises:
            QuantizationError: Always, naming the offending positions.
        """
        if bool(torch.isnan(x).any()):
            raise QuantizationError(f"NaN at {torch.where(torch.isnan(x))}")
        raise QuantizationError(f"Inf at {torch.where(torch.isinf(x))}")

    @staticmethod
    def _validate_finite(x: torch.Tensor) -> None:
        """Raise :class:`QuantizationError` if ``x`` contains NaN or Inf.

        Args:
            x: Tensor to validate.

        Raises:
            QuantizationError: With the exact offending positions.

        Note:
            Hot paths instead check the (tiny) reduced absmax/min/max tensor —
            NaN/Inf propagate through amax/amin — and call :meth:`_raise_nonfinite`
            only on failure, avoiding a full extra pass over large weights.
        """
        if bool(torch.isfinite(x).all()):
            return
        BaseQuantizer._raise_nonfinite(x)
