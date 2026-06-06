"""
Module: quantlib.quantizers.int8

INT8 quantizer supporting symmetric (int8 output) and asymmetric (uint8 output)
schemes at per-tensor, per-channel, and per-group granularity.

Mathematical Background:
    Symmetric (theory.md Eq. 1-3):
        scale = absmax / 127
        x_q   = clamp(round(x / scale), -128, 127)            -> int8
        x_r   = x_q * scale
    Asymmetric (theory.md Eq. 4-7):
        scale      = (x_max - x_min) / 255
        zero_point = clamp(round(-x_min / scale), 0, 255)
        x_q        = clamp(round(x / scale) + zero_point, 0, 255)  -> uint8
        x_r        = (x_q - zero_point) * scale

References:
    Jacob et al., 2018 — "Quantization and Training of Neural Networks ...".

Example:
    >>> import torch
    >>> from quantlib.quantizers.int8 import Int8Quantizer
    >>> q = Int8Quantizer(scheme="symmetric")
    >>> x_q, p = q.quantize(torch.tensor([-1.0, 0.0, 1.0]))
    >>> q.dequantize(x_q, p).tolist()
    [-1.0, 0.0, 1.0]
"""

from __future__ import annotations

from typing import Literal

import torch

from quantlib.core.dtypes import QuantDtype
from quantlib.core.qparams import QuantParams
from quantlib.quantizers.base import BaseQuantizer
from quantlib.quantizers.utils import (
    INT8_MAX,
    INT8_MIN,
    REDUCED_INT8_MAX,
    REDUCED_INT8_MIN,
    UINT8_MAX,
    UINT8_MIN,
    broadcast_for_apply,
    compute_absmax,
    compute_minmax,
    compute_scale_asymmetric,
    compute_scale_symmetric,
    restore_shape,
)

# ┌─ PLAN: Int8Quantizer ───────────────────────────────────────────────────────┐
# │ Mathematical Invariants:                                                      │
# │   • symmetric  -> zero_point == 0 everywhere; output dtype int8.              │
# │   • asymmetric -> affine map; output dtype uint8; dequant inverts exactly     │
# │     up to rounding (x_r = (x_q - zp) * scale).                                │
# │   • scale strictly positive (QuantParams enforces; we clamp + warn).          │
# │   • per_channel scale.shape == (x.shape[channel_dim],).                       │
# │   • per_group   scale.numel() == x.numel() // group_size.                     │
# │ Numerical Edge Cases:                                                         │
# │   • all-zeros        -> scale = 1.0, zp = 0  (both schemes).                  │
# │   • constant != 0    -> sym scale = |v|/127; asym scale = 1.0, zp = 0.        │
# │   • scale < 1e-8     -> clamp to 1e-8 + UserWarning.                          │
# │   • NaN / Inf        -> raise QuantizationError with exact positions.         │
# │   • reduce_range     -> q_max = 63, q_min = -64 (symmetric, VNNI).            │
# │ Module Dependencies:                                                          │
# │   • quantlib.quantizers.utils: scale formulas, granularity reduce/broadcast.  │
# │   • quantlib.core.qparams.QuantParams, quantlib.core.dtypes.QuantDtype.       │
# │ Hand-Computed Verification (symmetric per_tensor):                            │
# │   Input:    x = [-2.3, -1.1, 0.0, 0.7, 1.8, 3.2]                              │
# │   absmax = 3.2, scale = 3.2/127 = 0.0251968...                                │
# │   Expected x_q (int8) = [-91, -44, 0, 28, 71, 127]                            │
# │   Tolerance: x_q exact (int equality); scale atol=1e-6, rtol=0.               │
# └──────────────────────────────────────────────────────────────────────────────┘


class Int8Quantizer(BaseQuantizer):
    """8-bit integer quantizer (symmetric int8 / asymmetric uint8).

    Args:
        scheme: ``"symmetric"`` (zero_point ≡ 0, int8) or ``"asymmetric"``
            (affine, uint8).
        granularity: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"``.
        channel_dim: Channel dimension used when ``granularity == "per_channel"``.
        group_size: Elements per group when ``granularity == "per_group"``.
        reduce_range: If True, clamp to ``[-64, 63]`` for VNNI kernels (symmetric).
    """

    def __init__(
        self,
        scheme: Literal["symmetric", "asymmetric"] = "symmetric",
        granularity: Literal["per_tensor", "per_channel", "per_group"] = "per_tensor",
        channel_dim: int = 0,
        group_size: int = 128,
        reduce_range: bool = False,
    ) -> None:
        self.scheme = scheme
        self.granularity = granularity
        self.channel_dim = channel_dim
        self.group_size = group_size
        self.reduce_range = reduce_range
        self._dtype = QuantDtype.INT8 if scheme == "symmetric" else QuantDtype.UINT8

    # ── range helpers ─────────────────────────────────────────────────────────
    def _sym_qmax(self) -> int:
        return REDUCED_INT8_MAX if self.reduce_range else INT8_MAX

    def _sym_clamp(self) -> tuple[int, int]:
        if self.reduce_range:
            return REDUCED_INT8_MIN, REDUCED_INT8_MAX
        return INT8_MIN, INT8_MAX

    def _asym_qmax(self) -> int:
        # Asymmetric reduce_range halves the range to [0, 127].
        return 127 if self.reduce_range else UINT8_MAX

    def _group_size_or_none(self) -> int | None:
        return self.group_size if self.granularity == "per_group" else None

    def compute_params(self, x: torch.Tensor) -> QuantParams:
        """Compute scale + zero_point without modifying ``x`` (pure function).

        Args:
            x: Float input tensor.

        Returns:
            QuantParams: scale/zero_point plus granularity & scheme metadata.

        Raises:
            QuantizationError: If ``x`` contains NaN or Inf.
        """
        gs = self._group_size_or_none()
        if self.scheme == "symmetric":
            absmax = compute_absmax(x, self.granularity, self.channel_dim, gs)
            if not bool(torch.isfinite(absmax).all()):
                self._raise_nonfinite(x)
            scale, zero_point = compute_scale_symmetric(absmax, self._sym_qmax())
        else:
            x_min, x_max = compute_minmax(x, self.granularity, self.channel_dim, gs)
            if not bool(torch.isfinite(x_min).all() & torch.isfinite(x_max).all()):
                self._raise_nonfinite(x)
            scale, zero_point = compute_scale_asymmetric(
                x_min, x_max, UINT8_MIN, self._asym_qmax()
            )
        return QuantParams(
            scale=scale,
            zero_point=zero_point,
            dtype=self._dtype,
            granularity=self.granularity,
            scheme=self.scheme,
            channel_dim=self.channel_dim,
            group_size=gs,
            original_shape=tuple(x.shape),
        )

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, QuantParams]:
        """Quantize ``x`` to int8 (symmetric) or uint8 (asymmetric).

        Args:
            x: Float input tensor.

        Returns:
            tuple[Tensor, QuantParams]: ``(x_q, params)``.
        """
        params = self.compute_params(x)
        x_view, scale_b, zp_b = broadcast_for_apply(
            x, params.scale, params.zero_point, self.granularity, self.channel_dim,
            self._group_size_or_none(),
        )
        if self.scheme == "symmetric":
            lo, hi = self._sym_clamp()
            # In-place round/clamp on the division temporary to avoid extra 64 MB allocations.
            buf = x_view / scale_b
            buf.round_().clamp_(lo, hi)
            x_q = buf.to(torch.int8)
        else:
            buf = x_view / scale_b
            buf.round_().add_(zp_b).clamp_(UINT8_MIN, self._asym_qmax())
            x_q = buf.to(torch.uint8)
        x_q = restore_shape(x_q, params.original_shape or tuple(x.shape), self.granularity)
        return x_q, params
        # ✓ VERIFY — paste-able snippet proving the core math:
        # x = torch.tensor([-2.3, -1.1, 0.0, 0.7, 1.8, 3.2])
        # x_q, p = Int8Quantizer("symmetric").quantize(x)
        # assert x_q.tolist() == [-91, -44, 0, 28, 71, 127]
        # torch.testing.assert_close(p.scale, torch.tensor(3.2 / 127), atol=1e-6, rtol=0)

    def dequantize(self, x_q: torch.Tensor, params: QuantParams) -> torch.Tensor:
        """Reconstruct float32 from quantized codes (inverse of :meth:`quantize`).

        Args:
            x_q: Quantized tensor (int8 or uint8).
            params: The :class:`QuantParams` returned by :meth:`quantize`.

        Returns:
            torch.Tensor: float32 reconstruction with the original shape.
        """
        x_view, scale_b, zp_b = broadcast_for_apply(
            x_q, params.scale, params.zero_point, params.granularity,
            params.channel_dim, params.group_size,
        )
        if params.scheme == "symmetric":
            x_r = x_view.to(torch.float32) * scale_b
        else:
            x_r = (x_view.to(torch.float32) - zp_b.to(torch.float32)) * scale_b
        return restore_shape(x_r, params.original_shape or tuple(x_q.shape), params.granularity)
        # ✓ VERIFY — paste-able snippet proving the core math:
        # q = Int8Quantizer("asymmetric")
        # x = torch.tensor([-2.3, -1.1, 0.0, 0.7, 1.8, 3.2])
        # x_q, p = q.quantize(x)
        # assert x_q.tolist() == [0, 56, 107, 139, 190, 255]
        # torch.testing.assert_close(q.dequantize(x_q, p), x, atol=0.05, rtol=0)
