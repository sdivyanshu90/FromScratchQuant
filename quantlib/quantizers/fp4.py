"""
Module: quantlib.quantizers.fp4

4-bit quantizers: FP4 (E2M1 float codebook) and NF4 (NormalFloat quantile
codebook from QLoRA), plus NF4 double quantization of the per-group scales.

Mathematical Background:
    Both quantizers are codebook (nearest-value) quantizers:
        scale  = absmax / table_absmax
        x_norm = x / scale                       (FP4 -> [-6, 6], NF4 -> [-1, 1])
        x_q    = argmin_k |x_norm - table[k]|     (4-bit index 0..15)
        x_r    = table[x_q] * scale
    FP4 E2M1 value = (-1)^S · 2^(E-bias) · (1 + M/2), bias = 1, with the E=00
    subnormal branch 2^(1-bias) · (M/2).  NF4 places its 16 levels at the
    quantiles of N(0,1), minimizing E[(x - dequant(quant(x)))^2] for Gaussian x.

References:
    Dettmers et al., 2023 — "QLoRA" (https://arxiv.org/abs/2305.14314), App. D/E.

Example:
    >>> import torch
    >>> from quantlib.quantizers.fp4 import NF4Quantizer
    >>> q = NF4Quantizer()
    >>> x_q, p = q.quantize(torch.tensor([-0.70, -0.53, 0.0, 0.08, 0.24, 0.72]))
    >>> x_q.tolist()
    [0, 1, 7, 8, 11, 15]
"""

from __future__ import annotations

from typing import Final, Literal

import torch

from quantlib.core.dtypes import QuantDtype
from quantlib.core.qparams import QuantParams
from quantlib.quantizers.base import BaseQuantizer
from quantlib.quantizers.int8 import Int8Quantizer
from quantlib.quantizers.utils import (
    broadcast_for_apply,
    compute_absmax,
    compute_scale_symmetric,
    restore_shape,
    unpack_int4,
)

# ── FP4 E2M1 codebook (index == 4-bit code) — theory.md Sec. 4 ─────────────────
FP4_E2M1_VALUES: Final[list[float]] = [
    0.0,   # 0  0 00 0   0.0   special zero
    0.5,   # 1  0 00 1   0.5   subnormal 2^(1-1)*0.5
    1.0,   # 2  0 01 0   1.0   2^(1-1)*(1+0.0)
    1.5,   # 3  0 01 1   1.5   2^(1-1)*(1+0.5)
    2.0,   # 4  0 10 0   2.0   2^(2-1)*(1+0.0)
    3.0,   # 5  0 10 1   3.0   2^(2-1)*(1+0.5)
    4.0,   # 6  0 11 0   4.0   2^(3-1)*(1+0.0)
    6.0,   # 7  0 11 1   6.0   2^(3-1)*(1+0.5)  <- max absolute value
    -0.0,  # 8  1 00 0  -0.0
    -0.5,  # 9  1 00 1  -0.5
    -1.0,  # 10 1 01 0  -1.0
    -1.5,  # 11 1 01 1  -1.5
    -2.0,  # 12 1 10 0  -2.0
    -3.0,  # 13 1 10 1  -3.0
    -4.0,  # 14 1 11 0  -4.0
    -6.0,  # 15 1 11 1  -6.0
]
FP4_MAX: Final[float] = 6.0  # absmax of all representable FP4 values

# ── NF4 quantile codebook — Dettmers et al., 2023, App. D ──────────────────────
# 16 quantiles of N(0,1) mapped to [-1, +1]; index 7 is exactly 0.0.
NF4_QUANTILE_TABLE: Final[list[float]] = [
    -1.0000000000000000,  # 0
    -0.6961928009986877,  # 1
    -0.5250730514526367,  # 2
    -0.3949174880981445,  # 3
    -0.2844413816928864,  # 4
    -0.1847734302282333,  # 5
    -0.0910500362515450,  # 6
    0.0000000000000000,   # 7  <- zero exactly
    0.0795802995562553,   # 8
    0.1609302014112473,   # 9
    0.2461123019456863,   # 10
    0.3379152417182922,   # 11
    0.4407098293304443,   # 12
    0.5626170039176941,   # 13
    0.7229568362236023,   # 14
    1.0000000000000000,   # 15
]

# ┌─ PLAN: FP4Quantizer / NF4Quantizer ─────────────────────────────────────────┐
# │ Mathematical Invariants:                                                      │
# │   • output index in [0, 15], dtype uint8; quantize() returns UNPACKED idx.    │
# │   • dequant(x_q) == table[x_q] * scale; exactly reversible for table values.  │
# │   • NF4 SNR > FP4 SNR on N(0,1) (quantile vs geometric placement).            │
# │   • scale strictly positive; absmax == 0 -> scale = 1.0.                      │
# │ Numerical Edge Cases:                                                         │
# │   • signed-zero in FP4 table: x_norm == 0 ties idx0(+0) vs idx8(-0);          │
# │     argmin picks the first (idx 0) -> deterministic.                          │
# │   • NaN / Inf -> QuantizationError (inherited guard).                         │
# │   • packed params (set by QuantizedLinear) -> unpack before table lookup.     │
# │ Module Dependencies:                                                          │
# │   • utils: compute_absmax, compute_scale_symmetric, broadcast/restore, unpack │
# │   • int8.Int8Quantizer for NF4 double quantization of scales.                 │
# │ Hand-Computed Verification (NF4 per_tensor):                                  │
# │   Input:    x = [-0.70, -0.53, 0.0, 0.08, 0.24, 0.72], absmax = 0.72          │
# │   x_norm  = [-0.9722, -0.7361, 0.0, 0.1111, 0.3333, 1.0]                      │
# │   Expected idx = [0, 1, 7, 8, 11, 15]                                         │
# │   Tolerance: idx exact; x_r atol=1e-3, rtol=0.                                │
# └──────────────────────────────────────────────────────────────────────────────┘


class _CodebookQuantizer(BaseQuantizer):
    """Shared nearest-codebook 4-bit quantizer (subclassed by FP4/NF4).

    Args:
        table: The 16-entry value codebook (index == 4-bit code).
        table_absmax: Maximum absolute value in ``table`` (normalization ceiling).
        dtype: :class:`QuantDtype` tag (FP4 or NF4) for the produced params.
        granularity: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"``.
        channel_dim: Channel dimension for ``per_channel``.
        group_size: Elements per group for ``per_group``.
    """

    def __init__(
        self,
        table: list[float],
        table_absmax: int,
        dtype: QuantDtype,
        granularity: Literal["per_tensor", "per_channel", "per_group"] = "per_tensor",
        channel_dim: int = 0,
        group_size: int = 64,
    ) -> None:
        self._table = table
        self._table_absmax = table_absmax
        self._dtype = dtype
        self.granularity = granularity
        self.channel_dim = channel_dim
        self.group_size = group_size

    def _effective_granularity(self, x: torch.Tensor) -> str:
        """Granularity actually used for ``x``.

        Falls back from ``per_group`` to ``per_tensor`` (a single group) when
        ``x.numel()`` is not divisible by ``group_size`` — e.g. tiny tensors in
        worked examples. This is documented graceful degradation, not a silent
        failure: the result is still correct, just coarser-grained.
        """
        if self.granularity == "per_group" and x.numel() % self.group_size != 0:
            return "per_tensor"
        return self.granularity

    def _codebook(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self._table, dtype=dtype, device=device)

    def compute_params(self, x: torch.Tensor) -> QuantParams:
        """Compute the per-group scale (symmetric; zero_point ≡ 0).

        Args:
            x: Float input tensor.

        Returns:
            QuantParams: scale + zero_point with the 4-bit dtype tag. The stored
            ``granularity``/``group_size`` reflect the *effective* granularity
            (see :meth:`_effective_granularity`).

        Raises:
            QuantizationError: If ``x`` contains NaN or Inf.
        """
        eff = self._effective_granularity(x)
        gs = self.group_size if eff == "per_group" else None
        absmax = compute_absmax(x, eff, self.channel_dim, gs)
        if not bool(torch.isfinite(absmax).all()):
            self._raise_nonfinite(x)
        scale, zero_point = compute_scale_symmetric(absmax, self._table_absmax)
        return QuantParams(
            scale=scale,
            zero_point=zero_point,
            dtype=self._dtype,
            granularity=eff,  # type: ignore[arg-type]
            scheme="symmetric",
            channel_dim=self.channel_dim,
            group_size=gs,
            original_shape=tuple(x.shape),
        )

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, QuantParams]:
        """Quantize ``x`` to unpacked 4-bit indices (uint8, values 0..15).

        Args:
            x: Float input tensor.

        Returns:
            tuple[Tensor, QuantParams]: ``(indices, params)`` with ``params.packed``
            False; nibble packing is the caller's responsibility.
        """
        params = self.compute_params(x)
        table = self._codebook(x.device, x.dtype)
        x_view, scale_b, _ = broadcast_for_apply(
            x, params.scale, params.zero_point, params.granularity,
            params.channel_dim, params.group_size,
        )
        x_norm = x_view / scale_b
        diffs = (x_norm.unsqueeze(-1) - table).abs()
        idx = diffs.argmin(dim=-1).to(torch.uint8)
        idx = restore_shape(idx, params.original_shape or tuple(x.shape), params.granularity)
        return idx, params
        # ✓ VERIFY — paste-able snippet proving the core math:
        # q = NF4Quantizer()
        # x = torch.tensor([-0.70, -0.53, 0.0, 0.08, 0.24, 0.72])
        # idx, _ = q.quantize(x)
        # assert idx.tolist() == [0, 1, 7, 8, 11, 15]

    def dequantize(self, x_q: torch.Tensor, params: QuantParams) -> torch.Tensor:
        """Reconstruct float32 from 4-bit indices (packed or unpacked).

        Args:
            x_q: uint8 indices (shape == original) or nibble-packed bytes
                (when ``params.packed`` is True).
            params: The :class:`QuantParams` from :meth:`quantize`.

        Returns:
            torch.Tensor: float32 reconstruction with the original shape.
        """
        table = self._codebook(x_q.device, torch.float32)
        original_shape = params.original_shape or tuple(x_q.shape)
        if params.packed:
            idx = unpack_int4(x_q, original_shape[-1]).long()
        else:
            idx = x_q.long()
        vals = table[idx]  # shape == original_shape
        v_view, scale_b, _ = broadcast_for_apply(
            vals, params.scale, params.zero_point, params.granularity,
            params.channel_dim, params.group_size,
        )
        x_r = v_view * scale_b
        return restore_shape(x_r, original_shape, params.granularity)
        # ✓ VERIFY — paste-able snippet proving the core math:
        # q = NF4Quantizer()
        # x = torch.tensor([-0.70, -0.53, 0.0, 0.08, 0.24, 0.72])
        # idx, p = q.quantize(x)
        # torch.testing.assert_close(
        #     q.dequantize(idx, p),
        #     torch.tensor([-0.72, -0.5013, 0.0, 0.0573, 0.2433, 0.72]),
        #     atol=1e-3, rtol=0)


class FP4Quantizer(_CodebookQuantizer):
    """FP4 (E2M1) quantizer — 16-value geometric float codebook.

    Args:
        granularity: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"``.
        channel_dim: Channel dimension for ``per_channel``.
        group_size: Elements per group for ``per_group``.

    Example:
        >>> import torch
        >>> q = FP4Quantizer()
        >>> idx, p = q.quantize(torch.tensor([0.0, 3.0, -6.0]))
        >>> q.dequantize(idx, p).tolist()
        [0.0, 3.0, -6.0]
    """

    def __init__(
        self,
        granularity: Literal["per_tensor", "per_channel", "per_group"] = "per_group",
        channel_dim: int = 0,
        group_size: int = 64,
    ) -> None:
        super().__init__(
            table=FP4_E2M1_VALUES,
            table_absmax=int(FP4_MAX),
            dtype=QuantDtype.FP4,
            granularity=granularity,
            channel_dim=channel_dim,
            group_size=group_size,
        )


class NF4Quantizer(_CodebookQuantizer):
    """NF4 (NormalFloat4) quantizer — 16 N(0,1)-quantile codebook (QLoRA).

    Args:
        granularity: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"``.
        channel_dim: Channel dimension for ``per_channel``.
        group_size: Elements per group for ``per_group`` (QLoRA default 64).

    Note:
        WHY NF4 BEATS FP4 ON NORMAL WEIGHTS:
        Pretrained LLM weights are approximately N(0, σ) after layer
        normalization. NF4 places its 16 quantization levels at the quantiles
        of N(0,1), so each bin captures equal probability mass 1/16. This
        minimizes E[||x - dequant(quant(x))||²] for normally-distributed x.
        FP4 E2M1 places levels geometrically — suboptimal for Gaussians. NF4
        SNR will always exceed FP4 SNR on N(0,1) data.

    Example:
        >>> import torch
        >>> q = NF4Quantizer()
        >>> idx, p = q.quantize(torch.zeros(4))
        >>> idx.tolist()
        [7, 7, 7, 7]
    """

    def __init__(
        self,
        granularity: Literal["per_tensor", "per_channel", "per_group"] = "per_group",
        channel_dim: int = 0,
        group_size: int = 64,
    ) -> None:
        super().__init__(
            table=NF4_QUANTILE_TABLE,
            table_absmax=1,
            dtype=QuantDtype.NF4,
            granularity=granularity,
            channel_dim=channel_dim,
            group_size=group_size,
        )


def double_quantize_scales(
    scales: torch.Tensor,
    inner_dtype: QuantDtype = QuantDtype.INT8,
) -> tuple[torch.Tensor, float]:
    """Quantize NF4 primary scales to reduce storage overhead (QLoRA double quant).

    A ``(4096, 4096)`` weight with ``group_size=64`` yields 262,144 fp32 scales
    (~1 MB). Quantizing them to int8 with one shared secondary scale shrinks that
    to ~256 KB (≈4×).

    Args:
        scales: ``(num_groups,)`` float32 primary scales.
        inner_dtype: Inner integer dtype (only INT8 is supported).

    Returns:
        tuple[Tensor, float]: ``(quantized_scales int8, secondary_scale float)``.
        Dequantize via ``quantized_scales.float() * secondary_scale``.

    Raises:
        NotImplementedError: If ``inner_dtype`` is not INT8.

    Example:
        >>> import torch
        >>> s = torch.tensor([0.1, 0.2, 0.4, 0.8])
        >>> s_q, sec = double_quantize_scales(s)
        >>> torch.testing.assert_close(s_q.float() * sec, s, atol=5e-3, rtol=0)
    """
    if inner_dtype != QuantDtype.INT8:
        raise NotImplementedError("double_quantize_scales only supports INT8 inner dtype")
    quantizer = Int8Quantizer(scheme="symmetric", granularity="per_tensor")
    s_q, params = quantizer.quantize(scales)
    return s_q, float(params.scale.item())
