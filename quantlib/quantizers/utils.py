"""
Module: quantlib.quantizers.utils

Pure, vectorized helper functions shared by every quantizer: 4-bit nibble
packing/unpacking, the symmetric/asymmetric scale formulas, a safe divide,
and dtype clamping.

Mathematical Background:
    Symmetric   scale = absmax / q_max                         (theory.md Eq. 1)
    Asymmetric  scale = (x_max - x_min) / (q_max - q_min)       (theory.md Eq. 4)
                zero_point = clamp(round(q_min - x_min/scale))  (theory.md Eq. 5)
    Nibble packing stores two 4-bit codes per uint8 byte
    (high nibble = even index, low nibble = odd index).

References:
    Jacob et al., 2018, Eq. 1-2.

Example:
    >>> import torch
    >>> from quantlib.quantizers.utils import pack_int4, unpack_int4
    >>> x = torch.tensor([3, 7, 12, 0, 5], dtype=torch.uint8)
    >>> unpack_int4(pack_int4(x), 5).tolist()
    [3, 7, 12, 0, 5]
"""

from __future__ import annotations

import warnings
from typing import Final

import torch

from quantlib.core.dtypes import QuantDtype
from quantlib.core.exceptions import ShapeError

# ── Named numerical constants (every magic number lives here) ──────────────────
INT8_MAX: Final[int] = 127  # torch.iinfo(torch.int8).max — Jacob et al. 2018, Eq.2
INT8_MIN: Final[int] = -128  # torch.iinfo(torch.int8).min
UINT8_MAX: Final[int] = 255  # torch.iinfo(torch.uint8).max
UINT8_MIN: Final[int] = 0  # torch.iinfo(torch.uint8).min
REDUCED_INT8_MAX: Final[int] = 63  # VNNI reduce_range upper bound
REDUCED_INT8_MIN: Final[int] = -64  # VNNI reduce_range lower bound
_MIN_SCALE: Final[float] = 1e-8  # smallest representable scale; below -> clamp + warn
_NIBBLE_MASK: Final[int] = 0x0F  # low-4-bits mask
_NIBBLE_BITS: Final[int] = 4  # bits per packed nibble


def clamp_scale(scale: torch.Tensor) -> torch.Tensor:
    """Clamp a scale tensor to ``_MIN_SCALE``, warning if any element is clamped.

    Args:
        scale: Tensor of (intended strictly-positive) scale values.

    Returns:
        torch.Tensor: ``scale`` with every element >= ``_MIN_SCALE``.

    Note:
        Silent clamping is forbidden (hides bugs); a ``UserWarning`` is raised
        whenever clamping actually changes a value.
    """
    needs_clamp = (scale > 0) & (scale < _MIN_SCALE)
    if bool(needs_clamp.any()):
        warnings.warn(
            f"scale {scale[needs_clamp].min().item():.2e} clamped to {_MIN_SCALE:.0e}",
            UserWarning,
            stacklevel=3,
        )
    return scale.clamp(min=_MIN_SCALE)


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Divide elementwise, clamping the denominator away from zero first.

    Args:
        numerator: Dividend tensor.
        denominator: Divisor tensor; clamped to ``_MIN_SCALE`` magnitude.

    Returns:
        torch.Tensor: ``numerator / denominator`` with no division by zero.
    """
    safe_denom = denominator.clamp(min=_MIN_SCALE)
    return numerator / safe_denom


def compute_scale_symmetric(
    absmax: torch.Tensor, q_max: int = INT8_MAX
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric scale + (always-zero) zero_point from an absmax tensor.

    Args:
        absmax: Non-negative tensor (any shape) of per-group absolute maxima.
        q_max: Positive integer quantization ceiling (127, or 63 for reduce_range).

    Returns:
        tuple[Tensor, Tensor]: ``(scale, zero_point)``. ``zero_point`` is int32
        zeros with the same shape as ``scale``.

    Note:
        ``absmax == 0`` (all-zero input) maps to ``scale = 1.0`` so the result
        stays strictly positive and dequant is the identity on zeros.

    Example:
        >>> s, z = compute_scale_symmetric(torch.tensor(3.2))
        >>> torch.testing.assert_close(s, torch.tensor(3.2 / 127), atol=1e-6, rtol=0)
        >>> int(z.item())
        0
    """
    scale = absmax / q_max
    scale = torch.where(absmax == 0, torch.ones_like(scale), scale)
    scale = clamp_scale(scale)
    zero_point = torch.zeros_like(scale, dtype=torch.int32)
    return scale, zero_point


def compute_scale_asymmetric(
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    q_min: int = UINT8_MIN,
    q_max: int = UINT8_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine scale + zero_point mapping ``[x_min, x_max]`` to ``[q_min, q_max]``.

    Args:
        x_min: Per-group minima (any shape, broadcastable to ``x_max``).
        x_max: Per-group maxima (same shape as ``x_min``).
        q_min: Integer quantization floor (0 for uint8).
        q_max: Integer quantization ceiling (255 for uint8).

    Returns:
        tuple[Tensor, Tensor]: ``(scale, zero_point)`` where ``zero_point`` is
        int32, clamped to ``[q_min, q_max]``.

    Note:
        A constant input (``x_max == x_min``) maps to ``scale = 1.0`` and
        ``zero_point = 0`` (per the INT8 edge-case table).

    Example:
        >>> s, z = compute_scale_asymmetric(torch.tensor(-2.3), torch.tensor(3.2))
        >>> int(z.item())
        107
    """
    rng = x_max - x_min
    is_const = rng == 0
    scale = rng / (q_max - q_min)
    scale = torch.where(is_const, torch.ones_like(scale), scale)
    scale = clamp_scale(scale)
    zero_point_f = torch.round(q_min - x_min / scale)
    zero_point_f = torch.where(is_const, torch.zeros_like(zero_point_f), zero_point_f)
    zero_point = zero_point_f.clamp(q_min, q_max).to(torch.int32)
    return scale, zero_point


def clamp_to_dtype(x_q_float: torch.Tensor, dtype: QuantDtype) -> torch.Tensor:
    """Clamp rounded values to a dtype's integer range and cast to storage dtype.

    Args:
        x_q_float: Float tensor of already-rounded quantized codes.
        dtype: Target :class:`QuantDtype` (INT8 -> int8, else -> uint8).

    Returns:
        torch.Tensor: Clamped tensor cast to ``dtype.storage_dtype``.
    """
    if dtype == QuantDtype.INT8:
        return x_q_float.clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    return x_q_float.clamp(UINT8_MIN, UINT8_MAX).to(torch.uint8)


def _check_group_divisible(x: torch.Tensor, group_size: int) -> None:
    """Raise :class:`ShapeError` if ``x.numel()`` is not a multiple of group_size."""
    if x.numel() % group_size != 0:
        raise ShapeError(
            f"numel {x.numel()} not divisible by group_size {group_size}"
        )


def compute_absmax(
    x: torch.Tensor,
    granularity: str,
    channel_dim: int,
    group_size: int | None,
) -> torch.Tensor:
    """Reduce ``x`` to per-group absolute maxima for symmetric quantization.

    Args:
        x: Input tensor.
        granularity: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"``.
        channel_dim: Channel dimension for ``per_channel``.
        group_size: Elements per group for ``per_group``.

    Returns:
        torch.Tensor: absmax with shape ``()`` / ``(C,)`` / ``(G,)`` respectively.

    Raises:
        ShapeError: If ``per_group`` and numel is not divisible by group_size.
    """
    if granularity == "per_tensor":
        return x.abs().amax()
    if granularity == "per_channel":
        dims = tuple(d for d in range(x.ndim) if d != channel_dim)
        return x.abs().amax(dim=dims) if dims else x.abs()
    assert group_size is not None
    _check_group_divisible(x, group_size)
    return x.reshape(-1, group_size).abs().amax(dim=1)


def compute_minmax(
    x: torch.Tensor,
    granularity: str,
    channel_dim: int,
    group_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``x`` to per-group (min, max) for asymmetric quantization.

    Args:
        x: Input tensor.
        granularity: ``"per_tensor"`` | ``"per_channel"`` | ``"per_group"``.
        channel_dim: Channel dimension for ``per_channel``.
        group_size: Elements per group for ``per_group``.

    Returns:
        tuple[Tensor, Tensor]: ``(x_min, x_max)`` with matching group shape.

    Raises:
        ShapeError: If ``per_group`` and numel is not divisible by group_size.
    """
    if granularity == "per_tensor":
        return x.amin(), x.amax()
    if granularity == "per_channel":
        dims = tuple(d for d in range(x.ndim) if d != channel_dim)
        if not dims:
            return x.clone(), x.clone()
        return x.amin(dim=dims), x.amax(dim=dims)
    assert group_size is not None
    _check_group_divisible(x, group_size)
    xr = x.reshape(-1, group_size)
    return xr.amin(dim=1), xr.amax(dim=1)


def broadcast_for_apply(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    granularity: str,
    channel_dim: int,
    group_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape ``x``/``scale``/``zero_point`` so elementwise ops broadcast cleanly.

    Args:
        x: Input tensor.
        scale: Compact scale ``()`` / ``(C,)`` / ``(G,)``.
        zero_point: Compact zero_point, same shape as ``scale``.
        granularity: Granularity string.
        channel_dim: Channel dimension for ``per_channel``.
        group_size: Elements per group for ``per_group``.

    Returns:
        tuple[Tensor, Tensor, Tensor]: ``(x_view, scale_b, zp_b)`` where
        ``x_view`` is ``x`` (or its ``(-1, group_size)`` reshape for per_group)
        and the params broadcast against ``x_view``. Use :func:`restore_shape`
        to map a result computed on ``x_view`` back to ``x``'s shape.
    """
    if granularity == "per_tensor":
        return x, scale, zero_point
    if granularity == "per_channel":
        shape = [1] * x.ndim
        shape[channel_dim] = -1
        return x, scale.reshape(shape), zero_point.reshape(shape)
    assert group_size is not None
    xr = x.reshape(-1, group_size)
    return xr, scale.unsqueeze(-1), zero_point.unsqueeze(-1)


def restore_shape(
    result: torch.Tensor,
    original_shape: tuple[int, ...],
    granularity: str,
) -> torch.Tensor:
    """Reshape a per-group result back to the original tensor shape.

    Args:
        result: Tensor produced on the ``broadcast_for_apply`` view.
        original_shape: Shape to restore (only consulted for ``per_group``).
        granularity: Granularity string.

    Returns:
        torch.Tensor: ``result`` reshaped to ``original_shape`` for ``per_group``,
        otherwise ``result`` unchanged.
    """
    if granularity == "per_group":
        return result.reshape(original_shape)
    return result


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 tensor of 4-bit codes (0..15) two-per-byte along the last dim.

    Args:
        x: uint8 tensor with values in ``[0, 15]``, shape ``(..., N)``. If ``N``
            is odd, one zero is appended before packing.

    Returns:
        torch.Tensor: uint8 tensor of shape ``(..., ceil(N/2))``. The high nibble
        holds even-indexed values, the low nibble holds odd-indexed values.

    Example:
        >>> pack_int4(torch.tensor([3, 7, 12, 0, 5], dtype=torch.uint8)).tolist()
        [55, 192, 80]
    """
    # ┌─ PLAN: pack_int4 ──────────────────────────────────────────────────────────┐
    # │ Mathematical Invariants:                                                     │
    # │   • byte = (hi << 4) | lo, hi/lo in [0,15] -> byte in [0,255], reversible.   │
    # │   • output length == ceil(N/2); content of trailing pad nibble is 0.         │
    # │ Numerical Edge Cases:                                                        │
    # │   • odd N -> append one 0 so pairs are well-formed.                          │
    # │   • values already masked with 0x0F (defensive against >15).                 │
    # │ Module Dependencies: none (torch only).                                      │
    # │ Hand-Computed Verification:                                                  │
    # │   Input:    [3, 7, 12, 0, 5] -> pad [3,7,12,0,5,0]                            │
    # │   Expected: [(3<<4)|7, (12<<4)|0, (5<<4)|0] = [55, 192, 80]                   │
    # │   Tolerance: exact integer equality.                                         │
    # └──────────────────────────────────────────────────────────────────────────────┘
    n = x.shape[-1]
    if n % 2 == 1:
        pad = x.new_zeros((*x.shape[:-1], 1))
        x = torch.cat([x, pad], dim=-1)
    hi = (x[..., 0::2] & _NIBBLE_MASK) << _NIBBLE_BITS
    lo = x[..., 1::2] & _NIBBLE_MASK
    return (hi | lo).to(torch.uint8)
    # ✓ VERIFY — paste-able snippet proving the core math:
    # x = torch.tensor([3, 7, 12, 0, 5], dtype=torch.uint8)
    # assert pack_int4(x).tolist() == [55, 192, 80]


def unpack_int4(x: torch.Tensor, n_original: int) -> torch.Tensor:
    """Unpack a nibble-packed uint8 tensor back to ``n_original`` 4-bit codes.

    Args:
        x: Packed uint8 tensor, shape ``(..., ceil(n_original/2))``.
        n_original: Number of codes before packing (used to trim odd-N padding).

    Returns:
        torch.Tensor: uint8 tensor with values in ``[0, 15]``, shape
        ``(..., n_original)``.

    Example:
        >>> unpack_int4(torch.tensor([55, 192, 80], dtype=torch.uint8), 5).tolist()
        [3, 7, 12, 0, 5]
    """
    # ┌─ PLAN: unpack_int4 ────────────────────────────────────────────────────────┐
    # │ Mathematical Invariants:                                                     │
    # │   • hi = (byte >> 4) & 0x0F, lo = byte & 0x0F; exact inverse of pack.        │
    # │   • interleave [hi0, lo0, hi1, lo1, ...] then trim to n_original.            │
    # │ Numerical Edge Cases:                                                        │
    # │   • odd n_original -> drop the final padded nibble after interleaving.       │
    # │ Module Dependencies: none (torch only).                                      │
    # │ Hand-Computed Verification:                                                  │
    # │   Input:    [55, 192, 80], n_original=5                                       │
    # │   Expected: [3, 7, 12, 0, 5, 0] -> trim -> [3, 7, 12, 0, 5]                   │
    # │   Tolerance: exact integer equality.                                         │
    # └──────────────────────────────────────────────────────────────────────────────┘
    hi = (x >> _NIBBLE_BITS) & _NIBBLE_MASK
    lo = x & _NIBBLE_MASK
    unpacked = torch.stack([hi, lo], dim=-1).flatten(start_dim=-2)
    return unpacked[..., :n_original].to(torch.uint8)
    # ✓ VERIFY — paste-able snippet proving the core math:
    # packed = torch.tensor([55, 192, 80], dtype=torch.uint8)
    # assert unpack_int4(packed, 5).tolist() == [3, 7, 12, 0, 5]
