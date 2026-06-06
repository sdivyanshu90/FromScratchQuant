"""
Module: quantlib.core.qparams

Immutable container (:class:`QuantParams`) holding everything required to
reconstruct a float32 tensor from its quantized storage representation.

Mathematical Background:
    Dequantization is fully determined by (scale, zero_point) plus the
    granularity/scheme metadata:
        symmetric  : x_r = x_q * scale                     (theory.md Eq. 3)
        asymmetric : x_r = (x_q - zero_point) * scale       (theory.md Eq. 7)

References:
    Jacob et al., 2018, Eq. 1-2 (affine mapping).

Example:
    >>> import torch
    >>> from quantlib.core.qparams import QuantParams
    >>> from quantlib.core.dtypes import QuantDtype
    >>> p = QuantParams(
    ...     scale=torch.tensor(0.025),
    ...     zero_point=torch.tensor(0, dtype=torch.int32),
    ...     dtype=QuantDtype.INT8, granularity="per_tensor", scheme="symmetric",
    ... )
    >>> p.scheme
    'symmetric'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from quantlib.core.dtypes import QuantDtype
from quantlib.core.exceptions import NumericalError, ShapeError


@dataclass(frozen=True)
class QuantParams:
    """Immutable container for all parameters needed to quantize/dequantize a tensor.

    Everything required to reconstruct float32 from quantized storage lives here.
    No external state required — pass this alongside any quantized tensor.

    Attributes:
        scale: Strictly positive. Shape ``()`` per_tensor | ``(C,)`` per_channel
            | ``(G,)`` per_group.
        zero_point: Integer-valued, ``dtype=torch.int32``, same shape as scale.
        dtype: Target quantization dtype.
        granularity: Scope of each ``(scale, zero_point)`` pair.
        scheme: ``"symmetric"`` (zero_point == 0 always) or ``"asymmetric"`` (affine).
        channel_dim: Dimension index treated as "channel" for per_channel.
        group_size: Elements per group; required iff ``granularity == "per_group"``.
        original_shape: Needed to reshape back after per_group quantization.
        packed: True if nibble-packed (FP4 / NF4 stored as uint8).

    Raises:
        NumericalError: If any ``scale`` value is <= 0.
        ShapeError: If the ``group_size`` / ``granularity`` invariant is violated.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    dtype: QuantDtype
    granularity: Literal["per_tensor", "per_channel", "per_group"]
    scheme: Literal["symmetric", "asymmetric"]
    channel_dim: int = 0
    group_size: int | None = None
    original_shape: tuple[int, ...] | None = None
    packed: bool = False

    def __post_init__(self) -> None:
        if bool((self.scale <= 0).any()):
            raise NumericalError(
                f"scale must be strictly positive, got min={self.scale.min().item():.6e}"
            )
        if self.granularity == "per_group" and self.group_size is None:
            raise ShapeError("group_size must be set when granularity='per_group'")
        if self.granularity != "per_group" and self.group_size is not None:
            raise ShapeError("group_size must be None unless granularity='per_group'")

    def to(self, device: torch.device | str) -> "QuantParams":
        """Return a new QuantParams with scale/zero_point moved to ``device``.

        Args:
            device: Target device for the scale and zero_point tensors.

        Returns:
            QuantParams: A copy with tensors relocated; metadata unchanged.
        """
        return QuantParams(
            scale=self.scale.to(device),
            zero_point=self.zero_point.to(device),
            dtype=self.dtype,
            granularity=self.granularity,
            scheme=self.scheme,
            channel_dim=self.channel_dim,
            group_size=self.group_size,
            original_shape=self.original_shape,
            packed=self.packed,
        )
