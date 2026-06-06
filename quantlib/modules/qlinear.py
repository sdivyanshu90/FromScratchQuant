"""
Module: quantlib.modules.qlinear

:class:`QuantizedLinear` — a drop-in ``nn.Linear`` replacement that stores its
weight quantized (int8, or nibble-packed uint8 for FP4/NF4) and dequantizes
on-the-fly in ``forward``. Weight-only quantization; bias stays float32.

Mathematical Background:
    forward: w_fp32 = dequantize(weight_q, params); y = x @ w_fp32^T + bias
    Trades one dequant per call for 4× (INT8) / 8× (FP4/NF4) smaller weights.

References:
    Dettmers et al., 2023 — "QLoRA" (4-bit nibble-packed weights).

Example:
    >>> import torch
    >>> from torch import nn
    >>> from quantlib.quantizers.int8 import Int8Quantizer
    >>> from quantlib.modules.qlinear import QuantizedLinear
    >>> lin = nn.Linear(8, 4)
    >>> ql = QuantizedLinear.from_linear(lin, Int8Quantizer("symmetric", "per_channel"))
    >>> ql(torch.randn(2, 8)).shape
    torch.Size([2, 4])
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from quantlib.core.qparams import QuantParams
from quantlib.quantizers.base import BaseQuantizer
from quantlib.quantizers.utils import pack_int4


class QuantizedLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with weight-only quantization.

    Registered buffers (move with ``.to(device)`` / ``.cuda()``):
        ``weight_q``: quantized weight, int8 or nibble-packed uint8.
        ``scale``, ``zero_point``: the QuantParams tensors.
        ``bias``: float32 (or absent).

    Attributes:
        weight_params: full :class:`QuantParams` metadata.
        quantizer: the :class:`BaseQuantizer` used to dequantize in ``forward``.
        in_features, out_features: linear layer dimensions.
    """

    def __init__(
        self,
        weight_q: torch.Tensor,
        weight_params: QuantParams,
        quantizer: BaseQuantizer,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_params = weight_params
        self.quantizer = quantizer
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", weight_params.scale)
        self.register_buffer("zero_point", weight_params.zero_point)
        self.register_buffer("bias", bias if bias is not None else None)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, quantizer: BaseQuantizer
    ) -> "QuantizedLinear":
        """Build a QuantizedLinear by quantizing an existing ``nn.Linear``.

        Does not modify ``linear``: its weight is copied and quantized, its bias
        is copied verbatim. 4-bit dtypes are nibble-packed for 8× compression.

        Args:
            linear: Source layer.
            quantizer: Quantizer applied to ``linear.weight``.

        Returns:
            QuantizedLinear: the quantized equivalent.
        """
        weight = linear.weight.detach().clone()
        weight_q, params = quantizer.quantize(weight)
        if params.dtype.bits == 4:
            weight_q = pack_int4(weight_q)
            params = replace(params, packed=True)
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        return cls(
            weight_q=weight_q,
            weight_params=params,
            quantizer=quantizer,
            bias=bias,
            in_features=linear.in_features,
            out_features=linear.out_features,
        )

    def _live_params(self) -> QuantParams:
        """QuantParams rebound to the current device buffers (device-safe)."""
        return replace(self.weight_params, scale=self.scale, zero_point=self.zero_point)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized float32 weight, reconstructed on access.

        Exposed so modules that read ``.weight`` directly (e.g. the fused path
        in ``nn.MultiheadAttention.out_proj``) remain drop-in compatible.

        Returns:
            torch.Tensor: the float32 reconstruction of the stored weight.
        """
        return self.quantizer.dequantize(self.weight_q, self._live_params())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize the weight and apply the linear map.

        Args:
            x: float32 input of shape ``(..., in_features)``.

        Returns:
            torch.Tensor: ``(..., out_features)`` output (x stays float32).
        """
        w_fp32 = self.quantizer.dequantize(self.weight_q, self._live_params())
        return F.linear(x, w_fp32.to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        """One-line description shown by ``print(model)``.

        Returns:
            str: includes in_features, out_features, dtype, scheme, granularity.
        """
        p = self.weight_params
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"dtype={p.dtype.name}, scheme={p.scheme}, granularity={p.granularity}"
        )
