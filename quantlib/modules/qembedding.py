"""
Module: quantlib.modules.qembedding

:class:`QuantizedEmbedding` — an ``nn.Embedding`` replacement that stores its
table quantized and dequantizes the looked-up rows (or the whole table) in
``forward``.

Mathematical Background:
    forward: w_fp32 = dequantize(weight_q, params); y = w_fp32[input]
    Identical scheme to QuantizedLinear, applied to the embedding table.

References:
    Dettmers et al., 2023 — "QLoRA".

Example:
    >>> import torch
    >>> from torch import nn
    >>> from quantlib.quantizers.int8 import Int8Quantizer
    >>> from quantlib.modules.qembedding import QuantizedEmbedding
    >>> emb = nn.Embedding(10, 4)
    >>> qe = QuantizedEmbedding.from_embedding(emb, Int8Quantizer("symmetric", "per_channel"))
    >>> qe(torch.tensor([0, 3, 9])).shape
    torch.Size([3, 4])
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from quantlib.core.qparams import QuantParams
from quantlib.quantizers.base import BaseQuantizer
from quantlib.quantizers.utils import pack_int4


class QuantizedEmbedding(nn.Module):
    """Weight-only quantized replacement for ``nn.Embedding``.

    Registered buffers: ``weight_q``, ``scale``, ``zero_point`` (move with the
    module).

    Attributes:
        weight_params: full :class:`QuantParams` metadata.
        quantizer: quantizer used to dequantize the table.
        num_embeddings, embedding_dim: table dimensions.
        padding_idx: forwarded to ``F.embedding`` if set on the source layer.
    """

    def __init__(
        self,
        weight_q: torch.Tensor,
        weight_params: QuantParams,
        quantizer: BaseQuantizer,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight_params = weight_params
        self.quantizer = quantizer
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("scale", weight_params.scale)
        self.register_buffer("zero_point", weight_params.zero_point)

    @classmethod
    def from_embedding(
        cls, embedding: nn.Embedding, quantizer: BaseQuantizer
    ) -> "QuantizedEmbedding":
        """Build a QuantizedEmbedding by quantizing an existing ``nn.Embedding``.

        Args:
            embedding: Source embedding layer (not modified).
            quantizer: Quantizer applied to ``embedding.weight``.

        Returns:
            QuantizedEmbedding: the quantized equivalent.
        """
        weight = embedding.weight.detach().clone()
        weight_q, params = quantizer.quantize(weight)
        if params.dtype.bits == 4:
            weight_q = pack_int4(weight_q)
            params = replace(params, packed=True)
        return cls(
            weight_q=weight_q,
            weight_params=params,
            quantizer=quantizer,
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
        )

    def _live_params(self) -> QuantParams:
        return replace(self.weight_params, scale=self.scale, zero_point=self.zero_point)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized float32 embedding table, reconstructed on access.

        Returns:
            torch.Tensor: the float32 reconstruction of the stored table.
        """
        return self.quantizer.dequantize(self.weight_q, self._live_params())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Look up dequantized embedding rows.

        Args:
            input: Long tensor of indices, any shape.

        Returns:
            torch.Tensor: ``input.shape + (embedding_dim,)`` float embeddings.
        """
        w_fp32 = self.quantizer.dequantize(self.weight_q, self._live_params())
        return F.embedding(input, w_fp32, self.padding_idx)

    def extra_repr(self) -> str:
        """One-line description for ``print(model)``."""
        p = self.weight_params
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"dtype={p.dtype.name}, scheme={p.scheme}, granularity={p.granularity}"
        )
