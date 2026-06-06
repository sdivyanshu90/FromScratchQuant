"""Unit tests for QuantizedLinear / QuantizedEmbedding and quantize_model."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from quantlib import FP4Quantizer, Int8Quantizer, NF4Quantizer
from quantlib.benchmarks.accuracy import compute_snr_db
from quantlib.modules.qembedding import QuantizedEmbedding
from quantlib.modules.qlinear import QuantizedLinear
from quantlib.modules.wrappers import get_quantizable_layers, quantize_model


class TestQuantizedLinear:
    def test_forward_shape_matches_linear(self) -> None:
        lin = nn.Linear(32, 64)
        ql = QuantizedLinear.from_linear(lin, Int8Quantizer("symmetric", "per_channel"))
        x = torch.randn(4, 32)
        assert ql(x).shape == lin(x).shape == (4, 64)

    def test_does_not_mutate_source_linear(self) -> None:
        lin = nn.Linear(8, 8)
        before = lin.weight.detach().clone()
        QuantizedLinear.from_linear(lin, Int8Quantizer("symmetric"))
        torch.testing.assert_close(lin.weight.detach(), before, atol=0, rtol=0)

    def test_weight_q_dtype_int8_for_symmetric(self) -> None:
        ql = QuantizedLinear.from_linear(nn.Linear(8, 8), Int8Quantizer("symmetric"))
        assert ql.weight_q.dtype == torch.int8

    def test_only_quantized_weight_stored_no_fp32(self) -> None:
        ql = QuantizedLinear.from_linear(nn.Linear(8, 8), Int8Quantizer("symmetric"))
        names = {n for n, _ in ql.named_buffers()}
        assert "weight_q" in names
        assert "weight_fp32" not in names and "weight" not in names

    def test_bias_preserved_float32(self) -> None:
        lin = nn.Linear(8, 8, bias=True)
        ql = QuantizedLinear.from_linear(lin, Int8Quantizer("symmetric"))
        assert ql.bias is not None and ql.bias.dtype == torch.float32

    def test_no_bias_handled(self) -> None:
        lin = nn.Linear(8, 8, bias=False)
        ql = QuantizedLinear.from_linear(lin, Int8Quantizer("symmetric"))
        assert ql.bias is None
        assert ql(torch.randn(2, 8)).shape == (2, 8)

    def test_extra_repr_has_required_fields(self) -> None:
        ql = QuantizedLinear.from_linear(
            nn.Linear(512, 2048), Int8Quantizer("symmetric", "per_channel")
        )
        r = ql.extra_repr()
        for field in ("in_features", "out_features", "dtype", "scheme", "granularity"):
            assert field in r

    def test_forward_accuracy_int8_per_channel(self) -> None:
        lin = nn.Linear(128, 128, bias=False)
        ql = QuantizedLinear.from_linear(lin, Int8Quantizer("symmetric", "per_channel"))
        x = torch.randn(8, 128)
        snr = compute_snr_db(lin(x), ql(x))
        assert snr > 30.0

    def test_weight_property_returns_dequantized(self) -> None:
        lin = nn.Linear(16, 16, bias=False)
        ql = QuantizedLinear.from_linear(lin, Int8Quantizer("symmetric", "per_channel"))
        assert ql.weight.shape == (16, 16)
        assert ql.weight.dtype == torch.float32

    @pytest.mark.parametrize("quantizer", [FP4Quantizer, NF4Quantizer])
    def test_four_bit_weight_is_packed(self, quantizer: type) -> None:
        lin = nn.Linear(128, 64, bias=False)  # 128 % 64 == 0
        ql = QuantizedLinear.from_linear(lin, quantizer())
        assert ql.weight_q.dtype == torch.uint8
        assert ql.weight_params.packed
        # packed along last (in) dim -> (64, 64)
        assert ql.weight_q.shape == (64, 64)
        assert ql(torch.randn(2, 128)).shape == (2, 64)


class TestQuantizedEmbedding:
    def test_forward_shape(self) -> None:
        emb = nn.Embedding(20, 8)
        qe = QuantizedEmbedding.from_embedding(emb, Int8Quantizer("symmetric", "per_channel"))
        idx = torch.tensor([0, 5, 19])
        assert qe(idx).shape == (3, 8)

    def test_accuracy_reasonable(self) -> None:
        emb = nn.Embedding(50, 32)
        qe = QuantizedEmbedding.from_embedding(emb, Int8Quantizer("symmetric", "per_channel"))
        idx = torch.arange(50)
        snr = compute_snr_db(emb(idx), qe(idx))
        assert snr > 30.0


class TestQuantizeModel:
    def test_replaces_all_linears(self) -> None:
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        qm = quantize_model(model, Int8Quantizer("symmetric"))
        assert isinstance(qm[0], QuantizedLinear)
        assert isinstance(qm[2], QuantizedLinear)

    def test_exclude_patterns_keep_layer_fp32(self) -> None:
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        qm = quantize_model(model, Int8Quantizer("symmetric"), exclude_patterns=[r"^2$"])
        assert isinstance(qm[0], QuantizedLinear)
        assert isinstance(qm[2], nn.Linear) and not isinstance(qm[2], QuantizedLinear)

    def test_not_inplace_leaves_original(self) -> None:
        model = nn.Sequential(nn.Linear(8, 8))
        qm = quantize_model(model, Int8Quantizer("symmetric"), inplace=False)
        assert isinstance(model[0], nn.Linear) and not isinstance(model[0], QuantizedLinear)
        assert isinstance(qm[0], QuantizedLinear)

    def test_get_quantizable_layers_lists_linears(self) -> None:
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        layers = get_quantizable_layers(model)
        assert sorted(layers) == ["0", "2"]

    def test_nested_module_replacement(self) -> None:
        model = nn.Sequential(nn.Sequential(nn.Linear(8, 8)), nn.Linear(8, 4))
        qm = quantize_model(model, Int8Quantizer("symmetric"))
        assert isinstance(qm[0][0], QuantizedLinear)
        assert isinstance(qm[1], QuantizedLinear)
