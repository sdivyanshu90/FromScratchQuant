"""End-to-end integration tests: quantize -> infer -> measure pipeline."""

from __future__ import annotations

import torch
from torch import nn

from quantlib import (
    Int8Quantizer,
    NF4Quantizer,
    QuantizedLinear,
    load_quantized,
    quantize_model,
    save_quantized,
)
from quantlib.benchmarks.accuracy import compute_cosine_similarity
from quantlib.benchmarks.profiler import memory_footprint_bytes


def test_no_unquantized_linears_remain(tiny_transformer: nn.Module) -> None:
    q = Int8Quantizer("symmetric", "per_channel")
    qm = quantize_model(tiny_transformer, q)
    for name, mod in qm.named_modules():
        is_unquantized_linear = isinstance(mod, nn.Linear) and not isinstance(
            mod, QuantizedLinear
        )
        assert not is_unquantized_linear, f"Unquantized nn.Linear found: {name}"


def test_quantized_forward_output_shape(tiny_transformer: nn.Module) -> None:
    qm = quantize_model(tiny_transformer, Int8Quantizer("symmetric"))
    x = torch.randn(2, 10, 128)
    with torch.no_grad():
        assert qm(x).shape == (2, 10, 128)


def test_quantized_transformer_tracks_fp32(tiny_transformer: nn.Module) -> None:
    x = torch.randn(2, 10, 128)
    with torch.no_grad():
        ref = tiny_transformer(x)
    qm = quantize_model(tiny_transformer, Int8Quantizer("symmetric", "per_channel"))
    with torch.no_grad():
        out = qm(x)
    # Whole-model output: weight-only INT8 error compounds through 2 layers +
    # attention, so absolute SNR is modest (~12 dB). Cosine similarity is the
    # robust "still tracks the FP32 model" check.
    assert compute_cosine_similarity(ref, out) > 0.9


def test_int8_memory_reduction_1p5x_to_4x() -> None:
    linear = nn.Linear(512, 2048, bias=False)
    fp32_bytes = linear.weight.numel() * 4
    ql = QuantizedLinear.from_linear(linear, Int8Quantizer("symmetric", "per_channel"))
    int8_bytes = ql.weight_q.numel() * 1
    ratio = fp32_bytes / int8_bytes
    assert 1.5 < ratio <= 4.0, f"Expected 1.5x-4x reduction, got {ratio:.2f}x"


def test_nf4_memory_reduction_near_8x() -> None:
    linear = nn.Linear(512, 2048, bias=False)
    fp32_bytes = linear.weight.numel() * 4
    ql = QuantizedLinear.from_linear(linear, NF4Quantizer())
    packed_bytes = ql.weight_q.numel() * 1
    assert fp32_bytes / packed_bytes >= 7.5


def test_module_footprint_smaller_after_quantization() -> None:
    linear = nn.Linear(256, 256, bias=False)
    ql = QuantizedLinear.from_linear(linear, Int8Quantizer("symmetric", "per_channel"))
    assert memory_footprint_bytes(ql) < memory_footprint_bytes(linear)


def test_serialization_roundtrip(tiny_transformer: nn.Module, tmp_path: object) -> None:
    q = Int8Quantizer("symmetric")
    qm = quantize_model(tiny_transformer, q)
    save_quantized(qm, tmp_path)  # type: ignore[arg-type]
    loaded = load_quantized(tmp_path)  # type: ignore[arg-type]
    assert set(loaded.keys()) == {
        n for n, m in qm.named_modules() if isinstance(m, QuantizedLinear)
    }


def test_loaded_weights_reconstruct_original(
    tiny_transformer: nn.Module, tmp_path: object
) -> None:
    q = Int8Quantizer("symmetric", "per_channel")
    qm = quantize_model(tiny_transformer, q)
    save_quantized(qm, tmp_path)  # type: ignore[arg-type]
    loaded = load_quantized(tmp_path)  # type: ignore[arg-type]
    name = next(iter(loaded))
    wq, params = loaded[name]
    recon = q.dequantize(wq, params)
    ref = dict(qm.named_modules())[name].weight
    torch.testing.assert_close(recon, ref, atol=1e-6, rtol=0)


def test_nf4_end_to_end_pipeline() -> None:
    model = nn.Sequential(nn.Linear(256, 256, bias=False), nn.ReLU(), nn.Linear(256, 128, bias=False))
    x = torch.randn(4, 256)
    with torch.no_grad():
        ref = model(x)
    qm = quantize_model(model, NF4Quantizer())
    with torch.no_grad():
        out = qm(x)
    assert out.shape == ref.shape


def test_quantize_then_calibrate_independent() -> None:
    from quantlib import Calibrator

    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16))
    with Calibrator(model, method="mse") as cal:
        for _ in range(2):
            model(torch.randn(8, 32))
    qparams = cal.get_qparams()
    qm = quantize_model(model, Int8Quantizer("symmetric", "per_channel"))
    assert len(qparams) == 2
    assert qm(torch.randn(2, 32)).shape == (2, 16)
