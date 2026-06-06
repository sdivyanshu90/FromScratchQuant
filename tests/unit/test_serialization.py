"""Unit tests for quantlib.serialization.checkpoint (save/load round-trips)."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from quantlib import Int8Quantizer, NF4Quantizer, quantize_model
from quantlib.core.exceptions import CalibrationError
from quantlib.modules.qlinear import QuantizedLinear
from quantlib.serialization.checkpoint import load_quantized, save_quantized


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


def test_save_creates_two_files(tmp_path: object) -> None:
    qm = quantize_model(_model(), Int8Quantizer("symmetric", "per_channel"))
    save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]
    assert (tmp_path / "m.safetensors").exists()  # type: ignore[operator]
    assert (tmp_path / "m.qconfig.json").exists()  # type: ignore[operator]


def test_roundtrip_keys_match_quantized_layers(tmp_path: object) -> None:
    qm = quantize_model(_model(), Int8Quantizer("symmetric", "per_channel"))
    save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]
    loaded = load_quantized(tmp_path, model_name="m")  # type: ignore[arg-type]
    expected = {n for n, m in qm.named_modules() if isinstance(m, QuantizedLinear)}
    assert set(loaded.keys()) == expected


def test_roundtrip_weight_and_scale_preserved(tmp_path: object) -> None:
    qm = quantize_model(_model(), Int8Quantizer("symmetric", "per_channel"))
    save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]
    loaded = load_quantized(tmp_path, model_name="m")  # type: ignore[arg-type]
    wq, params = loaded["0"]
    ref = qm[0]
    torch.testing.assert_close(wq, ref.weight_q, atol=0, rtol=0)
    torch.testing.assert_close(params.scale, ref.scale, atol=0, rtol=0)


def test_dequantized_reconstruction_matches(tmp_path: object) -> None:
    qm = quantize_model(_model(), Int8Quantizer("symmetric", "per_channel"))
    save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]
    loaded = load_quantized(tmp_path, model_name="m")  # type: ignore[arg-type]
    q = Int8Quantizer("symmetric", "per_channel")
    wq, params = loaded["0"]
    recon = q.dequantize(wq, params)
    torch.testing.assert_close(recon, qm[0].weight, atol=1e-6, rtol=0)


def test_packed_nf4_roundtrip(tmp_path: object) -> None:
    model = nn.Sequential(nn.Linear(128, 64, bias=False))  # 128 % 64 == 0
    qm = quantize_model(model, NF4Quantizer())
    save_quantized(qm, tmp_path, model_name="nf4")  # type: ignore[arg-type]
    loaded = load_quantized(tmp_path, model_name="nf4")  # type: ignore[arg-type]
    wq, params = loaded["0"]
    assert params.packed and wq.dtype == torch.uint8
    recon = NF4Quantizer().dequantize(wq, params)
    torch.testing.assert_close(recon, qm[0].weight, atol=1e-6, rtol=0)


def test_no_silent_overwrite(tmp_path: object) -> None:
    qm = quantize_model(_model(), Int8Quantizer("symmetric"))
    save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]
    with pytest.raises(FileExistsError):
        save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]


def test_missing_files_raise(tmp_path: object) -> None:
    with pytest.raises(FileNotFoundError):
        load_quantized(tmp_path, model_name="absent")  # type: ignore[arg-type]


def test_malformed_json_raises_calibration_error(tmp_path: object) -> None:
    qm = quantize_model(_model(), Int8Quantizer("symmetric"))
    save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]
    (tmp_path / "m.qconfig.json").write_text("{ not valid json")  # type: ignore[operator]
    with pytest.raises(CalibrationError):
        load_quantized(tmp_path, model_name="m")  # type: ignore[arg-type]


def test_incompatible_version_raises(tmp_path: object) -> None:
    qm = quantize_model(_model(), Int8Quantizer("symmetric"))
    save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]
    cfg_path = tmp_path / "m.qconfig.json"  # type: ignore[operator]
    cfg = json.loads(cfg_path.read_text())
    cfg["quantlib_version"] = "9.9.9"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(CalibrationError, match="incompatible"):
        load_quantized(tmp_path, model_name="m")  # type: ignore[arg-type]


def test_config_json_is_human_readable(tmp_path: object) -> None:
    qm = quantize_model(_model(), Int8Quantizer("symmetric", "per_channel"))
    save_quantized(qm, tmp_path, model_name="m")  # type: ignore[arg-type]
    cfg = json.loads((tmp_path / "m.qconfig.json").read_text())  # type: ignore[operator]
    assert cfg["quantlib_version"]
    assert isinstance(cfg["layers"]["0"]["scale"], list)
    assert cfg["layers"]["0"]["dtype"] == "int8"
