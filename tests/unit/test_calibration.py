"""Unit tests for calibration methods and the Calibrator context manager."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from quantlib.calibration.calibrator import Calibrator
from quantlib.calibration.methods import (
    calibrate_entropy,
    calibrate_minmax,
    calibrate_mse,
    calibrate_percentile,
)
from quantlib.calibration.sensitivity import layer_sensitivity_analysis
from quantlib.core.observer import HistogramObserver, MinMaxObserver
from quantlib.quantizers.int8 import Int8Quantizer


def _hist_stats(x: torch.Tensor, bins: int = 512) -> object:
    obs = HistogramObserver(bins)
    obs.update(x)
    return obs.compute_stats()


def _minmax_stats(x: torch.Tensor) -> object:
    obs = MinMaxObserver()
    obs.update(x)
    return obs.compute_stats()


class TestCalibrationMethods:
    def test_minmax_scale_positive(self) -> None:
        scale, zp = calibrate_minmax(_minmax_stats(torch.randn(1000)))
        assert bool((scale > 0).all())

    def test_percentile_clips_outliers(self) -> None:
        x = torch.randn(5000)
        x[0] = 100.0  # extreme outlier
        scale_p, _ = calibrate_percentile(_hist_stats(x), p=99.0)
        scale_m, _ = calibrate_minmax(_minmax_stats(x))
        assert float(scale_p.reshape(-1)[0]) < float(scale_m.reshape(-1)[0])

    def test_percentile_falls_back_when_no_histogram(self) -> None:
        scale, _ = calibrate_percentile(_minmax_stats(torch.randn(100)))
        assert bool((scale > 0).all())

    def test_entropy_returns_symmetric_zero_point(self) -> None:
        _, zp = calibrate_entropy(_hist_stats(torch.randn(4000)))
        assert int(zp.reshape(-1)[0].item()) == 0

    def test_entropy_scale_positive(self) -> None:
        scale, _ = calibrate_entropy(_hist_stats(torch.randn(4000)))
        assert bool((scale > 0).all())

    def test_mse_symmetric_zero_point(self) -> None:
        _, zp = calibrate_mse(torch.randn(2000), scheme="symmetric")
        assert int(zp[0].item()) == 0

    def test_mse_scale_shape_is_one(self) -> None:
        scale, zp = calibrate_mse(torch.randn(2000))
        assert scale.shape == (1,) and zp.shape == (1,)

    def test_mse_asymmetric_runs(self) -> None:
        scale, _ = calibrate_mse(torch.randn(2000).abs(), scheme="asymmetric")
        assert bool((scale > 0).all())


class TestCalibrator:
    def _model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))

    @pytest.mark.parametrize("method", ["minmax", "percentile", "entropy", "mse"])
    def test_collects_qparams_for_every_linear(self, method: str) -> None:
        model = self._model()
        with Calibrator(model, method=method) as cal:  # type: ignore[arg-type]
            for _ in range(3):
                model(torch.randn(4, 16))
        qparams = cal.get_qparams()
        assert sorted(qparams) == ["0", "2"]

    def test_hooks_removed_after_context(self) -> None:
        model = self._model()
        with Calibrator(model, method="minmax") as cal:
            model(torch.randn(4, 16))
        assert all(len(m._forward_hooks) == 0 for m in model.modules())
        assert cal is not None

    def test_hooks_removed_even_on_exception(self) -> None:
        model = self._model()
        with pytest.raises(RuntimeError):
            with Calibrator(model, method="minmax"):
                model(torch.randn(4, 16))
                raise RuntimeError("boom")
        assert all(len(m._forward_hooks) == 0 for m in model.modules())

    def test_exclude_patterns_skip_layers(self) -> None:
        model = self._model()
        with Calibrator(model, method="minmax", exclude_patterns=[r"^2$"]) as cal:
            model(torch.randn(4, 16))
        assert sorted(cal.get_qparams()) == ["0"]

    def test_qparams_have_positive_scale(self) -> None:
        model = self._model()
        with Calibrator(model, method="percentile") as cal:
            model(torch.randn(8, 16))
        for p in cal.get_qparams().values():
            assert bool((p.scale > 0).all())

    def test_get_stats_returns_observer_stats(self) -> None:
        model = self._model()
        with Calibrator(model, method="minmax") as cal:
            model(torch.randn(4, 16))
        stats = cal.get_stats()
        assert sorted(stats) == ["0", "2"]


class TestSensitivity:
    def test_returns_score_per_linear(self) -> None:
        model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16))
        scores = layer_sensitivity_analysis(
            model, None, Int8Quantizer("symmetric", "per_channel")
        )
        assert sorted(scores) == ["0", "1"]
        assert all(v > 0 for v in scores.values())

    def test_cosine_metric_near_one(self) -> None:
        model = nn.Sequential(nn.Linear(64, 64))
        scores = layer_sensitivity_analysis(
            model, None, Int8Quantizer("symmetric", "per_channel"), metric="cosine"
        )
        assert all(0.99 < v <= 1.0 for v in scores.values())
