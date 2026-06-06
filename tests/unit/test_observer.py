"""Unit tests for quantlib.core.observer (MinMax + Histogram observers)."""

from __future__ import annotations

import pytest
import torch

from quantlib.core.exceptions import CalibrationError
from quantlib.core.observer import HistogramObserver, MinMaxObserver


def test_minmax_tracks_running_extremes() -> None:
    obs = MinMaxObserver()
    obs.update(torch.tensor([-1.0, 2.0]))
    obs.update(torch.tensor([3.0, -5.0]))
    stats = obs.compute_stats()
    assert stats.running_min.item() == -5.0
    assert stats.running_max.item() == 3.0


def test_minmax_raises_without_data() -> None:
    with pytest.raises(CalibrationError):
        MinMaxObserver().compute_stats()


def test_minmax_ignores_empty_batches() -> None:
    obs = MinMaxObserver()
    obs.update(torch.empty(0))
    obs.update(torch.tensor([1.0, 4.0]))
    stats = obs.compute_stats()
    assert stats.running_max.item() == 4.0


def test_histogram_count_is_conserved_single_batch() -> None:
    obs = HistogramObserver(num_bins=64)
    obs.update(torch.randn(1000))
    stats = obs.compute_stats()
    assert int(stats.histogram.sum().item()) == 1000


def test_histogram_count_conserved_across_widening_batches() -> None:
    obs = HistogramObserver(num_bins=128)
    obs.update(torch.linspace(-1.0, 1.0, 500))
    obs.update(torch.linspace(-5.0, 5.0, 700))  # widens the range -> rebinning
    stats = obs.compute_stats()
    assert int(stats.histogram.sum().item()) == 1200


def test_histogram_edges_span_observed_range() -> None:
    obs = HistogramObserver(num_bins=32)
    obs.update(torch.tensor([-2.0, 0.0, 3.0]))
    stats = obs.compute_stats()
    assert stats.bin_edges is not None
    torch.testing.assert_close(stats.bin_edges[0], torch.tensor(-2.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(stats.bin_edges[-1], torch.tensor(3.0), atol=1e-6, rtol=0)


def test_histogram_shape() -> None:
    obs = HistogramObserver(num_bins=256)
    obs.update(torch.randn(100))
    stats = obs.compute_stats()
    assert stats.histogram is not None and stats.histogram.shape == (256,)
    assert stats.bin_edges is not None and stats.bin_edges.shape == (257,)


def test_histogram_raises_without_data() -> None:
    with pytest.raises(CalibrationError):
        HistogramObserver().compute_stats()
