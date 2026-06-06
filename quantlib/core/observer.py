"""
Module: quantlib.core.observer

Streaming statistics collectors used during calibration. Observers consume
tensors batch-by-batch and accumulate the running min/max (and, for the
histogram observer, a fixed-bin histogram) needed by the calibration methods.

Mathematical Background:
    running_min = min over all observed elements
    running_max = max over all observed elements
    histogram   = element counts over `num_bins` equal-width bins spanning
                  [running_min, running_max]; bins are re-mapped when the
                  observed range grows (TensorRT-style dynamic histogram).

References:
    Migacz, 2017 — "8-bit Inference with TensorRT" (KL calibration histogram).

Example:
    >>> import torch
    >>> from quantlib.core.observer import MinMaxObserver
    >>> obs = MinMaxObserver()
    >>> obs.update(torch.tensor([-1.0, 2.0]))
    >>> obs.update(torch.tensor([3.0]))
    >>> stats = obs.compute_stats()
    >>> (stats.running_min.item(), stats.running_max.item())
    (-1.0, 3.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from quantlib.core.exceptions import CalibrationError

_DEFAULT_BINS: Final[int] = 512


@dataclass(frozen=True)
class ObserverStats:
    """Snapshot of collected calibration statistics.

    Attributes:
        running_min: Scalar tensor, minimum observed value.
        running_max: Scalar tensor, maximum observed value.
        histogram: ``(num_bins,)`` float counts, or None for MinMaxObserver.
        bin_edges: ``(num_bins + 1,)`` bin boundaries, or None.
    """

    running_min: torch.Tensor
    running_max: torch.Tensor
    histogram: torch.Tensor | None = None
    bin_edges: torch.Tensor | None = None


class MinMaxObserver:
    """Tracks the running min and max of every element it observes.

    Example:
        >>> obs = MinMaxObserver()
        >>> obs.update(torch.randn(10))
        >>> _ = obs.compute_stats()
    """

    def __init__(self) -> None:
        self._min: torch.Tensor | None = None
        self._max: torch.Tensor | None = None

    def update(self, x: torch.Tensor) -> None:
        """Fold a new tensor into the running min/max.

        Args:
            x: Any-shape float tensor; reduced to scalars internally.
        """
        x = x.detach().flatten().float()
        if x.numel() == 0:
            return
        batch_min, batch_max = x.min(), x.max()
        if self._min is None or self._max is None:
            self._min, self._max = batch_min, batch_max
        else:
            self._min = torch.minimum(self._min, batch_min)
            self._max = torch.maximum(self._max, batch_max)

    def compute_stats(self) -> ObserverStats:
        """Return the collected statistics.

        Returns:
            ObserverStats: with histogram/bin_edges left as None.

        Raises:
            CalibrationError: If no data was ever observed.
        """
        if self._min is None or self._max is None:
            raise CalibrationError("MinMaxObserver.update was never called with data")
        return ObserverStats(running_min=self._min, running_max=self._max)


class HistogramObserver:
    """Accumulates a fixed-bin histogram with dynamic range extension.

    On the first batch the range is set to that batch's [min, max]. If a later
    batch widens the range, the existing histogram is re-binned (counts are
    scattered into the bins of the new, wider grid) before the new batch is
    added. This keeps total counts exactly equal to the number of observed
    elements while spanning the full observed range.

    Args:
        num_bins: Number of histogram bins (default 512).

    Example:
        >>> obs = HistogramObserver(num_bins=8)
        >>> obs.update(torch.linspace(-1, 1, 100))
        >>> stats = obs.compute_stats()
        >>> int(stats.histogram.sum().item())
        100
    """

    def __init__(self, num_bins: int = _DEFAULT_BINS) -> None:
        if num_bins < 1:
            raise CalibrationError(f"num_bins must be >= 1, got {num_bins}")
        self.num_bins: Final[int] = num_bins
        self._min: torch.Tensor | None = None
        self._max: torch.Tensor | None = None
        self._hist: torch.Tensor | None = None

    def update(self, x: torch.Tensor) -> None:
        """Fold a new tensor into the histogram, extending the range if needed.

        Args:
            x: Any-shape float tensor.
        """
        x = x.detach().flatten().float()
        if x.numel() == 0:
            return
        batch_min, batch_max = x.min(), x.max()

        if self._min is None or self._max is None or self._hist is None:
            self._min, self._max = batch_min, batch_max
            self._hist = self._histc(x, batch_min, batch_max)
            return

        new_min = torch.minimum(self._min, batch_min)
        new_max = torch.maximum(self._max, batch_max)

        if bool(new_min == self._min) and bool(new_max == self._max):
            self._hist = self._hist + self._histc(x, self._min, self._max)
        else:
            rebinned = self._rebin(self._hist, self._min, self._max, new_min, new_max)
            self._hist = rebinned + self._histc(x, new_min, new_max)
            self._min, self._max = new_min, new_max

    def _histc(self, x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Histogram of ``x`` over [lo, hi]; degenerate range -> all mass in bin 0."""
        lo_f, hi_f = lo.item(), hi.item()
        if hi_f <= lo_f:
            hist = torch.zeros(self.num_bins, device=x.device)
            hist[0] = float(x.numel())
            return hist
        return torch.histc(x, bins=self.num_bins, min=lo_f, max=hi_f)

    def _rebin(
        self,
        hist: torch.Tensor,
        old_min: torch.Tensor,
        old_max: torch.Tensor,
        new_min: torch.Tensor,
        new_max: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter ``hist`` (defined over [old_min, old_max]) onto the new grid."""
        old_edges = torch.linspace(
            old_min.item(), old_max.item(), self.num_bins + 1, device=hist.device
        )
        centers = (old_edges[:-1] + old_edges[1:]) * 0.5
        new_edges = torch.linspace(
            new_min.item(), new_max.item(), self.num_bins + 1, device=hist.device
        )
        idx = torch.bucketize(centers, new_edges) - 1
        idx = idx.clamp(0, self.num_bins - 1)
        out = torch.zeros(self.num_bins, device=hist.device)
        out.scatter_add_(0, idx, hist)
        return out

    def compute_stats(self) -> ObserverStats:
        """Return the collected statistics including histogram and bin edges.

        Returns:
            ObserverStats: with ``histogram`` ``(num_bins,)`` and
            ``bin_edges`` ``(num_bins + 1,)``.

        Raises:
            CalibrationError: If no data was ever observed.
        """
        if self._min is None or self._max is None or self._hist is None:
            raise CalibrationError("HistogramObserver.update was never called with data")
        bin_edges = torch.linspace(
            self._min.item(), self._max.item(), self.num_bins + 1, device=self._hist.device
        )
        return ObserverStats(
            running_min=self._min,
            running_max=self._max,
            histogram=self._hist,
            bin_edges=bin_edges,
        )
