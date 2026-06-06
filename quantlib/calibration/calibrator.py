"""
Module: quantlib.calibration.calibrator

The :class:`Calibrator` context manager: registers forward hooks on a model's
target layers, collects activation statistics during forward passes, and turns
them into per-layer :class:`QuantParams`.

Mathematical Background:
    See quantlib.calibration.methods for the per-method scale/zero_point math.

References:
    Migacz, 2017 — "8-bit Inference with TensorRT".

Example:
    >>> import torch
    >>> from torch import nn
    >>> from quantlib.calibration.calibrator import Calibrator
    >>> model = nn.Sequential(nn.Linear(4, 4))
    >>> with Calibrator(model, method="minmax") as cal:
    ...     _ = model(torch.randn(8, 4))
    >>> qparams = cal.get_qparams()
    >>> sorted(qparams)        # one entry per nn.Linear
    ['0']
"""

from __future__ import annotations

import re
from types import TracebackType
from typing import Final, Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from quantlib.core.dtypes import QuantDtype
from quantlib.core.exceptions import CalibrationError
from quantlib.core.observer import HistogramObserver, MinMaxObserver, ObserverStats
from quantlib.core.qparams import QuantParams
from quantlib.calibration.methods import (
    calibrate_entropy,
    calibrate_minmax,
    calibrate_mse,
    calibrate_percentile,
)

# Schemes implied by each method (entropy/mse are symmetric, min/max-based are affine).
_SYMMETRIC_METHODS: Final[frozenset[str]] = frozenset({"entropy", "mse"})
_MAX_MSE_SAMPLE: Final[int] = 100_000  # cap stored activations for the MSE method


class _SampleObserver:
    """Stores a capped flat sample of activations for the MSE calibration method."""

    def __init__(self, max_elems: int = _MAX_MSE_SAMPLE) -> None:
        self._chunks: list[torch.Tensor] = []
        self._count = 0
        self._max = max_elems

    def update(self, x: torch.Tensor) -> None:
        if self._count >= self._max:
            return
        flat = x.detach().flatten().float()
        take = flat[: self._max - self._count]
        self._chunks.append(take)
        self._count += take.numel()

    def get_sample(self) -> torch.Tensor:
        if not self._chunks:
            raise CalibrationError("no activations were observed for MSE calibration")
        return torch.cat(self._chunks)


class Calibrator:
    """Context manager collecting activation stats and producing QuantParams.

    All forward hooks are removed unconditionally in :meth:`__exit__`, even on
    exception, so the model is left exactly as it was found.

    Args:
        model: Module to calibrate.
        method: ``"minmax"`` | ``"percentile"`` | ``"entropy"`` | ``"mse"``.
        percentile: Central percentile for the percentile method.
        target_module_types: Layer types to hook (default ``(nn.Linear,)``).
        exclude_patterns: Regex patterns; a layer is skipped if any matches its
            dotted name. ``None`` includes every target layer.
        num_histogram_bins: Bins for histogram-based methods.
    """

    def __init__(
        self,
        model: nn.Module,
        method: Literal["minmax", "percentile", "entropy", "mse"] = "percentile",
        percentile: float = 99.9,
        target_module_types: tuple[type[nn.Module], ...] = (nn.Linear,),
        exclude_patterns: list[str] | None = None,
        num_histogram_bins: int = 512,
    ) -> None:
        self.model = model
        self.method = method
        self.percentile = percentile
        self.target_module_types = target_module_types
        self.exclude_patterns = exclude_patterns
        self.num_histogram_bins = num_histogram_bins
        self._observers: dict[str, MinMaxObserver | HistogramObserver | _SampleObserver] = {}
        self._handles: list[RemovableHandle] = []

    def _is_excluded(self, name: str) -> bool:
        if not self.exclude_patterns:
            return False
        return any(re.search(p, name) for p in self.exclude_patterns)

    def _make_observer(self) -> MinMaxObserver | HistogramObserver | _SampleObserver:
        if self.method == "minmax":
            return MinMaxObserver()
        if self.method == "mse":
            return _SampleObserver()
        return HistogramObserver(self.num_histogram_bins)

    def __enter__(self) -> "Calibrator":
        """Register a forward hook on every (non-excluded) target layer."""
        for name, module in self.model.named_modules():
            if not isinstance(module, self.target_module_types):
                continue
            if self._is_excluded(name):
                continue
            observer = self._make_observer()
            self._observers[name] = observer

            def hook(
                _mod: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                _out: torch.Tensor,
                _obs: MinMaxObserver | HistogramObserver | _SampleObserver = observer,
            ) -> None:
                if inputs:
                    _obs.update(inputs[0])

            self._handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Remove every registered hook, unconditionally."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_stats(self) -> dict[str, ObserverStats]:
        """Return raw collected statistics per layer (for debugging/visualization).

        Returns:
            dict[str, ObserverStats]: keyed by layer name. MSE layers are omitted
            (they store raw samples, not summary stats).
        """
        stats: dict[str, ObserverStats] = {}
        for name, obs in self._observers.items():
            if isinstance(obs, _SampleObserver):
                continue
            stats[name] = obs.compute_stats()
        return stats

    def _params_for(
        self, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> QuantParams:
        symmetric = self.method in _SYMMETRIC_METHODS
        scheme: Literal["symmetric", "asymmetric"] = (
            "symmetric" if symmetric else "asymmetric"
        )
        dtype = QuantDtype.INT8 if symmetric else QuantDtype.UINT8
        return QuantParams(
            scale=scale.reshape(()),
            zero_point=zero_point.reshape(()).to(torch.int32),
            dtype=dtype,
            granularity="per_tensor",
            scheme=scheme,
        )

    def get_qparams(self) -> dict[str, QuantParams]:
        """Run the chosen calibration method on every layer's stats.

        Call after the context has exited.

        Returns:
            dict[str, QuantParams]: per-layer per-tensor parameters.

        Raises:
            CalibrationError: If a layer observed no data.
        """
        out: dict[str, QuantParams] = {}
        for name, obs in self._observers.items():
            if self.method == "minmax":
                assert isinstance(obs, MinMaxObserver)
                scale, zp = calibrate_minmax(obs.compute_stats())
            elif self.method == "percentile":
                assert isinstance(obs, HistogramObserver)
                scale, zp = calibrate_percentile(obs.compute_stats(), self.percentile)
            elif self.method == "entropy":
                assert isinstance(obs, HistogramObserver)
                scale, zp = calibrate_entropy(obs.compute_stats())
            else:  # mse
                assert isinstance(obs, _SampleObserver)
                scale, zp = calibrate_mse(obs.get_sample(), scheme="symmetric")
            out[name] = self._params_for(scale, zp)
        return out
