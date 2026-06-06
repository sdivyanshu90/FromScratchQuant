"""
Module: quantlib.benchmarks.profiler

Memory- and latency-profiling helpers for quantized tensors and modules.

Mathematical Background:
    memory_footprint_bytes = Σ (numel · element_size) over tensors/buffers/params
    latency_ms             = mean wall-clock per call over timed iterations

References:
    None.

Example:
    >>> import torch
    >>> from quantlib.benchmarks.profiler import memory_footprint_bytes
    >>> memory_footprint_bytes(torch.zeros(10, dtype=torch.int8))
    10
"""

from __future__ import annotations

import time
from typing import Callable, Literal

import torch
from torch import nn


def memory_footprint_bytes(obj: torch.Tensor | nn.Module) -> int:
    """Total storage in bytes of a tensor or every param/buffer of a module.

    Args:
        obj: A tensor, or an ``nn.Module`` whose parameters and buffers are summed.

    Returns:
        int: total bytes (``numel · element_size`` summed).
    """
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    total = 0
    for param in obj.parameters():
        total += param.numel() * param.element_size()
    for buffer in obj.buffers():
        total += buffer.numel() * buffer.element_size()
    return total


def latency_ms(
    fn: Callable[..., object],
    *args: object,
    n_warmup: int = 5,
    n_iter: int = 50,
    reduce: Literal["mean", "min"] = "mean",
) -> float:
    """Wall-clock latency of ``fn(*args)`` in milliseconds.

    Synchronizes CUDA around each timed call so device kernels are included.

    Args:
        fn: Callable to time.
        *args: Positional arguments forwarded to ``fn``.
        n_warmup: Untimed warm-up calls (JIT/caches/allocator/page-faults).
        n_iter: Timed iterations.
        reduce: ``"mean"`` for average latency, ``"min"`` for best-case
            steady-state (robust to OS/allocator jitter — standard for
            microbenchmarks).

    Returns:
        float: milliseconds per call under the chosen reduction.
    """
    for _ in range(n_warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_iter):
        start = time.perf_counter()
        fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)

    if reduce == "min":
        return min(times)
    return sum(times) / max(len(times), 1)
