"""Throughput benchmarks (performance floors from the spec, Section 14).

These assert the CPU budgets directly so they run under plain ``pytest`` even
without the ``pytest-benchmark`` plugin. They are timing-sensitive; if the host
is heavily loaded they may be flaky — treat failures as a prompt to re-run on an
idle machine rather than a correctness regression.
"""

from __future__ import annotations

import torch

from quantlib.quantizers.int8 import Int8Quantizer
from quantlib.quantizers.utils import pack_int4, unpack_int4
from quantlib.benchmarks.profiler import latency_ms


def test_quantize_4096x4096_under_100ms() -> None:
    x = torch.randn(4096, 4096)
    q = Int8Quantizer("symmetric", "per_channel")
    ms = latency_ms(lambda t: q.quantize(t), x, n_warmup=5, n_iter=15, reduce="min")
    assert ms < 100.0, f"quantize (4096,4096) took {ms:.1f} ms (budget 100 ms)"


def test_pack_int4_under_20ms() -> None:
    x = torch.randint(0, 16, (4096, 4096), dtype=torch.uint8)
    ms = latency_ms(lambda t: pack_int4(t), x, n_warmup=5, n_iter=15, reduce="min")
    assert ms < 20.0, f"pack_int4 took {ms:.1f} ms (budget 20 ms)"


def test_unpack_int4_under_20ms() -> None:
    packed = pack_int4(torch.randint(0, 16, (4096, 4096), dtype=torch.uint8))
    ms = latency_ms(lambda t: unpack_int4(t, 4096), packed, n_warmup=5, n_iter=15, reduce="min")
    assert ms < 20.0, f"unpack_int4 took {ms:.1f} ms (budget 20 ms)"
