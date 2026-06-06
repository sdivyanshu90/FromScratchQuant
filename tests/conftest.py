"""Shared pytest fixtures for the quantlib test suite.

All tests are made deterministic by the autouse ``set_seed`` fixture.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed() -> None:
    torch.manual_seed(42)  # ALL tests deterministic — no exceptions


@pytest.fixture
def weight_normal() -> torch.Tensor:
    return torch.randn(64, 64)  # N(0, 1), shape (64, 64)


@pytest.fixture
def weight_uniform() -> torch.Tensor:
    return torch.rand(64, 64) * 2.0 - 1.0  # U(-1, 1)


@pytest.fixture
def weight_positive() -> torch.Tensor:
    return torch.randn(64, 64).abs()  # positive-skewed (simulates activations)


@pytest.fixture
def tiny_transformer() -> torch.nn.Module:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

    enc = TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
    return TransformerEncoder(enc, num_layers=2)


@pytest.fixture(params=["cpu"])  # extend: pytest.param("cuda", marks=skipif(...))
def device(request: pytest.FixtureRequest) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)
