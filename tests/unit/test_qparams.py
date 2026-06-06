"""Unit tests for quantlib.core.qparams.QuantParams invariants."""

from __future__ import annotations

import pytest
import torch

from quantlib.core.dtypes import QuantDtype
from quantlib.core.exceptions import NumericalError, ShapeError
from quantlib.core.qparams import QuantParams


def _params(**overrides: object) -> QuantParams:
    base: dict[str, object] = dict(
        scale=torch.tensor(0.025),
        zero_point=torch.tensor(0, dtype=torch.int32),
        dtype=QuantDtype.INT8,
        granularity="per_tensor",
        scheme="symmetric",
    )
    base.update(overrides)
    return QuantParams(**base)  # type: ignore[arg-type]


def test_valid_params_construct() -> None:
    p = _params()
    assert p.scheme == "symmetric"


def test_nonpositive_scale_raises() -> None:
    with pytest.raises(NumericalError, match="strictly positive"):
        _params(scale=torch.tensor(0.0))


def test_negative_scale_raises() -> None:
    with pytest.raises(NumericalError):
        _params(scale=torch.tensor(-1.0))


def test_per_group_requires_group_size() -> None:
    with pytest.raises(ShapeError, match="group_size"):
        _params(granularity="per_group", group_size=None)


def test_group_size_forbidden_unless_per_group() -> None:
    with pytest.raises(ShapeError):
        _params(granularity="per_tensor", group_size=128)


def test_frozen_is_immutable() -> None:
    p = _params()
    with pytest.raises(Exception):
        p.scale = torch.tensor(1.0)  # type: ignore[misc]


def test_to_moves_tensors_and_preserves_metadata() -> None:
    p = _params(granularity="per_group", group_size=32)
    moved = p.to("cpu")
    assert moved.group_size == 32
    assert moved.granularity == "per_group"
    assert moved.scale.device.type == "cpu"
