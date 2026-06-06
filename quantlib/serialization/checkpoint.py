"""
Module: quantlib.serialization.checkpoint

Save / load quantized models to a two-file on-disk format: a ``.safetensors``
file holding the quantized weight (and bias) tensors, and a ``.qconfig.json``
sidecar holding the human-readable :class:`QuantParams` metadata.

Mathematical Background:
    None — (de)serialization only. Reconstruction math lives in the quantizers.

References:
    safetensors format — https://github.com/huggingface/safetensors

Example:
    >>> import torch, tempfile
    >>> from torch import nn
    >>> from quantlib.quantizers.int8 import Int8Quantizer
    >>> from quantlib.modules.wrappers import quantize_model
    >>> from quantlib.serialization.checkpoint import save_quantized, load_quantized
    >>> m = quantize_model(nn.Sequential(nn.Linear(4, 4)), Int8Quantizer("symmetric"))
    >>> d = tempfile.mkdtemp()
    >>> save_quantized(m, d, model_name="demo")
    >>> sorted(load_quantized(d, model_name="demo"))
    ['0']
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal, cast

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from quantlib._version import __version__
from quantlib.core.dtypes import QuantDtype
from quantlib.core.exceptions import CalibrationError
from quantlib.core.qparams import QuantParams
from quantlib.modules.qembedding import QuantizedEmbedding
from quantlib.modules.qlinear import QuantizedLinear

_QuantModule = (QuantizedLinear, QuantizedEmbedding)
_WEIGHTS_SUFFIX: Final[str] = ".safetensors"
_CONFIG_SUFFIX: Final[str] = ".qconfig.json"


def _params_to_json(params: QuantParams) -> dict[str, object]:
    """Serialize QuantParams metadata to a JSON-friendly dict."""
    return {
        "dtype": params.dtype.value,
        "scheme": params.scheme,
        "granularity": params.granularity,
        "channel_dim": params.channel_dim,
        "group_size": params.group_size,
        "packed": params.packed,
        "original_shape": list(params.original_shape) if params.original_shape else None,
        "scale": params.scale.detach().cpu().reshape(-1).tolist(),
        "zero_point": params.zero_point.detach().cpu().reshape(-1).tolist(),
    }


def _json_to_params(meta: dict[str, object], device: str | torch.device) -> QuantParams:
    """Rebuild QuantParams from a JSON metadata dict."""
    scale_list = cast("list[float]", meta["scale"])
    zp_list = cast("list[int]", meta["zero_point"])
    scale = torch.tensor(scale_list, dtype=torch.float32, device=device)
    zero_point = torch.tensor(zp_list, dtype=torch.int32, device=device)
    granularity = cast('Literal["per_tensor", "per_channel", "per_group"]', meta["granularity"])
    original_shape = cast("list[int] | None", meta["original_shape"])
    group_size = cast("int | None", meta["group_size"])
    # Per-tensor scales are stored as 1-element lists; restore the scalar shape.
    if granularity == "per_tensor":
        scale = scale.reshape(())
        zero_point = zero_point.reshape(())
    return QuantParams(
        scale=scale,
        zero_point=zero_point,
        dtype=QuantDtype(cast(str, meta["dtype"])),
        granularity=granularity,
        scheme=cast('Literal["symmetric", "asymmetric"]', meta["scheme"]),
        channel_dim=cast(int, meta["channel_dim"]),
        group_size=group_size,
        original_shape=tuple(original_shape) if original_shape else None,
        packed=cast(bool, meta["packed"]),
    )


def save_quantized(
    model: nn.Module,
    directory: str | Path,
    model_name: str = "model",
) -> None:
    """Save quantized layers to ``{directory}/{model_name}.safetensors`` + ``.qconfig.json``.

    Args:
        model: Model containing :class:`QuantizedLinear` / :class:`QuantizedEmbedding`.
        directory: Target directory (created if missing).
        model_name: Base filename for the two output files.

    Raises:
        FileExistsError: If either output file already exists (no silent overwrite).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / f"{model_name}{_WEIGHTS_SUFFIX}"
    config_path = directory / f"{model_name}{_CONFIG_SUFFIX}"
    if weights_path.exists() or config_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing {weights_path} / {config_path}"
        )

    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict[str, object]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, _QuantModule):
            continue
        tensors[f"{name}.weight_q"] = module.weight_q.detach().cpu().contiguous()
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor):
            tensors[f"{name}.bias"] = bias.detach().cpu().contiguous()
        layers[name] = _params_to_json(module.weight_params)

    config = {"quantlib_version": __version__, "layers": layers}
    save_file(tensors, str(weights_path))
    config_path.write_text(json.dumps(config, indent=2))


def load_quantized(
    directory: str | Path,
    model_name: str = "model",
    device: str | torch.device = "cpu",
) -> dict[str, tuple[torch.Tensor, QuantParams]]:
    """Load quantized weights + params from disk.

    Args:
        directory: Directory holding the two files.
        model_name: Base filename used at save time.
        device: Device for the loaded tensors.

    Returns:
        dict[str, tuple[Tensor, QuantParams]]: ``{layer_name: (weight_q, params)}``.

    Raises:
        FileNotFoundError: If the ``.safetensors`` or ``.qconfig.json`` is missing.
        CalibrationError: If the JSON is malformed or the version is incompatible.
    """
    directory = Path(directory)
    weights_path = directory / f"{model_name}{_WEIGHTS_SUFFIX}"
    config_path = directory / f"{model_name}{_CONFIG_SUFFIX}"
    if not weights_path.exists():
        raise FileNotFoundError(f"missing weights file: {weights_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"missing config file: {config_path}")

    try:
        config = json.loads(config_path.read_text())
        saved_version = config["quantlib_version"]
        layers = config["layers"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CalibrationError(f"malformed qconfig: {config_path}: {exc}") from exc

    saved_major_minor = ".".join(str(saved_version).split(".")[:2])
    current_major_minor = ".".join(__version__.split(".")[:2])
    if saved_major_minor != current_major_minor:
        raise CalibrationError(
            f"incompatible quantlib version: file={saved_version} lib={__version__}"
        )

    tensors = load_file(str(weights_path), device=str(device))
    out: dict[str, tuple[torch.Tensor, QuantParams]] = {}
    for name, meta in layers.items():
        weight_q = tensors[f"{name}.weight_q"]
        params = _json_to_params(meta, device)
        out[name] = (weight_q, params)
    return out
