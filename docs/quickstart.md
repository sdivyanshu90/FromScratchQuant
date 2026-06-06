# quantlib — Quickstart

Five fully-runnable examples. Each block runs as-is (`python example.py`) with
only `torch` and `quantlib` installed.

---

## Example 1 — Single tensor, INT8 symmetric

```python
import torch
from quantlib import Int8Quantizer
from quantlib.benchmarks.accuracy import compute_snr_db

q = Int8Quantizer(scheme="symmetric", granularity="per_tensor")
x = torch.randn(64, 64)
x_q, params = q.quantize(x)
x_r = q.dequantize(x_q, params)

print(f"SNR: {compute_snr_db(x, x_r):.1f} dB")
print(f"Memory: {x.nbytes} B -> {x_q.nbytes} B ({x.nbytes / x_q.nbytes:.1f}x)")
```

---

## Example 2 — Full model with percentile calibration

```python
import torch
from torch import nn
from quantlib import Calibrator, quantize_model, Int8Quantizer

model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
calibration_dataloader = [torch.randn(8, 32) for _ in range(10)]

with Calibrator(model, method="percentile", percentile=99.9) as cal:
    for batch in calibration_dataloader:
        model(batch)
qparams = cal.get_qparams()                 # activation params, per layer
print({name: tuple(p.scale.shape) for name, p in qparams.items()})

# Weight-only quantization of the same model:
qmodel = quantize_model(model, Int8Quantizer("symmetric", "per_channel"))
print(qmodel)
```

---

## Example 3 — NF4 vs FP4 SNR comparison

```python
import torch
from quantlib import NF4Quantizer, FP4Quantizer
from quantlib.benchmarks.accuracy import compute_snr_db

w = torch.randn(1024, 1024)                 # simulate an LLM weight (N(0,1))
xr_fp4 = FP4Quantizer().quantize_dequantize(w)
xr_nf4 = NF4Quantizer().quantize_dequantize(w)

print(f"FP4 SNR: {compute_snr_db(w, xr_fp4):.1f} dB")
print(f"NF4 SNR: {compute_snr_db(w, xr_nf4):.1f} dB")   # higher — quantile spacing
```

---

## Example 4 — Save and reload a quantized model

```python
import tempfile
from torch import nn
from quantlib import Int8Quantizer, quantize_model, save_quantized, load_quantized

qmodel = quantize_model(nn.Sequential(nn.Linear(32, 32)), Int8Quantizer("symmetric"))
directory = tempfile.mkdtemp()

save_quantized(qmodel, directory, model_name="my_model")
loaded = load_quantized(directory, model_name="my_model")
for name, (weight_q, params) in loaded.items():
    print(name, weight_q.dtype, params.granularity, tuple(params.scale.shape))
```

---

## Example 5 — Layer sensitivity analysis

```python
import torch
from torch import nn
from quantlib import Int8Quantizer
from quantlib.calibration import layer_sensitivity_analysis

model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64))
calib_loader = [torch.randn(8, 128) for _ in range(4)]

sensitivity = layer_sensitivity_analysis(
    model, calib_loader, Int8Quantizer("symmetric", "per_channel"), metric="snr_db"
)
# {"0": 44.1, "2": 43.2, ...} — low score = sensitive = keep in FP32
print(sensitivity)
worst = min(sensitivity, key=sensitivity.get)
print(f"Most sensitive layer: {worst} ({sensitivity[worst]:.1f} dB)")
```
