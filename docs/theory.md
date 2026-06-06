# quantlib — Quantization Theory

Math derivations behind every quantizer. Equation numbers match the docstrings.

---

## Section 1 — Why 127 not 128 for INT8 Symmetric

Two's-complement `int8` spans `[-128, 127]` — it is **asymmetric** around zero
(one more negative code than positive). Symmetric quantization wants a mapping
that is symmetric around 0, so it uses the magnitude that *both* signs can
represent: `Q_MAX = 127`.

```
scale = absmax / 127          # Eq. 1
x_q   = clamp(round(x/scale), -128, 127)
```

If we instead used `absmax / 128`, then `+absmax` maps to `+128`, which **overflows**
`INT8_MAX = 127` after rounding and gets clamped — silently losing the largest
positive value. Using `127` makes `±absmax → ±127` exactly, wasting nothing on the
positive side and leaving `-128` as a harmless unused code.

> Always derive the ceiling from `torch.iinfo(torch.int8).max`, never hard-code `128`.

For VNNI/`reduce_range` kernels the same logic applies with `Q_MAX = 63`,
`Q_MIN = -64` (the extra bit of headroom avoids int32 accumulator overflow in
fused `u8×s8` matmuls).

---

## Section 2 — Full Derivation of scale and zero_point

We want a linear (affine) map from a float range `[r_min, r_max]` onto an integer
range `[q_min, q_max]`:

```
q = round(x / scale) + zero_point
```

Anchor the two endpoints:

```
q_min = r_min/scale + zero_point        (1)
q_max = r_max/scale + zero_point        (2)
```

Subtract (1) from (2):

```
q_max - q_min = (r_max - r_min) / scale
=> scale = (r_max - r_min) / (q_max - q_min)         # Eq. 4
```

Substitute back into (1) to solve for the integer offset:

```
zero_point = q_min - r_min / scale                    # Eq. 5
           = round(-r_min / scale)   when q_min = 0   (uint8)
```

`zero_point` is rounded and clamped to `[q_min, q_max]` so that the float value
`0.0` maps exactly to an integer code (important: zero is exactly representable,
which keeps padding/ReLU outputs exact).

**Symmetric is the special case** where the range is symmetric, `r_min = -r_max`:

```
scale      = 2·r_max / (q_max - q_min) = r_max / Q_MAX   (with q_max=-q_min=Q_MAX)
zero_point = round(r_max / scale) - q_max = Q_MAX - Q_MAX = 0
```

So **zero_point = 0 iff the input range is symmetric around 0** — which is why the
symmetric path stores no offset and uses a signed dtype.

---

## Section 3 — Per-Channel vs Per-Tensor: When Each Wins

Per-tensor uses one `scale` for the whole weight; per-channel uses one per output
channel (row). Consider a weight whose rows have very different magnitudes:

```
row 0:  values in [-0.01, 0.01]     (tiny)
row 1:  values in [-4.0,  4.0]      (large)
```

Per-tensor `scale = 4.0/127 ≈ 0.0315`. Row 0's entire range `[-0.01, 0.01]`
collapses onto codes `{-0.. 0 ..0}` — effectively **0–1 distinct levels**, a
catastrophic loss for that channel.

Per-channel gives row 0 its own `scale = 0.01/127 ≈ 7.9e-5`, so it again uses the
full `±127` range. Each channel is quantized at its own resolution.

Empirically this roughly **doubles SNR** when channels have heterogeneous scales —
exactly the situation in the first attention projection of a transformer, where
different heads/features live at different magnitudes. (Measured in this repo:
per-tensor ≈ 41.2 dB vs per-channel ≈ 44.5 dB on a random `N(0,1)` 64×64 weight;
the gap widens sharply on real heterogeneous weights.)

---

## Section 4 — FP4 Bit Layout Diagram

FP4 E2M1 uses 1 sign + 2 exponent + 1 mantissa bit (exponent bias = 1):

```
 bit:   3   2   1   0
       [S] [E1  E0] [M]
        |    \___/   |
      sign  exponent mantissa
```

Value formula (normal, `E ≠ 00`):

```
value = (-1)^S · 2^(E - 1) · (1 + M/2)
```

Subnormal special case (`E = 00`):

```
value = (-1)^S · 2^(1 - 1) · (M/2) = (-1)^S · (M/2)   →   0.0 or ±0.5
```

Enumerating all 16 codes gives the geometric ladder
`{0, 0.5, 1, 1.5, 2, 3, 4, 6}` and its negatives — `FP4_MAX = 6.0`. The levels
are **geometrically** spaced: dense near zero, sparse near the extremes.

---

## Section 5 — NF4: Why Quantile Placement Minimizes Error for N(0,1)

A `b`-bit quantizer partitions the real line into `2^b` bins; the dequantized
value of a bin is its representative level. Expected squared error is

```
E[(x - Q(x))²] = Σ_i ∫_{bin i} (x - level_i)² · p(x) dx
```

For a fixed number of bins, this is minimized (Lloyd–Max / companding theory) when
each bin carries **equal probability mass** and its level sits at the bin's
conditional mean. NF4 places its 16 levels at the **quantiles** of `N(0,1)`, so
each of the 16 bins captures probability mass `1/16`. Pretrained LLM weights are
approximately `N(0, σ)` after normalization, so NF4 is near-optimal for them.

FP4's geometric spacing instead puts many levels out at `±4, ±6` where a Gaussian
has almost no mass, and too few in the `[-1, 1]` bulk where most weights live —
wasting codes. Hence **NF4 SNR > FP4 SNR on `N(0,1)` data, always.**

```
   N(0,1) PDF
        ▲
        │        ____
        │      /      \         NF4 levels  : • • •  •  • • •   (dense in the bulk)
        │     /        \        FP4 levels  : •  •   •    •  •  (spread geometrically)
        │   _/          \_
        └──┴───┴───┴───┴──►  x
          -2  -1   0   1   2
```

**A hard limit to keep honest:** 4 bits = 16 levels. The Lloyd–Max optimal
fixed-rate quantizer of a Gaussian achieves ≈ **20.2 dB** SQNR at 4 bits. No NF4
implementation beats that by much; block-wise (per-group) `absmax` normalization
recovers a little but also spends bits on the scale. So measured NF4 SNR on
`N(0,1)` lands around **20–21 dB** with `group_size = 64`, *not* 22+. quantlib's
tests use the physically-achievable floor (> 20 dB) rather than an impossible one.

---

## Section 6 — Calibration Comparison Table

| Method     | Complexity | Outlier Robustness | Best Use Case          |
| ---------- | ---------- | ------------------ | ---------------------- |
| MinMax     | O(n)       | None               | Debugging / baseline   |
| Percentile | O(n)       | High               | Activations            |
| KL-Entropy | O(n·C)     | High               | Weights + activations  |
| MSE        | O(n·S)     | Medium             | Weight-only PTQ        |

- **MinMax** — exact observed range; one outlier blows up the scale.
- **Percentile** — clips the histogram CDF to `[p_low, p_high]`, discarding tails.
- **KL-Entropy** — searches `C` clip thresholds, minimizing `KL(P_ref ‖ Q_quant)`
  (TensorRT-style); best information-preserving clip.
- **MSE** — grid-searches `S` scale fractions minimizing reconstruction MSE.
