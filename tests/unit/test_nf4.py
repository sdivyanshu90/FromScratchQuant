"""Unit tests for NF4 quantization (exact table + worked example).

The NF4 N(0,1) SNR floor is set to the physically-achievable ~20 dB (4-bit
Gaussian quantization is bounded near the Lloyd-Max limit of ~20.2 dB). See the
project README for the rationale; the QLoRA-standard group_size=64 is used.
"""

from __future__ import annotations

import torch

from quantlib import FP4Quantizer, NF4Quantizer, double_quantize_scales
from quantlib.benchmarks.accuracy import compute_snr_db
from quantlib.quantizers.fp4 import NF4_QUANTILE_TABLE


class TestNF4Table:
    def test_quantile_table_length(self) -> None:
        assert len(NF4_QUANTILE_TABLE) == 16

    def test_quantile_table_is_sorted(self) -> None:
        assert NF4_QUANTILE_TABLE == sorted(NF4_QUANTILE_TABLE)

    def test_quantile_table_endpoints_exact(self) -> None:
        assert NF4_QUANTILE_TABLE[0] == -1.0
        assert NF4_QUANTILE_TABLE[7] == 0.0  # zero exactly at index 7
        assert NF4_QUANTILE_TABLE[15] == 1.0

    def test_zero_is_exactly_representable(self) -> None:
        q = NF4Quantizer()
        idx, _ = q.quantize(torch.zeros(8))
        assert idx.tolist() == [7] * 8


class TestNF4WorkedExample:
    def test_worked_example_exact_indices(self) -> None:
        q = NF4Quantizer()
        x = torch.tensor([-0.70, -0.53, 0.00, 0.08, 0.24, 0.72])
        idx, _ = q.quantize(x)
        assert idx.tolist() == [0, 1, 7, 8, 11, 15], f"Got {idx.tolist()}"

    def test_worked_example_reconstruction(self) -> None:
        q = NF4Quantizer()
        x = torch.tensor([-0.70, -0.53, 0.00, 0.08, 0.24, 0.72])
        idx, p = q.quantize(x)
        expected = torch.tensor([-0.72, -0.5013, 0.0, 0.0573, 0.2433, 0.72])
        torch.testing.assert_close(q.dequantize(idx, p), expected, atol=1e-3, rtol=0)

    def test_small_tensor_uses_per_tensor_fallback(self) -> None:
        q = NF4Quantizer()
        _, p = q.quantize(torch.tensor([-0.70, -0.53, 0.0, 0.08, 0.24, 0.72]))
        assert p.granularity == "per_tensor"


class TestNF4Quality:
    def test_snr_above_20db(self, weight_normal: torch.Tensor) -> None:
        q = NF4Quantizer()
        idx, p = q.quantize(weight_normal)
        snr = compute_snr_db(weight_normal, q.dequantize(idx, p))
        assert snr > 20.0, f"NF4 SNR={snr:.2f} dB, expected >20"

    def test_nf4_snr_beats_fp4_on_normal_weights(self, weight_normal: torch.Tensor) -> None:
        q_fp4 = FP4Quantizer()
        q_nf4 = NF4Quantizer()
        idf, pf = q_fp4.quantize(weight_normal)
        idn, pn = q_nf4.quantize(weight_normal)
        snr_fp4 = compute_snr_db(weight_normal, q_fp4.dequantize(idf, pf))
        snr_nf4 = compute_snr_db(weight_normal, q_nf4.dequantize(idn, pn))
        assert snr_nf4 > snr_fp4

    def test_default_granularity_is_per_group_64(self, weight_normal: torch.Tensor) -> None:
        _, p = NF4Quantizer().quantize(weight_normal)  # 4096 % 64 == 0
        assert p.granularity == "per_group"
        assert p.scale.numel() == weight_normal.numel() // 64


class TestNF4DoubleQuantization:
    def test_double_quantize_reduces_scale_dtype_to_int8(self) -> None:
        scales = torch.rand(256) * 0.05 + 1e-3
        s_q, secondary = double_quantize_scales(scales)
        assert s_q.dtype == torch.int8
        assert isinstance(secondary, float)

    def test_double_quantize_reconstruction_close(self) -> None:
        scales = torch.tensor([0.1, 0.2, 0.4, 0.8])
        s_q, secondary = double_quantize_scales(scales)
        # Worst-case int8 reconstruction error is one half-step: 0.8/127/2 ≈ 3.2e-3.
        torch.testing.assert_close(
            s_q.float() * secondary, scales, atol=5e-3, rtol=0
        )
