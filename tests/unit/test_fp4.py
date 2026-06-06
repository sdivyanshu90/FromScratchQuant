"""Unit tests for FP4 quantization and the int4 pack/unpack utilities."""

from __future__ import annotations

import pytest
import torch

from quantlib import FP4Quantizer, NF4Quantizer
from quantlib.benchmarks.accuracy import compute_snr_db
from quantlib.quantizers.fp4 import FP4_E2M1_VALUES, FP4_MAX
from quantlib.quantizers.utils import pack_int4, unpack_int4


class TestPackUnpack:
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 9, 15, 16, 100, 101])
    def test_round_trip_exact_all_lengths(self, n: int) -> None:
        x = torch.randint(0, 16, (n,), dtype=torch.uint8)
        assert unpack_int4(pack_int4(x), n).tolist() == x.tolist(), (
            f"Round-trip failed at length n={n}"
        )

    def test_known_bytes_exact(self) -> None:
        x = torch.tensor([3, 7, 12, 0, 5], dtype=torch.uint8)
        result = pack_int4(x)
        assert result.tolist() == [0x37, 0xC0, 0x50], (
            f"Expected [55, 192, 80], got {result.tolist()}"
        )

    def test_packed_shape_is_ceil_n_div_2(self) -> None:
        for n in [1, 2, 3, 7, 8, 9]:
            x = torch.zeros(n, dtype=torch.uint8)
            assert pack_int4(x).shape[0] == (n + 1) // 2

    def test_pack_2d_along_last_dim(self) -> None:
        x = torch.randint(0, 16, (4, 10), dtype=torch.uint8)
        packed = pack_int4(x)
        assert packed.shape == (4, 5)
        assert unpack_int4(packed, 10).tolist() == x.tolist()


class TestFP4Table:
    def test_table_length_is_16(self) -> None:
        assert len(FP4_E2M1_VALUES) == 16

    def test_max_value_is_6(self) -> None:
        assert FP4_MAX == 6.0
        assert max(abs(v) for v in FP4_E2M1_VALUES) == 6.0

    def test_positive_half_is_geometric_e2m1(self) -> None:
        assert FP4_E2M1_VALUES[:8] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


class TestFP4Quantizer:
    def test_indices_in_range(self, weight_normal: torch.Tensor) -> None:
        idx, _ = FP4Quantizer().quantize(weight_normal)
        assert int(idx.min().item()) >= 0 and int(idx.max().item()) <= 15
        assert idx.dtype == torch.uint8

    def test_exact_table_values_reconstruct(self) -> None:
        q = FP4Quantizer()
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -6.0])
        idx, p = q.quantize(x)
        torch.testing.assert_close(q.dequantize(idx, p), x, atol=1e-5, rtol=0)

    def test_snr_above_18db(self, weight_normal: torch.Tensor) -> None:
        q = FP4Quantizer()
        idx, p = q.quantize(weight_normal)
        snr = compute_snr_db(weight_normal, q.dequantize(idx, p))
        assert snr > 18.0, f"FP4 SNR={snr:.2f} dB, expected >18"

    def test_quantize_dequantize_matches_manual(self, weight_normal: torch.Tensor) -> None:
        q = FP4Quantizer()
        idx, p = q.quantize(weight_normal)
        torch.testing.assert_close(
            q.quantize_dequantize(weight_normal), q.dequantize(idx, p), atol=1e-6, rtol=0
        )

    def test_small_tensor_falls_back_to_per_tensor(self) -> None:
        q = FP4Quantizer()  # default per_group(64); 5 elems not divisible
        _, p = q.quantize(torch.tensor([0.0, 3.0, -6.0, 1.5, 2.0]))
        assert p.granularity == "per_tensor"

    def test_device_agnostic(self, device: torch.device) -> None:
        q = FP4Quantizer()
        x = torch.randn(128, device=device)
        idx, p = q.quantize(x)
        assert idx.device == device
        assert q.dequantize(idx, p).device == device


class TestNF4BeatsFP4:
    def test_nf4_snr_beats_fp4_on_normal_weights(self, weight_normal: torch.Tensor) -> None:
        q_fp4 = FP4Quantizer()
        q_nf4 = NF4Quantizer()
        idf, pf = q_fp4.quantize(weight_normal)
        idn, pn = q_nf4.quantize(weight_normal)
        snr_fp4 = compute_snr_db(weight_normal, q_fp4.dequantize(idf, pf))
        snr_nf4 = compute_snr_db(weight_normal, q_nf4.dequantize(idn, pn))
        assert snr_nf4 > snr_fp4, (
            f"NF4 {snr_nf4:.2f} dB must exceed FP4 {snr_fp4:.2f} dB on N(0,1)"
        )
