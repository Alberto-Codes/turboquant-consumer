"""Tests for Lloyd-Max codebook solver and quantizer."""

import pytest
import torch

from turboquant_consumer.lloyd_max import LloydMaxCodebook, solve_lloyd_max

from .conftest import BITS, DIM


@pytest.mark.unit
class TestLloydMaxCodebook:
    """Validate Lloyd-Max codebook mathematical properties."""

    def test_codebook_centroids_sorted(self, codebook_3bit: LloydMaxCodebook) -> None:
        """Centroids must be strictly sorted ascending."""
        assert torch.all(codebook_3bit.centroids[1:] > codebook_3bit.centroids[:-1])

    def test_codebook_boundaries_sorted(self, codebook_3bit: LloydMaxCodebook) -> None:
        """Boundaries must be strictly sorted ascending."""
        assert torch.all(codebook_3bit.boundaries[1:] > codebook_3bit.boundaries[:-1])

    def test_codebook_symmetry(self, codebook_3bit: LloydMaxCodebook) -> None:
        """Centroids should be approximately symmetric around zero."""
        assert abs(codebook_3bit.centroids.mean().item()) < 1e-4

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_correct_number_of_levels(self, bits: int) -> None:
        """2^bits centroids and 2^bits - 1 boundaries."""
        centroids, boundaries = solve_lloyd_max(DIM, bits)
        assert len(centroids) == 2**bits
        assert len(boundaries) == 2**bits - 1

    def test_quantize_dequantize_shape(
        self, codebook_3bit: LloydMaxCodebook, device: torch.device
    ) -> None:
        """Quantize and dequantize preserve tensor shape."""
        x = (torch.randn(10, DIM) * 0.1).to(device)
        indices = codebook_3bit.quantize(x)
        assert indices.shape == x.shape

        reconstructed = codebook_3bit.dequantize(indices)
        assert reconstructed.shape == x.shape

    def test_indices_in_valid_range(
        self, codebook_3bit: LloydMaxCodebook, device: torch.device
    ) -> None:
        """All quantized indices must be in [0, 2^bits - 1]."""
        x = (torch.randn(100, DIM) * 0.1).to(device)
        indices = codebook_3bit.quantize(x)
        assert indices.min() >= 0
        assert indices.max() < 2**BITS
