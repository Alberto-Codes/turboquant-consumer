"""Tests for heterogeneous head_dim support in CompressedDynamicCache.

Gemma 4 uses head_dim=256 for sliding attention and head_dim=512 for
global attention. CompressedDynamicCache must create per-head_dim
compressors lazily to handle this.
"""

from __future__ import annotations

import pytest
import torch

from turboquant_vllm.kv_cache import CompressedDynamicCache

from .conftest import BITS

pytestmark = [pytest.mark.unit]

DIM_SMALL = 256  # e.g. Gemma 4 sliding attention layers
DIM_LARGE = 512  # e.g. Gemma 4 global attention layers


class TestHeterogeneousHeadDim:
    """Per-layer head_dim support for mixed architectures."""

    def test_different_head_dims_compress_without_error(self) -> None:
        """Layers with different head_dims should compress successfully."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM_SMALL, bits=BITS)

        cache.layers.clear()
        cache.layers.extend([DynamicLayer(), DynamicLayer()])

        # Layer 0: head_dim=256
        k0 = torch.randn(1, 4, 1, DIM_SMALL)
        v0 = torch.randn(1, 4, 1, DIM_SMALL)
        cache.update(k0, v0, layer_idx=0)

        # Layer 1: head_dim=512
        k1 = torch.randn(1, 2, 1, DIM_LARGE)
        v1 = torch.randn(1, 2, 1, DIM_LARGE)
        cache.update(k1, v1, layer_idx=1)

        assert len(cdc._compressed_keys) == 2
        assert cdc._compressed_keys[0] is not None
        assert cdc._compressed_keys[1] is not None

    def test_compressors_created_lazily(self) -> None:
        """Compressor for unseen head_dim is created on first use."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM_SMALL, bits=BITS)

        # Only primary head_dim compressor exists initially
        assert DIM_SMALL in cdc._key_compressors
        assert DIM_LARGE not in cdc._key_compressors

        # After compressing a layer with DIM_LARGE, compressor is created
        k = torch.randn(1, 2, 1, DIM_LARGE)
        v = torch.randn(1, 2, 1, DIM_LARGE)
        cache.update(k, v, layer_idx=0)

        assert DIM_LARGE in cdc._key_compressors
        assert DIM_LARGE in cdc._value_compressors

    def test_backward_compat_properties(self) -> None:
        """key_compressor/value_compressor properties return primary head_dim."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM_SMALL, bits=BITS)

        assert cdc.key_compressor.quantizer.rotation.shape == (
            DIM_SMALL,
            DIM_SMALL,
        )
        assert cdc.value_compressor.quantizer.rotation.shape == (
            DIM_SMALL,
            DIM_SMALL,
        )

    def test_round_trip_quality_per_head_dim(self) -> None:
        """Compression quality is reasonable for both head_dims."""
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer

        cache = DynamicCache()
        CompressedDynamicCache(cache, head_dim=DIM_SMALL, bits=BITS)
        cache.layers.clear()
        cache.layers.extend([DynamicLayer(), DynamicLayer()])

        for dim, layer_idx in [(DIM_SMALL, 0), (DIM_LARGE, 1)]:
            k = torch.randn(1, 4, 8, dim)
            v = torch.randn(1, 4, 8, dim)
            k_out, v_out = cache.update(k.clone(), v.clone(), layer_idx=layer_idx)

            # Cosine similarity: >0.8 at 3-bit (quality degrades with dim)
            k_flat = k.reshape(-1, dim).float()
            k_out_flat = k_out[:, :, -8:, :].reshape(-1, dim).float()
            cos = torch.nn.functional.cosine_similarity(k_flat, k_out_flat, dim=-1)
            assert cos.mean() > 0.8, (
                f"head_dim={dim}: key cosine {cos.mean():.3f} below 0.8"
            )
