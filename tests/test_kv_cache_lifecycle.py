"""Tests for TurboQuant KV cache lifecycle: context managers and double-wrap detection."""

from __future__ import annotations

import warnings

import pytest

from turboquant_vllm.kv_cache import CompressedDynamicCache, TurboQuantKVCache

from .conftest import BITS, DIM

pytestmark = [pytest.mark.unit]


class TestTQKVCacheLifecycle:
    """Validate TurboQuantKVCache lifecycle: double-wrap detection and context manager."""

    def test_double_wrap_warns(self) -> None:
        """Wrapping an already-wrapped cache should emit UserWarning."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

        with pytest.warns(UserWarning, match="already wrapped by TurboQuant"):
            _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

    def test_context_manager_restores(self) -> None:
        """Exiting `with` block should restore original update method."""
        from transformers import DynamicCache

        cache = DynamicCache()
        original_update = cache.update

        with TurboQuantKVCache(cache, head_dim=DIM, bits=BITS):
            assert cache.update != original_update

        assert cache.update == original_update

    def test_context_manager_restores_on_exception(self) -> None:
        """Restore should happen even when an exception is raised inside `with`."""
        from transformers import DynamicCache

        cache = DynamicCache()
        original_update = cache.update

        with pytest.raises(RuntimeError, match="deliberate"):
            with TurboQuantKVCache(cache, head_dim=DIM, bits=BITS):
                raise RuntimeError("deliberate")

        assert cache.update == original_update


class TestCDCLifecycle:
    """Validate CompressedDynamicCache lifecycle: double-wrap, cross-class, context manager."""

    def test_double_wrap_warns(self) -> None:
        """Wrapping an already-wrapped cache with same class should warn."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

        with pytest.warns(UserWarning, match="already wrapped by TurboQuant"):
            _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

    def test_cross_class_double_wrap_warns(self) -> None:
        """TurboQuantKVCache then CompressedDynamicCache should warn."""
        from transformers import DynamicCache

        cache = DynamicCache()
        _ = TurboQuantKVCache(cache, head_dim=DIM, bits=BITS)

        with pytest.warns(UserWarning, match="already wrapped by TurboQuant"):
            _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)

    def test_context_manager_restores(self) -> None:
        """Exiting `with` block should restore both original methods."""
        from transformers import DynamicCache

        cache = DynamicCache()
        original_update = cache.update
        original_get_seq = cache.get_seq_length

        with CompressedDynamicCache(cache, head_dim=DIM, bits=BITS):
            assert cache.update != original_update
            assert cache.get_seq_length != original_get_seq

        assert cache.update == original_update
        assert cache.get_seq_length == original_get_seq

    def test_context_manager_restores_on_exception(self) -> None:
        """Restore should happen even when an exception is raised inside `with`."""
        from transformers import DynamicCache

        cache = DynamicCache()
        original_update = cache.update
        original_get_seq = cache.get_seq_length

        with pytest.raises(RuntimeError, match="deliberate"):
            with CompressedDynamicCache(cache, head_dim=DIM, bits=BITS):
                raise RuntimeError("deliberate")

        assert cache.update == original_update
        assert cache.get_seq_length == original_get_seq

    def test_rewrap_after_restore_no_warning(self) -> None:
        """After restore(), re-wrapping the same cache should NOT warn."""
        from transformers import DynamicCache

        cache = DynamicCache()
        cdc = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
        cdc.restore()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ = CompressedDynamicCache(cache, head_dim=DIM, bits=BITS)
