"""Unit tests for benchmark._detect_model_config division guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from turboquant_vllm.benchmark import _detect_model_config

pytestmark = [pytest.mark.unit]


def _make_config(
    *,
    head_dim: int | None = 128,
    num_heads: int = 8,
    hidden_size: int = 1024,
    num_kv_heads: int = 8,
    num_layers: int = 32,
    has_head_dim: bool = True,
) -> SimpleNamespace:
    """Build a mock model with config attributes."""
    cfg_attrs: dict = {
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": num_layers,
    }
    if has_head_dim:
        cfg_attrs["head_dim"] = head_dim
    return SimpleNamespace(config=SimpleNamespace(**cfg_attrs))


def _make_vlm_config(
    *,
    head_dim: int | None = 128,
    num_heads: int = 8,
    hidden_size: int = 1024,
    num_kv_heads: int = 8,
    num_layers: int = 32,
) -> SimpleNamespace:
    """Build a mock VLM model with nested text_config."""
    text_cfg = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=num_layers,
    )
    return SimpleNamespace(config=SimpleNamespace(text_config=text_cfg))


class TestDetectModelConfig:
    """Tests for _detect_model_config head_dim resolution and division guard."""

    def test_explicit_head_dim(self) -> None:
        model = _make_config(head_dim=128)
        result = _detect_model_config(model)
        assert result["head_dim"] == 128

    def test_head_dim_none_falls_back(self) -> None:
        model = _make_config(head_dim=None, hidden_size=1024, num_heads=8)
        result = _detect_model_config(model)
        assert result["head_dim"] == 128

    def test_num_heads_zero_with_explicit_head_dim(self) -> None:
        model = _make_config(head_dim=128, num_heads=0)
        result = _detect_model_config(model)
        assert result["head_dim"] == 128

    def test_num_heads_zero_without_head_dim_raises(self) -> None:
        model = _make_config(head_dim=None, num_heads=0)
        with pytest.raises(ValueError, match="num_attention_heads=0"):
            _detect_model_config(model)

    def test_explicit_head_dim_zero_raises(self) -> None:
        model = _make_config(head_dim=0)
        with pytest.raises(ValueError, match="head_dim=0"):
            _detect_model_config(model)

    def test_explicit_head_dim_negative_raises(self) -> None:
        model = _make_config(head_dim=-1)
        with pytest.raises(ValueError, match="head_dim=-1"):
            _detect_model_config(model)

    def test_head_dim_absent_falls_back(self) -> None:
        model = _make_config(has_head_dim=False, hidden_size=1024, num_heads=8)
        result = _detect_model_config(model)
        assert result["head_dim"] == 128

    def test_vlm_nested_text_config(self) -> None:
        model = _make_vlm_config(head_dim=128)
        result = _detect_model_config(model)
        assert result["head_dim"] == 128
