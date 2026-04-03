"""Tests for VALIDATED_MODELS registry (extracted from test_verify.py)."""

from __future__ import annotations

import json

import pytest

from turboquant_vllm.verify import VALIDATED_MODELS, main

from .test_verify import _make_result

pytestmark = [pytest.mark.unit]


class TestValidatedModels:
    def test_molmo2_exact_match(self) -> None:
        assert "molmo2" in VALIDATED_MODELS
        assert VALIDATED_MODELS["molmo2"] == "Molmo2"

    def test_mistral_exact_match(self) -> None:
        assert "mistral" in VALIDATED_MODELS
        assert VALIDATED_MODELS["mistral"] == "Mistral"

    def test_llama_exact_match(self) -> None:
        assert "llama" in VALIDATED_MODELS
        assert VALIDATED_MODELS["llama"] == "Llama"

    def test_qwen2_exact_match(self) -> None:
        assert "qwen2" in VALIDATED_MODELS
        assert VALIDATED_MODELS["qwen2"] == "Qwen2.5"

    def test_phi3_exact_match(self) -> None:
        assert "phi3" in VALIDATED_MODELS
        assert VALIDATED_MODELS["phi3"] == "Phi"

    def test_gemma2_exact_match(self) -> None:
        assert "gemma2" in VALIDATED_MODELS
        assert VALIDATED_MODELS["gemma2"] == "Gemma 2"

    def test_gemma3_exact_match(self) -> None:
        assert "gemma3" in VALIDATED_MODELS
        assert VALIDATED_MODELS["gemma3"] == "Gemma 3"

    def test_gemma4_exact_match(self) -> None:
        assert "gemma4" in VALIDATED_MODELS
        assert VALIDATED_MODELS["gemma4"] == "Gemma 4"

    def test_unvalidated_for_unknown_type(self) -> None:
        assert "gpt2" not in VALIDATED_MODELS

    def test_no_substring_match(self) -> None:
        # "molmo2" should not match "molmo2-extended" or "xmolmo2"
        assert "molmo2-extended" not in VALIDATED_MODELS
        assert "xmolmo2" not in VALIDATED_MODELS

    def test_display_name_mapping(self) -> None:
        for model_type, display_name in VALIDATED_MODELS.items():
            assert isinstance(model_type, str)
            assert isinstance(display_name, str)
            assert len(display_name) > 0

    def test_validated_result_field(self, mocker, capsys) -> None:
        result = _make_result(validation="VALIDATED", family_name="Molmo2")
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=result,
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["validation"] == "VALIDATED"

    def test_unvalidated_result_field(self, mocker, capsys) -> None:
        result = _make_result(validation="UNVALIDATED", family_name=None)
        mocker.patch(
            "turboquant_vllm.verify._run_verification",
            return_value=result,
        )
        with pytest.raises(SystemExit):
            main(["--model", "test/m", "--bits", "4", "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["validation"] == "UNVALIDATED"
