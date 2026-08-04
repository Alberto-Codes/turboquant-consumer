"""Tests for the public API surface and HuggingFace path without vLLM."""

import sys
import importlib

EXPECTED_PUBLIC_API = [
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantCompressorMSE",
    "TurboQuantCompressorV2",
    "TurboQuantKVCache",
    "CompressedDynamicCache",
    "LloydMaxCodebook",
    "solve_lloyd_max",
]


def test_public_exports_match_expected():
    """Verify `from turboquant_vllm import *` produces exactly the 8 expected symbols."""
    import turboquant_vllm

    public_symbols = [name for name in dir(turboquant_vllm) if not name.startswith("_")]
    assert sorted(public_symbols) == sorted(EXPECTED_PUBLIC_API), (
        f"Expected {EXPECTED_PUBLIC_API}, got {public_symbols}"
    )


def test_hf_path_importable_without_vllm():
    """Mock `vllm` as unimportable, verify core HF classes still import and instantiate."""
    # Clear turboquant_vllm from cache to force fresh import
    for key in list(sys.modules.keys()):
        if key.startswith("turboquant_vllm"):
            del sys.modules[key]

    vllm_present = "vllm" in sys.modules
    saved_vllm = sys.modules.get("vllm")
    saved_vllm_subs = {k: v for k, v in sys.modules.items() if k.startswith("vllm.")}

    # Temporarily remove vllm to simulate uninstalled environment
    if vllm_present:
        sys.modules.pop("vllm", None)
        for k in list(sys.modules.keys()):
            if k.startswith("vllm."):
                del sys.modules[k]

    try:
        mod = importlib.import_module("turboquant_vllm")

        # Verify core classes are accessible
        assert hasattr(mod, "CompressedDynamicCache")
        assert hasattr(mod, "TurboQuantKVCache")
        assert hasattr(mod, "TurboQuantMSE")
        assert hasattr(mod, "TurboQuantProd")
        assert hasattr(mod, "LloydMaxCodebook")
        assert callable(mod.solve_lloyd_max)
        assert callable(mod.TurboQuantCompressorMSE)
        assert callable(mod.TurboQuantCompressorV2)
    finally:
        # Restore vllm state
        if vllm_present:
            sys.modules["vllm"] = saved_vllm
            sys.modules.update(saved_vllm_subs)


def test_all_matches_dir():
    """Verify `__all__` matches the public API contract."""
    import turboquant_vllm

    assert hasattr(turboquant_vllm, "__all__"), "turboquant_vllm must define __all__"
    assert sorted(turboquant_vllm.__all__) == sorted(EXPECTED_PUBLIC_API), (
        f"__all__ mismatch: expected {EXPECTED_PUBLIC_API}, got {turboquant_vllm.__all__}"
    )
