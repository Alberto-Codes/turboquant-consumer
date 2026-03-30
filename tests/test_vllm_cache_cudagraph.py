"""Tests for D7 CUDA graph buffer pre-allocation in TQ4 backend.

Extracted from ``test_vllm_cache.py`` (Test Maturity Priority 1) to keep
both files under the 500-line module gate.
"""

from __future__ import annotations

import logging

import pytest

vllm = pytest.importorskip("vllm", reason="vLLM not installed")

import torch  # noqa: E402

from turboquant_vllm.vllm.tq4_backend import (  # noqa: E402  # isort: skip
    TQ4AttentionBackend,
    TQ4MetadataBuilder,
)

from tests.helpers.vllm_impl import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_SIZE,
    NUM_HEADS,
    NUM_KV_HEADS,
    make_cache,
    make_impl,
)

pytestmark = [pytest.mark.unit]


class TestCUDAGraphBufferPreallocation:
    """Tests for D7 CUDA graph buffer pre-allocation."""

    def test_get_cudagraph_support_returns_single_token_decode(self) -> None:
        """TQ4 builder reports UNIFORM_SINGLE_TOKEN_DECODE (AC 2)."""
        from vllm.v1.attention.backend import AttentionCGSupport

        result = TQ4MetadataBuilder.get_cudagraph_support(None, None)
        assert result == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def test_backend_returns_tq4_builder(self) -> None:
        """TQ4AttentionBackend.get_builder_cls() returns TQ4MetadataBuilder."""
        assert TQ4AttentionBackend.get_builder_cls() is TQ4MetadataBuilder

    def test_init_cg_buffers_shapes(self, tq4_quantizer) -> None:
        """_init_cg_buffers creates buffers of correct shape/dtype (AC 1)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        assert not impl._cg_buffers_ready
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)
        assert impl._cg_buffers_ready

        max_tokens = 4 * BLOCK_SIZE
        H = NUM_KV_HEADS
        D = HEAD_SIZE

        assert impl._cg_decompress_k.shape == (max_tokens, H, D)
        assert impl._cg_decompress_k.dtype == torch.float16
        assert impl._cg_decompress_v.shape == (max_tokens, H, D)
        assert impl._cg_decompress_v.dtype == torch.float16

        assert impl._cg_compress_packed.shape == (1, H, D // 2)
        assert impl._cg_compress_packed.dtype == torch.uint8
        assert impl._cg_compress_norms.shape == (1, H, 1)
        assert impl._cg_compress_norms.dtype == torch.float32

        assert impl._cg_q_rot.shape == (1, NUM_HEADS, D)
        assert impl._cg_q_rot.dtype == torch.float32
        assert impl._cg_q_rot_cast.shape == (1, NUM_HEADS, D)
        assert impl._cg_q_rot_cast.dtype == torch.float16

        assert impl._cg_compress_row.shape == (1, impl._total_bytes)
        assert impl._cg_compress_row.dtype == torch.uint8

    def test_preallocated_decompress_matches_dynamic(self, tq4_quantizer) -> None:
        """Blocking acceptance test: pre-allocated buffers match dynamic (AC 4).

        Uses float16 (matching pre-allocated buffer dtype and real forward path).
        """
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        # Write some tokens
        key = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        slot_mapping = torch.tensor([0, 5, 33])
        impl._compress_and_store(key, value, kv_cache, slot_mapping)

        # Dynamic allocation (no out= buffers)
        k_dynamic, v_dynamic = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )

        # Pre-allocated buffers (max-size, sliced by decompress)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)
        k_prealloc, v_prealloc = impl._decompress_cache(
            kv_cache,
            torch.float16,
            apply_rotation=False,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )

        # Must be IDENTICAL (not just close)
        assert torch.equal(k_dynamic, k_prealloc), (
            "Pre-allocated K decompress differs from dynamic"
        )
        assert torch.equal(v_dynamic, v_prealloc), (
            "Pre-allocated V decompress differs from dynamic"
        )

    def test_preallocated_decompress_bfloat16(self, tq4_quantizer) -> None:
        """Decompress buffers use compute_dtype, not hardcoded float16 (F2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        key = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0, 5, 33]))

        # Dynamic path with bfloat16
        k_dynamic, v_dynamic = impl._decompress_cache(
            kv_cache, torch.bfloat16, apply_rotation=False
        )
        assert k_dynamic.dtype == torch.bfloat16

        # Pre-allocated path with bfloat16 compute_dtype
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.bfloat16)
        assert impl._cg_decompress_k.dtype == torch.bfloat16

        k_prealloc, v_prealloc = impl._decompress_cache(
            kv_cache,
            torch.bfloat16,
            apply_rotation=False,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )
        assert k_prealloc.dtype == torch.bfloat16
        assert torch.equal(k_dynamic, k_prealloc), (
            "BF16 pre-allocated K differs from dynamic"
        )
        assert torch.equal(v_dynamic, v_prealloc), (
            "BF16 pre-allocated V differs from dynamic"
        )

    def test_decode_path_uses_preallocated_compress(self, tq4_quantizer) -> None:
        """Decode compress uses pre-allocated buffers (AC 3)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=2)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        slot_mapping = torch.tensor([0])

        # Compress with pre-allocated buffers
        impl._compress_and_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            compress_out=(impl._cg_compress_packed, impl._cg_compress_norms),
            row_out=impl._cg_compress_row,
        )

        # Verify data was written to cache
        flat = kv_cache.view(-1, impl._total_bytes)
        assert flat[0].any(), "Slot 0 should have data after pre-allocated compress"

        # Decompress and verify round-trip
        k_cache, v_cache = impl._decompress_cache(kv_cache, torch.float32)
        recon_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)[0]
        for h in range(NUM_KV_HEADS):
            cos_k = torch.nn.functional.cosine_similarity(
                key[0, h].unsqueeze(0), recon_k[h].unsqueeze(0)
            ).item()
            assert cos_k > 0.85, f"Pre-alloc compress K head {h} cosine {cos_k:.4f}"

    def test_prefill_path_unchanged(self, tq4_quantizer) -> None:
        """Prefill (multi-token) path still uses dynamic allocation (AC 5)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        N = 5
        key = torch.randn(N, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(N, NUM_KV_HEADS, HEAD_SIZE)
        slot_mapping = torch.tensor([0, 1, 2, 3, 4])

        # Without pre-allocated buffers (prefill path)
        impl._compress_and_store(key, value, kv_cache, slot_mapping)
        k_cache, _ = impl._decompress_cache(kv_cache, torch.float32)
        flat_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)

        for i in range(N):
            for h in range(NUM_KV_HEADS):
                cos = torch.nn.functional.cosine_similarity(
                    key[i, h].unsqueeze(0), flat_k[i, h].unsqueeze(0)
                ).item()
                assert cos > 0.85, f"Prefill token {i} head {h} cosine {cos:.4f}"

    def test_buffer_reuse_consecutive_decode_steps(self, tq4_quantizer) -> None:
        """Same buffers reused on consecutive decode steps (AC 3)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=2)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        compress_out = (impl._cg_compress_packed, impl._cg_compress_norms)
        row_out = impl._cg_compress_row

        # Step 1
        key1 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        val1 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(
            key1,
            val1,
            kv_cache,
            torch.tensor([0]),
            compress_out=compress_out,
            row_out=row_out,
        )

        # Step 2 (same buffers, different slot)
        key2 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        val2 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(
            key2,
            val2,
            kv_cache,
            torch.tensor([1]),
            compress_out=compress_out,
            row_out=row_out,
        )

        # Both slots should have data
        flat = kv_cache.view(-1, impl._total_bytes)
        assert flat[0].any(), "Slot 0 should have data"
        assert flat[1].any(), "Slot 1 should have data"

        # Verify each slot decompresses correctly
        k_cache, _ = impl._decompress_cache(kv_cache, torch.float32)
        flat_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_SIZE)
        cos_1 = torch.nn.functional.cosine_similarity(
            key1[0, 0].unsqueeze(0), flat_k[0, 0].unsqueeze(0)
        ).item()
        cos_2 = torch.nn.functional.cosine_similarity(
            key2[0, 0].unsqueeze(0), flat_k[1, 0].unsqueeze(0)
        ).item()
        assert cos_1 > 0.85, f"Step 1 cosine {cos_1:.4f}"
        assert cos_2 > 0.85, f"Step 2 cosine {cos_2:.4f}"

    def test_stale_data_immunity(self, tq4_quantizer) -> None:
        """Stale data in decompress buffers doesn't affect output (AC 4).

        Fill decompress buffers with garbage before calling decompress+
        flash_attn. Validates that tq4_decompress overwrites all positions
        and that stale data in unused positions doesn't leak.
        """
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Write some data
        key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([5]))

        # Fill decompress buffers with garbage
        impl._cg_decompress_k.fill_(99.0)
        impl._cg_decompress_v.fill_(99.0)

        # Decompress into garbage-filled buffers
        k_stale, _ = impl._decompress_cache(
            kv_cache,
            torch.float16,
            apply_rotation=False,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )

        # Clean run (fresh buffers)
        k_clean, _ = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )

        # The written slot (5) must be identical regardless of stale data
        stale_k5 = k_stale.view(-1, NUM_KV_HEADS, HEAD_SIZE)[5]
        clean_k5 = k_clean.view(-1, NUM_KV_HEADS, HEAD_SIZE)[5]
        assert torch.equal(stale_k5, clean_k5), (
            "Stale data in decompress buffer affected written slot output"
        )

    def test_prefill_buffer_shapes(self, tq4_quantizer) -> None:
        """_init_cg_buffers allocates prefill buffers (AC 1).

        Sized min(max_prefill_len, max_tokens).
        """
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        max_tokens = 4 * BLOCK_SIZE
        prefill_tokens = min(impl._max_prefill_len, max_tokens)
        H = NUM_KV_HEADS
        D = HEAD_SIZE

        assert impl._cg_prefill_k.shape == (prefill_tokens, H, D)
        assert impl._cg_prefill_k.dtype == torch.float16
        assert impl._cg_prefill_v.shape == (prefill_tokens, H, D)
        assert impl._cg_prefill_v.dtype == torch.float16
        assert impl._max_prefill_blocks == prefill_tokens // BLOCK_SIZE

    def test_decompress_buffers_unchanged_by_prefill_addition(
        self, tq4_quantizer
    ) -> None:
        """_init_cg_buffers does NOT change _cg_decompress_k/_v sizing (AC 1, regression)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Non-fused path: decompress buffers should be full cache size
        max_tokens = 4 * BLOCK_SIZE
        H = NUM_KV_HEADS
        D = HEAD_SIZE
        assert impl._cg_decompress_k.shape == (max_tokens, H, D)
        assert impl._cg_decompress_v.shape == (max_tokens, H, D)

    def test_prefill_buffers_fused_paged_mode(self, tq4_quantizer) -> None:
        """Prefill buffers same size regardless of fused_paged flag (AC 1)."""
        impl_nonfused = make_impl(tq4_quantizer)
        impl_fused = make_impl(tq4_quantizer)
        impl_fused._fused_paged_available = True

        # Need enough blocks so max_tokens > max_prefill_len (2048)
        kv_cache = make_cache(num_blocks=200)
        impl_nonfused._init_cg_buffers(kv_cache, compute_dtype=torch.float16)
        impl_fused._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Prefill buffers should be identical size in both modes
        assert impl_nonfused._cg_prefill_k.shape == impl_fused._cg_prefill_k.shape
        assert impl_nonfused._cg_prefill_v.shape == impl_fused._cg_prefill_v.shape

        # But decompress buffers differ (fused downsizes them)
        assert (
            impl_nonfused._cg_decompress_k.shape[0]
            > impl_fused._cg_decompress_k.shape[0]
        )


class TestPagedDecompress:
    """Tests for _decompress_cache_paged method (AC 2-7)."""

    def test_paged_decompress_common_case(self, tq4_quantizer) -> None:
        """Paged decompress of 5 unique blocks from a 100-block cache (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=100)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Write data to specific blocks (blocks 10, 20, 30, 40, 50)
        target_blocks = [10, 20, 30, 40, 50]
        for blk in target_blocks:
            for pos in range(BLOCK_SIZE):
                slot = blk * BLOCK_SIZE + pos
                key = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
                value = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
                impl._compress_and_store(key, value, kv_cache, torch.tensor([slot]))

        # Build a block_table referencing only those 5 blocks
        block_table = torch.tensor([target_blocks], dtype=torch.int32)
        seq_lens = torch.tensor([5 * BLOCK_SIZE], dtype=torch.int32)

        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        # Should contain exactly 5 decompressed blocks
        assert k_paged.shape == (5, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        assert v_paged.shape == (5, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

        # Verify cosine parity with full decompress reference
        k_full, v_full = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )
        for i, blk in enumerate(target_blocks):
            assert torch.equal(k_paged[i], k_full[blk]), (
                f"Block {blk}: paged K differs from full decompress"
            )
            assert torch.equal(v_paged[i], v_full[blk]), (
                f"Block {blk}: paged V differs from full decompress"
            )

    def test_remapped_block_table_correctness(self, tq4_quantizer) -> None:
        """Remapped block table maps logical blocks to compact indices (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=20)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Sequence using blocks 5, 10, 15 (non-contiguous)
        block_table = torch.tensor([[5, 10, 15]], dtype=torch.int32)
        seq_lens = torch.tensor([3 * BLOCK_SIZE], dtype=torch.int32)

        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        # unique sorted = [5, 10, 15] -> compact [0, 1, 2]
        expected_remap = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        assert torch.equal(remap_bt, expected_remap), (
            f"Expected remapped block table {expected_remap}, got {remap_bt}"
        )

        # Verify Flash Attention would read same data:
        # k_paged[remap_bt[0, j]] should equal full_decompress[block_table[0, j]]
        k_full, _ = impl._decompress_cache(
            kv_cache, torch.float16, apply_rotation=False
        )
        for j in range(3):
            compact_idx = int(remap_bt[0, j])
            original_idx = int(block_table[0, j])
            assert torch.equal(k_paged[compact_idx], k_full[original_idx])

    def test_paged_decompress_fallback(self, tq4_quantizer) -> None:
        """Dynamic fallback when unique blocks exceed max_prefill_blocks (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=200)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Force fallback: reference more blocks than max_prefill_blocks
        num_blocks_needed = impl._max_prefill_blocks + 5
        block_indices = list(range(num_blocks_needed))
        block_table = torch.tensor([block_indices], dtype=torch.int32)
        seq_lens = torch.tensor([num_blocks_needed * BLOCK_SIZE], dtype=torch.int32)

        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        # Should still return correctly shaped output (dynamically allocated)
        assert k_paged.shape == (
            num_blocks_needed,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
        )
        assert v_paged.shape == k_paged.shape

    def test_decompress_cache_backward_compat(self, tq4_quantizer) -> None:
        """Existing _decompress_cache method unchanged (AC 5)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=4)

        key = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        value = torch.randn(3, NUM_KV_HEADS, HEAD_SIZE)
        impl._compress_and_store(key, value, kv_cache, torch.tensor([0, 5, 33]))

        # Dynamic allocation (no out=)
        k_dyn, v_dyn = impl._decompress_cache(
            kv_cache, torch.float32, apply_rotation=False
        )
        assert k_dyn.shape == (4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)

        # With out= buffers
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float32)
        k_pre, v_pre = impl._decompress_cache(
            kv_cache,
            torch.float32,
            apply_rotation=False,
            out_k=impl._cg_decompress_k,
            out_v=impl._cg_decompress_v,
        )
        assert torch.equal(k_dyn, k_pre)
        assert torch.equal(v_dyn, v_pre)

    def test_multi_sequence_batch_dedup(self, tq4_quantizer) -> None:
        """Multi-sequence batch with overlapping physical blocks (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=20)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        # Two sequences sharing block 5, padding with zeros in unused slots
        # Seq 0: blocks [3, 5], seq_len=2*BS
        # Seq 1: blocks [5, 7], seq_len=2*BS
        block_table = torch.tensor([[3, 5, 0], [5, 7, 0]], dtype=torch.int32)
        seq_lens = torch.tensor([2 * BLOCK_SIZE, 2 * BLOCK_SIZE], dtype=torch.int32)

        k_paged, v_paged, remap_bt = impl._decompress_cache_paged(
            kv_cache,
            block_table,
            seq_lens,
            torch.float16,
            out_k=impl._cg_prefill_k,
            out_v=impl._cg_prefill_v,
        )

        # Unique blocks: {3, 5, 7} -> 3 compact blocks
        assert k_paged.shape[0] == 3

        # Block 5 appears in both sequences but decompressed only once
        # remap_bt should map: 3->0, 5->1, 7->2
        assert remap_bt[0, 0] == 0  # block 3 -> compact 0
        assert remap_bt[0, 1] == 1  # block 5 -> compact 1
        assert remap_bt[1, 0] == 1  # block 5 -> compact 1 (same)
        assert remap_bt[1, 1] == 2  # block 7 -> compact 2

        # Padding columns (beyond blocks_needed) must be zero (safe sentinel)
        assert remap_bt[0, 2] == 0
        assert remap_bt[1, 2] == 0

    def test_paged_decompress_fallback_logs_warning(
        self, tq4_quantizer, caplog
    ) -> None:
        """Warning logged when fallback to dynamic allocation (AC 2)."""
        impl = make_impl(tq4_quantizer)
        kv_cache = make_cache(num_blocks=200)
        impl._init_cg_buffers(kv_cache, compute_dtype=torch.float16)

        num_blocks_needed = impl._max_prefill_blocks + 1
        block_table = torch.tensor([list(range(num_blocks_needed))], dtype=torch.int32)
        seq_lens = torch.tensor([num_blocks_needed * BLOCK_SIZE], dtype=torch.int32)

        with caplog.at_level(
            logging.WARNING, logger="turboquant_vllm.vllm.tq4_backend"
        ):
            impl._decompress_cache_paged(
                kv_cache,
                block_table,
                seq_lens,
                torch.float16,
                out_k=impl._cg_prefill_k,
                out_v=impl._cg_prefill_v,
            )

        assert any("dynamic fallback" in msg for msg in caplog.messages)
