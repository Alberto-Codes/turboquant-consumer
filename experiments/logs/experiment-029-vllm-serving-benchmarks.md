# Experiment 029: End-to-End vLLM Serving Benchmarks

**Date:** 2026-04-01
**Hardware:** NVIDIA RTX 4090 (24 GiB), AMD Ryzen 9 7950X
**vLLM:** 0.18.0 (V1 engine)
**TQ4:** turboquant-vllm 1.3.0 (K4/V4 default, `--attention-backend CUSTOM`)
**Baseline:** Separate venv, vLLM 0.18.0, no TQ4 plugin (FP16 KV cache)

## Objective

Measure TQ4 KV cache compression overhead in a real vLLM serving context,
including throughput, latency, and VRAM impact. Compare against uncompressed
FP16 baseline and evaluate fused paged decode kernel benefit.

## Methodology

### 3-Layer Benchmark Protocol

| Layer | Tool | Question Answered | Key Metric |
|-------|------|-------------------|------------|
| 1. Offline throughput | `vllm bench throughput` | Pure engine-level TQ4 overhead | tok/s (no HTTP) |
| 2. Online saturation | `vllm bench serve --request-rate inf` | Peak capacity with batching | tok/s, TTFT, TPOT |
| 3. Online latency sweep | `vllm bench serve --request-rate 1,2,4,8,16` | Latency under realistic load | TTFT p50, TPOT p50 |

### Parameters

- `--gpu-memory-utilization 0.85`, `--max-model-len 4096`, `--enforce-eager`
- `--dataset-name random`, input=512 tokens, output=128 tokens
- `--num-warmups 10` for server warmup
- `--num-prompts`: 200 (offline/saturation), 100 (sweep)
- **Reproducibility:** 3 independent runs per configuration (fresh server restart each run)
- **Reporting:** Median with P25/P75 range across runs
- **Primary metric:** TQ4/baseline ratio (stable across hardware)

### Baseline Isolation

TQ4 auto-registers via `vllm.general_plugins` entry point. Baseline MUST use a
separate venv with the exact same vLLM version (0.18.0) but no turboquant-vllm
installed. Verified: `import turboquant_vllm` raises `ModuleNotFoundError` in
baseline venv.

## Results

### TQ4 vs Baseline — Saturation (rate=inf)

| Metric | Molmo2-4B TQ4 | Molmo2-4B Base | Ratio | Llama-8B TQ4 | Llama-8B Base | Ratio |
|--------|--------------|----------------|-------|-------------|---------------|-------|
| Output throughput (tok/s) | 1,232 | 1,747 | **0.71** | 967 | 1,042 | **0.93** |
| TTFT p50 (ms) | 4,172 | 2,987 | 1.40 | 6,977 | 9,324 | **0.75** |
| TPOT p50 (ms) | 127 | 44 | **2.87** | 144 | 48 | **3.02** |
| Peak VRAM (MiB) | 23,984 | 20,242 | +3,742 | 24,018 | 22,387 | +1,631 |

### TQ4 vs Baseline — Latency Sweep

**Molmo2-4B — Throughput (tok/s)**

| QPS | TQ4 | Baseline | Ratio |
|-----|-----|----------|-------|
| 1 | 123 | 125 | 0.98 |
| 2 | 222 | 247 | 0.90 |
| 4 | 437 | 470 | 0.93 |
| 8 | 726 | 891 | 0.82 |
| 16 | 1,046 | 1,473 | 0.71 |

**Molmo2-4B — TPOT p50 (ms)**

| QPS | TQ4 | Baseline | Ratio |
|-----|-----|----------|-------|
| 1 | 31 | 15 | 2.03 |
| 4 | 34 | 16 | 2.08 |
| 16 | 50 | 26 | 1.95 |

**Llama-3.1-8B — Throughput (tok/s)**

| QPS | TQ4 | Baseline | Ratio |
|-----|-----|----------|-------|
| 1 | 122 | 125 | 0.98 |
| 2 | 225 | 246 | 0.91 |
| 4 | 430 | 463 | 0.93 |
| 8 | 727 | 818 | 0.89 |
| 16 | 994 | 1,001 | 0.99 |

**Llama-3.1-8B — TPOT p50 (ms)**

| QPS | TQ4 | Baseline | Ratio |
|-----|-----|----------|-------|
| 1 | 36 | 20 | 1.81 |
| 4 | 38 | 25 | 1.53 |
| 16 | 54 | 46 | 1.16 |

### Offline Throughput (Layer 1)

| Model | TQ4 tok/s | Notes |
|-------|-----------|-------|
| Molmo2-4B | 6,170 (median, 3 runs) | Peak VRAM 23,874 MiB |
| Llama-3.1-8B | OOM | 16 GiB weights + activations exceed 24 GiB at 0.85 util |

Baseline offline not collected (same OOM limitation for 8B model; Molmo2
baseline offline omitted for consistency).

### Fused vs Decompress-All (Llama-3.1-8B)

| Metric | Decompress-All (eager) | Fused (eager) | Fused/Decomp Ratio |
|--------|----------------------|---------------|-------------------|
| Output throughput (tok/s) | 967 | 941 | 0.97 |
| TTFT p50 (ms) | 6,977 | 7,114 | 1.02 |
| TPOT p50 (ms) | 144 | 147 | 1.03 |

**Fused + CUDA Graphs: BLOCKED.** Server crashes during CUDA graph capture:
`cudaErrorStreamCaptureUnsupported`. The Triton-based fused paged decode kernel
does not support CUDA graph stream capture despite the backend reporting
`UNIFORM_SINGLE_TOKEN_DECODE` CG support level. This means the primary benefit
of the fused path (enabling CUDA graphs for reduced Python overhead) is not
available.

**Sweep rates show similar parity** — fused/decomp ratios are within ±3% across
all QPS levels (1-16), indicating the kernel-level 1.7x speedup (Exp 020) does
not translate to serving-level benefit. vLLM's batching and scheduling overhead
dominates kernel execution time.

## Analysis

### TQ4 Overhead Profile

1. **TPOT is the dominant overhead**: 2-3x baseline at saturation, 1.2-2.0x at
   low QPS. This is the decompression cost — each decode step must decompress
   KV cache before attention.

2. **Throughput overhead scales with load**: At low QPS (1-4), throughput is
   near parity (0.93-0.98x). At high QPS (16+), overhead increases to 0.71x
   for Molmo2-4B. vLLM's continuous batching helps but can't fully hide the
   decompression cost at saturation.

3. **VRAM: TQ4 uses MORE, not less**, on this 24 GiB GPU. TQ4's compressed
   blocks allow 3.76x more KV blocks in the same pool, but the decompression
   buffers, rotation matrices, and codebooks add fixed overhead. Net VRAM is
   +1.6 to +3.7 GiB above baseline. The compression benefit manifests as
   **capacity** (more sequences/longer context) not as memory savings at fixed
   workload.

4. **Llama vs Molmo2 asymmetry**: Llama-8B shows less throughput overhead
   (0.93x vs 0.71x) despite having 2x the parameters. This is because Llama's
   KV pool is only ~3 GiB (model weights dominate), making the KV
   compression overhead relatively smaller. Molmo2-4B has more KV headroom
   (smaller model, larger pool), so the decompression cost is more visible.

### Fused Kernel: Serving-Level Verdict

The fused paged decode kernel, which showed 1.7x speedup at kernel level
(Exp 020, 1K context), provides **no measurable benefit in serving context**.
Two factors explain this:

1. **Batching amortizes kernel overhead**: vLLM processes multiple requests per
   decode step. The kernel execution time is a small fraction of total step time.
2. **CUDA graph capture fails**: The Triton kernel cannot be captured in CUDA
   graphs, eliminating the main benefit (Python overhead elimination). The fused
   path reports CG support but the kernel itself is not capturable.

**Recommendation:** The fused kernel is not production-ready for serving.
Continue using the decompress-all path until either (a) the Triton kernel
supports CUDA graph capture, or (b) a CUDA/C++ kernel replaces it.

### Where TQ4 Wins

TQ4's value proposition on constrained GPUs is **capacity, not speed**:

- 3.76x more KV cache blocks → longer contexts or more concurrent sequences
- At the same context length and concurrency, TQ4 is slower but can serve
  workloads that wouldn't fit in FP16 KV cache at all
- The overhead is most visible on small GPUs (24 GiB) where the KV pool is
  small relative to model weights; on larger GPUs (80 GiB), the ratio improves

## Known Limitations

1. **Layer 1 (offline throughput) OOMs for Llama-3.1-8B** at 0.85 gpu-util on
   24 GiB GPU. The vLLM engine's batch scheduler requires more activation memory
   than available after model + KV pool allocation.

2. **Fused + CUDA graphs not working**: `cudaErrorStreamCaptureUnsupported`
   during graph capture. Triton kernels may not support stream capture.

3. **Molmo2-4B baseline** requires `--max-num-batched-tokens 4096` for the
   multimodal encoder budget, even with text-only prompts. Added as CLI flag.

4. **vLLM 0.18.0 V1 engine** uses `/v1/chat/completions` endpoint (not
   `/v1/completions`). Benchmark client uses `--backend openai-chat`.

## Recommended Configurations

| Use Case | Config | Why |
|----------|--------|-----|
| Production (latency-sensitive) | Baseline FP16 | 2-3x lower TPOT |
| Memory-constrained (long context) | TQ4 decompress-all, eager | 3.76x KV capacity, 7-30% throughput cost |
| Latency-tolerant batch | TQ4 decompress-all, eager | Near-parity throughput at low QPS |

Do NOT use TQ4 fused paged decode in production — no benefit over decompress-all
in serving context and CUDA graphs are blocked.

## Quality Gate Checkpoint

- **Test suite:** 457 passed, 10 skipped (0 failures)
- **Coverage:** 96.93% (95% `fail_under` gate passing)
- **Ruff lint/format:** Clean
- **Type check (ty):** Clean

## File Inventory

| File | Description |
|------|-------------|
| `experiment_029_vllm_serving_benchmark.py` | Benchmark orchestrator script |
| `experiment-029-molmo2-4b-tq4-*.json` | Molmo2-4B TQ4 per-run results |
| `experiment-029-molmo2-4b-baseline-*.json` | Molmo2-4B baseline per-run results |
| `experiment-029-llama-8b-tq4-*.json` | Llama-3.1-8B TQ4 per-run results |
| `experiment-029-llama-8b-baseline-*.json` | Llama-3.1-8B baseline per-run results |
| `experiment-029-llama-8b-fused-eager-*.json` | Fused eager per-run results |
| `experiment-029-llama-8b-fused-cg-server-run1.log` | Fused+CG crash log |
