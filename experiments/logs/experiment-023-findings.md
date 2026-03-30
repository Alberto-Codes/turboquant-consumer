# Experiment 023 — Frame Count Sweep: Quality vs Frame Density

**Date:** 2026-03-30
**Hardware:** RTX 4090 (24 GiB), NVIDIA driver 595.x
**Episode:** Seinfeld S05E12 "The Stall" (1371s)
**Start offset:** 120s (skip cold open/credits)
**Max tokens:** 512
**vLLM:** v0.18.0, TQ4 via vllm-turboquant:1.2.2

## Goal

Determine the ideal combination of clip duration, frame sampling rate, and backend (baseline FP8 KV vs TQ4) for video inference on Molmo2-8B with 24 GiB VRAM.

## Setup

| Config | Baseline | TQ4 v1.2.2 |
|---|---|---|
| Image | `vllm/vllm-openai:v0.18.0` | `vllm-turboquant:1.2.2` |
| KV cache | `--kv-cache-dtype fp8` | `--attention-backend CUSTOM` (TQ4) |
| Model len | 6144 | 6144 |
| GPU util | 0.90 | 0.90 |
| Eager | yes | yes |

Frame sampling controlled via `--media-io-kwargs '{"video": {"num_frames": N}}'`.

## Key Discovery: Molmo2 Video Token Math

Molmo2's `Molmo2VideoProcessor` config:
- **83 tokens per frame** (9x9 grid after 14px patches + 3x3 pooling + 2 special tokens)
- `sampling_fps=2`, `max_fps=2`, `num_frames=384` (model capacity)
- `frame_sample_mode=uniform_last_frame`

vLLM caps frames via: `max_frames = max_model_len // tokens_per_frame`

| max_model_len | Max frames | At 2fps, max clip |
|---|---|---|
| 6144 | 74 | 37s |
| 8192 | 98 | 49s |

The `--media-io-kwargs` `num_frames` override works when set **above** the default (~31 for a 30s clip). Setting it below default has no effect (sampler already picks fewer).

## Results — Molmo2-8B Baseline (FP8 KV)

### Default fps (~1fps, ~31 frames)

| Clip | Tokens | Frames | Elapsed | Characters |
|---|---|---|---|---|
| 5s | 1,798 | 21 | 9.7s | jerry:5, elaine:4 |
| 15s | 2,334 | 28 | 8.5s | jerry:2 |
| 30s | 2,609 | 31 | 12.8s | jerry:5, george:3, elaine:6 |

### 2fps (num_frames=60)

| Clip | Tokens | Frames | Elapsed | Characters |
|---|---|---|---|---|
| 5s | 3,294 | 39 | 13.1s | jerry:9, elaine:3 |
| 15s | 4,366 | 52 | 14.6s | jerry:2, george:3, elaine:4 |
| 30s | 4,826 | 58 | 16.9s | **(zero)** — generic "two couples" |

**Finding:** 2fps improved 5s (jerry:9!) and 15s (added george+elaine) but destroyed 30s quality. More frames spread attention too thin at longer durations.

## Results — Molmo2-8B TQ4 v1.2.2

### Default fps (~1fps, ~31 frames)

| Clip | Tokens | Frames | Elapsed | Characters |
|---|---|---|---|---|
| 5s | 1,798 | 21 | 9.6s | jerry:3, elaine:2 |
| 15s | 2,334 | 28 | 5.6s | elaine:5 |
| **30s** | **2,609** | **31** | **15.8s** | **jerry:8, george:6, elaine:6** |
| 45s | 2,701 | 32 | 14.8s | jerry:4, elaine:4 |
| **55s** | **2,702** | **32** | **14.1s** | **jerry:5, george:4, elaine:6** |
| 60s | — | — | — | OOM (ViT GELU) |

### 2fps (num_frames=60)

| Clip | Tokens | Frames | Elapsed | Characters |
|---|---|---|---|---|
| 5s | 3,294 | 39 | 10.5s | **(zero)** — hallucinated "Friends" |
| 15s | — | — | — | OOM (ViT GELU) |
| 30s | 4,826 | 58 | 19.9s | jerry:6, elaine:6 |

### 0.5fps (num_frames=15)

| Clip | Tokens | Frames | Elapsed | Characters |
|---|---|---|---|---|
| 30s | 1,190 | 14 | 9.8s | **(zero)** — hallucinated "Jennifer Aniston" |
| 60s | 1,281 | 15 | 11.3s | elaine:1 |
| **120s** | **1,284** | **15** | **14.4s** | **jerry:7, elaine:7** |

**Finding:** 0.5fps unlocks 2-minute clips — no ViT OOM (only 15 frames). Quality is poor at 30s (too sparse) but recovers at 120s because 15 frames spanning 2 minutes catch more scene transitions and character appearances.

## Head-to-Head: Baseline vs TQ4 at 30s Default FPS

| Metric | Baseline (FP8 KV) | TQ4 v1.2.2 |
|---|---|---|
| Elapsed | 12.8s | 15.8s |
| Input tokens | 2,609 | 2,609 |
| Jerry mentions | 5 | **8** |
| George mentions | 3 | **6** |
| Elaine mentions | 6 | **6** |
| Total characters | 14 | **20** |

TQ4 produces 43% more character mentions at the same frame count. The output text correctly names "Seinfeld" and describes scene transitions with specific character actions.

## ViT OOM Ceiling

The vision encoder (not TQ4 KV cache) is the VRAM bottleneck for high frame counts:

| Backend | Max frames before ViT OOM | Max clip at default fps |
|---|---|---|
| Baseline FP8 | ~85 (4B), not tested (8B) | 30s+ works at default |
| TQ4 | ~60 (8B 2fps), ~32 (8B default) | 55s works, 60s OOMs |

TQ4's uint8 KV cache lets vLLM provision more blocks than FP8, leaving slightly less free VRAM for the ViT encoder. This is why TQ4 8B OOMs at 60s while baseline 8B can serve 60s clips (from Experiment 022 data).

## Conclusions

1. **Default ~1fps is the best frame rate for 8B.** 2fps hurts quality at 30s+ (attention dilution). 0.5fps is too sparse for short clips but enables long clips.

2. **30s is the quality sweet spot at default fps.** Both baseline and TQ4 peak here. TQ4 gets all 3 characters (jerry:8, george:6, elaine:6); baseline gets 2.5 (jerry:5, george:3, elaine:6).

3. **TQ4 pushes usable clip length to 55s** at default fps — all 3 characters recognized. Baseline was not tested at 55s but Experiment 022 showed baseline hallucinating at 30s+ with 2fps equivalent frame counts.

4. **0.5fps unlocks 120s clips** on TQ4 with good quality (jerry:7, elaine:7). Fewer frames reduce ViT VRAM pressure and give each frame more attention weight at long durations.

5. **4B is useless for character recognition.** Hallucinated "Cruel Intentions" at every frame count tested (default through 85 frames). The 8B model is required for video quality tasks.

6. **`--media-io-kwargs` works** for controlling Molmo2 frame sampling in vLLM, but only increases past the default. It cannot reduce below what the sampler naturally selects.

## Recommended Configurations

| Use Case | FPS | Clip Duration | Config |
|---|---|---|---|
| **Best quality** | default (~1fps) | 30s | `--max-model-len 6144` (no override) |
| **Longer clips, good quality** | default (~1fps) | 45-55s | `--max-model-len 6144` (no override) |
| **Maximum duration** | 0.5fps | 120s+ | `--media-io-kwargs '{"video": {"num_frames": 15}}'` |
| **Dense short clips** | 2fps | 5-15s | `--media-io-kwargs '{"video": {"num_frames": 60}}'` |

All configs use Molmo2-8B + TQ4 (`--attention-backend CUSTOM`) + `--enforce-eager` + `--gpu-memory-utilization 0.90`.

## Molmo2-4B Frame Sweep (supplemental)

4B was tested first for VRAM headroom. Frame override confirmed working but quality was unusable:

| num_frames | Tokens | Frames | Elapsed | Quality |
|---|---|---|---|---|
| default | 2,609 | 31 | 8.9s | "Cruel Intentions" halluc. |
| 60 | 4,826 | 58 | 16.9s | "Willie and I" halluc. |
| 75 | 6,067 | 73 | 14.9s | "Cruel Intentions" halluc. |
| 85 | 6,865 | 83 | 21.7s | "Cruel Intentions" halluc. |
| 90 | — | — | — | OOM (ViT GELU) |

ViT OOM ceiling on 4B baseline FP8: ~85-90 frames.
