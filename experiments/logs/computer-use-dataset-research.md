# Computer Use Dataset Research — TQ4 Quality Benchmarking

**Date:** 2026-03-30
**Status:** Parked — resume after frame sampling experiments

## Goal

Use open-source computer use datasets (video recordings of humans using computers) as a TQ4 quality benchmark. GUI-heavy visual content (dense text, small UI elements, precise spatial layout) is a harder test for KV cache compression than TV video.

## Top Candidate: PSAI Computer Use Data

- **Dataset:** [anaisleila/computer-use-data-psai](https://huggingface.co/datasets/anaisleila/computer-use-data-psai)
- **GitHub:** [anaishowland/computeruse-data-psai](https://github.com/anaishowland/computeruse-data-psai)
- **Scale:** 3,167 completed tasks, all with MP4 screen recordings
- **Size:** 16.9 GB videos (49.2 GB total with DOM/screenshots)
- **Split:** 70% browser tasks, 30% desktop tasks
- **Difficulty:** Easy 79.4%, Medium 16.7%, Hard 3.9%
- **Ground truth:** `task_name`, `events` (action sequences), DOM snapshots
- **LLM Judge:** [llm-judge-psai](https://github.com/anaishowland/llm-judge-psai) — ready-made eval

### What each task contains
- **Video** — MP4 screen recording of the human performing the task
- **Screenshots** — individual frames (42.6% coverage)
- **DOM snapshots** — HTML element tree at each step (55.8% coverage)
- **Interaction events** — clicks, keystrokes, scrolls tied to DOM elements

### Estimated video durations
16.9 GB / 3,167 videos = ~5.3 MB avg. At 2-5 Mbps screen recording bitrate, roughly **10-20 seconds per video**.

## Other Candidates

| Dataset | Format | Scale | Link |
|---------|--------|-------|------|
| **VideoGUI** (NeurIPS 2024) | Instructional video + actions | 86 tasks, 2.7K annotations, professional software (Photoshop, SD WebUI) | [GitHub](https://github.com/showlab/videogui) |
| **ScreenAgent** (IJCAI 2024) | Screenshots + action sequences | Daily computer tasks (file ops, browsing, gaming) | [GitHub](https://github.com/niuzaisheng/ScreenAgent) |
| **AgentNet (OpenCUA)** | Human demonstrations | 3 OSes, 200+ apps | [GitHub](https://github.com/xlang-ai/OpenCUA) |
| **ScreenSpot-Pro** | Screenshots (image only) | 23 apps, 3 OSes, 5 industries | [GitHub](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding) |
| **OSWorld** | Real desktop environments | 369 tasks | [Site](https://os-world.github.io/) |

## Experiment Design (draft)

1. Download PSAI videos (16.9 GB)
2. Sample subset (e.g., 100 tasks across easy/medium/hard)
3. Serve via vLLM: baseline FP8 KV vs TQ4
4. Prompt: "What is the user doing in this video?" or use `task_name` as context
5. Eval: LLM judge comparing response against ground truth `task_name` + `events`
6. Metrics: accuracy (did model identify the correct action), wall-clock time, tokens

## Key Considerations

- Molmo2 samples up to **128 frames at up to 2 fps** — not a fixed 10 frames. The token plateau seen in Experiment 022 (~2,900 tokens) may be a vLLM `num_frames` default cap, not a Molmo2 limit.
- vLLM's `--media-io-kwargs '{"video": {"num_frames": N}}'` flag may be tunable.
- GUI screenshots are text-heavy — exactly where quantization errors show up. Harder test for TQ4 than TV video.
- Need to probe actual video durations and token counts on a small sample before committing to full run.
- `max_model_len=6144` may be tight if frame count is increased. At 2 fps × 15s × 250 tokens/frame = 7,500 tokens — would need to bump `max_model_len`.

## Molmo2 Video Benchmarks (for reference)

Molmo2 is evaluated on these video benchmarks (structured, downloadable):

**Short video:** MVBench, MotionBench, TempCompass, PerceptionTest, EgoSchema, NeXTQA
**Long video:** VideoMME, LongVideoBench, LVBench, MLVU
**Molmo2's own:** PixMo Dense Video Captions (104K videos), PixMo Video Grounding (520K instances)

None of these are computer-use focused. PSAI would be novel ground.
