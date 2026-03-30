r"""Experiment 023 -- Frame count sweep: quality vs frame density at fixed clip duration.

Tests how Molmo2's output quality and token count scale with the number of
video frames extracted by vLLM.  The frame count is a server-side config
(``--media-io-kwargs``), so the vLLM server must be restarted between runs
with different ``num_frames`` values.

Workflow (repeat for each backend: baseline FP8, TQ4):

    # 1. Default frames (no override -- discover vLLM's default):
    vllm serve allenai/Molmo2-4B [backend flags] --max-model-len 16384
    uv run python experiments/experiment_023_frame_count_sweep.py \
        --tag baseline-default --frames-label default

    # 2. 32 frames:
    vllm serve allenai/Molmo2-4B [backend flags] --max-model-len 16384 \
        --media-io-kwargs '{"video": {"num_frames": 32}}'
    uv run python experiments/experiment_023_frame_count_sweep.py \
        --tag baseline-32f --frames-label 32

    # 3. 64 frames:
    vllm serve ... --media-io-kwargs '{"video": {"num_frames": 64}}'
    uv run python experiments/experiment_023_frame_count_sweep.py \
        --tag baseline-64f --frames-label 64

    # 4. 128 frames:
    vllm serve ... --media-io-kwargs '{"video": {"num_frames": 128}}'
    uv run python experiments/experiment_023_frame_count_sweep.py \
        --tag baseline-128f --frames-label 128

    Then repeat 1-4 with TQ4 backend (--attention-backend CUSTOM).

Notes:
    - Start with Molmo2-4B for VRAM headroom at high frame counts.
    - Use --max-model-len 16384 to accommodate higher token counts.
    - Input token count is the ground truth for how many frames were processed.
    - At 128 frames x ~250 tokens/frame = ~32K tokens -- may exceed 16384.
      If OOM or token overflow, the script logs the error and continues.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

_EPISODE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "molmo-video-analyzer"
    / "data"
    / "tv"
    / "Seinfeld - S05E12 - The Stall - [WEBDL-720P][AAC 2.0][H264]-NTB.mkv"
)

_PROMPT = (
    "Describe what is happening in this video clip in detail. "
    "Include the names of any characters you recognize, the setting, "
    "and any notable actions or dialogue."
)

_CHARACTERS = [
    "jerry",
    "george",
    "kramer",
    "elaine",
    "newman",
    "puddy",
    "jane",
]

_START_OFFSET_S = 120
_CLIP_DURATION_S = 30


def _get_duration(video_path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _extract_clip(
    video_path: Path,
    start_s: float,
    duration_s: int,
    output_path: Path,
) -> Path:
    """Extract a single clip from the video at a specific offset."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
            "-i",
            str(video_path),
            "-t",
            str(duration_s),
            "-c",
            "copy",
            "-an",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def _count_characters(text: str) -> dict[str, int]:
    """Count Seinfeld character name mentions in text."""
    lower = text.lower()
    return {name: lower.count(name) for name in _CHARACTERS if lower.count(name) > 0}


def _send_clip(
    clip_path: Path,
    url: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Send a clip to the vLLM API and return results."""
    b64 = base64.b64encode(clip_path.read_bytes()).decode("ascii")
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{b64}"},
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()
    resp = requests.post(f"{url}/chat/completions", json=payload, timeout=600)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "elapsed_s": round(elapsed, 2),
        "output_text": text,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "characters": _count_characters(text),
    }


def main() -> None:
    """CLI entry point for Experiment 023."""
    parser = argparse.ArgumentParser(
        description="Experiment 023: Frame count sweep at fixed clip duration",
    )
    parser.add_argument("--episode", type=Path, default=_EPISODE_PATH)
    parser.add_argument(
        "--clip-duration",
        type=int,
        default=_CLIP_DURATION_S,
        help="Clip duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration for consistency (default: 3)",
    )
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="allenai/Molmo2-4B")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--tag",
        required=True,
        help="Run tag (e.g., 'baseline-default', 'tq4-64f')",
    )
    parser.add_argument(
        "--frames-label",
        required=True,
        help="Frame count label for this run (e.g., 'default', '32', '64', '128')",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=_PROMPT,
        help="Prompt to send with each clip",
    )
    parser.add_argument(
        "--start-offset",
        type=float,
        default=_START_OFFSET_S,
        help="Seconds into episode to start (default: 120)",
    )
    args = parser.parse_args()

    if not args.episode.exists():
        print(f"Episode not found: {args.episode}")
        sys.exit(1)

    episode_duration = _get_duration(args.episode)

    if args.start_offset + args.clip_duration > episode_duration:
        print(
            f"Clip exceeds episode length: {args.start_offset} + {args.clip_duration} "
            f"> {episode_duration:.0f}s",
        )
        sys.exit(1)

    # Health check
    try:
        resp = requests.get(f"{args.vllm_url}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()
        print(f"vLLM ready — serving: {[m['id'] for m in models['data']]}")
    except requests.RequestException as exc:
        print(f"vLLM not reachable at {args.vllm_url}: {exc}")
        sys.exit(1)

    print("\nExperiment 023: Frame Count Sweep")
    print(f"{'=' * 60}")
    print(f"Episode: {args.episode.name} ({episode_duration:.0f}s)")
    print(f"Clip: {args.clip_duration}s from {args.start_offset}s offset")
    print(f"Model: {args.model}")
    print(f"Frames label: {args.frames_label}")
    print(f"Runs: {args.runs}")
    print(f"Tag: {args.tag}")

    results: dict[str, Any] = {
        "experiment": "023-frame-count-sweep",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "model_id": args.model,
        "max_new_tokens": args.max_new_tokens,
        "prompt": args.prompt,
        "episode": args.episode.name,
        "episode_duration_s": round(episode_duration, 1),
        "start_offset_s": args.start_offset,
        "clip_duration_s": args.clip_duration,
        "frames_label": args.frames_label,
        "runs": [],
    }

    with tempfile.TemporaryDirectory(prefix="exp023_") as tmpdir:
        tmp = Path(tmpdir)
        clip_path = tmp / f"clip_{args.clip_duration}s.mp4"

        _extract_clip(args.episode, args.start_offset, args.clip_duration, clip_path)
        clip_size_mb = clip_path.stat().st_size / (1024 * 1024)
        print(f"Clip size: {clip_size_mb:.1f} MB")

        for run_idx in range(args.runs):
            print(f"\n{'=' * 60}")
            print(f"RUN {run_idx + 1}/{args.runs} — frames={args.frames_label}")
            print(f"{'=' * 60}")

            try:
                result = _send_clip(
                    clip_path,
                    args.vllm_url,
                    args.model,
                    args.prompt,
                    args.max_new_tokens,
                )
                result["run"] = run_idx + 1
                result["clip_size_mb"] = round(clip_size_mb, 1)
                results["runs"].append(result)

                print(f"  Elapsed: {result['elapsed_s']}s")
                print(
                    f"  Tokens:  {result['input_tokens']} in, "
                    f"{result['output_tokens']} out",
                )
                print(f"  Characters: {result['characters']}")
                print(f"  Output: {result['output_text'][:200]}...")
            except requests.RequestException as exc:
                print(f"  FAILED: {exc}")
                results["runs"].append({"run": run_idx + 1, "error": str(exc)})

    # Summary
    successful = [r for r in results["runs"] if "error" not in r]
    if successful:
        avg_tokens = sum(r["input_tokens"] for r in successful) / len(successful)
        avg_elapsed = sum(r["elapsed_s"] for r in successful) / len(successful)
        all_chars: dict[str, int] = {}
        for r in successful:
            for name, count in r["characters"].items():
                all_chars[name] = all_chars.get(name, 0) + count

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Frames label: {args.frames_label}")
        print(f"Successful runs: {len(successful)}/{args.runs}")
        print(f"Avg input tokens: {avg_tokens:.0f}")
        print(f"Avg elapsed: {avg_elapsed:.1f}s")
        print(f"Characters (total across runs): {all_chars}")
        print(
            f"Est. frames processed: ~{(avg_tokens - 23) / 83:.0f} "
            f"(at ~83 tokens/frame)",
        )

        results["summary"] = {
            "successful_runs": len(successful),
            "avg_input_tokens": round(avg_tokens),
            "avg_output_tokens": round(
                sum(r["output_tokens"] for r in successful) / len(successful),
            ),
            "avg_elapsed_s": round(avg_elapsed, 2),
            "total_characters": all_chars,
            "est_frames": round((avg_tokens - 23) / 83.0),
        }

    output_path = Path(f"experiments/logs/experiment-023-{args.tag}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
