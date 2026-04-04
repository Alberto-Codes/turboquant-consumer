#!/usr/bin/env python3
"""GPU smoke test: upload video → chat completion via vllm-file:// → Molmo2.

Proves the full loop — that a video uploaded via POST /v1/files can be
referenced in a chat completion as {"url": "vllm-file://<id>"} and
actually consumed by the multimodal model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8101"


def smoke_test(video_path: Path) -> None:
    print("=== GPU smoke test ===")
    print(f"server:  {URL}")
    print(
        f"video:   {video_path.name} ({video_path.stat().st_size / 1024 / 1024:.1f} MiB)"
    )
    print()

    with httpx.Client(base_url=URL, timeout=120.0) as client:
        # List models
        r = client.get("/v1/models")
        assert r.status_code == 200, r.text
        model_id = r.json()["data"][0]["id"]
        print(f"model:   {model_id}")

        # Upload
        print("\n[1/3] POST /v1/files ...")
        t0 = time.time()
        with open(video_path, "rb") as f:
            r = client.post(
                "/v1/files",
                files={"file": (video_path.name, f, "video/mp4")},
                data={"purpose": "vision"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        file_id = body["id"]
        print(
            f"        {file_id}  ({body['bytes']} bytes, {(time.time() - t0) * 1000:.0f}ms)"
        )

        # Chat completion referencing the uploaded file
        print(f"\n[2/3] POST /v1/chat/completions  (video_url=vllm-file://{file_id})")
        t0 = time.time()
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": f"vllm-file://{file_id}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Describe what is happening in this video clip "
                                    "in 2-3 sentences. What scene or setting does it depict?"
                                ),
                            },
                        ],
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.2,
            },
        )
        elapsed_ms = (time.time() - t0) * 1000
        assert r.status_code == 200, r.text
        answer = r.json()["choices"][0]["message"]["content"]
        usage = r.json().get("usage", {})
        print(
            f"        ({elapsed_ms:.0f}ms, {usage.get('prompt_tokens')} prompt tokens, "
            f"{usage.get('completion_tokens')} completion tokens)"
        )
        print()
        print("MODEL RESPONSE:")
        print(f"  {answer}")

        # Cleanup
        print(f"\n[3/3] DELETE /v1/files/{file_id}")
        r = client.delete(f"/v1/files/{file_id}")
        assert r.status_code == 200, r.text

    print("\n✓ FULL LOOP WORKS — video uploaded, referenced, consumed by Molmo2")


def main() -> None:
    video = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(
            "/var/home/Alberto-Codes/Projects/molmo-video-analyzer/data/tv/"
            "soup-nazi-clip01-early.mp4"
        )
    )
    if not video.exists():
        print(f"Video not found: {video}", file=sys.stderr)
        sys.exit(1)
    smoke_test(video)


if __name__ == "__main__":
    main()
