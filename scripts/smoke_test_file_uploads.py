#!/usr/bin/env python3
"""Standalone smoke test for the /v1/files endpoint from vllm#38531.

Spins up a minimal FastAPI app wrapping the files router (no engine, no
GPU), uploads a real video, and verifies the full round-trip. This is
the "show it works with real data" check that sits between the unit
tests and the full GPU integration test.

Usage:
    python scripts/smoke_test_file_uploads.py <path-to-video.mp4>
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import httpx

# Hook the local vllm fork onto the import path.
sys.path.insert(0, str(Path("/var/home/Alberto-Codes/Projects/vllm")))

SERVER_PORT = 8765
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"


def build_app(upload_dir: Path):
    from fastapi import FastAPI
    from vllm.config import FileUploadConfig
    from vllm.entrypoints.openai.files.api_router import attach_router
    from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
    from vllm.entrypoints.openai.files.store import FileUploadStore, register_store

    config = FileUploadConfig(
        enabled=True,
        dir=str(upload_dir),
        ttl_seconds=3600,
        max_size_mb=128,  # 128 MB cap
        max_total_gb=1,
        max_concurrent=4,
        scope_header="",
        disable_listing=False,
    )
    store = FileUploadStore(config)
    register_store(store)

    app = FastAPI()
    app.state.openai_serving_files = OpenAIServingFiles(store, config)
    attach_router(app)
    return app


def run_server(upload_dir: Path):
    """Run uvicorn in a subprocess with the above app."""
    harness = Path("/tmp/vllm_files_smoke_harness.py")
    harness.write_text(
        f"""
import sys
sys.path.insert(0, '/var/home/Alberto-Codes/Projects/vllm')
sys.path.insert(0, '{Path(__file__).parent}')
from smoke_test_file_uploads import build_app
app = build_app('{upload_dir}')
"""
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "vllm_files_smoke_harness:app",
            "--port",
            str(SERVER_PORT),
            "--host",
            "127.0.0.1",
            "--log-level",
            "info",
        ],
        cwd="/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    return proc


def wait_for_server(timeout: float = 15.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            httpx.get(f"{SERVER_URL}/v1/files", timeout=0.5)
            return
        except httpx.RequestError:
            time.sleep(0.2)
    raise RuntimeError("server never came up")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def smoke_test(video_path: Path) -> None:
    print("\n=== vLLM /v1/files smoke test ===")
    print(f"File:     {video_path}")
    print(f"Size:     {video_path.stat().st_size / 1024 / 1024:.1f} MiB")
    original_hash = sha256_of(video_path)
    print(f"sha256:   {original_hash}")
    print()

    with httpx.Client(base_url=SERVER_URL, timeout=60.0) as client:
        # 1. Upload
        print("POST /v1/files ...")
        t0 = time.time()
        with open(video_path, "rb") as f:
            r = client.post(
                "/v1/files",
                files={"file": (video_path.name, f, "video/mp4")},
                data={"purpose": "vision"},
            )
        upload_ms = (time.time() - t0) * 1000
        print(f"  status: {r.status_code}  ({upload_ms:.0f} ms)")
        assert r.status_code == 200, r.text
        body = r.json()
        print(f"  id:       {body['id']}")
        print(f"  bytes:    {body['bytes']}")
        print(f"  filename: {body['filename']}")
        print(f"  purpose:  {body['purpose']}")
        print(f"  status:   {body['status']}")
        file_id = body["id"]
        assert body["bytes"] == video_path.stat().st_size

        # 2. List
        print("\nGET /v1/files ...")
        r = client.get("/v1/files")
        assert r.status_code == 200, r.text
        data = r.json()["data"]
        print(f"  listed {len(data)} file(s)")
        assert any(f["id"] == file_id for f in data)

        # 3. Get metadata
        print(f"\nGET /v1/files/{file_id} ...")
        r = client.get(f"/v1/files/{file_id}")
        assert r.status_code == 200, r.text
        assert r.json()["id"] == file_id

        # 4. Download content, verify sha256 matches
        print(f"\nGET /v1/files/{file_id}/content ...")
        t0 = time.time()
        r = client.get(f"/v1/files/{file_id}/content")
        download_ms = (time.time() - t0) * 1000
        assert r.status_code == 200, r.text
        downloaded_hash = hashlib.sha256(r.content).hexdigest()
        print(f"  status: {r.status_code}  ({download_ms:.0f} ms)")
        print(f"  bytes:  {len(r.content)}")
        print(f"  sha256: {downloaded_hash}")
        print(f"  content-type: {r.headers.get('content-type')}")
        assert downloaded_hash == original_hash, (
            f"hash mismatch: uploaded {original_hash}, got {downloaded_hash}"
        )
        assert r.headers["content-type"].startswith("video/")

        # 5. Delete
        print(f"\nDELETE /v1/files/{file_id} ...")
        r = client.delete(f"/v1/files/{file_id}")
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] is True

        # 6. Retrieve-after-delete = 404
        print(f"\nGET /v1/files/{file_id} (post-delete) ...")
        r = client.get(f"/v1/files/{file_id}")
        print(f"  status: {r.status_code}")
        assert r.status_code == 404

    print(
        "\n✓ SMOKE TEST PASSED — full upload → list → get → download → delete round-trip"
    )


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-video.mp4>")
        sys.exit(1)
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"File not found: {video_path}")
        sys.exit(1)

    upload_dir = Path("/tmp/vllm-smoke-uploads")
    upload_dir.mkdir(exist_ok=True)

    proc = run_server(upload_dir)
    try:
        wait_for_server()
        smoke_test(video_path)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
