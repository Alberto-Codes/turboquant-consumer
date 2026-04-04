#!/usr/bin/env python3
"""Scope-header smoke test: verify capability non-disclosure.

Simulates a gateway-fronted deployment (Apigee/Kong injecting the
OpenAI-Project header) with listing disabled, proving that
cross-scope operations return 404 rather than 403.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path("/var/home/Alberto-Codes/Projects/vllm")))

PORT = 8766
URL = f"http://127.0.0.1:{PORT}"


def build_app(upload_dir: Path):
    from fastapi import FastAPI
    from vllm.config import FileUploadConfig
    from vllm.entrypoints.openai.files.api_router import attach_router
    from vllm.entrypoints.openai.files.serving import OpenAIServingFiles
    from vllm.entrypoints.openai.files.store import FileUploadStore, register_store

    config = FileUploadConfig(
        enabled=True,
        dir=str(upload_dir),
        max_size_mb=128,
        scope_header="OpenAI-Project",  # simulate Apigee injection
        disable_listing=True,  # compliance-hardened
    )
    store = FileUploadStore(config)
    register_store(store)
    app = FastAPI()
    app.state.openai_serving_files = OpenAIServingFiles(store, config)
    attach_router(app)
    return app


def run_server(upload_dir: Path):
    harness = Path("/tmp/vllm_scope_smoke_harness.py")
    harness.write_text(
        f"""
import sys
sys.path.insert(0, '/var/home/Alberto-Codes/Projects/vllm')
sys.path.insert(0, '{Path(__file__).parent}')
from smoke_test_scope import build_app
app = build_app('{upload_dir}')
"""
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "vllm_scope_smoke_harness:app",
            "--port",
            str(PORT),
            "--host",
            "127.0.0.1",
            "--log-level",
            "warning",
        ],
        cwd="/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_server(timeout: float = 15.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            # /v1/files returns 404 (listing disabled) but that means the
            # server is up.
            httpx.get(f"{URL}/v1/files", headers={"OpenAI-Project": "x"}, timeout=0.5)
            return
        except httpx.RequestError:
            time.sleep(0.2)
    raise RuntimeError("server never came up")


def smoke_test(clip_a: Path, clip_b: Path) -> None:
    print("\n=== Scope-header smoke test ===")
    print("Server config: --file-upload-scope-header OpenAI-Project")
    print("               --file-upload-disable-listing")
    print()

    with httpx.Client(base_url=URL, timeout=60.0) as client:
        # 1. Upload without scope header → 400
        print("1. POST /v1/files  (no scope header)")
        with open(clip_a, "rb") as f:
            r = client.post(
                "/v1/files",
                files={"file": (clip_a.name, f, "video/mp4")},
                data={"purpose": "vision"},
            )
        print(f"   status: {r.status_code}  (expected 400)")
        assert r.status_code == 400, r.text
        assert "OpenAI-Project" in r.json()["error"]["message"]

        # 2. Upload as team-alpha
        print("\n2. POST /v1/files  (OpenAI-Project: team-alpha)")
        with open(clip_a, "rb") as f:
            r = client.post(
                "/v1/files",
                files={"file": (clip_a.name, f, "video/mp4")},
                data={"purpose": "vision"},
                headers={"OpenAI-Project": "team-alpha"},
            )
        assert r.status_code == 200, r.text
        alpha_id = r.json()["id"]
        print(f"   id: {alpha_id}")

        # 3. Upload as team-bravo
        print("\n3. POST /v1/files  (OpenAI-Project: team-bravo)")
        with open(clip_b, "rb") as f:
            r = client.post(
                "/v1/files",
                files={"file": (clip_b.name, f, "video/mp4")},
                data={"purpose": "vision"},
                headers={"OpenAI-Project": "team-bravo"},
            )
        assert r.status_code == 200, r.text
        bravo_id = r.json()["id"]
        print(f"   id: {bravo_id}")

        # 4. team-bravo tries to retrieve team-alpha's file → 404
        print(f"\n4. GET /v1/files/{alpha_id}  (as team-bravo)")
        r = client.get(
            f"/v1/files/{alpha_id}", headers={"OpenAI-Project": "team-bravo"}
        )
        print(f"   status: {r.status_code}  (expected 404 — NOT 403)")
        assert r.status_code == 404

        # 5. team-alpha retrieves own file → 200
        print(f"\n5. GET /v1/files/{alpha_id}  (as team-alpha)")
        r = client.get(
            f"/v1/files/{alpha_id}", headers={"OpenAI-Project": "team-alpha"}
        )
        print(f"   status: {r.status_code}  (expected 200)")
        assert r.status_code == 200

        # 6. List is disabled → 404 even with correct scope
        print("\n6. GET /v1/files  (listing disabled)")
        r = client.get("/v1/files", headers={"OpenAI-Project": "team-alpha"})
        print(f"   status: {r.status_code}  (expected 404 — disable_listing)")
        assert r.status_code == 404

        # 7. team-bravo tries to delete team-alpha's file → 404
        print(f"\n7. DELETE /v1/files/{alpha_id}  (as team-bravo)")
        r = client.delete(
            f"/v1/files/{alpha_id}", headers={"OpenAI-Project": "team-bravo"}
        )
        print(f"   status: {r.status_code}  (expected 404)")
        assert r.status_code == 404

        # 8. team-alpha's file still there
        r = client.get(
            f"/v1/files/{alpha_id}", headers={"OpenAI-Project": "team-alpha"}
        )
        assert r.status_code == 200, "team-alpha's file should still exist"

    print("\n✓ SCOPE TEST PASSED — capability non-disclosure enforced")


def main():
    clips_dir = Path("/var/home/Alberto-Codes/Projects/molmo-video-analyzer/data/tv/")
    clip_a = clips_dir / "soup-nazi-clip01-early.mp4"
    clip_b = clips_dir / "soup-nazi-clip02-soupstand.mp4"
    upload_dir = Path("/tmp/vllm-scope-smoke")
    upload_dir.mkdir(exist_ok=True)

    proc = run_server(upload_dir)
    try:
        wait_for_server()
        smoke_test(clip_a, clip_b)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
