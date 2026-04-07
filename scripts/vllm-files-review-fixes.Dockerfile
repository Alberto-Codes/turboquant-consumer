# Layer over vllm-openai that applies the feat/direct-multipart-video-
# upload branch PLUS the post-review hardening commits (6 new commits
# on top of the 2 original), so the patched container reflects exactly
# what the PR #39003 branch contains after my self-review pass.
#
# Strategy:
#   1. COPY new subsystem files verbatim (no deps on the rest of vllm).
#   2. OVERLAY connector.py — my changes touched structured regions
#      (url parsing, executor dispatch on media_io.load_bytes, new
#      async method) where surgical sed patches would be fragile.
#      Stock image's connector.py is upstream/main, which my branch is
#      based on, so overlay should be compatible.
#   3. SURGICAL PATCH cli_args.py, api_server.py, config/__init__.py —
#      these are small targeted additions to stable stock files.
#
# Build context must be the vLLM project root: ~/Projects/vllm

FROM docker.io/vllm/vllm-openai:latest

ARG VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm

# 1. New subsystem files — full drop-ins.
COPY vllm/config/file_upload.py ${VLLM_PKG}/config/file_upload.py
COPY vllm/entrypoints/openai/files/ ${VLLM_PKG}/entrypoints/openai/files/

# 2. Overlay the modified connector.py. My changes: regex-validated
#    file_id, url_spec.host over netloc, media_io.load_bytes dispatched
#    to global_thread_pool in the async path.
COPY vllm/multimodal/media/connector.py ${VLLM_PKG}/multimodal/media/connector.py

# 2b. The overlaid connector.py depends on env vars that are newer than
#     the stock vllm-openai image's envs.py. Backfill them here.
RUN python3 - <<'PY'
path = "/usr/local/lib/python3.12/dist-packages/vllm/envs.py"
src = open(path).read()
old = "VLLM_MEDIA_LOADING_THREAD_COUNT: int = 8"
new = """VLLM_MEDIA_LOADING_THREAD_COUNT: int = 8
    VLLM_MEDIA_CACHE: str = ""
    VLLM_MEDIA_CACHE_MAX_SIZE_MB: int = 5120
    VLLM_MEDIA_CACHE_TTL_HOURS: float = 24
    VLLM_MEDIA_URL_ALLOW_REDIRECTS: bool = False"""
src = src.replace(old, new, 1)
old2 = '''"VLLM_MEDIA_LOADING_THREAD_COUNT": lambda: int(
        os.getenv("VLLM_MEDIA_LOADING_THREAD_COUNT", "8")
    ),'''
new2 = '''"VLLM_MEDIA_LOADING_THREAD_COUNT": lambda: int(
        os.getenv("VLLM_MEDIA_LOADING_THREAD_COUNT", "8")
    ),
    "VLLM_MEDIA_CACHE": lambda: os.getenv("VLLM_MEDIA_CACHE", ""),
    "VLLM_MEDIA_CACHE_MAX_SIZE_MB": lambda: int(os.getenv("VLLM_MEDIA_CACHE_MAX_SIZE_MB", "5120")),
    "VLLM_MEDIA_CACHE_TTL_HOURS": lambda: float(os.getenv("VLLM_MEDIA_CACHE_TTL_HOURS", "24")),
    "VLLM_MEDIA_URL_ALLOW_REDIRECTS": lambda: os.getenv("VLLM_MEDIA_URL_ALLOW_REDIRECTS", "0").lower() in ("1", "true", "yes"),'''
src = src.replace(old2, new2, 1)
open(path, "w").write(src)
print("envs.py patched (backfilled VLLM_MEDIA_CACHE_* env vars)")
PY

# 3a. Expose FileUploadConfig from vllm.config (preserves the stock
#     import chain; FileUploadConfig's own module has no heavy imports).
RUN sed -i "s|from vllm.config.ec_transfer import ECTransferConfig|from vllm.config.ec_transfer import ECTransferConfig\nfrom vllm.config.file_upload import FileUploadConfig|" \
    ${VLLM_PKG}/config/__init__.py && \
    sed -i 's|    "ECTransferConfig",|    "ECTransferConfig",\n    "FileUploadConfig",|' \
    ${VLLM_PKG}/config/__init__.py

# 3b. Append file-upload CLI fields to FrontendArgs.
RUN python3 - <<'PY'
path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/cli_args.py"
src = open(path).read()
new_fields = '''    # ---- /v1/files upload endpoint (#38531) ----
    enable_file_uploads: bool = False
    """Enable the `/v1/files` OpenAI-compatible file upload endpoint."""
    file_upload_dir: str = ""
    """Directory for uploaded file bytes."""
    file_upload_ttl_seconds: int = 3600
    """TTL (seconds) measured from last access. -1 disables time expiry."""
    file_upload_max_size_mb: int = 512
    """Per-file size cap (MB)."""
    file_upload_max_total_gb: int = 5
    """Total on-disk quota (GB). Evicts oldest via LRU."""
    file_upload_max_concurrent: int = 4
    """Max in-flight POST /v1/files operations."""
    file_upload_scope_header: str = ""
    """Request header whose value scopes uploaded files."""
    file_upload_disable_listing: bool = False
    """When True, GET /v1/files returns 404."""

    @classmethod
    def _customize_cli_kwargs('''
src2 = src.replace("    @classmethod\n    def _customize_cli_kwargs(", new_fields, 1)
assert src2 != src, "pattern not found"
open(path, "w").write(src2)
print("cli_args.py patched")
PY

# 3c. Add the files router + state init to build_app and init_app_state.
#     NOTE: uses `await init_files_state(...)` (async post-review).
RUN python3 - <<'PY'
path = "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py"
src = open(path).read()

# build_app: attach router after transcription block
router_block = '''        register_speech_to_text_api_router(app)

    if getattr(args, "enable_file_uploads", False):
        from vllm.entrypoints.openai.files.api_router import (
            attach_router as register_files_api_router,
        )

        register_files_api_router(app)'''
src2 = src.replace("        register_speech_to_text_api_router(app)", router_block, 1)
assert src2 != src, "register_speech_to_text_api_router pattern not found"
src = src2

# init_app_state: init files state after transcription state (async now).
init_block = '''        init_transcription_state(
            engine_client, state, args, request_logger, supported_tasks
        )

    if args.enable_file_uploads:
        from vllm.entrypoints.openai.files.api_router import init_files_state

        await init_files_state(state, args)'''
src2 = src.replace(
    "        init_transcription_state(\n            engine_client, state, args, request_logger, supported_tasks\n        )",
    init_block, 1
)
assert src2 != src, "init_transcription_state pattern not found"
open(path, "w").write(src2)
print("api_server.py patched (async init_files_state)")
PY

# Drop stale .pyc caches so our new .py files get picked up.
RUN find ${VLLM_PKG} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Sanity check: all patched files parse + key identifiers present.
RUN python3 -c "\
import ast; \
ast.parse(open('${VLLM_PKG}/config/file_upload.py').read()); \
ast.parse(open('${VLLM_PKG}/config/__init__.py').read()); \
ast.parse(open('${VLLM_PKG}/entrypoints/openai/cli_args.py').read()); \
ast.parse(open('${VLLM_PKG}/entrypoints/openai/api_server.py').read()); \
ast.parse(open('${VLLM_PKG}/entrypoints/openai/files/store.py').read()); \
ast.parse(open('${VLLM_PKG}/entrypoints/openai/files/serving.py').read()); \
ast.parse(open('${VLLM_PKG}/entrypoints/openai/files/api_router.py').read()); \
ast.parse(open('${VLLM_PKG}/entrypoints/openai/files/mime.py').read()); \
ast.parse(open('${VLLM_PKG}/multimodal/media/connector.py').read()); \
assert 'vllm-file' in open('${VLLM_PKG}/multimodal/media/connector.py').read(); \
assert '_VLLM_FILE_ID_RE' in open('${VLLM_PKG}/multimodal/media/connector.py').read(), 'post-review regex missing'; \
assert 'await init_files_state' in open('${VLLM_PKG}/entrypoints/openai/api_server.py').read(), 'async wiring missing'; \
print('✅ all patched files parse OK + post-review markers present')"
