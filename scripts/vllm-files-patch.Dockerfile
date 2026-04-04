# Thin layer over the stock vllm-openai image that adds the /v1/files
# endpoint from the feat/direct-multipart-video-upload branch
# (vllm-project/vllm#38531). Everything new is pure Python — no
# C++/CUDA rebuild.
#
# Strategy: drop in our new files verbatim, then use sed to patch the
# image's own cli_args.py and api_server.py to add our 8 CLI flags and
# the files-router wire-up. This avoids the fork-divergence cascade
# that would come from overlaying whole files.
FROM docker.io/vllm/vllm-openai:latest

ARG VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm

# New subsystem files — these have no dependencies on the rest of our
# fork, so they drop in clean.
COPY vllm/config/file_upload.py ${VLLM_PKG}/config/file_upload.py
COPY vllm/entrypoints/openai/files/ ${VLLM_PKG}/entrypoints/openai/files/

# Expose FileUploadConfig from vllm.config
RUN sed -i "s|from vllm.config.ec_transfer import ECTransferConfig|from vllm.config.ec_transfer import ECTransferConfig\nfrom vllm.config.file_upload import FileUploadConfig|" \
    ${VLLM_PKG}/config/__init__.py && \
    sed -i 's|    "ECTransferConfig",|    "ECTransferConfig",\n    "FileUploadConfig",|' \
    ${VLLM_PKG}/config/__init__.py

# Append file-upload CLI fields to FrontendArgs. We insert BEFORE the
# _customize_cli_kwargs classmethod, detected by its signature.
RUN python3 - <<'PY'
import re
path = "${VLLM_PKG}/entrypoints/openai/cli_args.py".replace("${VLLM_PKG}", "/usr/local/lib/python3.12/dist-packages/vllm")
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

# Add the files router + state init to build_app and init_app_state.
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
src = src.replace("        register_speech_to_text_api_router(app)", router_block, 1)

# init_app_state: init files state after transcription state
init_block = '''        init_transcription_state(
            engine_client, state, args, request_logger, supported_tasks
        )

    if getattr(args, "enable_file_uploads", False):
        from vllm.entrypoints.openai.files.api_router import init_files_state

        init_files_state(state, args)'''
src = src.replace(
    "        init_transcription_state(\n            engine_client, state, args, request_logger, supported_tasks\n        )",
    init_block, 1
)
open(path, "w").write(src)
print("api_server.py patched")
PY

# Patch MediaConnector to resolve vllm-file://<id> URLs. Surgical sed
# rather than overlaying the whole file, because the image's connector
# is slightly older than our fork's (doesn't have VLLM_MEDIA_CACHE).
RUN python3 - <<'PY'
path = "/usr/local/lib/python3.12/dist-packages/vllm/multimodal/media/connector.py"
src = open(path).read()

# Add scheme branch to both load_from_url methods.
sync_branch = '''        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)

        if url_spec.scheme == "vllm-file":
            return self._load_vllm_file_url(url_spec, media_io)
'''
async_branch = '''        if url_spec.scheme == "file":
            future = loop.run_in_executor(
                global_thread_pool, self._load_file_url, url_spec, media_io
            )
            return await future
        if url_spec.scheme == "vllm-file":
            future = loop.run_in_executor(
                global_thread_pool, self._load_vllm_file_url, url_spec, media_io
            )
            return await future
'''
src = src.replace(
    '''        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)
''', sync_branch, 1)
src = src.replace(
    '''        if url_spec.scheme == "file":
            future = loop.run_in_executor(
                global_thread_pool, self._load_file_url, url_spec, media_io
            )
            return await future
''', async_branch, 1)

# Inject the resolver method BEFORE _assert_url_in_allowed_media_domains.
method = '''    def _load_vllm_file_url(
        self,
        url_spec,
        media_io,
    ):
        """Resolve vllm-file://<id> via the process-wide upload store."""
        from vllm.entrypoints.openai.files.store import get_store

        store = get_store()
        if store is None:
            raise RuntimeError(
                "vllm-file:// URLs require --enable-file-uploads on the server."
            )

        file_id = url_spec.netloc or ""
        if not file_id and url_spec.path:
            file_id = url_spec.path.lstrip("/")

        data = store.read_bytes_by_id(file_id)
        if data is None:
            raise ValueError(f"Unknown vllm-file id: {file_id!r}")
        return media_io.load_bytes(data)

    def _assert_url_in_allowed_media_domains'''
src = src.replace("    def _assert_url_in_allowed_media_domains", method, 1)

open(path, "w").write(src)
print("connector.py patched")
PY

# Drop stale .pyc caches
RUN find ${VLLM_PKG} -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Sanity check: parse all patched files + imports work standalone
RUN python3 -c "import ast; \
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
    print('all patched files parse OK')"
