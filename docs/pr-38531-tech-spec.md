# Tech Spec — vLLM Issue #38531: Direct File Upload for Multimodal API

**Issue**: [vllm-project/vllm#38531](https://github.com/vllm-project/vllm/issues/38531)
**Author (this spec)**: @Alberto-Codes
**Date**: 2026-04-04
**Status**: Draft / pre-PR

## 1. Problem

The `/v1/chat/completions` endpoint accepts multimodal content (video/image/audio) via exactly three URL schemes today: `http(s)://`, `data:...;base64,...`, and `file://` (gated by `--allowed-local-media-path`). For a client with a local file this forces one of three bad options:

| Option | Failure mode |
|---|---|
| Base64 data URL | 18 MB video → ~24 MB payload → `E2BIG` on shell, ~33% wire overhead, JSON parsing blows the roof on large videos |
| `file://` | Requires `--allowed-local-media-path` server flag AND the file to live on the **server** machine — useless when client and server are on different hosts |
| Stand-up an HTTP server | User has to run a second process just to hand a file to vLLM |

This is felt most acutely for **video** workloads (tens to hundreds of MB per clip), which is the motivating workload for issue #38531.

### What maintainers asked (@DarkLight1337, comment)

> "how do you propose to manage the lifecycle of the uploaded files? We also need to be careful of security risks of letting users upload files to the server."

The spec answers both explicitly in §4 and §5.

## 2. Design overview

**Add a new `/v1/files` endpoint + a `vllm-file://<id>` URL scheme, both gated behind an opt-in server flag.** This is the OpenAI Files API pattern, giving vLLM natural lifecycle semantics (create/list/retrieve/delete) and reusing the existing `MediaConnector` URL-dispatch mechanism via `MEDIA_CONNECTOR_REGISTRY`.

```bash
# 1. upload once
$ curl -X POST http://host:8000/v1/files \
    -F "file=@local/video.mp4" \
    -F "purpose=vision"
{"id": "file-abc123...", "object": "file", "bytes": 18432100,
 "created_at": 1743800000, "expires_at": 1743803600,
 "filename": "video.mp4", "purpose": "vision", "status": "processed"}

# 2. reference in any number of chat-completion calls
{"type":"video_url","video_url":{"url":"vllm-file://file-abc123..."}}
```

### Design philosophy

The upload store is a **capability-based file store**, not a tenancy system. File IDs are 128-bit unguessable capability handles — possession of the ID implies authority to access. All additional controls (scope header, disable-listing, TTL, quota) are **defense-in-depth knobs** that compose independently. vLLM remains identity-agnostic; per-caller data partitioning is an operator concern expressed via a configurable request header the operator's gateway already populates.

### Why not just multipart in `/v1/chat/completions`?

Considered and rejected for v1:
- `/v1/chat/completions` currently has `Depends(validate_json_request)` which rejects multipart — changing that guard is intrusive and changes the error-message contract for every misuse of the endpoint
- Multipart-in-chat-completions forces the user to re-upload on every request (no reuse across turns in a video-chat session)
- The OpenAI Files API is the established ecosystem pattern; SGLang does NOT have this endpoint today, so implementing it is *additive* value, not parity
- Future work can still add it as a convenience wrapper once `/v1/files` is landed

### Why not SGLang's "accept any string ending in .mp4 as a local path"?

SGLang's `load_image` (`python/sglang/srt/utils/common.py`) treats any string ending in a known extension as a server-side file path — no opt-in flag. This is exactly the attack surface vLLM's `--allowed-local-media-path` was designed to close. Not porting that pattern.

## 3. API contract

All endpoints return `404` unless the server is launched with `--enable-file-uploads`.

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/v1/files` | `multipart/form-data` (`file`, `purpose`) | `File` object with `id`, `bytes`, `expires_at` |
| `GET` | `/v1/files` | — | `{"data": [File, ...], "object": "list"}` (disabled when `--file-upload-disable-listing`) |
| `GET` | `/v1/files/{file_id}` | — | `File` object |
| `GET` | `/v1/files/{file_id}/content` | — | raw bytes (`application/octet-stream`, streamed) |
| `DELETE` | `/v1/files/{file_id}` | — | `{"id": ..., "deleted": true}` |

Response schema (OpenAI-compatible subset):

```json
{
  "id": "file-<32hex>",
  "object": "file",
  "bytes": 18432100,
  "created_at": 1743800000,
  "expires_at": 1743803600,
  "filename": "video.mp4",
  "purpose": "vision",
  "status": "processed"
}
```

Supported `purpose` values for v1: `"vision"` and `"user_data"`. Rejects unknown values with `400` and a message listing allowed values.

The `vllm-file://<id>` scheme is resolved by a new `MediaConnector` handler registered via `@MEDIA_CONNECTOR_REGISTRY.register("vllm-file")` — it reads the upload store, streams bytes into the existing `media_io.load_bytes` path, and integrates with the existing `VLLM_MEDIA_CACHE` if present.

## 4. Lifecycle (answer to @DarkLight1337)

1. **Default TTL**: 1 hour after last access. Configurable via `--file-upload-ttl-seconds` (default `3600`) or `VLLM_FILE_UPLOAD_TTL_SECONDS`. Set to `-1` to disable time-based expiry (quota eviction still applies; `expires_at` omitted from responses).
2. **Auto-eviction**: a background task scans the upload dir every 5 minutes; removes files where `now - atime > TTL`.
3. **Total-size quota**: `--file-upload-max-total-gb` (default `5 GB`). When uploads would exceed quota, evict oldest (LRU) to make room; reject if single upload alone exceeds quota.
4. **Per-file size cap**: `--file-upload-max-size-mb` (default `512 MB`).
5. **Explicit `DELETE`**: client can always free a file before TTL expiry.
6. **Touched on use**: every `vllm-file://` resolution in a chat-completion request updates `atime`, so active files in a conversation don't expire mid-session.
7. **Non-persistent across restarts**: upload dir is cleared on server startup. This bounds the attack surface and avoids orphaned files from previous runs of a different operator.

### 4.1 Deployment patterns

All patterns are equivalent — vLLM accepts any configurable request header as the scope key, so operators pick whichever their gateway already populates. Leave `--file-upload-scope-header` unset for server-global behavior (capability handle is the only access control).

**Pattern A — Direct OpenAI SDK client (no gateway)**

```bash
# server
vllm serve my-model --enable-file-uploads \
  --file-upload-scope-header OpenAI-Project

# client (openai-python, or any SDK wrapping OpenAI-compat)
from openai import OpenAI
client = OpenAI(api_key="<key>", base_url="http://host:8000/v1",
                project="team-alpha")
# SDK automatically sets `OpenAI-Project: team-alpha` on every request
file = client.files.create(file=open("video.mp4","rb"), purpose="vision")
```

**Pattern B — Apigee gateway (AssignMessage policy)**

```xml
<!-- inject consumer identifier into OpenAI-Project header -->
<AssignMessage name="Add-OpenAI-Project">
  <AssignTo createNew="false" type="request"/>
  <Set>
    <Headers>
      <Header name="OpenAI-Project">{apiproduct.developer.app.name}</Header>
    </Headers>
  </Set>
</AssignMessage>
```

Server: `vllm serve ... --file-upload-scope-header OpenAI-Project`

**Pattern C — Kong gateway (native consumer propagation)**

```yaml
# kong.yaml — Kong injects X-Consumer-ID on authenticated routes
services:
  - name: vllm
    url: http://vllm-backend:8000
    plugins:
      - name: key-auth   # Kong populates X-Consumer-ID post-auth
```

Server: `vllm serve ... --file-upload-scope-header X-Consumer-ID`

**Pattern D — Envoy / oauth2-proxy (JWT `sub` claim)**

```yaml
# oauth2-proxy extracts JWT `sub` claim into X-Auth-Request-User;
# Envoy ext_authz forwards the header to the upstream vLLM service.
```

Server: `vllm serve ... --file-upload-scope-header X-Auth-Request-User`

**When `--file-upload-scope-header` is set but the header is missing from a request → `400 Bad Request` with message `"Scope header '<NAME>' required"`**. Fail loud on misconfiguration; no silent fallback to global scope.

## 5. Security (answer to @DarkLight1337)

1. **Off by default**: `--enable-file-uploads` must be explicitly set. Default build behaves identically to today.
2. **128-bit capability handles**: file IDs are `file-<32hex>` (128 bits of entropy from `os.urandom`). Possession of the ID implies authority. No enumeration via brute force.
3. **Enumeration control**: `--file-upload-disable-listing` removes `GET /v1/files` entirely (returns `404`). Individual file operations still work via known ID. Compliance-oriented deployments enable this by default.
4. **Per-request scope header** (§4.1): when `--file-upload-scope-header <NAME>` is set, files are tagged with the header value at upload and filtered on every subsequent operation. Scope mismatch returns `404` (not `403`) — capability non-disclosure pattern.
5. **Path confinement**: files stored under a server-chosen directory (`--file-upload-dir`, default `$TMPDIR/vllm-uploads-<pid>`). On-disk filenames are `sha256(random)` — the client `filename` field is stored only in metadata, never touches any filesystem path. Zero path-traversal surface.
6. **Client filename sanitization**: on upload, the `filename` metadata field is stripped of path components (`os.path.basename`), truncated to 255 chars, and control characters are removed. Prevents PII exfiltration via filename echoing in `LIST` responses and prevents terminal-injection from malicious clients.
7. **MIME validation**: **inline magic-byte check** against an allowlist of supported formats (mp4, webm/matroska, png, jpeg, gif, webp, wav, mp3) — no new required dependencies. Optional upgrade path: if `python-magic` is installed, the sniffer uses it for comprehensive detection. Rejects uploads whose sniffed type isn't in the allowlist. Client-declared `Content-Type` is advisory only.
8. **Streaming size caps**: we do NOT read the entire upload into memory. Stream chunks to a temp file and abort + unlink if we exceed `--file-upload-max-size-mb`.
9. **No execution semantics**: files are opaque bytes. No shell-out, no `eval`, no server-side decompression (no zip/gzip).
10. **Concurrent-upload limit**: `--file-upload-max-concurrent` (default `4`) caps simultaneous POST operations to prevent disk-fill via concurrent max-size uploads.
11. **Structured audit log**: every file operation emits one INFO-level JSON line:

    ```json
    {
      "op": "file.upload",
      "file_id": "file-<32hex>",
      "bytes": 18432100,
      "scope": "<header value or null>",
      "client_host": "<remote ip>",
      "request_id": "<string or null>",
      "ts": "2026-04-04T12:34:56Z"
    }
    ```

    Valid `op` values: `file.upload`, `file.retrieve`, `file.delete`, `file.list`, `file.reject`. On rejection, the log line includes a `reason` field (`size_exceeded`, `mime_mismatch`, `quota_full`, `scope_mismatch`, `scope_missing`, `invalid_purpose`).

## 6. Implementation sketch

### 6.1 New files

- `vllm/config/file_upload.py` — new `FileUploadConfig` `@config` dataclass (matches vLLM's subsystem pattern: `CacheConfig`, `LoRAConfig`). Fields: `enabled`, `dir`, `ttl_seconds`, `max_size_mb`, `max_total_gb`, `max_concurrent`, `scope_header`, `disable_listing`. CLI flags auto-generated from field names via `BaseFrontendArgs.add_cli_args` pattern.
- `vllm/entrypoints/openai/files/__init__.py`
- `vllm/entrypoints/openai/files/protocol.py` — `FileObject`, `FileList`, `FileDeleteResponse` Pydantic models
- `vllm/entrypoints/openai/files/store.py` — `FileUploadStore` (file-backed, LRU eviction, TTL sweeper task, audit logging)
- `vllm/entrypoints/openai/files/api_router.py` — FastAPI router with POST/GET/DELETE handlers
- `vllm/entrypoints/openai/files/serving.py` — business logic (scope header extraction, validation, MIME sniff, streaming write, filename sanitization)
- `vllm/entrypoints/openai/files/mime.py` — inline magic-byte sniffing with optional `python-magic` fallback when installed

### 6.2 Modified files

- `vllm/entrypoints/openai/api_server.py` — wire up new router when `FileUploadConfig.enabled` is True
- `vllm/entrypoints/openai/cli_args.py` — compose `FileUploadConfig` into `FrontendArgs`; flags auto-register as `--file-upload-*` (e.g. `--file-upload-ttl-seconds`, `--file-upload-scope-header`)
- `vllm/envs.py` — matching `VLLM_FILE_UPLOAD_*` env-var equivalents
- `vllm/multimodal/media/connector.py` — add `_load_vllm_file_url` branch in `load_from_url`/`load_from_url_async`; register the scheme handler
- `docs/features/multimodal_inputs.md` — user guide (new "Uploading local media files" subsection)
- `docs/serving/openai_compatible_server.md` — endpoint + flag reference

### 6.3 Vocabulary note

The codebase uses the noun **`scope`** throughout — never `project`, `tenant`, `user`, `organization`, or `owner`. This matches vLLM's existing architectural stance: multi-tenancy is an **operator concern** expressed at the gateway, not a primitive inside vLLM. Reference points in the current vLLM tree:

- `vllm/config/cache.py:87` — explicit warning that vLLM acknowledges "multi-tenant environments" exist but delegates risk management to the operator
- `vllm/distributed/utils.py:582` — the only other use of `multi_tenant`, an internal torch distributed `Store` flag (not API-layer tenancy)

This PR is the first user-facing per-caller data partition in vLLM. The neutral `scope` vocabulary keeps the code open to any partition dimension (project, team, service, use-case) the operator chooses, without vLLM taking a position on identity semantics.

### 6.4 Wire-up points

1. **App builder** (`build_app` in `api_server.py`, around line 230): add a new block
   ```python
   if args.enable_file_uploads:
       from vllm.entrypoints.openai.files.api_router import attach_router as attach_files_router
       attach_files_router(app)
   ```
2. **MediaConnector** (`load_from_url` in `media/connector.py`, around line 312): new branch
   ```python
   if url_spec.scheme == "vllm-file":
       return self._load_vllm_file_url(url_spec, media_io)
   ```
   The store singleton is exposed via `app.state` and passed into the connector at construction time in `chat_utils.py` (next to `allowed_local_media_path`).

## 7. Test plan

### Unit tests (`tests/entrypoints/openai/files/`)

- `test_store.py`: TTL eviction, LRU eviction under quota pressure, touch-on-read, path confinement (reject `../`), sha256 filename generation, concurrent write limiting
- `test_protocol.py`: `FileObject` serialization matches OpenAI shape
- `test_mime_validation.py`: magic-byte sniffing rejects executables masquerading as `video/mp4`, accepts real mp4/webm/png/jpeg/wav
- `test_filename_sanitization_drops_path_components`: input `"../etc/passwd/video.mp4"` → stored metadata `"video.mp4"`
- `test_filename_sanitization_caps_length`: 500-char filename → truncated to 255
- `test_file_ids_are_128_bit_random`: sample 10k IDs, verify uniform distribution, verify `len == 37` (`"file-" + 32 hex`), verify no collisions

### Integration tests (`tests/entrypoints/openai/test_files_api.py`)

- Flag off by default → `POST /v1/files` returns 404
- Flag on → upload → list → retrieve → delete round-trip
- Upload exceeds `max-size-mb` → 413 (streaming, so memory doesn't spike)
- Upload with wrong magic bytes → 400
- Upload with `purpose="assistants"` or unknown → 400 listing allowed values
- Concurrent upload limit → 503 after N simultaneous
- TTL expiry → GET after sleep returns 404
- `DELETE` removes both metadata and on-disk bytes
- `test_disable_listing_returns_404_on_GET_v1_files`: server with `--file-upload-disable-listing` → `GET /v1/files` returns 404; individual file ops still work
- `test_scope_header_mismatch_returns_404_not_403`: upload with scope `"alpha"`, retrieve with scope `"bravo"` → 404 (capability non-disclosure)
- `test_scope_header_missing_returns_400`: `--file-upload-scope-header X-Foo` set, request without `X-Foo` → 400
- `test_audit_log_emits_on_every_operation_with_required_fields`: every upload/retrieve/delete/reject emits one INFO JSON line with `op`, `file_id`, `scope`, `client_host`, `request_id`, `ts` present

### End-to-end (`tests/entrypoints/openai/chat_completion/test_video.py` extension)

- Start server with `--enable-file-uploads`
- Upload a test video fixture
- Issue chat-completion with `vllm-file://file-...` reference
- Assert the referenced file is retrievable by the multimodal processor and frames are fed to the model

### Security regression tests

- Flag off → GET /v1/files returns 404 (not 401, not 500)
- Path traversal in `filename` field → filename is stored in metadata but never used for filesystem writes (verified by mock fs and assertion)
- Oversize upload → file handle closed, temp file unlinked, no disk leak

## 8. Design decisions

Consensus positions reached during internal design review. Listed here so maintainers can concur or challenge.

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Scoping model | Server-global by default. Opt-in `--file-upload-scope-header <NAME>` for gateway-fronted deployments. Stacking knobs: 128-bit capability IDs (always), `--file-upload-disable-listing`, structured audit log. | vLLM has no user/tenant primitives in its entrypoints today (only LoRA adapter multiplexing). Matches vLLM's existing operator-owned multi-tenancy stance (`config/cache.py:87`). Header-based scoping lets operators plug in any gateway convention without vLLM taking a position on identity. |
| 2 | `purpose` enum | `{"vision", "user_data"}`. Reject others with 400 listing allowed values. | `assistants`, `fine-tune`, `batch` are specific to OpenAI's hosted services and meaningless in a local inference server. Silent aliasing hides behavior — explicit rejection is safer. |
| 3 | `expires_at` when TTL disabled | Default TTL = 3600s. Disable via `--file-upload-ttl-seconds=-1` (sentinel, not 0). When disabled, omit `expires_at` from responses and skip time-based sweeper. Quota-based LRU eviction remains. | Matches OpenAI convention (field omitted when no expiry). Sentinel `-1` is unambiguous; `0` could mean "expire immediately". Quota eviction prevents disk-fill DoS in long-running servers. |
| 4 | Streaming download | Always stream via `StreamingResponse` + async 64KB chunks. No size-based branching. | One code path, easier to test, memory-safe at any file size. FastAPI/Starlette handle both small and large files efficiently. |
| 5 | Docs location | Both: user guide in `docs/features/multimodal_inputs.md` ("Uploading local media files" subsection); endpoint/flag/security reference in `docs/serving/openai_compatible_server.md`. Cross-linked. | Guide where users search ("how do I send a local video"), reference where maintainers review. Separation-of-concerns. |
| 6 | MIME validation dependency | Inline magic-byte sniffer for the allowlist (mp4, webm/matroska, png, jpeg, gif, webp, wav, mp3). Optional `python-magic` used if installed. No new required dep. | Avoids maintainer friction of adding a dep to a minimal-surface feature. Inline covers the "reject binary-as-media" attack vector. Power users can `pip install python-magic` for comprehensive detection. |
| 7 | CLI flag implementation pattern | New `FileUploadConfig` `@config` dataclass, composed into `FrontendArgs`. Flags auto-generated from typed fields. | Matches vLLM's subsystem-config pattern (`CacheConfig`, `LoRAConfig`). Cleaner test fixtures, clearer deprecation path, better architectural boundary than bolting 8 flags onto `FrontendArgs` directly. |

### Items where maintainer input would be welcome

- **Default TTL value**: 3600s feels right for video-chat sessions but may be long for high-churn deployments. Happy to change the default.
- **Header allowlist in OpenAPI schema**: the `--file-upload-scope-header` value is operator-defined and therefore cannot be statically documented in the FastAPI OpenAPI schema. Open to suggestions if there's a preferred pattern.
- **`purpose` value alignment with future OpenAI changes**: OpenAI may add/remove values over time. Proposal: keep a conservative local allowlist, update as the ecosystem does.

## 9. Release plan

- **v1 (this PR)**: `/v1/files` CRUD + `vllm-file://` scheme + opt-in flag + all scope/security knobs + docs
- **v2 (follow-up PR)**: multipart-in-`/v1/chat/completions` convenience wrapper for one-shot clients — only if there's demand after v1 lands
- **v3 (speculative)**: additional scope dimensions (e.g., multiple headers, composite keys) if gateway-fronted deployments need it

### 9.1 PR body distillation

This tech-spec stays in `turboquant-vllm` as internal planning. The upstream PR body (in `vllm-project/vllm`) is a ~80-120 line distillation matching vLLM's PR template:

1. **Purpose** — problem paragraph + link to issue #38531
2. **Approach** — 4-5 bullets summarizing `/v1/files` + `vllm-file://` + opt-in flag
3. **Design decisions** — inline the §8 decisions table (7 rows, brief rationales)
4. **Security posture** — defense-in-depth controls bulleted
5. **Deployment patterns** — inline the §4.1 four recipes
6. **Test Plan** — list of new test files + count, not every test ID
7. **Test Result** — filled in after CI

The full spec is not published externally. Reviewers asking for more detail get specific sections pasted inline on request.

## 10. References

- Issue: [vllm-project/vllm#38531](https://github.com/vllm-project/vllm/issues/38531)
- OpenAI Files API reference: https://platform.openai.com/docs/api-reference/files
- Existing vLLM multipart pattern: `vllm/entrypoints/openai/speech_to_text/api_router.py:52-82`
- Existing `MediaConnector` scheme dispatch: `vllm/multimodal/media/connector.py:286-368`
- Existing media-cache pattern we're mirroring for lifecycle: `vllm/multimodal/media/connector.py:127-234`
- vLLM's operator-owned multi-tenancy stance: `vllm/config/cache.py:87`
