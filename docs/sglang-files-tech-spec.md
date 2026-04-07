# Tech Spec — SGLang `/v1/files` upload endpoint

**Target repo**: [sgl-project/sglang](https://github.com/sgl-project/sglang)
**Branch**: `feat/direct-multipart-video-upload` on `Alberto-Codes/sglang`
**Status**: ready for draft PR
**Reference implementation**: vllm-project/vllm#39003 (ours; not cited in SGLang PR per cross-ecosystem etiquette)

## 1. Problem

SGLang's chat-completion multimodal content (video_url / image_url / audio_url) accepts three URL shapes today via `load_image` / `load_video` / `load_audio` in `sglang/srt/utils/common.py`:

- `http://`, `https://` (network fetch)
- `data:` (inline base64)
- `file://` (server-side filesystem path; requires the file to live on the server machine)

For a client with a **local** file (video on laptop → SGLang server on a different host), this forces one of:

| Option | Failure mode |
|---|---|
| Base64 data URL | 24 MB video → ~33% wire overhead, exceeds shell `ARG_MAX` for curl, blows JSON parsers |
| `file://` | File must live on the **server** machine, not useful cross-host |
| Stand up an HTTP server | Run an extra process just to hand a file to SGLang |

Video workloads (tens of MB per clip) feel this acutely.

## 2. Design

**Add `/v1/files` endpoint (off by default, opt-in via `--enable-file-uploads`) + `sgl-file://<id>` URL scheme.** OpenAI-compatible Files API shape, uploaded files referenced in chat completions via the new scheme in the existing `video_url` / `image_url` / `audio_url` content types.

### Vendor-specific scheme (deliberate)

Follows the `s3://` / `gs://` / `ipfs://` precedent: the scheme name IS the routing hint. `sgl-file://<id>` tells the reader "resolve via SGLang's upload store." The vLLM sibling uses `vllm-file://`, LMDeploy would use `lmdeploy-file://`, etc. **No cross-ecosystem scheme consolidation** — readability beats synthetic standardization.

### Capability-handle trust model

File IDs are 128-bit random handles (`file-<32 hex>`). Possession = authority. The `sgl-file://` resolver in `load_image`/`load_video`/`load_audio` is **scope-bypassing** — scope headers apply to `/v1/files` CRUD (to prevent enumeration via LIST), but NOT to multimodal resolution, because the chat-completion layer does not receive request headers in the current architecture. Every resolution emits a `file.resolve` audit log line.

## 3. Port deltas vs vLLM #39003

Same feature, adapted to SGLang's conventions:

| Area | vLLM | SGLang |
|---|---|---|
| Config class | `@config` pydantic dataclass | plain `@dataclass` + `__post_init__` validation |
| Router structure | Modular `api_router.py` + `attach_router(app)` | Inline handlers in monolithic `http_server.py` |
| Scheme name | `vllm-file://` | `sgl-file://` |
| URL dispatcher | `MediaConnector.load_from_url` with `_load_vllm_file_url` method | `_resolve_sgl_file_bytes` helper called from `load_image`/`load_video`/`load_audio` elif branches |
| CLI args pattern | Typed dataclass fields + auto-generated argparse | Plain dataclass fields + manual `add_argument()` calls |
| ErrorResponse shape | Nested (`ErrorInfo` inside `ErrorResponse`) | Flat (`message`/`type`/`code` on `ErrorResponse` directly) |
| Serving class inheritance | Standalone | Standalone (does NOT inherit `OpenAIServingBase` — file I/O doesn't need `tokenizer_manager`) |

Shared verbatim (~60% of the ported LOC):
- `store.py` — capability IDs, streaming uploads, TTL+LRU quota, audit log, marker-file safety, two-phase eviction, async read_bytes_by_id
- `mime.py` — magic-byte sniffer (8 formats) + optional `python-magic` fallback
- `protocol.py` — Pydantic response models (Literal-constrained `purpose`, `expires_at` omission)
- Test suite — 94 tests, direct port with import renames

## 4. Trust model decision

Documented in the SGLang docs page (`docs/basic_usage/openai_api_files.md`) as a **deliberate trade-off**:

> `sgl-file://<id>` URLs resolving through the multimodal loaders are
> capability-based: possession of the 128-bit file ID is the access
> control. Scope headers are enforced on the `/v1/files` CRUD endpoints
> (which could leak IDs via LIST) but **not** on multimodal resolution —
> the chat-completion layer does not receive request headers in the
> current architecture. Every resolution emits a `file.resolve` audit
> log line, so access remains traceable.

Same trade-off as the vLLM PR. If reviewers push back, the follow-up `--file-upload-strict-scope` flag threads headers through the chat path (+~15 LOC).

## 5. Deployment patterns

Matches vLLM port — four equal-weight gateway recipes (Direct SDK / Apigee / Kong / Envoy+oauth2-proxy). All use a configurable scope header (`OpenAI-Project`, `X-Consumer-ID`, `X-Auth-Request-User`).

See `docs/basic_usage/openai_api_files.md` for the full table.

## 6. Test plan

```bash
cd sglang
pytest test/registered/unit/entrypoints/openai/files/ -v
```

Coverage (94 tests, ~0.6s CPU):
- Config validation (`__post_init__`, extra-field rejection)
- MIME sniffer (8 formats + ELF/PE/HTML/script rejection, pymagic fallback)
- Store CRUD (128-bit IDs, path confinement, filename sanitization, LRU quota, TTL sweep)
- Scope non-disclosure (404 vs 403)
- Audit log schema + required fields
- Startup directory safety marker
- Concurrency limit → `ConcurrencyLimitExceeded`
- TOCTOU: stream_content eager open, read_bytes_by_id/async FileNotFoundError handling
- Async race: event-loop-only metadata access in `read_bytes_by_id_async`

**GPU end-to-end** (operator-runnable):

```bash
python -m sglang.launch_server \
    --model-path allenai/Molmo2-8B --trust-remote-code \
    --enable-file-uploads --file-upload-max-size-mb 128 --port 30000

python examples/runtime/openai_file_upload_client.py path/to/clip.mp4
```

Expected: 24 MiB video uploads in ~150 ms, chat completion returns a coherent video description. Same smoke test shape as our vLLM validation against Molmo2-8B.

## 7. Design decisions (ported from vLLM review consensus)

| # | Question | Decision |
|---|---|---|
| 1 | Scoping model | Server-global default, opt-in `--file-upload-scope-header`. 128-bit IDs always. `--file-upload-disable-listing` available. |
| 2 | `purpose` enum | `{"vision", "user_data"}`. Reject others with 400. |
| 3 | `expires_at` when TTL disabled | Default 3600s. `-1` disables. Omits `expires_at` from responses when disabled. |
| 4 | Streaming download | Always `StreamingResponse` + 64 KiB chunks. |
| 5 | MIME validation | Inline magic-byte + optional `python-magic`. No new required dep. |
| 6 | Config pattern | Plain `@dataclass` (vLLM used pydantic `@config`) matching SGLang's `ServerArgs` conventions. |
| 7 | URL scheme | `sgl-file://` — vendor-specific, follows `s3://`/`gs://` precedent. |

## 8. PR strategy

SGLang's contributing guide allows direct PRs (no issue-first requirement for features). Workflow:

1. **Push branch to `Alberto-Codes/sglang`**
2. **Squash → atomic commits** (SGLang convention: typically 1-2 commits for a feature of this size)
3. **Draft PR** following their template (Motivation / Modifications / Accuracy Tests / Speed Tests / Checklist)
4. **Don't cross-reference vLLM PR #39003** in the PR body (respect separate ecosystems)
5. **Await Merge Oncall engagement** — ping when CI is green + CODEOWNERS review requested

## 9. Files changed (SGLang fork)

```
python/sglang/srt/entrypoints/openai/files/
├── __init__.py
├── config.py                (FileUploadConfig + __post_init__)
├── protocol.py              (FileObject, FileList, FileDeleteResponse)
├── mime.py                  (magic-byte sniffer, 8 formats)
├── store.py                 (upload store with TTL/LRU/audit/marker)
└── serving_files.py         (OpenAIServingFiles handler)

python/sglang/srt/server_args.py           (+ 8 --file-upload-* flags)
python/sglang/srt/entrypoints/http_server.py  (+ 5 routes + state init)
python/sglang/srt/utils/common.py           (+ sgl-file:// in 3 loaders)

test/registered/unit/entrypoints/openai/files/
├── __init__.py
├── test_config.py
├── test_mime.py
├── test_protocol.py
└── test_store.py            (94 tests total)

examples/runtime/openai_file_upload_client.py    (runnable example)
docs/basic_usage/openai_api_files.md             (full reference)
docs/basic_usage/openai_api.rst                  (+ toctree entry)
```

**17 files, +2726 LOC vs `sgl-project/sglang:main`**.

## 10. References

- Reference implementation: vllm-project/vllm#39003 (our prior work)
- SGLang contributing guide: https://docs.sglang.io/developer_guide/contribution_guide.html
- SGLang PR template: `.github/pull_request_template.md` in repo
- Local branch: `Alberto-Codes/sglang:feat/direct-multipart-video-upload`
