## Purpose

Implements the `/v1/files` upload endpoint requested in **RFC issue
#38531**. Lets clients upload multimodal files once (via multipart
form) and reference them in subsequent chat completions through a new
`vllm-file://<id>` URL scheme — an alternative to base64 data URLs
(which inflate payloads and can exceed shell `ARG_MAX` for videos) and
to `file://` URLs (which require the file to already exist on the
server machine). The endpoint is **off by default**; operators
opt in with `--enable-file-uploads`.

Closes #38531.

### Answers to @DarkLight1337's questions

> **Lifecycle:** `--file-upload-ttl-seconds` (default 1h, atime-based
> expiry) + `--file-upload-max-total-gb` quota with LRU eviction
> (default 5 GB). Upload dir is cleared on server startup, so no state
> persists across restarts. Explicit `DELETE /v1/files/{id}` also
> works. Opportunistic sweeper runs inside `create_file` so no
> background task is needed.

> **Security:** Off-by-default + 128-bit capability handles
> (`file-<32 hex>`) + MIME magic-byte allowlist (video/image/audio
> only) + path confinement (sha256 on-disk names, client filename is
> metadata only, 255-byte cap, control chars stripped) + streaming
> size enforcement (no memory spikes) + structured JSON audit log +
> optional `--file-upload-scope-header` for gateway-fronted
> deployments + optional `--file-upload-disable-listing` to remove
> the enumeration surface.

### Design decisions (consensus during implementation)

| # | Question | Decision |
|---|---|---|
| 1 | Scoping model | Server-global by default. Opt-in `--file-upload-scope-header` for gateway-fronted deployments. Scope mismatches return 404, not 403 (capability non-disclosure). |
| 2 | `purpose` enum | `{"vision", "user_data"}`. Rejects OpenAI-specific values (`assistants`, `batch`, `fine-tune`) with 400. |
| 3 | `expires_at` | Default TTL 3600s. `--file-upload-ttl-seconds=-1` disables time-based expiry; `expires_at` is omitted from responses in that mode. Quota LRU still applies. |
| 4 | Streaming download | Always `StreamingResponse` + 64 KiB chunks. No size-based branching. |
| 5 | MIME validation | Inline magic-byte sniffer (no new required dependency). Optionally uses `python-magic` if installed for broader detection. |
| 6 | Config pattern | New `FileUploadConfig` `@config` dataclass, matching vLLM's `CacheConfig`/`LoRAConfig` subsystem pattern. |

### Trust model for `vllm-file://` resolution

`vllm-file://<id>` URLs resolving through `MediaConnector` are
**capability-based**: possession of the 128-bit file ID is the access
control. Scope headers are enforced on the `/v1/files` CRUD endpoints
(which could leak IDs via LIST) but **not** on multimodal resolution —
the chat-completion layer does not receive request headers in the
current architecture. Every resolution emits a `file.resolve` audit
log line, so access remains traceable.

If stricter semantics are desired for a specific deployment, a
follow-up `--file-upload-strict-scope` flag can thread the header
through `MediaConnector` (requires ~15 lines of changes to
`chat_utils.py`).

### Deployment patterns

Works out of the box with the existing OpenAI SDK (`project=` maps to
`OpenAI-Project` header) and standard gateways:

| Deployment | `--file-upload-scope-header` | Notes |
| --- | --- | --- |
| Direct OpenAI SDK client | `OpenAI-Project` | SDK auto-sends from `OPENAI_PROJECT_ID` env or `project=...` client param |
| Apigee (AssignMessage) | `OpenAI-Project` or `X-Consumer-ID` | One-line policy |
| Kong | `X-Consumer-ID` | Native on authenticated routes |
| Envoy / oauth2-proxy | `X-Auth-Request-User` | JWT `sub` claim |

Docs added in this PR:

- `docs/features/multimodal_inputs.md` — new **"Uploading Local Media
  Files"** subsection under Online Serving Video Inputs, with OpenAI
  SDK + curl examples.
- `docs/serving/openai_compatible_server.md` — new **"Files API"**
  section with the endpoint table, all `--file-upload-*` flags with
  defaults, the four gateway deployment patterns, and the full
  security posture.
- `examples/online_serving/openai_file_upload_client.py` — minimal
  runnable example (upload → reference via `vllm-file://<id>` →
  chat completion → delete).

---

## Test Plan

**118 unit + integration tests** across 6 files under
`tests/entrypoints/openai/files/`:

```bash
pytest tests/entrypoints/openai/files/ -v
```

Covers:

- `FileUploadConfig` validation (defaults, size constraints, extra-field rejection)
- MIME magic-byte sniffer (8 supported formats + ELF/PE/HTML/script rejection, pymagic fallback path)
- Store behaviour: streaming upload, 128-bit IDs, path confinement, filename sanitisation, LRU quota eviction, TTL sweep, scope non-disclosure, audit-log schema
- Pydantic protocol models (purpose allowlist, `expires_at` omission)
- Router via FastAPI `TestClient`: full round-trip, scope header enforcement (missing/present/mismatch), `disable_listing`, purpose validation, X-Request-Id propagation, feature-off 404 contract, **concurrency-limit → 503**
- `MediaConnector` `vllm-file://` scheme dispatch (sync + async, scope bypass, unknown-id/unregistered-store errors, atime touching)
- **Concurrency + TOCTOU races**: fail-fast semaphore rejection, concurrent-eviction handling in `read_bytes_by_id`/`read_bytes_by_id_async`, eager open in `stream_content`, eviction-vs-reads invariant, startup refusal to wipe unmarked user directory

**GPU end-to-end** (operator-runnable via the example client added in
this PR):

```bash
# Launch server with the feature enabled
vllm serve allenai/Molmo2-8B --trust-remote-code --max-model-len 6144 \
    --enable-file-uploads --file-upload-max-size-mb 128

# Upload + chat-completion round-trip against a real video
python examples/online_serving/openai_file_upload_client.py path/to/clip.mp4
```

---

## Test Result

**All 118 tests pass** (CPU-only, ~25s on a developer laptop):

```
118 passed, 2 warnings in 28.40s
```

**GPU end-to-end** on RTX 4090 (24 GB) with Molmo2-8B:

```
POST /v1/files (24.5 MiB Seinfeld clip, multipart form)  → 200  149 ms
POST /v1/chat/completions  (video_url=vllm-file://...)   → 200  7421 ms
    2784 prompt tokens, 87 completion tokens
    Model output: "This scene depicts a group of friends walking
    together on a busy city street. The setting is a typical urban
    environment with storefronts, parked cars, and pedestrians going
    about their day. The characters are engaged in conversation as
    they stroll along the sidewalk, with one woman gesturing
    animatedly while the others listen and respond..."
DELETE /v1/files/{id}  → 200  deleted:true
```

This is the exact failure case from #38531 (24 MB video → base64 →
`ARG_MAX` exceeded), now a 149 ms multipart upload.

---

## AI-Assisted Contribution Disclosure

This PR was developed with assistance from **Claude (Anthropic)** per
vLLM's [AI Assisted Contributions guide](https://docs.vllm.ai/en/latest/contributing/#ai-assisted-contributions).
The human author reviewed all code changes, ran all 118 tests locally,
and validated the GPU integration end-to-end against Molmo2-8B on a
24 MiB video clip. Commits carry `Co-authored-by: Claude` trailers as
documented in the guide.

---

<details>
<summary>Essential Elements of an Effective PR Description Checklist</summary>

- [x] The purpose of the PR, and link to existing issue (#38531)
- [x] The test plan, including the pytest command
- [x] The test results (all 118 tests passing + GPU end-to-end output)
- [x] Documentation update (user guide + API reference + example client)
- [ ] Release notes update (will add once this PR is close to merge)
</details>
