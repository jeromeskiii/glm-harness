# Changelog

All notable changes to the GLM harness ship in this file. Versions
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- `GitHubProvider.list_pull_request_files` dropped every PR file that carried
  no (or a small) `patch` field — the append sat inside the oversized-patch
  branch. Binary adds, renames, and small-diff files now appear in the tool
  result, with oversized patches still truncated.
- `record_feedback` no longer trips the stdlib logger's reserved `message`
  LogRecord key; the message now rides in the log line itself. Listeners
  registered via `on_feedback` actually fire (previously the blanket
  exception handler swallowed the `KeyError` and telemetry was dead).
- Protocol server: the per-request `sessions` service slot is restored in
  `finally` so a later request can never observe a stale session, and the
  `state_dir` guard coerces to `Path` so a string-constructed config cannot
  crash `session/import` with `AttributeError`.

### Added

- Telemetry feedback is wired into the runtime: turn failures emit a `bug`
  event and tool handler failures emit a `gap` event through
  `record_feedback`. The module is exported from the package root
  (`FEEDBACK_KINDS`, `on_feedback`, `record_feedback`) and covered by
  `tests/test_telemetry_feedback.py`.
- TypeScript entry point: `src/index.ts` (Node >= 22, zero dependencies, ESM,
  no build step) mirrors the CLI initialization sequence — parse argv, merge
  config, validate, resolve runtime, build child argv, dispatch — and
  delegates to the resolved runtime with inherited stdio, so the stream
  contract and exit codes (0/2/3/4/130) are unchanged. `--trace-init` /
  `GLMH_TRACE_INIT=1` trace the sequence on stderr; `GLMH_HARNESS_BIN` and
  `GLMH_PYTHON` pin the runtime. Documented in `README.md` and
  `README-HARNESS.md`.
- Reliability hardening: `write_file` / `edit_file` write atomically (temp
  file + fsync + `os.replace`) and reject payloads above 10 MiB; the
  protocol server bounds its in-memory session cache via `GLMH_MAX_SESSIONS`
  (default 256, oldest non-default evicted with a warning) and runs
  `agent/send` under a whole-turn deadline `GLMH_TURN_TIMEOUT_S` (default
  3600 s; 0 disables) that closes the turn with a durable `TURN_TIMEOUT`
  marker. Covered by `tests/test_hardening.py`.
- Identity strings are single-sourced in `identity.brand`: `HARNESS_AUTHOR`,
  `HARNESS_REPO`, `HARNESS_HF`, and `MODEL_VENDOR` now feed the CLI help,
  doctor output, the `fetch_url` user agent, the JSON-RPC
  `runtimeInfo.vendor`, and the missing-weights error message; `LICENSE`
  adds the harness copyright.

## [0.4.4] - 2026-09-20

The verdict layer is wired into the tool registry, so every tool call is
judged against a named consequence and the judgment is durable.

### Added

- `VerificationRun` now lives on `ToolRegistry` (`registry.verification_run`);
  every executed call records a `Consequence(tool, expect, arguments)` and
  the folded verdict (`ok` / `error` / `unknown`).
- The `tool/result` fact carries `verdict` plus a `steps` audit trail built
  from the flow-step vocabulary (`pre-execute`, `post-execute`).
- `Consequence`, `VerificationRun`, and `judge` are exported from the
  package root; `tests/test_verdict.py` covers the layer and its wiring.

## [0.4.3] - 2026-09-12

Reproducible end-to-end coverage for the remote model path, and a
single-sourced package version.

### Added

- ``tests/test_e2e_remote.py``: drives the full CLI against a local
  OpenAI-compatible streaming stub over a real socket — covering one-shot
  streaming and a streamed tool call whose result is dispatched and fed back
  to the endpoint. Replaces the ad-hoc stub that verified Phase 8 and was
  never committed.

### Fixed

- ``pyproject.toml`` still declared ``0.4.0`` while ``CHANGELOG`` had already
  shipped ``0.4.1`` and ``0.4.2``. The version now derives from
  ``glmharness.__version__`` through ``[tool.hatch.version]``, so the two can
  no longer drift apart.

## [0.4.2] - 2026-09-12

HTTP / Web Document Reader tool added to the built-in tool battery.

### Added

- ``fetch_url``: async-safe HTTP/HTTPS content extraction tool with HTML-to-markdown
  stripping, plain-text / JSON handling, and SSRF loopback/private IP filtering.
- ``clean_html_to_markdown``: zero-dependency HTML parser for structured extraction
  of page titles, headers, links, and code blocks.
- Tool factory ``make_fetch_url_tool`` and ``clean_html_to_markdown`` exported
  from ``glmharness`` and mounted in ``BuiltinToolsPlugin``.

## [0.4.1] - 2026-09-12

Workspace codebase search tools added to the built-in tool battery.

### Added

- ``find_files``: glob pattern matching across workspace directories with
  file/directory type filtering and automatic vendor/cache directory pruning.
- ``grep_search``: recursive text and regular expression search within workspace
  files with case sensitivity toggle, file glob filter, and line citations.
- Tool factories ``make_find_files_tool`` and ``make_grep_search_tool`` exported
  from ``glmharness`` and mounted in ``BuiltinToolsPlugin``.

## [0.4.0] - 2026-09-09

Semantic embedding service wired into the harness.

### Added

- ``EmbeddingProvider``: lazy loader over a local sentence-transformers
  snapshot (``all-MiniLM-L6-v2``) with a fail-fast snapshot check and an
  actionable error when the optional ``[embeddings]`` extra is missing.
- Semantic tool battery mounted by ``EmbeddingsPlugin``: ``embed_text``,
  ``semantic_similarity``, and ``semantic_rank`` (read-only, cosine-based,
  normalized vectors).
- ``GLMH_EMBED_MODEL_PATH`` / ``--embed-model-path`` config; when unset the
  resolver picks up ``~/all-MiniLM-L6-v2`` or ``./all-MiniLM-L6-v2`` when
  present, so the tools mount with zero configuration on operator machines.
- CLI convenience actions ``--embed TEXT`` and ``--similarity A B``, REPL
  commands ``/embed`` and ``/similar``, and a ``--doctor`` embeddings row
  that fails when the snapshot or the ``[embeddings]`` extra is missing.
- JSON-RPC server mounts the plugin and advertises ``embeddings`` in
  ``runtimeCapabilities`` when the model resolves; the semantic tools are
  reachable over ``tools/list`` and ``tools/execute``.
- New ``[embeddings]`` extra (``sentence-transformers>=3.0``).
- ``cosine_similarity`` helper (numpy-free) and unit tests for the provider,
  tools, plugin mount, and CLI wiring.

## [0.3.3] - 2026-09-09

Local snapshot loading fixed for the multimodal GLM-5.3-Flash architecture.

### Fixed

- ``TransformersGLM`` loaded the snapshot with ``AutoModelForCausalLM`` /
  ``AutoTokenizer``, which transformers rejects for ``glm5_next``
  (``Unrecognized configuration class``). The adapter now loads through the
  image-text-to-text stack — ``AutoProcessor`` +
  ``AutoModelForImageTextToText`` — and applies the chat template on the
  processor.
- ``turn failed`` logs include the exception type in text mode, so a failed
  turn is diagnosable without switching to JSON logging.

### Changed

- The ``[inference]`` extra now includes ``accelerate``, ``torchvision`` and
  ``pillow`` alongside ``torch`` and ``transformers`` so the multimodal
  processor and ``device_map="auto"`` load out of the box.
- A config-only snapshot (no ``model-*.safetensors`` shards) fails fast with
  an actionable ``ConfigError`` (exit 2) pointing at the disk/RAM requirement
  and the ``--api-base`` alternative, instead of a cryptic loader traceback
  after retries.
- ``HarnessError`` messages from a failed turn are logged to stderr before
  the CLI returns the mapped exit code, so the reason is visible in text mode.

## [0.3.2] - 2026-09-05

Operator default: request timeout is on.

### Changed

- ``HarnessConfig.request_timeout_s`` (and therefore the CLI / ``GLMH_REQUEST_TIMEOUT_S``)
  defaults to **300 seconds**. ``0`` still disables the bound for long local
  generations. ``AgentLoop`` constructed directly still defaults to ``0`` so
  library callers are not silently bounded.
- ``--doctor`` only prints the timeout ``WARN`` when the bound is disabled.

## [0.3.1] - 2026-09-05

Second production pass: timeouts, empty prompts, and doctor honesty.

### Changed

- A spent per-request timeout is now ``ProviderTimeout`` (CLI exit 3,
  retryable until ``max_retries`` is exhausted) instead of a raw
  ``TimeoutError`` (exit 4).
- An empty prompt on a non-TTY (or EOF) is a config error, not a hang
  on ``input()``.
- ``--doctor`` resolves the snapshot the same way as ``run``, prints
  ``WARN`` when ``request_timeout_s`` is 0, and fails when cwd is not a
  GLM snapshot and ``--mock`` is unset.

## [0.3.0] - 2026-09-05

Production hardening for the local runner. Public API stays backward
compatible; new names are additive (`SafetyPlugin`,
`validate_tool_arguments`, `looks_like_snapshot`, `resolve_model_path`).

### Added

- `GLMH_MOCK` is now a real env knob (it was documented, not wired).
- Tool-argument JSON Schema subset (`type` / `required` / `properties` /
  `additionalProperties` / `items`) runs before every handler. Bad args
  return `INVALID_ARGS` and never invoke the tool.
- `SafetyPlugin` + `GLMH_TOOL_ALLOWLIST` / `--tool-allowlist`: an explicit
  name list denies every other tool at `tools/pre-execute`. Empty list is
  a no-op so the plugin is always mounted.
- `glm-harness --doctor` validates config, snapshot layout, and the
  optional `inference` extra without running a turn.
- Cwd that looks like a GLM snapshot (`config.json` +
  `tokenizer_config.json`) is the default model path, matching the README.

### Changed

- Package version `0.3.0`.

## [0.2.0] - 2026-09-05

The harness is now production-ready: installable Python package,
tooled + tested, strict types, CI-integrated. Backward-compatible at
the public-API surface (public exception classes, `MockLLM`,
`TransformersGLM`, `AgentLoop`, `BasePlugin`, `Tool`, `ToolRegistry`,
`parse_tool_calls`, `EventBus`, `Context`, `PluginLoader`,
`SessionLog`, `HarnessConfig`).

### Added

- `pyproject.toml` with `glmharness` install metadata, console script
  entry point, optional `[inference]` and `[dev]` extras.
- Strict type checking (`pyright --strict src`) wired into CI.
- 72 new unit tests across 11 files (`tests/`).
- Opt-in integration suite (`tests/integration/`, gated by
  `GLMH_RUN_INTEGRATION=1`) that exercises the LLM-bridge surface
  without loading the model.
- Structured logging with `text` and `json` formats on stderr.
- `HarnessConfig.from_env()` reads `GLMH_*` env vars with unknown-key
  warnings (typo guard).
- Retry/backoff with deterministic jitter for transient provider
  errors; non-retryable `ProviderError`s fail fast.
- Per-request and per-tool timeouts (`asyncio.timeout` /
  `asyncio.wait_for`).
- Three-way corruption policy for `SessionLog`: `skip`, `rename`, `fail`.

### Changed

- `harness.py` is now a thin compat shim over the package — new code
  should `import glmharness` directly.
- LLM thread-bridge is race-free: a feeder thread and a generation
  thread post to an asyncio queue; both share an error box written
  **before** the done sentinel so the consumer always observes a
  generation failure after the stream ends.
- Retried provider attempts never commit their tokens to the
  session log — the durable history contains only the recovered
  response.
- `AgentLoop` enforces a `max_rounds >= 1` configuration constraint
  so the loop body is always entered.
- CLI exit-code map documented and tested: 0=ok, 2=config, 3=provider,
  4=other, 130=cancelled.

### Fixed

- `pyright --strict src`: 46 errors -> 0.
- `MockLLM` regression that conflated with `TransformersGLM`'s body.
- Tool waterfall denial dropped the call's `id`/`name`; now merged
  with original so `tool/result` always identifies the call.
- `SessionLog` corruption-recovery rename policy preserves the
  in-memory prefix and continues writing to the original path.

## [0.1.0] - 2026-09-03

Initial MVP single-file `harness.py`. Three kernel primitives
(Context, EventBus, PluginLoader) and the contract that
"model-visible means logged."

[0.2.0]: https://huggingface.co/ohmskiii/GLM-5.3-Flash/compare/c8fda3e...59bff63
