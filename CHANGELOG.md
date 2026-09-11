# Changelog

All notable changes to the GLM harness ship in this file. Versions
follow [Semantic Versioning](https://semver.org/).

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
