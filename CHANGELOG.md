# Changelog

All notable changes to the GLM harness ship in this file. Versions
follow [Semantic Versioning](https://semver.org/).

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
