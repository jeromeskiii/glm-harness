# Architecture summary

The kernel is three primitives: `Context` (typed service bag), `EventBus`
(four dispatch modes: `emit`, `parallel`, `serial`, `waterfall`), and
`PluginLoader` (mount + fail-loud + LIFO close). Everything else lives
behind a service:

- `glmharness.session.SessionLog` — append-only JSONL with `fsync`,
  corruption policy, surface-event projection (`derive_messages`).
- `glmharness.tools.ToolRegistry` — openai-shape schema export, guarded
  pipeline (`pre-execute` -> tool -> `post-execute`), per-tool timeout.
- `glmharness.loop.AgentLoop` — one-shot runner with per-request
  retry/backoff, max-round cap, and durable status markers on every
  outcome.
- `glmharness.llm.TransformersGLM` — local GLM-5.3-Flash adapter with
  real streaming via `TextIteratorStreamer` bridged to asyncio through a
  two-thread bridge and an error box; `MockLLM` for tests.
- `glmharness.config.HarnessConfig` — defaults, env-var loading, CLI
  override, validation at startup.
- `glmharness.errors.*` — typed error hierarchy (`ConfigError`,
  `ProviderError`, `SessionCorruptError`...). Every wire/log vocabulary
  is centralized.
- `glmharness.cli` — argv parser, layered config, signal handlers, exit
  codes 0/2/3/4/130.

The single invariant — *model-visible means logged* — is preserved by
deriving provider history from the session log only. Provider requests
record `request/header` and `request/context` on the first round;
streamed chunks are buffered per attempt and committed only when an
attempt succeeds, so a retried attempt never pollutes the log.
