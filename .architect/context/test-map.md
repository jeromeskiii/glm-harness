# Test map

- `tests/test_bus.py` — every EventBus dispatch mode (emit / parallel /
  serial / waterfall) including short-circuit semantics; off() unsubscribes;
  unknown modes raise.
- `tests/test_context.py` — Context provider duplicate/get; LIFO disposer
  order; loader mounts in order and closes the context on a plugin boot
  failure.
- `tests/test_session.py` — append-only JSONL; surface-event projection
  (`derive_messages`); corruption policies (`skip` / `rename` / `fail`).
- `tests/test_tools.py` — schema export, unknown/policy-deny, pre-execute
  waterfall denial, post-execute rewrite, handler exceptions, per-tool
  timeout, sync+async handlers.
- `tests/test_tools_parser.py` — every shape of tool-call block the model
  can emit (single, multi, object values, undecodable fall-back, empty).
- `tests/test_config.py` — defaults validate; reasoning-budget enum; knobs
  in range; env-var loading, typos warned about, bad values fail.
- `tests/test_loop.py` — happy path completes; reconstruction epoch
  recorded only on first round; tool-calling loop drives the model back
  until it stops; transient errors retry; non-retryable provider errors
  propagate; request timeout fires; cancellation closes the turn;
  `max_rounds` exhaustion records `step/end status="max_rounds_reached"`.
- `tests/test_logging.py` — text and JSON formats write to stderr with
  the right shape; unknown formats rejected.
- `tests/test_llm_adapters.py` — `MockLLM` yields its response;
  `TransformersGLM._load` raises a clear `ConfigError` when transformers
  is unavailable; idempotent if model already loaded.
- `tests/test_cli.py` — argparse flag wiring, version flag exits cleanly,
  config layering catches bad envs, exit-code map (0/2/3/4/130).
- `test_harness.py` (legacy shim) — preserved for audit comparability;
  same coverage as the original single-file harness.

Fast check: `python3 -m pytest -q`. No live-model test runs by default
because the snapshot is large and hardware-dependent; opt in by installing
`pip install -e ".[inference]"` and writing a test against a fixture that
patches `TransformersGLM._load`.
