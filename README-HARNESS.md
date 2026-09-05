# GLM-5.3-Flash harness

A small, auditable H1-style plugin harness for a local GLM-5.3-Flash
snapshot. The kernel owns three primitives — `Context`, `EventBus`, and
`PluginLoader` — and everything else (session log, tool registry, agent
loop, model adapter) is a replaceable service behind them.

## Install

```bash
# Optional local virtualenv
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[inference,dev]"
```

The `inference` extra pulls in `transformers>=5.0` and `torch`. Without it
the harness falls back to `MockLLM` for offline smoke runs.

## Run

```bash
# Deterministic mock (no model weights, no GPU required)
glm-harness --mock 'hello from the mock adapter' 'say hello'

# Live local snapshot — defaults to the repo root, override with
# GLMH_MODEL_PATH or --model-path.
glm-harness 'Explain this repository'

# Persist the conversation for replay / debugging
glm-harness --session .sessions/demo.jsonl 'Plan a release'
```

The CLI composes with pipes: the final answer is on stdout, every
diagnostic record on stderr.

### Configuration

Knobs are layered: built-in defaults → `GLMH_*` env vars → CLI flags. The
set of recognized envs with their defaults:

| Env var | Default | Purpose |
| --- | --- | --- |
| `GLMH_MODEL_PATH` | repo root | path to a GLM-5.3-Flash snapshot |
| `GLMH_MOCK` | — | if set, use a deterministic response instead of the model |
| `GLMH_REASONING_EFFORT` | `max` | one of `low`, `high`, `max` |
| `GLMH_MAX_NEW_TOKENS` | 8192 | per-request token budget |
| `GLMH_MAX_ROUNDS` | 12 | tool-calling rounds per turn |
| `GLMH_REQUEST_TIMEOUT_S` | 0 (off) | per-request wall clock; 0 disables |
| `GLMH_TOOL_TIMEOUT_S` | 30 | per-tool timeout |
| `GLMH_MAX_RETRIES` | 2 | retries on transient provider failure |
| `GLMH_RETRY_BASE_DELAY_S` | 1.0 | exponential-backoff base |
| `GLMH_RETRY_MAX_DELAY_S` | 30.0 | backoff cap |
| `GLMH_RETRY_JITTER` | 0.25 | fractional jitter in `[0, 1)` |
| `GLMH_LOG_FORMAT` | `text` | `text` for humans, `json` for shippers |
| `GLMH_LOG_LEVEL` | `INFO` | standard logging levels |
| `GLMH_CORRUPT_POLICY` | `skip` | `skip` / `rename` / `fail` on bad session JSONL |

Unknown `GLMH_*` variables are logged and ignored — typos won't crash the
harness.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 2 | configuration error |
| 3 | provider failure |
| 4 | tool/pipeline failure |
| 130 | cancelled (SIGINT/SIGTERM) |

## Develop

```bash
pytest -q                                                 # all tests
pytest --cov=glmharness --cov-report=term-missing        # coverage
ruff check src tests                                      # lint
```

## Deliberate MVP boundaries

This is a one-shot local runner. The kernel stays minimal on purpose:

- **Sandbox / approval** are not in the kernel. Add them as a plugin that
  listens on `tools/pre-execute`.
- **Network protocol** is out of scope; this CLI is the only entry point.
- **Streaming backpressure** is bounded only by the consumer's iterator;
  a hosting carrier would add explicit flow control.
- **Multi-image / video** inputs require the multimodal chat template;
  text-only is what this CLI exercises.

Tool outcomes are appended as `tool/result` facts; failed or cancelled
provider calls close their turn with a durable status marker.
