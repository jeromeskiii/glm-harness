# glm-harness

A small, auditable H1-style plugin agent harness for a local
GLM-5.3-Flash snapshot. The kernel owns three primitives —
`Context`, `EventBus`, and `PluginLoader` — and everything else
(session log, tool registry, loop, model adapter) is a replaceable
service behind them.

For full documentation see **[`README-HARNESS.md`](./README-HARNESS.md)**.

For the GLM-5.3-Flash model card and citations see
[`MODEL_CARD.md`](./MODEL_CARD.md).

For the Hugging Face mirror that ships the vendored model
artifacts, see
[ohmskiii/GLM-5.3-Flash](https://huggingface.co/ohmskiii/GLM-5.3-Flash).

## Local Agent Harness

This repository includes a small, auditable **H1-style plugin agent harness** for running GLM-5.3-Flash locally. The harness provides a lightweight turn-taking agent loop with tool calling, session logging, and a pluggable architecture — all designed for transparency and easy extension.

The kernel exposes three primitives:

- **`Context`** — shared state and service registry
- **`EventBus`** — pub/sub for plugin lifecycle hooks
- **`PluginLoader`** — dynamic plugin discovery and mounting

Everything else (session log, tool registry, agent loop, model adapter) is a replaceable service behind them.

### Quick Start

```bash
# Install with inference support (requires torch + transformers)
pip install -e ".[inference,dev]"

# Deterministic mock run (no GPU needed)
glm-harness --mock 'hello from the mock adapter' 'say hello'

# Run against the local model (cwd must be a GLM snapshot)
glm-harness 'Explain this repository'

# Pre-flight
glm-harness --doctor

# Persist conversation for replay / debugging
glm-harness --session .sessions/demo.jsonl 'Plan a release'

# GitHub tools: scope to a repo (token picked up from $GITHUB_TOKEN)
glm-harness --github octocat/Hello-World 'list the open issues'
```

### Configuration

Knobs are layered: built-in defaults → `GLMH_*` env vars → CLI flags. See [`README-HARNESS.md`](./README-HARNESS.md) for the full reference.

### Why a harness?

This is a **one-shot local runner** — not a production serving framework. The kernel stays minimal on purpose:

- **OS sandbox** is not in the kernel; `SafetyPlugin` is the allowlist gate (`GLMH_TOOL_ALLOWLIST`)
- **No network protocol** — the CLI is the only entry point
- **Bounded streaming** — flow control is left to the consumer
- **Text-only by default** — multimodal inputs require the chat template

For full details, including the configuration reference, exit codes, and development guide, see **[`README-HARNESS.md`](./README-HARNESS.md)**.
