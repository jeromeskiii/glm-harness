# GLM-5.3-Flash harness

A small, auditable H1-style plugin harness for a local GLM-5.3-Flash
snapshot. The kernel owns three primitives — `Context`, `EventBus`, and
`PluginLoader` — and everything else (session log, tool registry, agent
loop, model adapter) is a replaceable service behind them.

The kernel is exposed through two front doors: the authoritative Python CLI
(`glm-harness`) and a zero-dependency TypeScript entry point
([`src/index.ts`](./src/index.ts)) that mirrors the CLI's initialization
sequence and delegates to it — for Node hosts such as IDE carriers, DMH
controllers, and sibling `-H` harnesses.

## Install

```bash
# Optional local virtualenv
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[inference,embeddings,dev]"
```

The `inference` extra pulls in `transformers>=5.0`, `torch`, `torchvision`,
`pillow` and `accelerate` — the stack the multimodal local snapshot path
needs. Without it the harness falls back to `MockLLM` for offline smoke
runs. The `embeddings` extra pulls in `sentence-transformers` for the
semantic tool battery.

## Run

# Connect to external vLLM / SGLang / Ollama / OpenAI-compatible endpoint
glm-harness --api-base http://127.0.0.1:8000/v1 'Explain this repository'

# Deterministic mock (no model weights, no GPU required)
glm-harness --mock 'hello from the mock adapter' 'say hello'

# Interactive multi-turn REPL
glm-harness --repl --api-base http://127.0.0.1:8000/v1

# Stdio JSON-RPC 2.0 protocol server (for DMH & IDE host integration)
glm-harness --serve

# Live local snapshot — cwd is used when it contains config.json +
# tokenizer_config.json; override with GLMH_MODEL_PATH or --model-path.
glm-harness 'Explain this repository'

# Pre-flight: config, snapshot layout, inference extra
glm-harness --doctor

# Persist the conversation for replay / debugging
glm-harness --session .sessions/demo.jsonl 'Plan a release'

# Stop-Slop: Prose quality analysis & deterministic de-slopping
glm-harness --slop-analyze "Here's the thing: we must navigate the fast-paced landscape."
glm-harness --slop-rewrite "Here's the thing: we must navigate uncertainty. Let that sink in."
glm-harness --slop-rules scoring
glm-harness --slop-examples

# Semantic embeddings: local sentence-transformers snapshot
glm-harness --embed 'hello world'
glm-harness --similarity 'machine learning' 'neural networks'
```

The CLI composes with pipes: the final answer is on stdout, every
diagnostic record on stderr.

### TypeScript entry point

`src/index.ts` is the Node-side front door: zero dependencies, ESM, no
build step, executed via Node's type stripping (Node ≥ 22; add
`--experimental-strip-types` on Node < 23.6):

```bash
# Pre-flight
node src/index.ts --doctor

# Deterministic mock run
node src/index.ts --mock 'hello from the mock adapter' 'say hello'

# Same flags and GLMH_* layering as the Python CLI
node src/index.ts --api-base http://127.0.0.1:8000/v1 'Explain this repository'

# Trace the initialization sequence (steps 1-6) on stderr
GLMH_TRACE_INIT=1 node src/index.ts --mock ok 'say hello'
```

It mirrors the Python initialization sequence step for step — parse argv →
merge config → validate → resolve runtime → build child argv → dispatch —
and executes the resolved runtime with inherited stdio, so the stream
contract is unchanged: stdout carries only the final answer, diagnostics
go to stderr, and exit codes match the CLI (`0` ok, `2` config, `3`
provider, `4` tool, `130` cancelled).

Runtime resolution order: `$GLMH_HARNESS_BIN` → `$GLMH_PYTHON`
(`-m glmharness.cli`) → `<repo>/.venv/bin/glm-harness` → `glm-harness` on
`PATH` → `python3 -m glmharness.cli`.

### Local snapshot requirements

The vendored repo is a **config-only snapshot** — `config.json`,
`tokenizer_config.json` and the processor files are present, but the
weights are not. The full GLM-5.3-Flash FP8 checkpoint is ~330 GB on disk
and dequantizes to ~660 GB in RAM at load, so it needs a GPU/XPU server or
an operator-class box. On a laptop, run the model through a served
endpoint instead:

```bash
# Z.ai API, or any local vLLM / SGLang / Ollama OpenAI-compatible server
glm-harness --api-base http://127.0.0.1:8000/v1 'Explain this repository'
```

When weights are missing the harness fails fast (exit 2) with an
actionable message instead of a cryptic loader traceback. To run the
snapshot locally, place the `model-*.safetensors` shards next to
`config.json` (or point `GLMH_MODEL_PATH` at a complete snapshot
directory).

### Semantic embeddings

The harness can mount a local sentence-transformers snapshot
(`all-MiniLM-L6-v2`, 384-dim, normalized) behind three read-only model
tools — `embed_text`, `semantic_similarity`, and `semantic_rank` — plus
the `--embed` / `--similarity` CLI actions and `/embed` / `/similar`
REPL commands. The provider loads lazily, so runs that never touch the
semantic tools pay nothing.

The model resolves from `--embed-model-path` / `GLMH_EMBED_MODEL_PATH`;
when unset, `~/all-MiniLM-L6-v2` (then `./all-MiniLM-L6-v2`) is picked
up automatically when present. `glm-harness --doctor` reports the
resolved path and the sentence-transformers version (and fails when
either is missing). The stdio JSON-RPC server advertises `embeddings` in
`runtimeCapabilities` and serves the tools over `tools/list` /
`tools/execute`.

```bash
# Embed and print the 384-dim vector
glm-harness --embed 'hello world'

# Cosine similarity between two texts
glm-harness --similarity 'machine learning' 'neural networks'
```

### GitHub tools

Setting a repo scope (`--github owner/name` / `GITHUB_REPO`) or a token
(`--github-token` / `GITHUB_TOKEN`) auto-mounts the zero-dependency GitHub
REST battery: 13 read-only tools (repo metadata, issues, issue comments,
files, PRs, PR files, branches, commits, code & issue search) and 5
mutating tools (`github_write_file`, `github_create_issue`,
`github_create_pull_request`, `github_create_branch`,
`github_add_issue_comment`) gated by the sandbox policy. The API base
defaults to `https://api.github.com` and must be HTTPS (plain HTTP is
accepted only for loopback development). `glm-harness --doctor` reports
the active scope and whether a token is configured.

### Configuration

Knobs are layered: built-in defaults → `GLMH_*` env vars → CLI flags. The
set of recognized envs with their defaults:

| Env var | Default | Purpose |
| --- | --- | --- |
| `GLMH_API_BASE` | — | base URL for OpenAI-compatible endpoint (`--api-base`) |
| `GLMH_API_KEY` | — | optional bearer token (`--api-key`) |
| `GLMH_MODEL` | `GLM-5.3-Flash` | model identifier for remote endpoint (`--model`) |
| `GLMH_MODEL_PATH` | cwd if it is a snapshot | path to a GLM-5.3-Flash snapshot |
| `GLMH_SESSION` | — | path to the append-only JSONL session log (`--session`) |
| `GLMH_MOCK` | — | if set, use this string as a deterministic response instead of the model |
| `GLMH_TOOL_ALLOWLIST` | empty (all registered) | comma-separated tool names the model may call |
| `GLMH_REASONING_EFFORT` | `max` | one of `low`, `high`, `max` |
| `GLMH_MAX_NEW_TOKENS` | 8192 | per-request token budget |
| `GLMH_MAX_ROUNDS` | 12 | tool-calling rounds per turn |
| `GLMH_REQUEST_TIMEOUT_S` | 300 | per-request wall clock in seconds; 0 disables |
| `GLMH_TOOL_TIMEOUT_S` | 30 | per-tool timeout |
| `GLMH_TURN_TIMEOUT_S` | `3600` | whole-turn deadline for server `agent/send`, covering every round and retry (env-only); 0 disables |
| `GLMH_MAX_RETRIES` | 2 | retries on transient provider failure |
| `GLMH_RETRY_BASE_DELAY_S` | 1.0 | exponential-backoff base |
| `GLMH_RETRY_MAX_DELAY_S` | 30.0 | backoff cap |
| `GLMH_RETRY_JITTER` | 0.25 | fractional jitter in `[0, 1)` |
| `GLMH_LOG_FORMAT` | `text` | `text` for humans, `json` for shippers |
| `GLMH_LOG_LEVEL` | `INFO` | standard logging levels |
| `GLMH_CORRUPT_POLICY` | `skip` | `skip` / `rename` / `fail` on bad session JSONL |
| `GLMH_WORKSPACE` | cwd | workspace root directory for tools (`--workspace`) |
| `GLMH_SANDBOX` | `deny` | sandbox policy (`allow`, `deny`, `ask`); mutating tools require explicit opt-in |
| `GLMH_COMPACTION_THRESHOLD` | `0` | token or turn threshold for automated history compaction (`--compaction-threshold`) |
| `GLMH_COMPACTION_KEEP_ROUNDS` | `4` | recent interaction turns to keep uncompacted (`--compaction-keep-rounds`) |
| `GLMH_COMPACTION_STRATEGY` | `summarize` | `summarize` or `truncate` (`--compaction-strategy`) |
| `GLMH_REPLAY_LOG` | — | JSONL session log to replay / reconstruct history from (`--replay-log`) |
| `GLMH_PROJECTION` | — | JSON array of prior messages to replay from (`--projection`) |
| `GLMH_SKILLS_DIR` | — | directory tree containing external `SKILL.md` bundles (`--skills-dir`) |
| `GLMH_SKILL_GATE_TOOLS` | `false` | enable skill-managed tool gating (`--skill-gate-tools`) |
| `GLMH_TASK_RISK` | `low` | task risk boundary used for skill activation (`--task-risk`) |
| `GLMH_EMBED_MODEL_PATH` | `~/all-MiniLM-L6-v2` if present | local sentence-transformers snapshot for semantic tools (`--embed-model-path`) |
| `GLMH_TEMPERATURE` | `0.7` | sampling temperature for remote endpoints, in [0.0, 2.0] (`--temperature`) |
| `GLMH_RPC_TOKEN` | auto-generated per process | bearer token required by the stdio JSON-RPC server (`--rpc-token`) |
| `GLMH_RPC_AUTO_TOKEN` | `true` | auto-generate an RPC token when none is set (`--rpc-auto-token`) |
| `GLMH_RPC_ALLOW_MUTATING` | `false` | permit mutating tool calls over the JSON-RPC server (`--rpc-allow-mutating`) |
| `GLMH_STATE_DIR` | — | root directory the server may read `session/import.logPath` files from (`--state-dir`) |
| `GLMH_MAX_SESSIONS` | `256` | cap on in-memory sessions held by the protocol server; the oldest non-default session is evicted with a warning (env-only) |
| `GITHUB_TOKEN` | — | bearer token for GitHub tools; also read from the environment directly (`--github-token`) |
| `GITHUB_REPO` | — | default repo scope `owner/name` for GitHub tools (`--github`) |
| `GITHUB_API_BASE` | `https://api.github.com` | GitHub REST API base URL, HTTPS required (`--github-api-base`) |

Unknown `GLMH_*` variables are logged and ignored — typos won't crash the
harness.

Bridge-only variables (consumed by the TypeScript entry before it delegates;
the Python CLI itself never reads them):

| Env var | Default | Purpose |
| --- | --- | --- |
| `GLMH_HARNESS_BIN` | — | explicit path to the `glm-harness` console script |
| `GLMH_PYTHON` | — | python interpreter used for the `-m glmharness.cli` path |
| `GLMH_TRACE_INIT` | — | set to `1` to trace the initialization sequence to stderr |

### JSON-RPC server security

The stdio protocol server (`--serve`) is authenticated by default:

- **Token auth.** With no `--rpc-token` / `GLMH_RPC_TOKEN`, a per-process
  token is generated with the OS CSPRNG, printed to stderr exactly once, and
  required by every RPC after `initialize`. Set `--rpc-auto-token=false` to
  run unauthenticated (loopback/IDE hosts only).
- **Mutating tools denied.** `bash`, `write_file`, `edit_file`, and the
  mutating GitHub tools are refused over RPC unless `--rpc-allow-mutating`
  is passed explicitly.
- **Import path scoping.** `session/import.logPath` reads are refused unless
  the path is relative and falls under `--state-dir`.
- **Bounded session cache.** At most `GLMH_MAX_SESSIONS` (default 256)
  sessions are held in memory; when the cap is reached the oldest
  non-default session is evicted with a warning (only `default` persists to
  `--session`).
- **Turn deadline.** `agent/send` runs under a whole-turn budget
  (`GLMH_TURN_TIMEOUT_S`, default 3600 s; 0 disables). Expiry closes the
  turn with a durable `TURN_TIMEOUT` marker and returns JSON-RPC error
  `-32008`.

### Runtime & safety flags

Sampling and behavior (`--temperature`, `--reasoning-effort`, `--max-rounds`,
`--max-new-tokens`), retries and timeouts (`--max-retries`,
`--request-timeout-s`, `--tool-timeout-s`), logging (`--log-format`,
`--log-level`, `--corrupt-policy`), and the safety surface (`--sandbox`,
`--tool-allowlist`, `--enable-bash` / `--no-enable-bash`,
`--no-enable-fetch-url`) are all wired through the configuration layer above.
The `bash` tool is on by default and prints a warning banner; pass
`--no-enable-bash` to omit it. Every mutating tool is additionally gated by
the `--sandbox` policy (default `deny`).

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 2 | configuration error |
| 3 | provider failure (including a spent request timeout) |
| 4 | tool/pipeline failure |
| 130 | cancelled (SIGINT/SIGTERM) |

## Stop-Slop Prose Quality Engine

The harness bundles an embedded, zero-dependency prose quality engine based on the `stop-slop` skill. It eliminates predictable AI tells, throat-clearing, emphasis crutches, business jargon, and adverbs.

### Features

- **5-Dimensional Scoring**: Evaluates prose across Directness, Rhythm, Trust, Authenticity, and Density (1–10 each, total /50; threshold <35 indicates revision needed).
- **Deterministic Rewriter**: Automatically strips throat-clearing openers, emphasis crutches ("Let that sink in.", "Full stop."), replaces corporate jargon with plain English, and cleans em dashes.
- **Dynamic Skill Integration**: When prompt topics mention `prose`, `writing`, `slop`, `draft`, `jargon`, or `buzzwords`, the `stop-slop` skill automatically activates, injecting quality guidance and unlocking model tools.
- **Model Tools**: `stop_slop_analyze`, `stop_slop_rewrite`, `stop_slop_rules`, and `stop_slop_examples`.
- **REPL Commands**: `/slop <text>` for instant dimensional audits, and `/deslop <text>` for instant rewrites.

## Develop

```bash
pytest -q                                                 # all tests
pytest --cov=glmharness --cov-report=term-missing        # coverage
ruff check src tests                                      # lint
```

The TypeScript entry has no build step and no dependencies; run it directly
(`node src/index.ts --help`).

## Deliberate MVP boundaries

This is a one-shot local runner. The kernel stays minimal on purpose:

- **OS sandbox** is not in the kernel. `SafetyPlugin` is the production
  allowlist gate on `tools/pre-execute`; add an approval plugin the same way.
- **Network protocol** is not exposed by the harness; the stdio JSON-RPC carrier
  and the TypeScript entry point (`src/index.ts`, which spawns the runtime with
  inherited stdio) are intended for a local host such as an IDE or DMH controller.
- **Streaming backpressure** is bounded only by the consumer's iterator;
  a hosting carrier would add explicit flow control.
- **Multi-image / video** inputs require the multimodal chat template;
  text-only is what this CLI exercises.

Tool outcomes are appended as `tool/result` facts; failed or cancelled
provider calls close their turn with a durable status marker.

### Verdict layer

Every tool execution is judged against a named consequence before it
becomes history. `ToolRegistry.verification_run` (a `VerificationRun`)
records a `Consequence(tool, expect, arguments)` for each call and folds
the result into an `ok` / `error` / `unknown` verdict; that verdict and
the flow-step audit trail are appended to the durable `tool/result` fact.

### Production safety baseline

The default sandbox policy is `deny`. To permit mutating tools, choose an
explicit policy for the deployment:

```bash
# Interactive operator approval for each mutating action
GLMH_SANDBOX=ask glm-harness --mock 'ok' 'inspect the workspace'

# Explicit unattended opt-in (use only with a dedicated workspace)
GLMH_SANDBOX=allow glm-harness --workspace /srv/glm-work 'run the task'
```

Keep the workspace dedicated to the harness. The built-in `bash` tool runs a
shell in that directory but is not an OS-level sandbox; use a container or
other OS isolation when commands must be confined against a malicious model
or untrusted prompt.
