# SPEC: GLM-5.3-Flash Harness

## Phase 1: Core Tool Battery & Execution Confinement (COMPLETED)
- Built-in filesystem tools (`read_file`, `write_file`, `edit_file`, `list_dir`) and shell tool (`bash`).
- Path confinement via `resolve_safe_path`.
- Sandbox execution policies (`allow`, `deny`, `ask`).

## Phase 2: Remote / OpenAI-Compatible LLM Adapter (COMPLETED)
- Zero-dependency `OpenAICompatibleGLM` using standard library `urllib.request` + background threading.
- Streaming SSE chunks and automated translation of OpenAI `tool_calls` into `<tool_call>` XML.
- Config wired for `--api-base`, `--api-key`, `--model`.

---

## Phase 3: Interactive Carriers — Terminal REPL & Stdio Protocol Server

### Goal
Extend `glmharness` beyond one-shot CLI execution by providing:
1. **Interactive Multi-Turn REPL (`glm-harness repl`)**: Continuous conversational session in terminal with state persistence and meta-commands (`/clear`, `/tools`, `/session`, `/exit`).
2. **JSON-RPC 2.0 Stdio Protocol Server (`glm-harness serve`)**: Standard line-delimited JSON-RPC 2.0 protocol over `stdin`/`stdout` compatible with IDE integrations and Dynamic Multi-Harness (`dmh`).

### Architecture & Components

#### 1. Interactive Terminal REPL (`src/glmharness/repl.py`)
- `run_repl(config: HarnessConfig) -> int`:
  - Instantiates `Context`, `SessionLog`, `ToolRegistry`, and `LLM`.
  - Mounts `BasePlugin`, `BuiltinToolsPlugin`, `SandboxPlugin`, `SafetyPlugin`.
  - Prompts operator via `glm> `.
  - Runs turns through `AgentLoop`, maintaining conversation history across turns.
  - REPL meta-commands:
    - `/help`: Show available commands.
    - `/tools`: List registered tools and descriptions.
    - `/clear`: Start a new conversation turn history.
    - `/session`: Display session event metrics.
    - `/exit`, `/quit`: Exit cleanly.

#### 2. JSON-RPC 2.0 Stdio Protocol Server (`src/glmharness/server.py`)
- `run_server(config: HarnessConfig) -> int`:
  - Reads line-delimited JSON objects from `sys.stdin`.
  - Writes line-delimited JSON responses or notification frames to `sys.stdout`.
  - Diagnostics logged to `sys.stderr`.
  - Standard methods:
    - `initialize`: Returns runtime metadata, ABI version (2), and capabilities dictionary.
    - `ping`: Health check returning `{"ok": true, "timestamp": float}`.
    - `session/list`: Returns active session identifiers.
    - `session/new`: Creates a new session context.
    - `agent/send`: Dispatches `text` into `sessionId`, drives `AgentLoop`, and returns final answer + status.
    - `tools/list`: Returns schemas of all registered tools.
    - `tools/execute`: Direct tool execution with `{"name": str, "arguments": dict}`.
    - `shutdown`: Terminates server cleanly with exit code 0.

#### 3. CLI Subcommands & Flags (`src/glmharness/cli.py`)
- `glm-harness repl` or `--repl`: Start REPL.
- `glm-harness serve` or `--serve`: Start JSON-RPC 2.0 stdio server.
- Default behavior: if `prompt` argument is given, run one-shot turn; if no prompt and `sys.stdin.isatty()`, start `repl`.

### Verification Gate
- Unit tests for REPL session loops and meta-commands.
- Unit tests for JSON-RPC 2.0 protocol server (requests, responses, notifications, error frames).
- CLI integration tests for `serve` and `repl` modes.
- Strict pyright 0 errors, ruff clean, 100% pytest pass rate.

---

## Phase 4: Dynamic Multi-Harness (DMH) Integration & Replay / Compaction

### Goal
1. Support multi-turn history reconstruction and replay from external session logs (`replay.from_log: true`).
2. Implement session token / message compaction to keep long-running multi-turn sessions within budget.
3. Integrate with Dynamic Multi-Harness (`dmh`), enabling runtime switching into `glm-5.3-flash`.

### Architecture & Components

#### 1. History Replay & Projection Import (`src/glmharness/session.py`)
- `SessionLog.import_projection(projection: list[dict[str, Any]])`:
  - Ingests pre-switch surface history (`user`, `assistant`, `tool` messages) and appends durable `SessionEvent`s.
- `SessionLog.import_log(path: Path | str)`:
  - Ingests foreign or prior JSONL logs.
- CLI flags: `--replay-log <path>` and `--projection <json_or_path>`.

#### 2. Session Compaction (`src/glmharness/compaction.py`)
- `CompactionPlugin` and `compact_session(log: SessionLog, threshold: int, keep_rounds: int, strategy: str)`:
  - Truncates or summarizes turns older than watermark.
  - Appends `"session/compacted"` event with `upto`, `dropped_count`, `summary`.
- `SessionLog.derive_messages()`:
  - Honors `session/compacted` watermark, rendering `[compacted history] <summary>` followed only by events >= watermark.
- Configurable via `GLMH_COMPACTION_THRESHOLD`, `GLMH_COMPACTION_KEEP_ROUNDS`, `--compaction-threshold`, `--compaction-keep-rounds`.

#### 3. Protocol Server & DMH Adapter (`src/glmharness/server.py` & `dmh/glm_runtime.py`)
- `src/glmharness/server.py`:
  - Advertises `replay.from_log: true` and `compaction: true` in `initialize`.
  - Accepts `projection` in `agent/send` and `session/import` RPC.
  - Exposes `session/compact` RPC.
- DMH `dmh/glm_runtime.py`:
  - Sets `GLM_CAPABILITIES["replay.from_log"] = True` and `GLM_CAPABILITIES["compaction"] = True`.
  - Replays prior projection history into session before turn execution, supporting seamless runtime switching.

