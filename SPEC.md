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

---

## Phase 5: Dynamic Skill Trigger & Tool Gating System

### Goal
Provide a deterministic, high-safety skill trigger and tool-gating subsystem inspired by SGAA:
1. **Deterministic 3-Tier Trigger Matching**: Alphanumeric token boundaries, morphological stemming (doubled consonants, silent `-e`, `-ies`), irregular verb groups (`run/ran`, `write/wrote`), and phrase boundary regex.
2. **Multi-Signal Activation Scoring & Risk Gating**: Mathematical scoring across `always`, explicit metadata, capability overlap, trigger ramps ($\min(1.0, 0.45 + 0.20 \times |hits|)$), and tags, bounded by `[min_risk, max_risk]`.
3. **Safe Tool Gating**: Effective allowed tools computed as $(T_{\text{all}} \setminus T_{\text{managed}}) \cup T_{\text{active}}$. Unmatched skill tools are locked out from both LLM schemas and execution interception.
4. **Dynamic `SKILL.md` Ingestion**: Recursive discovery of external Agent Skills with YAML frontmatter parsing and heuristic trigger auto-extraction.
5. **Plugin & Carrier Integration**: `SkillsPlugin` mounted across CLI, REPL, and Stdio Server (`skills/list`, `skills/match`, `skills/import`).

---

## Phase 6: Stop-Slop Prose Quality Engine & Tool Integration

### Goal
Install and wire the `stop-slop` agent skill from `/Users/ohmskiii/Desktop/stop-slop` into `/Users/ohmskiii/GLM-5.3-Flash-H`, providing deterministic prose pattern analysis, dimensional scoring, and automatic de-slopping.

### Architecture & Components

#### 1. Installed Skill Bundle (`skills/stop-slop/`)
- Ingested `SKILL.md` with complete YAML frontmatter (tools, triggers, tags, category).
- Ingested reference corpus: `references/phrases.md`, `references/structures.md`, and `references/examples.md`.

#### 2. Stop-Slop Engine & Toolset (`src/glmharness/stop_slop.py`)
- `StopSlopEngine`:
  - `analyze(text: str) -> dict[str, Any]`: Evaluates prose across 5 dimensions: Directness, Rhythm, Trust, Authenticity, and Density (1–10 each, total /50; threshold <35). Identifies throat-clearing, emphasis crutches, corporate jargon, adverbs/fillers, binary contrasts, meta-commentary, vague declaratives, and em-dashes.
  - `rewrite(text: str) -> dict[str, Any]`: Deterministic cleanup replacing jargon with plain English, removing openers and crutches, and normalizing punctuation.
  - `get_rules(category: str | None) -> dict[str, Any]`: Structured style guide rules.
  - `get_examples() -> list[dict[str, str]]`: 5 canonical before/after transformations.
- Tool Factories:
  - `make_stop_slop_analyze_tool()`
  - `make_stop_slop_rewrite_tool()`
  - `make_stop_slop_rules_tool()`
  - `make_stop_slop_examples_tool()`

#### 3. Builtin Tools & Skills Integration
- `BuiltinToolsPlugin`: Automatically mounts all 4 stop-slop tools into `ToolRegistry`.
- `default_skills()`: Registers `stop-slop` skill with triggers (`slop`, `de-slop`, `prose`, `writing`, `draft`, `buzzword`, `jargon`, `adverb`).
- `SkillsPlugin.apply()`: Automatically discovers `skills/` directory when `--skills-dir` is not explicitly provided.

#### 4. Carriers & CLI Controls
- CLI: `--slop-analyze <text>`, `--slop-rewrite <text>`, `--slop-rules [category]`, `--slop-examples`.
- REPL: `/slop <text>` and `/deslop <text>` commands.



