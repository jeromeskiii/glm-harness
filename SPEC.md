# SPEC: Phase 1 — Core Tool Battery & Execution Confinement

## Goal
Transform `glmharness` from an empty-tool runner into a capable, sandboxed agent harness with a battery of built-in filesystem and execution tools, directory confinement, and an approval/safety plane.

## Architecture & Principles
1. **H1 Plugin Architecture**: Core tools and sandbox policies are implemented as plugins (`BuiltinToolsPlugin`, `SandboxPlugin`) mounting to `Context` and registering tools to `ToolRegistry`.
2. **Fail-Closed Security**: Path traversal (`..`, symlinks outside workspace), unauthorized writes, and command executions default to denied if constraints fail.
3. **Model-Visible Durability**: All tool outcomes (success, invalid args, permission denial, timeout, error) produce structured `tool/result` events logged to `SessionLog`.

## Components & Interfaces

### 1. Built-in Tools
Registered to `ToolRegistry` with strict JSON schemas:
- `read_file`:
  - Args: `path` (string, required), `offset` (integer, optional, default 0), `limit` (integer, optional, default 2000 lines)
  - Returns: `{"content": str, "lines": int, "truncated": bool}`
- `write_file`:
  - Args: `path` (string, required), `content` (string, required)
  - Returns: `{"path": str, "bytes_written": int}`
- `edit_file`:
  - Args: `path` (string, required), `old_string` (string, required), `new_string` (string, required)
  - Returns: `{"path": str, "replacements": int}`
- `list_dir`:
  - Args: `path` (string, optional, default ".")
  - Returns: `{"entries": [{"name": str, "type": "file"|"dir", "size": int}]}`
- `bash`:
  - Args: `command` (string, required), `timeout_s` (number, optional, default 30.0)
  - Returns: `{"exit_code": int, "stdout": str, "stderr": str, "timed_out": bool}`

### 2. Confinement & Sandbox (`SandboxPlugin`)
- Enforces `workspace_dir` (default: current working directory or explicit path).
- Canonical path resolution against root to prevent path traversal.
- Modes:
  - `allow`: Allows execution within workspace bounds.
  - `deny`: Rejects any mutating operations (`write_file`, `edit_file`, `bash`) at `tools/pre-execute`.
  - `ask`: Interactive confirmation for mutating actions (falls back to deny if not a TTY).

### 3. Configuration & CLI Wiring
- `HarnessConfig`:
  - `workspace_dir: Path`
  - `sandbox_mode: Literal["allow", "deny", "ask"]`
- CLI flags: `--workspace <path>`, `--sandbox <mode>`.
- Env vars: `GLMH_WORKSPACE`, `GLMH_SANDBOX`.
- CLI `--doctor` checks workspace existence and sandbox mode.

## Verification Gate
- 100% test pass rate across new and existing tests.
- `pyright --strict src tests` passes with 0 errors.
- `ruff check src tests` passes with 0 errors.
