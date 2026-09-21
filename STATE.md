# Project State: GLM-5.3-Flash Harness

## Phase 1: Core Tool Battery & Execution Confinement (COMPLETED)
- Shipped `builtin_tools.py` and `sandbox.py`.
- 108 tests passing.

## Phase 2: Remote / OpenAI-Compatible LLM Adapter (COMPLETED)
- Shipped `OpenAICompatibleGLM` with zero dependencies.
- 115 tests passing.

## Phase 3: Interactive Carriers (REPL & Stdio Protocol Server) (COMPLETED)
- Shipped `src/glmharness/repl.py` (interactive multi-turn terminal REPL with meta commands `/help`, `/tools`, `/clear`, `/session`, `/exit`).
- Shipped `src/glmharness/server.py` (JSON-RPC 2.0 stdio protocol server supporting ABI v2, `initialize`, `agent/send`, `tools/list`, `tools/execute`, `session/list`, `session/new`).
- Wired `--repl` and `--serve` flags in `src/glmharness/cli.py`.
- 125 tests passing, 0 pyright errors, 0 ruff warnings.

## Phase 4: Dynamic Multi-Harness (DMH) Integration & Replay / Compaction (COMPLETED)
- Shipped `import_projection` and `import_log` on `SessionLog` (`src/glmharness/session.py`) supporting multi-turn replay and foreign JSONL ingestion.
- Shipped `CompactionPlugin` and `compact_session` (`src/glmharness/compaction.py`) with summarize/truncate strategies and watermark-aware `SessionLog.derive_messages`.
- Wired compaction and replay CLI flags (`--compaction-threshold`, `--compaction-keep-rounds`, `--compaction-strategy`, `--replay-log`, `--projection`) and JSON-RPC RPC endpoints (`session/import`, `session/compact`).
- Connected with DMH (`/Users/ohmskiii/Dynamic-Multi-Harness`): updated `dmh/glm_runtime.py` with `replay.from_log: true` and `compaction: true`, passing DMH `test_phase10_glm.py` (6 tests passing including multi-runtime switch).
- 136 tests passing in `glmharness`, 0 pyright errors, 0 ruff warnings.

## Phase 5: Dynamic Skill Trigger & Tool Gating System (COMPLETED)
- Shipped `src/glmharness/skills.py` (`Skill`, `SkillCatalog`, `SkillsPlugin`, `RiskLevel`, morphological stemmer, irregular verb groups, `SKILL.md` parser and directory ingestion).
- Shipped waterfall hooks on `agent/request` (dynamic trigger matching, guidance injection, and tool allowlist filtering) and `tools/pre-execute` (enforcement gating for skill-managed tools).
- Wired CLI flags (`--skills-dir`, `--skill-gate-tools`, `--task-risk`), REPL command (`/skills`), and JSON-RPC protocol endpoints (`skills/list`, `skills/match`, `skills/import`).
- Added comprehensive unit tests in `tests/test_skills.py` (9/9 passed).
- Full test suite: 143 passed, 2 skipped (100% pass rate on unit tests).
- 0 pyright errors, 0 ruff warnings across `src` and `tests`.

## Phase 6: Stop-Slop Prose Quality Engine & Tool Integration (COMPLETED)
- Installed `stop-slop` skill bundle into `skills/stop-slop` with clean YAML metadata, reference files (`phrases.md`, `structures.md`, `examples.md`).
- Implemented `src/glmharness/stop_slop.py` (`StopSlopEngine`, `make_stop_slop_analyze_tool`, `make_stop_slop_rewrite_tool`, `make_stop_slop_rules_tool`, `make_stop_slop_examples_tool`).
- Wired into `BuiltinToolsPlugin` (`src/glmharness/builtin_tools.py`) to auto-register all 4 stop-slop tools into `ToolRegistry`.
- Wired into `default_skills()` (`src/glmharness/skills.py`) to include `stop-slop` in the baseline skill battery with multi-signal triggers (`slop`, `de-slop`, `prose`, `writing`, `draft`, `buzzword`, `jargon`, `adverb`).
- Enhanced `SkillsPlugin.apply()` with automatic directory discovery for repo/cwd `skills/` bundles when `--skills-dir` is omitted.
- Wired CLI convenience actions (`--slop-analyze`, `--slop-rewrite`, `--slop-rules`, `--slop-examples`) and REPL commands (`/slop <text>`, `/deslop <text>`).
- Shipped comprehensive unit & integration tests in `tests/test_stop_slop.py` (14/14 passed).
- Full test suite: 161 passed, 2 skipped (100% pass rate).
- 0 pyright errors, 0 ruff warnings across `src` and `tests`.

## Phase 7: Local Snapshot Loader Fix (COMPLETED)
- Fixed `TransformersGLM` (`src/glmharness/llm.py`) to load the GLM-5.3-Flash snapshot through the multimodal image-text-to-text stack (`AutoProcessor` + `AutoModelForImageTextToText`) instead of `AutoModelForCausalLM` / `AutoTokenizer`, which transformers rejects for `glm5_next` (`Unrecognized configuration class`).
- Extended the `[inference]` extra with `accelerate`, `torchvision`, `pillow` so the multimodal processor and `device_map="auto"` load out of the box.
- Surfaced the exception type in text-mode `turn failed` logs (`src/glmharness/loop.py`).
- Added regression test `test_transformers_glm_load_uses_multimodal_stack`.
- Verified end-to-end: `--doctor` fully green; the snapshot path now reaches weight loading and fails cleanly with `FileNotFoundError: model-00001-of-00062.safetensors` until shards are placed.
- 162 tests passing, 0 pyright errors, 0 ruff warnings.

## Phase 8: Weight-Presence Gate & Endpoint Verification (COMPLETED)
- The vendored repo is a **config-only snapshot**: the full FP8 checkpoint is 328 GB (62 shards), which dequantizes to ~660 GB at load — infeasible on this host (M4 Pro, 25.8 GB RAM, 331 GB free disk). Weights were not downloaded.
- Added `TransformersGLM._require_shards()` (`src/glmharness/llm.py`): a snapshot without `model-*.safetensors` shards now fails fast with an actionable `ConfigError` (exit 2) explaining the disk/RAM requirement and the `--api-base` alternative.
- `HarnessError` messages from a failed turn now reach stderr in text mode before the CLI returns the mapped exit code (`src/glmharness/cli.py`).
- Added regression test `test_transformers_glm_load_fails_fast_without_weights`.
- Verified the supported real-model path end-to-end: ran `glm-harness --api-base http://127.0.0.1:8777/v1 'say hello'` against a local OpenAI-compatible streaming stub — full pipeline (skills, plugins, agent loop) streamed the answer, exit 0. Opt-in integration tests (`GLMH_RUN_INTEGRATION=1`) pass.
- Documented the snapshot requirements in `README-HARNESS.md`.
- 163 tests passing, 2 skipped; 0 pyright errors, 0 ruff warnings.

## Phase 9: Semantic Embeddings Integration (COMPLETED)
- Shipped `src/glmharness/embeddings.py`: `EmbeddingProvider` (lazy sentence-transformers loader with fail-fast snapshot check), `cosine_similarity` (numpy-free), `EmbeddingsPlugin`, and tool factories `embed_text` / `semantic_similarity` / `semantic_rank`.
- Added `[embeddings]` extra (`sentence-transformers>=3.0`) to `pyproject.toml`; installed sentence-transformers 6.0.1 via `uv sync`.
- Wired config (`GLMH_EMBED_MODEL_PATH` / `--embed-model-path`), auto-resolution of `~/all-MiniLM-L6-v2`, plugin mounting in all three carriers (`cli.py` run, `repl.py`, `server.py`), `--doctor` embeddings row (FAIL when the snapshot or the extra is missing), CLI actions `--embed` / `--similarity`, and REPL commands `/embed` / `/similar`.
- JSON-RPC server advertises `embeddings: true` in `runtimeCapabilities` when the provider mounts; semantic tools are reachable over `tools/list` + `tools/execute`.
- Verified against the real snapshot at `/Users/ohmskiii/all-MiniLM-L6-v2`: 384-dim vectors, similarity 0.566 (ML pair) vs 0.149 (distant) vs 1.0 (identical); doctor reports the resolved path; REPL `/tools` lists the three semantic tools; server `tools/execute` returns `{"similarity": 0.6916}` for the ML pair.
- 173 tests passing, 2 skipped; 0 pyright errors, 0 ruff warnings. Bumped to v0.4.0.

## Phase 10: Workspace Search Tools Integration (COMPLETED)
- Shipped `make_find_files_tool` and `make_grep_search_tool` in `src/glmharness/builtin_tools.py`.
- Auto-registered both tools in `BuiltinToolsPlugin.apply()`.
- Exported tool factories in `src/glmharness/__init__.py`.
- Added unit tests in `tests/test_search_tools.py` covering glob patterns, directory recursion, regex searching, case sensitivity, file filters, and boundary security.
- Full test suite: 187 passed, 2 skipped (100% pass rate). 0 ruff errors.

## Phase 11: HTTP / Web Document Reader Tool Integration (COMPLETED)
- Shipped `clean_html_to_markdown`, `is_blocked_host`, and `make_fetch_url_tool` in `src/glmharness/builtin_tools.py`.
- Implemented robust HTML parsing with tag stripping (`<script>`, `<style>`, `<nav>`, etc.) and Markdown conversion for headers, links, and code blocks.
- Implemented SSRF filtering for loopback, local, and private IPv4/IPv6 address blocks.
- Auto-registered `fetch_url` in `BuiltinToolsPlugin.apply()` with `enable_fetch_url` configuration.
- Exported tool factory and HTML cleaner from `src/glmharness/__init__.py`.
- Added unit tests in `tests/test_fetch_url.py` (7/7 passed).
- Full test suite: 194 passed, 2 skipped (100% pass rate). 0 ruff errors.

## Phase 12: Remote Path End-to-End Coverage & Version Single-Sourcing (COMPLETED)
- Added `tests/test_e2e_remote.py`: stands up a local OpenAI-compatible SSE stub on a real socket and drives the full CLI (config merge, plugin mount, skill discovery, agent loop, tool dispatch) — one-shot streaming plus a streamed tool call whose result is dispatched and fed back to the endpoint. Replaces the ad-hoc Phase 8 stub that was never committed, which left the supported real-model path without reproducible coverage.
- Verified the shipped binary end-to-end: `glm-harness --api-base http://127.0.0.1:8777/v1 --workspace <ws> 'list the workspace'` against the stub — 2 provider calls, 15 tools advertised, `list_dir` executed against the real workspace, tool result returned to the endpoint, exit 0.
- Fixed version drift: `pyproject.toml` declared `0.4.0` while `CHANGELOG` had already shipped `0.4.1` and `0.4.2`. The version now derives from `glmharness.__version__` through `[tool.hatch.version]` (single source), bumped to 0.4.3.
- Full test suite: 196 passed, 2 skipped (100% pass rate). 0 ruff errors, 0 pyright errors.

## Phase 13: GitHub REST Client & Tool Battery (COMPLETED)
- Shipped `src/glmharness/github.py` (~1.16k lines): zero-dependency GitHub REST provider (`GitHubOptions`, `GitHubProvider`) with an injectable transport seam and hardened inputs — repo slug / git ref / commit SHA validation, HTTPS-only API base (HTTP allowed only for loopback), clamped paging (1–100), 64K file / 8K patch truncation, and response-shape coercion.
- Shipped 18 model-facing tools: 13 read-only (`github_get_repo`, `github_list_issues`, `github_get_issue`, `github_list_issue_comments`, `github_get_file`, `github_list_pull_requests`, `github_get_pull_request`, `github_list_pull_request_files`, `github_list_branches`, `github_list_commits`, `github_get_commit`, `github_search_code`, `github_search_issues`) and 5 mutating (`github_write_file`, `github_create_issue`, `github_create_pull_request`, `github_create_branch`, `github_add_issue_comment`) gated by `SandboxPlugin` (included in the default mutating set).
- Wired CLI flags (`--github`, `--github-token`, `--github-api-base`), env vars (`GITHUB_TOKEN`, `GITHUB_REPO`, `GITHUB_API_BASE`), and auto-mount in `cli.py` when a repo scope or token is present; `doctor` reports the active GitHub scope and auth status.
- Added unit tests in `tests/test_github_tools.py` (8/8 passed).
- Fixed env loading: bare conventional names (`GITHUB_TOKEN`, `GITHUB_REPO`, `GITHUB_API_BASE`) are now honored by `HarnessConfig.from_env` alongside the `GLMH_GITHUB_*` forms; previously only `GLMH_GITHUB_*` reached the config layer even though `cli.py`/`doctor` read the bare names directly.
- Full test suite: 204 passed, 2 skipped (100% pass rate). 0 ruff errors, 0 pyright errors (`pyright src`, strict).
- Known nit: `GITHUB_MUTATING_TOOLS` in `github.py` duplicates the GitHub entries in `sandbox._MUTATING_TOOLS`; consolidate to a single source to avoid drift. → **Resolved**: `sandbox._MUTATING_TOOLS` is now composed as `{write_file, edit_file, bash} | GITHUB_MUTATING_TOOLS` (single source in `github.py`), with a drift-guard test in `tests/test_sandbox.py` asserting `GITHUB_MUTATING_TOOLS <= SandboxPlugin().mutating_tools`.

## Summary of Completed Phases
- **Phase 1**: Core Tool Battery & Execution Confinement.
- **Phase 2**: Remote / OpenAI-Compatible LLM Adapter.
- **Phase 3**: Interactive Carriers (REPL & JSON-RPC 2.0 stdio protocol server).
- **Phase 4**: Dynamic Multi-Harness (DMH) Integration & Replay / Token Compaction.
- **Phase 5**: Dynamic Skill Trigger & Tool Gating System.
- **Phase 6**: Stop-Slop Prose Quality Engine & Tool Integration.
- **Phase 7**: Local Snapshot Loader Fix (multimodal image-text-to-text stack).
- **Phase 8**: Weight-Presence Gate & Endpoint Verification.
- **Phase 9**: Semantic Embeddings Integration (all-MiniLM-L6-v2).
- **Phase 10**: Workspace Search Tools (`find_files` & `grep_search`).
- **Phase 11**: HTTP / Web Document Reader Tool (`fetch_url`).
- **Phase 12**: Remote Path End-to-End Coverage & Version Single-Sourcing.
- **Phase 13**: GitHub REST Client & Tool Battery (`github.py`, 18 tools, sandbox-gated mutations).








## Phase 14: Verdict Layer Wiring (COMPLETED)
- Wired `VerificationRun` into `ToolRegistry` (`src/glmharness/tools.py`): every
  executed call records a `Consequence(tool, expect, arguments)` and the folded
  verdict (`ok`/`error`/`unknown`) is exposed via `registry.verification_run`.
- The durable `tool/result` fact now carries `verdict` plus a flow-step audit
  trail built with `step_step_event` and the `pre-execute`/`post-execute` stage
  vocabulary from `src/glmharness/artifacts/flow_step_tool.py`.
- Exported `Consequence`, `VerificationRun`, and `judge` from the package root
  (`src/glmharness/__init__.py`).
- Added `tests/test_verdict.py` (6 tests): judge mapping, run folding,
  flow-step records, and registry integration for ok/error verdicts.
- Documented in `README-HARNESS.md` (Verdict layer section) and bumped to
  0.4.4 (single source: `src/glmharness/identity/brand.py`).
- Full test suite: 225 passed, 2 skipped; 0 ruff errors, 0 pyright errors.
