# Project State: GLM-5.3-Flash Harness

## Phase 1: Core Tool Battery & Execution Confinement (COMPLETED)
- Shipped `builtin_tools.py` with `read_file`, `write_file`, `edit_file`, `list_dir`, `bash`.
- Shipped `sandbox.py` with `SandboxPlugin` (`allow`, `deny`, `ask`).
- 108 tests passing, strict type checking passing.

## Current Phase: Phase 2 — Remote / OpenAI-Compatible LLM Adapter
- **Status**: COMPLETED
- **SPEC**: `SPEC.md` (Phase 2 FINALIZED)

### Tasks
- [x] Task 1: Implement `OpenAICompatibleGLM` in `src/glmharness/llm.py` with streaming HTTP/SSE and tool call translation.
- [x] Task 2: Wire `api_base`, `api_key`, `model_name` into `HarnessConfig` and `glmharness/cli.py`.
- [x] Task 3: Add comprehensive unit tests in `tests/test_llm_adapters.py` (text streaming, tool-calls streaming, error handling).
- [x] Task 4: Empirical verification (`pytest` 115 passed, `pyright` strict 0 errors, `ruff` clean).
