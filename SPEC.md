# SPEC: GLM-5.3-Flash Harness

## Phase 1: Core Tool Battery & Execution Confinement (COMPLETED)
- Built-in filesystem tools (`read_file`, `write_file`, `edit_file`, `list_dir`) and shell tool (`bash`).
- Path confinement via `resolve_safe_path`.
- Sandbox execution policies (`allow`, `deny`, `ask`).

---

## Phase 2: Remote / OpenAI-Compatible LLM Adapter (`OpenAICompatibleGLM`)

### Goal
Decouple `glmharness` from local PyTorch/Transformers weight loading by adding a zero-dependency, streaming OpenAI-compatible client adapter. This enables connecting to vLLM, SGLang, Ollama, TokenSpeed, or remote GLM API endpoints via HTTP/SSE.

### Architecture & Components
1. **`OpenAICompatibleGLM` in `src/glmharness/llm.py`**:
   - Implements `LLM` protocol: `stream(messages, tools) -> AsyncIterator[str]`.
   - Uses standard library `urllib.request` in a daemon background feeder thread posting lines to `asyncio.Queue` (zero external dependencies).
   - Handles Server-Sent Events (`data: {...}`, `data: [DONE]`).
   - Streams text chunks from `delta.content`.
   - Accumulates structured `delta.tool_calls` (if the remote server emits OpenAI tool-call format) and formats them into `<tool_call>name<arg_key>...` XML blocks so `AgentLoop` and `parse_tool_calls` consume them uniformly.
   - Centralizes error handling into `ProviderError` with proper `retryable` markings (e.g. 429, 500, 502, 503, 504 are retryable; 400, 401, 404 fail fast).
2. **Configuration & CLI**:
   - `HarnessConfig`:
     - `api_base: str | None = None` (`GLMH_API_BASE`, `--api-base`)
     - `api_key: str | None = None` (`GLMH_API_KEY`, `--api-key`)
     - `model_name: str = "GLM-5.3-Flash"` (`GLMH_MODEL`, `--model`)
   - Precedence:
     - If `--mock` / `GLMH_MOCK`: `MockLLM`.
     - Else if `api_base` / `GLMH_API_BASE`: `OpenAICompatibleGLM`.
     - Else: `TransformersGLM` (requires snapshot directory).
   - `--doctor` reports `provider: openai-compatible (url, model)` when configured.
3. **Verification Gate**:
   - Unit tests mocking HTTP SSE responses (streaming text, streaming tool calls, HTTP errors, connection drops, retries).
   - CLI flags verification (`--api-base`, `--api-key`, `--model`).
   - 100% pytest pass rate, pyright strict 0 errors, ruff clean.
