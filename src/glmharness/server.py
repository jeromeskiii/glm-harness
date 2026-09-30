"""JSON-RPC 2.0 stdio protocol server for GLM-5.3-Flash agent integration.

Speaks line-delimited JSON-RPC 2.0 over stdio for IDE extensions and
Dynamic Multi-Harness (DMH) host controllers.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TextIO, cast

from . import __version__
from .builtin_tools import BuiltinToolsPlugin
from .compaction import CompactionPlugin, compact_session
from .config import HarnessConfig, generate_rpc_token, resolve_embed_model_path, resolve_model_path
from .context import Context, PluginLoader
from .embeddings import EmbeddingProvider, EmbeddingsPlugin
from .github import GITHUB_MUTATING_TOOLS, GitHubOptions, GitHubPlugin, GitHubProvider
from .identity.brand import MODEL_VENDOR
from .llm import MockLLM, OpenAICompatibleGLM, TransformersGLM
from .logging import configure_logging, get_logger
from .loop import AgentLoop
from .plugins import BasePlugin, SafetyPlugin
from .sandbox import SandboxPlugin
from .session import SessionLog
from .skills import RiskLevel, SkillCatalog, SkillsPlugin
from .tools import ToolRegistry
from .wire.constants.constants import SERVER_CAPABILITIES

#: Tools that are reachable through ``tools/execute`` only when the operator
#: has explicitly enabled ``rpc_allow_mutating``. These are the tools that
#: would let a same-UID peer escalate to local code execution or remote
#: commits, so they are denied by default over RPC. The local CLI loop is
#: unaffected; this gate applies only to JSON-RPC requests.
_RPC_MUTATING_TOOLS: frozenset[str] = frozenset({
    "bash",
    "write_file",
    "edit_file",
}) | GITHUB_MUTATING_TOOLS

_SERVER_CAPABILITIES: dict[str, bool] = {
    **SERVER_CAPABILITIES,
    "runtime.multiSession": True,
}


class ProtocolServer:
    def __init__(
        self,
        config: HarnessConfig,
        reader: TextIO = sys.stdin,
        writer: TextIO = sys.stdout,
    ):
        self.config = config
        self.reader = reader
        self.writer = writer
        self.running = True
        self.sessions: dict[str, SessionLog] = {}
        self.ctx: Context | None = None
        self.tools: ToolRegistry | None = None
        self.llm: Any = None
        self.embedding_path: Path | None = None
        self.github_mounted: bool = False
        # When ``rpc_token`` is unset and ``rpc_auto_token`` is true (the
        # default), the server generates a per-process bearer token via the
        # OS CSPRNG and prints it to stderr exactly once. Stdio is loopback
        # but a same-UID peer can still write to the fd, so authentication
        # is no longer opt-in. To deliberately run unauthenticated, set
        # ``rpc_token=""`` and ``rpc_auto_token=False`` together.
        if config.rpc_token is None and config.rpc_auto_token:
            self._auto_token: str | None = generate_rpc_token()
        else:
            self._auto_token = None
        # ``initialize`` flips this once; the configured (or auto-generated)
        # token, if any, was verified against the request at that moment.
        self._auth_ok: bool = (config.rpc_token is None and not config.rpc_auto_token)

    def _write_frame(self, frame: dict[str, Any]) -> None:
        raw = json.dumps(frame, ensure_ascii=False)
        self.writer.write(raw + "\n")
        self.writer.flush()

    def _success(self, req_id: Any, result: Any) -> None:
        self._write_frame({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _error(self, req_id: Any, code: int, message: str, data: Any = None) -> None:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._write_frame({"jsonrpc": "2.0", "id": req_id, "error": err})

    async def initialize_runtime(self) -> None:
        resolve_model_path(self.config)
        self.ctx = Context()
        initial_log = SessionLog(path=self.config.session_path, corrupt_policy=self.config.corrupt_policy)
        default_session_id = "default"
        self.sessions[default_session_id] = initial_log

        self.tools = ToolRegistry(self.ctx, tool_timeout_s=self.config.tool_timeout_s)

        if self.config.mock is not None:
            self.llm = MockLLM(self.config.mock)
        elif self.config.api_base is not None:
            self.llm = OpenAICompatibleGLM(
                api_base=self.config.api_base,
                api_key=self.config.api_key,
                model=self.config.model_name,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                timeout_s=self.config.request_timeout_s,
            )
        else:
            assert self.config.model_path is not None
            self.llm = TransformersGLM(
                self.config.model_path,
                reasoning_effort=self.config.reasoning_effort,
                max_new_tokens=self.config.max_new_tokens,
            )

        loader = PluginLoader(self.ctx)
        plugins: list[Any] = [
            BasePlugin(initial_log, self.tools),
            BuiltinToolsPlugin(
                self.config.workspace_dir,
                enable_bash=self.config.enable_bash,
                enable_fetch_url=self.config.enable_fetch_url,
            ),
            SandboxPlugin(mode=self.config.sandbox_mode),  # type: ignore[arg-type]
            SafetyPlugin(self.config.tool_allowlist),
            CompactionPlugin(
                threshold=self.config.compaction_threshold,
                keep_rounds=self.config.compaction_keep_rounds,
                strategy=self.config.compaction_strategy,
            ),
            SkillsPlugin(
                skills_dir=self.config.skills_dir,
                gate_tools=self.config.skill_gate_tools,
                default_risk=RiskLevel.from_str(self.config.task_risk),
            ),
        ]
        self.embedding_path = resolve_embed_model_path(self.config)
        if self.embedding_path is not None:
            plugins.append(EmbeddingsPlugin(EmbeddingProvider(self.embedding_path)))
        gh_token = self.config.github_token or os.environ.get("GITHUB_TOKEN")
        self.github_mounted = False
        if self.config.github_repo or gh_token:
            owner, repo = None, None
            if self.config.github_repo and "/" in self.config.github_repo:
                parts = self.config.github_repo.split("/", 1)
                owner, repo = parts[0], parts[1]
            gh_opts = GitHubOptions(
                token=gh_token,
                api_base=self.config.github_api_base or "https://api.github.com",
                owner=owner,
                repo=repo,
            )
            plugins.append(GitHubPlugin(GitHubProvider(gh_opts)))
            self.github_mounted = True
        await loader.mount(plugins)

    def _register_session(self, session_id: str, log: SessionLog) -> None:
        """Insert a session, evicting the oldest non-default one when the
        in-memory cache exceeds ``max_sessions``.

        Server memory must stay bounded for months-long stdio hosting; only
        the ``default`` session is persisted, so dynamic sessions are
        RAM-only and safe to drop (with a warning) under pressure.
        """
        self.sessions[session_id] = log
        max_sessions = max(1, int(self.config.max_sessions))
        while len(self.sessions) > max_sessions:
            evict = next(
                (sid for sid in self.sessions if sid != "default" and sid != session_id),
                None,
            )
            if evict is None:
                break
            del self.sessions[evict]
            get_logger().warning(
                "session evicted (max_sessions reached)",
                extra={"sessionId": evict},
            )

    @property
    def auto_token(self) -> str | None:
        """The per-process auto-generated bearer token, or ``None``.

        ``None`` when the operator supplies ``rpc_token`` explicitly or
        disabled auto tokens with ``rpc_auto_token=False``.
        """
        return self._auto_token

    @property
    def _effective_rpc_token(self) -> str | None:
        """The bearer token required by ``initialize``.

        Prefers the operator-supplied ``rpc_token``; falls back to the
        per-process auto-generated token when ``rpc_auto_token`` is enabled.
        ``None`` only when the operator has explicitly opted out of auth
        (``rpc_token is None`` and ``rpc_auto_token is False``).
        """
        if self.config.rpc_token is not None:
            return self.config.rpc_token
        return self._auto_token

    def _require_state_relative(self, raw: str) -> Path:
        """Resolve a caller-supplied path against ``state_dir``.

        Rejects absolute paths and any path that escapes ``state_dir`` after
        resolution. Without this, an authenticated peer could ask the server
        to ingest ``/etc/passwd`` or any file the operator can read.

        ``state_dir`` is typed as ``Path`` but dataclasses do not coerce at
        assignment time, so a caller constructing ``HarnessConfig`` directly
        with a string leaves a ``str`` here. We coerce defensively so an
        authenticated peer cannot crash the server with ``AttributeError``
        from an operator-side mis-configuration.
        """
        if self.config.state_dir is None:
            raise PermissionError(
                "logPath imports require state_dir to be configured on the server"
            )
        state_root = Path(self.config.state_dir).resolve()
        candidate = Path(raw)
        if candidate.is_absolute():
            raise PermissionError(
                f"logPath must be relative to state_dir ({state_root}); got absolute path"
            )
        resolved = (state_root / candidate).resolve()
        if resolved != state_root and state_root not in resolved.parents:
            raise PermissionError(
                f"logPath '{raw}' escapes state_dir ({state_root})"
            )
        return resolved

    async def handle_request(self, frame: dict[str, Any]) -> None:
        req_id: object = frame.get("id")
        method = frame.get("method")
        params_raw: object = frame.get("params")
        params: dict[str, Any] = cast(dict[str, Any], params_raw) if isinstance(params_raw, dict) else {}

        if not isinstance(method, str):
            self._error(req_id, -32600, "Invalid Request: method must be string")
            return

        # Auth gate: when a token is configured (or auto-generated), every
        # method except ``initialize`` (which itself authenticates the peer)
        # is rejected until a successful ``initialize`` was processed.
        effective_token = self._effective_rpc_token
        if effective_token is not None and not self._auth_ok and method != "initialize":
            self._error(req_id, -32001, "Unauthorized: send initialize first")
            return

        if method == "initialize":
            expected = effective_token
            if expected is not None:
                header_token = str(params.get("authToken", ""))
                # hmac.compare_digest keeps the comparison constant-time so a
                # local timing probe cannot enumerate the configured token.
                if not header_token or not hmac.compare_digest(header_token, expected):
                    self._error(req_id, -32001, "Unauthorized: authToken missing or wrong")
                    return
                self._auth_ok = True
            capabilities = dict(_SERVER_CAPABILITIES)
            capabilities["embeddings"] = self.embedding_path is not None
            capabilities["github"] = self.github_mounted
            capabilities["auth"] = expected is not None
            capabilities["rpcAllowMutating"] = self.config.rpc_allow_mutating
            self._success(
                req_id,
                {
                    "abiVersion": 2,
                    "runtimeInfo": {
                        "name": "glm-5.3-flash",
                        "version": __version__,
                        "vendor": MODEL_VENDOR,
                        "protocol": "glm-jsonrpc-stdio",
                    },
                    "runtimeCapabilities": capabilities,
                    "authMethods": [],
                },
            )
            return

        if method == "ping":
            self._success(req_id, {"ok": True, "timestamp": time.time()})
            return

        if method == "session/list":
            self._success(req_id, list(self.sessions.keys()))
            return

        if method == "session/new":
            new_id = str(params.get("sessionId") or f"session-{uuid.uuid4().hex[:8]}")
            s_log = SessionLog()
            proj_new: object = params.get("projection")
            if isinstance(proj_new, list):
                s_log.import_projection(cast(list[dict[str, Any]], proj_new))
            self._register_session(new_id, s_log)
            self._success(req_id, {"sessionId": new_id})
            return

        if method == "session/import":
            session_id = str(params.get("sessionId", "default"))
            session_log = self.sessions.get(session_id)
            if session_log is None:
                session_log = SessionLog()
                self._register_session(session_id, session_log)
            imported = 0
            proj_raw: object = params.get("projection")
            if isinstance(proj_raw, list):
                imported += session_log.import_projection(cast(list[dict[str, Any]], proj_raw))
            log_path_raw: object = params.get("logPath")
            if isinstance(log_path_raw, str):
                try:
                    safe_path = self._require_state_relative(log_path_raw)
                except PermissionError as exc:
                    self._error(req_id, -32003, f"Forbidden: {exc}")
                    return
                imported += session_log.import_log(safe_path)
            self._success(req_id, {"sessionId": session_id, "imported": imported})
            return

        if method == "session/compact":
            session_id = str(params.get("sessionId", "default"))
            session_log = self.sessions.get(session_id)
            if session_log is None:
                self._error(req_id, -32602, f"Session not found: {session_id}")
                return
            threshold = int(params.get("threshold", 4000))
            keep_rounds = int(params.get("keepRounds", 4))
            strategy = str(params.get("strategy", "summarize"))
            res = compact_session(
                session_log,
                threshold=threshold,
                keep_rounds=keep_rounds,
                strategy=strategy,
            )
            self._success(req_id, res)
            return

        if method == "tools/list":
            assert self.tools is not None
            self._success(req_id, self.tools.schemas())
            return

        if method == "tools/execute":
            assert self.tools is not None
            name = str(params.get("name", ""))
            if (
                name in _RPC_MUTATING_TOOLS
                and not self.config.rpc_allow_mutating
            ):
                self._error(
                    req_id,
                    -32004,
                    (
                        f"Forbidden: tool '{name}' is mutating and "
                        "rpc_allow_mutating is disabled"
                    ),
                )
                return
            args = cast(dict[str, Any], params.get("arguments") or {})
            res = await self.tools.execute(name, args)
            self._success(req_id, res)
            return

        if method == "agent/send":
            session_id = str(params.get("sessionId", "default"))
            text = str(params.get("text", ""))
            if not text:
                self._error(req_id, -32602, "Invalid params: 'text' is required")
                return

            session_log = self.sessions.get(session_id)
            if session_log is None:
                session_log = SessionLog()
                self._register_session(session_id, session_log)

            proj_raw: object = params.get("projection")
            if isinstance(proj_raw, list) and not session_log.events:
                session_log.import_projection(cast(list[dict[str, Any]], proj_raw))

            assert self.ctx is not None
            assert self.tools is not None
            # Scoped override: the tool pipeline, skills, and compaction all
            # resolve the *current* session via the ``sessions`` service
            # (tools.py, skills.py, compaction.py). Restore the previous value
            # in ``finally`` so the slot never holds a stale session after the
            # request completes.
            prev_sessions = self.ctx.services.get("sessions")
            self.ctx.services["sessions"] = session_log

            agent = AgentLoop(
                self.ctx,
                self.llm,
                session_log,
                self.tools,
                max_rounds=self.config.max_rounds,
                request_timeout_s=self.config.request_timeout_s,
                max_retries=self.config.max_retries,
                retry_base_delay_s=self.config.retry_base_delay_s,
                retry_max_delay_s=self.config.retry_max_delay_s,
                retry_jitter=self.config.retry_jitter,
            )

            turn_ctx = (
                asyncio.timeout(self.config.turn_timeout_s)
                if self.config.turn_timeout_s > 0
                else nullcontext()
            )
            try:
                try:
                    async with turn_ctx:
                        answer = await agent.run(text)
                except TimeoutError:
                    # Keep the session log coherent: cancellation closes the
                    # turn with a durable failed marker (loop contract).
                    session_log.append(
                        "turn/end", {"status": "failed", "error": "TURN_TIMEOUT"}
                    )
                    self._error(
                        req_id,
                        -32008,
                        f"Agent turn timed out after {self.config.turn_timeout_s}s",
                        data={"sessionId": session_id},
                    )
                    return
                self._success(
                    req_id,
                    {
                        "ok": True,
                        "answer": answer,
                        "sessionId": session_id,
                        "eventCount": len(session_log.events),
                    },
                )
            except Exception as exc:
                self._error(
                    req_id,
                    -32603,
                    f"Agent execution failed: {type(exc).__name__}: {exc}",
                    data={"sessionId": session_id},
                )
            finally:
                if prev_sessions is None:
                    self.ctx.services.pop("sessions", None)
                else:
                    self.ctx.services["sessions"] = prev_sessions
            return

        if method == "skills/list":
            if self.ctx is None or "skills" not in self.ctx.services:
                self._error(req_id, -32603, "Skills subsystem not initialized")
                return
            catalog = cast(SkillCatalog, self.ctx.get("skills"))
            skills_data = [
                {
                    "name": s.name,
                    "description": s.description,
                    "triggers": sorted(s.triggers),
                    "tools": sorted(s.tools),
                    "capabilities": sorted(s.capabilities),
                    "tags": sorted(s.tags),
                    "always": s.always,
                    "min_risk": s.min_risk.name,
                    "max_risk": s.max_risk.name,
                }
                for s in catalog.all_skills()
            ]
            self._success(req_id, {"skills": skills_data, "gateTools": catalog.gate_tools})
            return

        if method == "skills/match":
            if self.ctx is None or "skills" not in self.ctx.services:
                self._error(req_id, -32603, "Skills subsystem not initialized")
                return
            objective = str(params.get("objective", ""))
            risk_str = str(params.get("risk", self.config.task_risk))
            risk_val = RiskLevel.from_str(risk_str)
            caps_raw = params.get("capabilities", [])
            caps = (
                frozenset(str(c) for c in cast(list[object], caps_raw) if c)
                if isinstance(caps_raw, list)
                else frozenset[str]()
            )
            catalog = cast(SkillCatalog, self.ctx.get("skills"))
            matches = catalog.match(objective=objective, capabilities=caps, risk=risk_val)
            res = [
                {
                    "name": s.name,
                    "score": score,
                    "tools": sorted(s.tools),
                    "triggers": sorted(s.triggers),
                }
                for s, score in matches
            ]
            self._success(req_id, {"matches": res})
            return

        if method == "skills/import":
            if self.ctx is None or "skills" not in self.ctx.services:
                self._error(req_id, -32603, "Skills subsystem not initialized")
                return
            dir_str = str(params.get("dir", ""))
            target_dir = Path(dir_str)
            loop = asyncio.get_running_loop()
            is_dir = await loop.run_in_executor(None, target_dir.is_dir)
            if not is_dir:
                self._error(req_id, -32602, f"Invalid params: directory not found: {dir_str}")
                return
            catalog = cast(SkillCatalog, self.ctx.get("skills"))
            count = await loop.run_in_executor(None, catalog.import_from_dir, target_dir)
            self._success(req_id, {"imported": count, "total": len(catalog.all_skills())})
            return

        if method == "shutdown":
            self.running = False
            self._success(req_id, {"ok": True})
            return

        self._error(req_id, -32601, f"Method not found: {method}")

    async def serve(self) -> int:
        await self.initialize_runtime()
        logger = get_logger()
        logger.info("protocol server ready")

        loop = asyncio.get_running_loop()

        while self.running:
            # Read line asynchronously in thread
            line = await loop.run_in_executor(None, self.reader.readline)
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue
            try:
                raw_json: object = json.loads(line_str)
            except json.JSONDecodeError as exc:
                self._error(None, -32700, f"Parse error: {exc}")
                continue

            if not isinstance(raw_json, dict):
                self._error(None, -32600, "Invalid Request: must be JSON object")
                continue

            frame_dict = cast(dict[str, Any], raw_json)
            req_id: object = frame_dict.get("id")
            if frame_dict.get("jsonrpc") != "2.0":
                self._error(req_id, -32600, "Invalid Request: must be JSON-RPC 2.0 object")
                continue

            await self.handle_request(frame_dict)

        if self.ctx is not None:
            await self.ctx.close()
        return 0


async def run_server(config: HarnessConfig) -> int:
    configure_logging(fmt=config.log_format, level=config.log_level)
    server = ProtocolServer(config)
    # When the server is running with an auto-generated bearer token, print
    # it to stderr exactly once so the operator can hand it to the peer.
    # We use stderr (not stdout) so the JSON-RPC framing on stdout stays
    # parseable for the peer.
    auto_token = server.auto_token
    if auto_token is not None:
        sys.stderr.write(
            f"[glmharness] auto-generated RPC token: {auto_token}\n"
        )
        sys.stderr.flush()
    return await server.serve()
