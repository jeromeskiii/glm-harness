"""Interactive Terminal REPL for GLM-5.3-Flash agent sessions."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from .builtin_tools import BuiltinToolsPlugin
from .config import HarnessConfig, resolve_model_path
from .context import Context, PluginLoader
from .llm import MockLLM, OpenAICompatibleGLM, TransformersGLM
from .logging import configure_logging
from .loop import AgentLoop
from .plugins import BasePlugin, SafetyPlugin
from .sandbox import SandboxPlugin
from .session import SessionLog
from .tools import ToolRegistry

_HELP_TEXT = """Available commands:
  /help       - Show this help message
  /tools      - List registered tools and descriptions
  /clear      - Reset current conversation history
  /session    - Show session metrics and persistence path
  /exit, /quit - Terminate the REPL session
"""


def _default_write(text: str) -> None:
    sys.stdout.write(text)


async def run_repl(
    config: HarnessConfig,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = _default_write,
) -> int:
    """Run an interactive conversational loop."""
    configure_logging(fmt=config.log_format, level=config.log_level)
    resolve_model_path(config)

    ctx = Context()
    sessions = SessionLog(path=config.session_path, corrupt_policy=config.corrupt_policy)
    tools = ToolRegistry(ctx, tool_timeout_s=config.tool_timeout_s)

    if config.mock is not None:
        llm: object = MockLLM(config.mock)
        provider_name = f"mock ({config.mock!r})"
    elif config.api_base is not None:
        llm = OpenAICompatibleGLM(
            api_base=config.api_base,
            api_key=config.api_key,
            model=config.model_name,
            max_new_tokens=config.max_new_tokens,
            timeout_s=config.request_timeout_s,
        )
        provider_name = f"openai-compatible ({config.api_base})"
    else:
        assert config.model_path is not None
        llm = TransformersGLM(
            config.model_path,
            reasoning_effort=config.reasoning_effort,
            max_new_tokens=config.max_new_tokens,
        )
        provider_name = f"local-transformers ({config.model_path.name})"

    ws = (config.workspace_dir or Path.cwd()).resolve()

    loader = PluginLoader(ctx)
    await loader.mount(
        [
            BasePlugin(sessions, tools),
            BuiltinToolsPlugin(config.workspace_dir),
            SandboxPlugin(mode=config.sandbox_mode),  # type: ignore[arg-type]
            SafetyPlugin(config.tool_allowlist),
        ]
    )

    agent = AgentLoop(
        ctx,
        llm,  # type: ignore[arg-type]
        sessions,
        tools,
        max_rounds=config.max_rounds,
        request_timeout_s=config.request_timeout_s,
        max_retries=config.max_retries,
        retry_base_delay_s=config.retry_base_delay_s,
        retry_max_delay_s=config.retry_max_delay_s,
        retry_jitter=config.retry_jitter,
    )

    output_func(
        f"GLM-5.3-Flash Agent REPL\n"
        f"Provider: {provider_name} | Workspace: {ws} | Sandbox: {config.sandbox_mode}\n"
        f"Type /help for commands, /exit to quit.\n\n"
    )

    try:
        while True:
            try:
                line = input_func("glm> ").strip()
            except (EOFError, KeyboardInterrupt):
                output_func("\nGoodbye!\n")
                break

            if not line:
                continue

            if line.startswith("/"):
                cmd = line.lower()
                if cmd in ("/exit", "/quit"):
                    output_func("Goodbye!\n")
                    break
                if cmd == "/help":
                    output_func(_HELP_TEXT + "\n")
                    continue
                if cmd == "/clear":
                    sessions = SessionLog(path=config.session_path, corrupt_policy=config.corrupt_policy)
                    agent.sessions = sessions
                    ctx.services["sessions"] = sessions
                    output_func("Conversation history cleared.\n\n")
                    continue
                if cmd == "/session":
                    path_str = str(config.session_path) if config.session_path else "in-memory"
                    output_func(f"Session path: {path_str} | Events recorded: {len(sessions.events)}\n\n")
                    continue
                if cmd == "/tools":
                    schemas = tools.schemas()
                    output_func(f"Registered tools ({len(schemas)}):\n")
                    for s in schemas:
                        fn = s.get("function", {})
                        output_func(f"  - {fn.get('name')}: {fn.get('description')}\n")
                    output_func("\n")
                    continue
                output_func(f"Unknown command: {line}. Type /help for assistance.\n\n")
                continue

            try:
                answer = await agent.run(line)
                output_func(f"{answer}\n\n")
            except Exception as exc:
                output_func(f"[error] {type(exc).__name__}: {exc}\n\n")

        return 0
    finally:
        await ctx.close()
