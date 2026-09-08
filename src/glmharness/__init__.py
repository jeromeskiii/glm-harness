"""GLM-5.3-Flash local agent harness.

The kernel exposes three primitives — :class:`~glmharness.bus.EventBus`,
:class:`~glmharness.context.Context`, and
:class:`~glmharness.context.PluginLoader` — and everything else (session
log, tool registry, loop, LLM adapter) is a service behind them. Add new
capabilities by writing a :class:`~glmharness.context.Plugin`; never edit
the kernel.
"""

from __future__ import annotations

from .bus import EventBus
from .config import HarnessConfig, looks_like_snapshot, resolve_model_path
from .context import Context, Plugin, PluginLoader
from .errors import (
    ConfigError,
    HarnessError,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    SessionCorruptError,
    ToolError,
)
from .llm import LLM, MockLLM, TransformersGLM
from .loop import AgentLoop
from .plugins import BasePlugin, SafetyPlugin
from .session import SessionEvent, SessionLog
from .tools import Tool, ToolRegistry, parse_tool_calls, validate_tool_arguments

__version__ = "0.3.2"

__all__ = [
    "LLM",
    "AgentLoop",
    "BasePlugin",
    "ConfigError",
    "Context",
    "EventBus",
    "HarnessConfig",
    "HarnessError",
    "MockLLM",
    "Plugin",
    "PluginLoader",
    "ProviderError",
    "ProviderTimeout",
    "ProviderUnavailable",
    "SafetyPlugin",
    "SessionCorruptError",
    "SessionEvent",
    "SessionLog",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "TransformersGLM",
    "__version__",
    "looks_like_snapshot",
    "parse_tool_calls",
    "resolve_model_path",
    "validate_tool_arguments",
]
