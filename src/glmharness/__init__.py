"""GLM-5.3-Flash local agent harness.

The kernel exposes three primitives — :class:`~glmharness.bus.EventBus`,
:class:`~glmharness.context.Context`, and
:class:`~glmharness.context.PluginLoader` — and everything else (session
log, tool registry, loop, LLM adapter) is a service behind them. Add new
capabilities by writing a :class:`~glmharness.context.Plugin`; never edit
the kernel.
"""

from __future__ import annotations

__version__ = "0.4.3"

from .builtin_tools import (
    BuiltinToolsPlugin,
    clean_html_to_markdown,
    make_fetch_url_tool,
    make_find_files_tool,
    make_grep_search_tool,
    resolve_safe_path,
)
from .bus import EventBus
from .compaction import CompactionPlugin, compact_session
from .config import (
    HarnessConfig,
    generate_rpc_token,
    looks_like_snapshot,
    resolve_embed_model_path,
    resolve_model_path,
)
from .context import Context, Plugin, PluginLoader
from .embeddings import (
    EmbeddingProvider,
    EmbeddingsPlugin,
    cosine_similarity,
    make_embed_text_tool,
    make_semantic_rank_tool,
    make_semantic_similarity_tool,
)
from .errors import (
    ConfigError,
    HarnessError,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    SessionCorruptError,
    ToolError,
)
from .github import (
    GitHubOptions,
    GitHubPlugin,
    GitHubProvider,
    make_github_tools,
)
from .llm import LLM, MockLLM, OpenAICompatibleGLM, TransformersGLM
from .loop import AgentLoop
from .plugins import BasePlugin, SafetyPlugin
from .repl import run_repl
from .sandbox import SandboxMode, SandboxPlugin
from .server import ProtocolServer, run_server
from .session import SessionEvent, SessionLog
from .skills import (
    RiskLevel,
    Skill,
    SkillCatalog,
    SkillsPlugin,
    default_skills,
    import_skills_from_dir,
)
from .stop_slop import (
    StopSlopEngine,
    make_stop_slop_analyze_tool,
    make_stop_slop_examples_tool,
    make_stop_slop_rewrite_tool,
    make_stop_slop_rules_tool,
)
from .tools import Tool, ToolRegistry, parse_tool_calls, validate_tool_arguments

__all__ = [
    "LLM",
    "AgentLoop",
    "BasePlugin",
    "BuiltinToolsPlugin",
    "CompactionPlugin",
    "ConfigError",
    "Context",
    "EmbeddingProvider",
    "EmbeddingsPlugin",
    "EventBus",
    "GitHubOptions",
    "GitHubPlugin",
    "GitHubProvider",
    "HarnessConfig",
    "HarnessError",
    "MockLLM",
    "OpenAICompatibleGLM",
    "Plugin",
    "PluginLoader",
    "ProtocolServer",
    "ProviderError",
    "ProviderTimeout",
    "ProviderUnavailable",
    "RiskLevel",
    "SafetyPlugin",
    "SandboxMode",
    "SandboxPlugin",
    "SessionCorruptError",
    "SessionEvent",
    "SessionLog",
    "Skill",
    "SkillCatalog",
    "SkillsPlugin",
    "StopSlopEngine",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "TransformersGLM",
    "__version__",
    "clean_html_to_markdown",
    "compact_session",
    "cosine_similarity",
    "default_skills",
    "generate_rpc_token",
    "import_skills_from_dir",
    "looks_like_snapshot",
    "make_embed_text_tool",
    "make_fetch_url_tool",
    "make_find_files_tool",
    "make_github_tools",
    "make_grep_search_tool",
    "make_semantic_rank_tool",
    "make_semantic_similarity_tool",
    "make_stop_slop_analyze_tool",
    "make_stop_slop_examples_tool",
    "make_stop_slop_rewrite_tool",
    "make_stop_slop_rules_tool",
    "parse_tool_calls",
    "resolve_embed_model_path",
    "resolve_model_path",
    "resolve_safe_path",
    "run_repl",
    "run_server",
    "validate_tool_arguments",
]
