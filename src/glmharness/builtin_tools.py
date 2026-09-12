"""Standard built-in tools for filesystem, search, HTTP fetch, and subprocess execution.

Provides ``read_file``, ``write_file``, ``edit_file``, ``list_dir``, ``find_files``,
``grep_search``, ``fetch_url``, and ``bash`` guarded by workspace and SSRF boundary checks.
"""

from __future__ import annotations

import asyncio
import fnmatch
import html
import ipaddress
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .context import Context
from .stop_slop import (
    make_stop_slop_analyze_tool,
    make_stop_slop_examples_tool,
    make_stop_slop_rewrite_tool,
    make_stop_slop_rules_tool,
)
from .tools import Tool, ToolRegistry

_MAX_OUTPUT_BYTES = 50_000
_DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
#: Cap a single ``read_file`` fetch at 10 MiB so a malicious or accidental
#: very large file cannot exhaust memory before slicing. The slice is still
#: bounded by ``limit`` lines, but the initial read was unbounded.
_READ_FILE_MAX_BYTES = 10 * 1024 * 1024


def resolve_safe_path(base: Path, relative_or_absolute: str | Path) -> Path:
    """Resolve a path and ensure it does not escape ``base``.

    Raises:
        PermissionError: if the resolved path is outside ``base``.
    """
    base_resolved = base.resolve()
    target_path = Path(relative_or_absolute)
    if target_path.is_absolute():
        resolved = target_path.resolve()
    else:
        resolved = (base_resolved / target_path).resolve()

    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise PermissionError(f"access denied: '{relative_or_absolute}' escapes workspace boundary")
    return resolved


def make_read_file_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative or absolute file path within workspace"},
            "offset": {"type": "integer", "description": "Starting line number (0-indexed, default 0)"},
            "limit": {"type": "integer", "description": "Maximum number of lines to return (default 2000)"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_safe_path(workspace, str(args["path"]))
        if not target.is_file():
            raise FileNotFoundError(f"file not found: {args['path']}")
        offset = int(args.get("offset", 0))
        limit = int(args.get("limit", 2000))
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise FileNotFoundError(f"file not readable: {args['path']}") from exc
        if size > _READ_FILE_MAX_BYTES:
            raise ValueError(
                f"file too large to read in one call: {size} bytes > {_READ_FILE_MAX_BYTES} bytes"
            )
        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        sliced = lines[offset : offset + limit]
        truncated = (offset + len(sliced)) < total_lines
        return {
            "path": str(target.relative_to(workspace.resolve())),
            "content": "".join(sliced),
            "lines": len(sliced),
            "total_lines": total_lines,
            "truncated": truncated,
        }

    return Tool(
        name="read_file",
        description="Read lines from a file within the workspace boundary.",
        schema=schema,
        handler=handler,
    )


def make_write_file_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path within workspace"},
            "content": {"type": "string", "description": "Full file content to write"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_safe_path(workspace, str(args["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        target.write_text(content, encoding="utf-8")
        return {
            "path": str(target.relative_to(workspace.resolve())),
            "bytes_written": len(content.encode("utf-8")),
        }

    return Tool(
        name="write_file",
        description="Create or overwrite a file within the workspace boundary.",
        schema=schema,
        handler=handler,
    )


def make_edit_file_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path within workspace"},
            "old_string": {"type": "string", "description": "Exact text to find and replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target = resolve_safe_path(workspace, str(args["path"]))
        if not target.is_file():
            raise FileNotFoundError(f"file not found: {args['path']}")
        content = target.read_text(encoding="utf-8")
        old_string = str(args["old_string"])
        new_string = str(args["new_string"])
        count = content.count(old_string)
        if count == 0:
            raise ValueError(f"target text not found in {args['path']}")
        if count > 1:
            raise ValueError(f"target text matches {count} times in {args['path']}; must be unique")
        updated = content.replace(old_string, new_string, 1)
        target.write_text(updated, encoding="utf-8")
        return {
            "path": str(target.relative_to(workspace.resolve())),
            "replacements": 1,
        }

    return Tool(
        name="edit_file",
        description="Replace a unique string occurrence in a file within the workspace.",
        schema=schema,
        handler=handler,
    )


def make_list_dir_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path within workspace (default: '.')"},
        },
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target_str = str(args.get("path", "."))
        target = resolve_safe_path(workspace, target_str)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {target_str}")
        entries: list[dict[str, Any]] = []
        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            entries.append(
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )
        rel = "." if target == workspace.resolve() else str(target.relative_to(workspace.resolve()))
        return {
            "path": rel,
            "entries": entries,
        }

    return Tool(
        name="list_dir",
        description="List contents of a directory within the workspace boundary.",
        schema=schema,
        handler=handler,
    )


def make_bash_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout_s": {"type": "number", "description": "Execution timeout in seconds (default: 30)"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    cwd_str = str(workspace.resolve())

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        command = str(args["command"])
        timeout_s = float(args.get("timeout_s", 30.0))
        if timeout_s < 0:
            raise ValueError("timeout_s must be >= 0")
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s if timeout_s > 0 else None
            )
        except TimeoutError:
            timed_out = True
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            stdout_bytes, stderr_bytes = b"", b"execution timed out"

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_BYTES]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_BYTES]
        return {
            "exit_code": proc.returncode if not timed_out else -1,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
        }

    return Tool(
        name="bash",
        description="Execute a shell command inside the workspace directory.",
        schema=schema,
        handler=handler,
    )


def make_find_files_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Starting directory within workspace (default: '.')"},
            "pattern": {"type": "string", "description": "Glob pattern to match (default: '*')"},
            "file_type": {
                "type": "string",
                "enum": ["file", "dir", "any"],
                "description": "Filter by entry type ('file', 'dir', 'any', default: 'any')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 100)",
            },
        },
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        target_str = str(args.get("path", "."))
        target = resolve_safe_path(workspace, target_str)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {target_str}")

        pattern = str(args.get("pattern", "*"))
        file_type = str(args.get("file_type", "any"))
        max_results = max(1, int(args.get("max_results", 100)))

        entries: list[dict[str, Any]] = []
        workspace_resolved = workspace.resolve()

        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if d not in _DEFAULT_IGNORED_DIRS]
            root_path = Path(root)

            if file_type in ("dir", "any"):
                for d in dirs:
                    d_path = root_path / d
                    rel_to_target = str(d_path.relative_to(target))
                    if fnmatch.fnmatch(d, pattern) or fnmatch.fnmatch(rel_to_target, pattern):
                        entries.append(
                            {
                                "path": str(d_path.relative_to(workspace_resolved)),
                                "type": "dir",
                                "size": None,
                            }
                        )

            if file_type in ("file", "any"):
                for f in files:
                    f_path = root_path / f
                    rel_to_target = str(f_path.relative_to(target))
                    if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(rel_to_target, pattern):
                        try:
                            size = f_path.stat().st_size
                        except OSError:
                            size = None
                        entries.append(
                            {
                                "path": str(f_path.relative_to(workspace_resolved)),
                                "type": "file",
                                "size": size,
                            }
                        )

        entries.sort(key=lambda x: x["path"].lower())
        total_matches = len(entries)
        truncated = total_matches > max_results
        sliced = entries[:max_results]

        rel = "." if target == workspace_resolved else str(target.relative_to(workspace_resolved))
        return {
            "path": rel,
            "pattern": pattern,
            "entries": sliced,
            "total_matches": total_matches,
            "truncated": truncated,
        }

    return Tool(
        name="find_files",
        description="Find files and directories matching a glob pattern within the workspace boundary.",
        schema=schema,
        handler=handler,
    )


def make_grep_search_tool(workspace: Path) -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text substring or regex pattern to search for"},
            "path": {
                "type": "string",
                "description": "Directory or file path within workspace to search (default: '.')",
            },
            "is_regex": {
                "type": "boolean",
                "description": "Treat query as a regular expression (default: false)",
            },
            "case_sensitive": {"type": "boolean", "description": "Case-sensitive matching (default: true)"},
            "include_pattern": {
                "type": "string",
                "description": "Glob pattern for files to include (e.g. '*.py', default: '*')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (default: 100)",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        target_str = str(args.get("path", "."))
        target = resolve_safe_path(workspace, target_str)
        is_regex = bool(args.get("is_regex", False))
        case_sensitive = bool(args.get("case_sensitive", True))
        include_pattern = str(args.get("include_pattern", "*"))
        max_results = max(1, int(args.get("max_results", 100)))

        flags = 0 if case_sensitive else re.IGNORECASE
        pattern_str = query if is_regex else re.escape(query)
        try:
            regex = re.compile(pattern_str, flags)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc

        workspace_resolved = workspace.resolve()
        matches: list[dict[str, Any]] = []
        truncated = False

        def search_single_file(file_path: Path) -> None:
            nonlocal truncated
            if truncated:
                return
            try:
                st = file_path.stat()
                if st.st_size > _READ_FILE_MAX_BYTES:
                    return
                with open(file_path, "rb") as bf:
                    chunk = bf.read(1024)
                    if b"\x00" in chunk:
                        return
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return

            rel_path = str(file_path.relative_to(workspace_resolved))
            for line_idx, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "path": rel_path,
                            "line_number": line_idx,
                            "line": line.strip(),
                        }
                    )
                    if len(matches) >= max_results:
                        truncated = True
                        return

        if target.is_file():
            search_single_file(target)
        elif target.is_dir():
            for root, dirs, files in os.walk(target):
                dirs[:] = [d for d in dirs if d not in _DEFAULT_IGNORED_DIRS]
                for f in sorted(files):
                    if fnmatch.fnmatch(f, include_pattern):
                        search_single_file(Path(root) / f)
                        if truncated:
                            break
                if truncated:
                    break
        else:
            raise FileNotFoundError(f"path not found: {target_str}")

        rel = "." if target == workspace_resolved else str(target.relative_to(workspace_resolved))
        return {
            "path": rel,
            "query": query,
            "matches": matches,
            "total_matches": len(matches),
            "truncated": truncated,
        }

    return Tool(
        name="grep_search",
        description="Search for text or regex patterns across files within the workspace boundary.",
        schema=schema,
        handler=handler,
    )


class _HTMLToMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title: str | None = None
        self._in_title = False
        self._ignore_stack: list[str] = []
        self._pieces: list[str] = []
        self._ignored_tags = {"script", "style", "noscript", "svg", "nav", "header", "footer"}

    @property
    def text(self) -> str:
        """Accumulated text captured so far, in document order."""
        return "".join(self._pieces)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self._in_title = True
        elif tag_lower in self._ignored_tags:
            self._ignore_stack.append(tag_lower)
        elif not self._ignore_stack:
            if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag_lower[1])
                self._pieces.append("\n\n" + "#" * level + " ")
            elif tag_lower in ("p", "div"):
                self._pieces.append("\n\n")
            elif tag_lower == "br":
                self._pieces.append("\n")
            elif tag_lower == "li":
                self._pieces.append("\n* ")
            elif tag_lower == "pre":
                self._pieces.append("\n\n```\n")

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self._in_title = False
        elif self._ignore_stack and self._ignore_stack[-1] == tag_lower:
            self._ignore_stack.pop()
        elif not self._ignore_stack:
            if tag_lower in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "div"):
                self._pieces.append("\n")
            elif tag_lower == "pre":
                self._pieces.append("\n```\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            t = data.strip()
            if t:
                self.title = (self.title + " " + t) if self.title else t
        elif not self._ignore_stack:
            self._pieces.append(data)


def clean_html_to_markdown(html_content: str) -> tuple[str | None, str]:
    """Parse HTML and extract page title and stripped markdown-like text."""
    parser = _HTMLToMarkdownParser()
    try:
        parser.feed(html_content)
        parser.close()
    except Exception:
        pass
    text = parser.text
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return parser.title, text


_BLOCKED_HOSTNAMES = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254", "::1"}


def is_blocked_host(host: str) -> bool:
    """Check if host is a local / private / loopback IP to mitigate SSRF."""
    host_clean = host.split(":")[0].strip().lower()
    if host_clean in _BLOCKED_HOSTNAMES or host_clean.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host_clean)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    except ValueError:
        pass
    return False


def make_fetch_url_tool() -> Tool:
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "HTTP or HTTPS URL to fetch content from"},
            "timeout_s": {
                "type": "number",
                "description": "Request timeout in seconds (default: 15)",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of content to return (default: 50000)",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        raw_url = str(args["url"]).strip()
        parsed = urllib.parse.urlparse(raw_url)
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme '{parsed.scheme}'; only http and https are allowed")
        if not parsed.netloc:
            raise ValueError(f"invalid URL: missing host in '{raw_url}'")
        if is_blocked_host(parsed.netloc):
            raise PermissionError(f"access denied: blocked target address '{parsed.netloc}'")

        timeout_s = float(args.get("timeout_s", 15.0))
        max_chars = max(100, int(args.get("max_chars", 50_000)))

        def _do_fetch() -> dict[str, Any]:
            req = urllib.request.Request(
                raw_url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; GLM-5.3-Flash-Harness/0.4.2)"},
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    status_code = getattr(resp, "code", 200) or getattr(resp, "status", 200)
                    content_type = str(resp.headers.get("Content-Type", "text/plain"))
                    raw_bytes = resp.read(10 * 1024 * 1024)
            except urllib.error.HTTPError as err:
                status_code = err.code
                content_type = str(err.headers.get("Content-Type", "text/plain"))
                raw_bytes = err.read(512 * 1024)
            except urllib.error.URLError as err:
                raise ValueError(f"failed to fetch URL: {err.reason}") from err

            text_content = raw_bytes.decode("utf-8", errors="replace")
            title = None
            if "html" in content_type.lower():
                title, text_content = clean_html_to_markdown(text_content)

            truncated = len(text_content) > max_chars
            sliced = text_content[:max_chars]

            return {
                "url": raw_url,
                "status_code": status_code,
                "content_type": content_type,
                "title": title,
                "content": sliced,
                "truncated": truncated,
            }

        return await asyncio.to_thread(_do_fetch)

    return Tool(
        name="fetch_url",
        description="Fetch and extract readable text/markdown from a web page or HTTP API endpoint.",
        schema=schema,
        handler=handler,
    )


class BuiltinToolsPlugin:
    """Mounts standard tools (read, write, edit, list, find, grep, fetch, bash, stop-slop) into registry."""

    id = "builtin-tools"

    def __init__(
        self,
        workspace: Path | None = None,
        enable_bash: bool = True,
        enable_fetch_url: bool = True,
        enable_stop_slop: bool = True,
    ):
        self.workspace = (workspace or Path.cwd()).resolve()
        self.enable_bash = enable_bash
        self.enable_fetch_url = enable_fetch_url
        self.enable_stop_slop = enable_stop_slop

    def apply(self, ctx: Context) -> None:
        ctx.provide("workspace", self.workspace)
        tools: ToolRegistry = ctx.get("tools")
        tools.register(make_read_file_tool(self.workspace))
        tools.register(make_write_file_tool(self.workspace))
        tools.register(make_edit_file_tool(self.workspace))
        tools.register(make_list_dir_tool(self.workspace))
        tools.register(make_find_files_tool(self.workspace))
        tools.register(make_grep_search_tool(self.workspace))
        if self.enable_fetch_url:
            tools.register(make_fetch_url_tool())
        if self.enable_bash:
            tools.register(make_bash_tool(self.workspace))
        if self.enable_stop_slop:
            tools.register(make_stop_slop_analyze_tool())
            tools.register(make_stop_slop_rewrite_tool())
            tools.register(make_stop_slop_rules_tool())
            tools.register(make_stop_slop_examples_tool())


