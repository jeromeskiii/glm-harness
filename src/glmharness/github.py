"""GitHub REST client and tool battery for GLM-5.3-Flash agent harness.

Provides a thin, zero-dependency GitHub provider client and model-facing tools:
- Read-only tools (github_get_repo, github_list_issues, github_get_issue,
  github_list_issue_comments, github_get_file, github_list_pull_requests,
  github_get_pull_request, github_list_pull_request_files, github_list_branches,
  github_list_commits, github_get_commit, github_search_code, github_search_issues)
- Mutating tools (github_write_file, github_create_issue, github_create_pull_request,
  github_create_branch, github_add_issue_comment) guarded by sandbox policy.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from .context import Context
from .errors import ConfigError, ToolError
from .tools import Tool, ToolRegistry

JsonObject = dict[str, Any]
JsonPayload = JsonObject | list[Any]

MAX_FILE_CHARS: int = 64 * 1024
MAX_PATCH_CHARS: int = 8 * 1024
DEFAULT_PAGE_SIZE: int = 30
MAX_PAGE_SIZE: int = 100
DEFAULT_API_BASE: str = "https://api.github.com"
LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

_REPO_SLUG_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_GIT_REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")

GITHUB_MUTATING_TOOLS: frozenset[str] = frozenset({
    "github_write_file",
    "github_create_issue",
    "github_create_pull_request",
    "github_create_branch",
    "github_add_issue_comment",
})


def is_repo_slug(slug: str) -> bool:
    """Validate owner/name slug."""
    if not _REPO_SLUG_RE.match(slug):
        return False
    parts = slug.split("/")
    if len(parts) != 2:
        return False
    owner, name = parts
    return owner not in (".", "..") and name not in (".", "..")


def is_git_ref_name(name: str) -> bool:
    """Validate git branch/tag reference."""
    if not name or len(name) > 250:
        return False
    if name.startswith("/") or name.endswith("/") or name.endswith("."):
        return False
    if ".." in name or "//" in name or "\\" in name:
        return False
    if name.startswith("refs/"):
        return False
    return bool(_GIT_REF_RE.match(name))


def is_commit_sha(sha: str) -> bool:
    """Validate commit SHA."""
    return bool(_COMMIT_SHA_RE.match(sha))


def normalize_api_base(raw: str) -> str:
    """Validate and normalize GitHub API base URL."""
    try:
        url = urllib.parse.urlsplit(raw)
    except Exception as exc:
        raise ConfigError(f"GitHub api_base must be a valid URL: {exc}") from exc

    if url.username or url.password or url.query or url.fragment:
        raise ConfigError("GitHub api_base must not contain credentials, query parameters, or fragments")

    if url.scheme != "https" and not (url.scheme == "http" and url.hostname in LOOPBACK_HOSTS):
        raise ConfigError("GitHub api_base must use HTTPS (HTTP is permitted only for loopback development)")

    return raw.rstrip("/")


def _coerce_dict_list(val: object) -> list[JsonObject]:
    if isinstance(val, list):
        out: list[JsonObject] = []
        for raw_item in cast(list[object], val):
            if isinstance(raw_item, dict):
                out.append(cast(JsonObject, raw_item))
        return out
    return []


@dataclass(frozen=True)
class GitHubOptions:
    """Configuration options for the GitHub provider."""

    token: str | None = None
    api_base: str = DEFAULT_API_BASE
    owner: str | None = None
    repo: str | None = None
    max_list_items: int = 20
    transport: Callable[[urllib.request.Request, float], tuple[int, dict[str, str], bytes]] | None = None


class GitHubProvider:
    """Thin REST client for the GitHub API."""

    def __init__(self, options: GitHubOptions | None = None) -> None:
        opts = options or GitHubOptions()
        self.token = opts.token
        self.api_base = normalize_api_base(opts.api_base)
        self.owner = opts.owner
        self.repo = opts.repo
        self.max_list_items = max(1, min(opts.max_list_items, 100))
        self.transport = opts.transport or self._default_transport

    @property
    def default_repo_slug(self) -> str | None:
        if self.owner and self.repo:
            return f"{self.owner}/{self.repo}"
        return None

    def resolve_repo(self, repo_arg: str | None) -> str:
        slug = repo_arg or self.default_repo_slug
        if not slug:
            raise ToolError(
                "Repository is required (specify 'repo' parameter or configure default repo scope)"
            )
        if not is_repo_slug(slug):
            raise ToolError(f"Invalid repository slug '{slug}'; expected 'owner/name'")
        return slug

    @staticmethod
    def _default_transport(
        req: urllib.request.Request, timeout_s: float
    ) -> tuple[int, dict[str, str], bytes]:
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.status, headers, resp.read()
        except urllib.error.HTTPError as exc:
            headers = {k.lower(): v for k, v in exc.headers.items()}
            return exc.code, headers, exc.read()
        except Exception as exc:
            raise ToolError(f"GitHub network request failed: {exc}") from exc

    def request(
        self,
        method: str,
        path: str,
        *,
        query: JsonObject | None = None,
        body: JsonObject | None = None,
        timeout_s: float = 30.0,
    ) -> JsonPayload:
        clean_path = path if path.startswith("/") else f"/{path}"
        url = f"{self.api_base}{clean_path}"

        filtered_query: dict[str, str] = {}
        if query:
            for k, v in query.items():
                if v is not None:
                    filtered_query[k] = str(v)
            if filtered_query:
                url = f"{url}?{urllib.parse.urlencode(filtered_query)}"

        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", "glmharness-github/1.0")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")

        status, _, resp_bytes = self.transport(req, timeout_s)

        if status == 204:
            return {}

        text = resp_bytes.decode("utf-8", errors="replace")
        try:
            parsed: JsonPayload = cast(JsonPayload, json.loads(text)) if text else {}
        except Exception:
            parsed = {"raw": text}

        if status >= 400:
            msg = parsed.get("message") if isinstance(parsed, dict) else text
            raise ToolError(f"GitHub API error {status}: {msg}")

        return parsed

    # Domain methods
    def get_repo(self, repo_slug: str | None) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        res = self.request("GET", f"/repos/{slug}")
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub get_repo")
        return {
            "full_name": res.get("full_name", slug),
            "description": res.get("description"),
            "default_branch": res.get("default_branch", "main"),
            "private": bool(res.get("private", False)),
            "html_url": res.get("html_url"),
            "language": res.get("language"),
            "stargazers_count": res.get("stargazers_count", 0),
        }

    def list_issues(
        self,
        repo_slug: str | None,
        *,
        state: str = "open",
        labels: str | list[str] | None = None,
        page: int = 1,
        per_page: int = 30,
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clamped_page = max(1, page)
        clamped_per_page = max(1, min(per_page, MAX_PAGE_SIZE))
        q: dict[str, Any] = {
            "state": state if state in ("open", "closed", "all") else "open",
            "page": clamped_page,
            "per_page": clamped_per_page,
        }
        if labels:
            q["labels"] = ",".join(labels) if isinstance(labels, list) else str(labels)

        res = self.request("GET", f"/repos/{slug}/issues", query=q)
        items = _coerce_dict_list(res)
        clean_items: list[JsonObject] = []
        for i in items:
            if "pull_request" not in i:
                user_data: JsonObject | None = i.get("user") if isinstance(i.get("user"), dict) else None
                login: str | None = user_data.get("login") if isinstance(user_data, dict) else None
                clean_items.append({
                    "number": i.get("number"),
                    "title": i.get("title"),
                    "state": i.get("state"),
                    "user": login,
                    "html_url": i.get("html_url"),
                })
        has_more = len(clean_items) >= clamped_per_page
        trimmed = clean_items[: self.max_list_items]
        return {
            "values": trimmed,
            "page": clamped_page,
            "per_page": clamped_per_page,
            "has_more": has_more or len(clean_items) > len(trimmed),
        }

    def get_issue(self, repo_slug: str | None, number: int) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        res = self.request("GET", f"/repos/{slug}/issues/{number}")
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub get_issue")
        user = res.get("user")
        user_obj: JsonObject | None = cast(JsonObject, user) if isinstance(user, dict) else None
        login: str | None = user_obj.get("login") if isinstance(user_obj, dict) else None
        labels = [lbl.get("name") for lbl in _coerce_dict_list(res.get("labels", []))]
        assignees = [a.get("login") for a in _coerce_dict_list(res.get("assignees", []))]
        return {
            "number": res.get("number"),
            "title": res.get("title"),
            "state": res.get("state"),
            "user": login,
            "body": res.get("body"),
            "labels": labels,
            "assignees": assignees,
            "comments": res.get("comments", 0),
            "html_url": res.get("html_url"),
        }

    def list_issue_comments(
        self, repo_slug: str | None, number: int, *, page: int = 1, per_page: int = 30
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clamped_page = max(1, page)
        clamped_per_page = max(1, min(per_page, MAX_PAGE_SIZE))
        res = self.request(
            "GET",
            f"/repos/{slug}/issues/{number}/comments",
            query={"page": clamped_page, "per_page": clamped_per_page},
        )
        items = _coerce_dict_list(res)
        values: list[JsonObject] = []
        for c in items:
            user = c.get("user")
            user_obj: JsonObject | None = cast(JsonObject, user) if isinstance(user, dict) else None
            values.append({
                "id": c.get("id"),
                "user": user_obj.get("login") if isinstance(user_obj, dict) else None,
                "created_at": c.get("created_at"),
                "body": c.get("body"),
            })
        has_more = len(values) >= clamped_per_page
        trimmed = values[: self.max_list_items]
        return {
            "values": trimmed,
            "page": clamped_page,
            "per_page": clamped_per_page,
            "has_more": has_more or len(values) > len(trimmed),
        }

    def get_file(self, repo_slug: str | None, path: str, *, ref: str | None = None) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clean_path = path.strip("/")
        q = {"ref": ref} if ref else None
        res = self.request("GET", f"/repos/{slug}/contents/{clean_path}", query=q)

        # Directory listing
        if isinstance(res, list):
            entries: list[JsonObject] = []
            for e in _coerce_dict_list(res):
                entries.append({"name": e.get("name"), "type": e.get("type"), "size": e.get("size")})
            return {"path": path, "listing": True, "entries": entries}

        assert isinstance(res, dict)

        content = res.get("content", "")
        encoding = res.get("encoding")
        if encoding == "base64" and isinstance(content, str):
            try:
                decoded = base64.b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                decoded = content
        else:
            decoded = str(content)

        truncated = False
        if len(decoded) > MAX_FILE_CHARS:
            decoded = decoded[:MAX_FILE_CHARS]
            truncated = True

        return {
            "path": path,
            "sha": res.get("sha"),
            "size": res.get("size"),
            "content": decoded,
            "truncated": truncated,
        }

    def list_pull_requests(
        self, repo_slug: str | None, *, state: str = "open", page: int = 1, per_page: int = 30
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clamped_page = max(1, page)
        clamped_per_page = max(1, min(per_page, MAX_PAGE_SIZE))
        res = self.request(
            "GET",
            f"/repos/{slug}/pulls",
            query={
                "state": state if state in ("open", "closed", "all") else "open",
                "page": clamped_page,
                "per_page": clamped_per_page,
            },
        )
        items = _coerce_dict_list(res)
        values: list[JsonObject] = []
        for p in items:
            head_raw = p.get("head")
            base_raw = p.get("base")
            head: JsonObject | None = cast(JsonObject, head_raw) if isinstance(head_raw, dict) else None
            base: JsonObject | None = cast(JsonObject, base_raw) if isinstance(base_raw, dict) else None
            values.append({
                "number": p.get("number"),
                "title": p.get("title"),
                "state": p.get("state"),
                "head": head.get("ref") if isinstance(head, dict) else None,
                "base": base.get("ref") if isinstance(base, dict) else None,
            })
        has_more = len(values) >= clamped_per_page
        trimmed = values[: self.max_list_items]
        return {
            "values": trimmed,
            "page": clamped_page,
            "per_page": clamped_per_page,
            "has_more": has_more or len(values) > len(trimmed),
        }

    def get_pull_request(self, repo_slug: str | None, number: int) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        res = self.request("GET", f"/repos/{slug}/pulls/{number}")
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub get_pull_request")
        user = res.get("user")
        head = res.get("head")
        base = res.get("base")
        user_obj: JsonObject | None = cast(JsonObject, user) if isinstance(user, dict) else None
        head_obj: JsonObject | None = cast(JsonObject, head) if isinstance(head, dict) else None
        base_obj: JsonObject | None = cast(JsonObject, base) if isinstance(base, dict) else None
        return {
            "number": res.get("number"),
            "title": res.get("title"),
            "state": res.get("state"),
            "user": user_obj.get("login") if isinstance(user_obj, dict) else None,
            "body": res.get("body"),
            "head": head_obj.get("ref") if isinstance(head_obj, dict) else None,
            "base": base_obj.get("ref") if isinstance(base_obj, dict) else None,
            "mergeable": res.get("mergeable"),
            "diff_url": res.get("diff_url"),
            "html_url": res.get("html_url"),
        }

    def list_pull_request_files(
        self, repo_slug: str | None, number: int, *, page: int = 1, per_page: int = 30
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clamped_page = max(1, page)
        clamped_per_page = max(1, min(per_page, MAX_PAGE_SIZE))
        res = self.request(
            "GET",
            f"/repos/{slug}/pulls/{number}/files",
            query={"page": clamped_page, "per_page": clamped_per_page},
        )
        items = _coerce_dict_list(res)
        values: list[JsonObject] = []
        for f in items:
            patch = f.get("patch")
            patch_truncated = False
            if isinstance(patch, str) and len(patch) > MAX_PATCH_CHARS:
                patch = patch[:MAX_PATCH_CHARS]
                patch_truncated = True
            entry: dict[str, Any] = {
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
            }
            if patch:
                entry["patch"] = patch
            if patch_truncated:
                entry["patch_truncated"] = True
            values.append(entry)
        has_more = len(values) >= clamped_per_page
        trimmed = values[: self.max_list_items]
        return {
            "values": trimmed,
            "page": clamped_page,
            "per_page": clamped_per_page,
            "has_more": has_more or len(values) > len(trimmed),
        }

    def list_branches(
        self, repo_slug: str | None, *, page: int = 1, per_page: int = 30
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clamped_page = max(1, page)
        clamped_per_page = max(1, min(per_page, MAX_PAGE_SIZE))
        res = self.request(
            "GET",
            f"/repos/{slug}/branches",
            query={"page": clamped_page, "per_page": clamped_per_page},
        )
        items = _coerce_dict_list(res)
        values: list[JsonObject] = []
        for b in items:
            commit = b.get("commit")
            commit_obj: JsonObject | None = cast(JsonObject, commit) if isinstance(commit, dict) else None
            sha: str | None = commit_obj.get("sha") if isinstance(commit_obj, dict) else None
            values.append({"name": b.get("name"), "sha": sha})
        has_more = len(values) >= clamped_per_page
        trimmed = values[: self.max_list_items]
        return {
            "values": trimmed,
            "page": clamped_page,
            "per_page": clamped_per_page,
            "has_more": has_more or len(values) > len(trimmed),
        }

    def list_commits(
        self, repo_slug: str | None, *, branch: str | None = None, page: int = 1, per_page: int = 30
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clamped_page = max(1, page)
        clamped_per_page = max(1, min(per_page, MAX_PAGE_SIZE))
        q: dict[str, Any] = {"page": clamped_page, "per_page": clamped_per_page}
        if branch:
            q["sha"] = branch
        res = self.request("GET", f"/repos/{slug}/commits", query=q)
        items = _coerce_dict_list(res)
        values: list[JsonObject] = []
        for c in items:
            commit = c.get("commit")
            commit_obj: JsonObject | None = cast(JsonObject, commit) if isinstance(commit, dict) else None
            msg: str | None = commit_obj.get("message") if isinstance(commit_obj, dict) else None
            author_raw = commit_obj.get("author") if isinstance(commit_obj, dict) else None
            author_dict: JsonObject | None = (
                cast(JsonObject, author_raw) if isinstance(author_raw, dict) else None
            )
            values.append({
                "sha": c.get("sha"),
                "message": msg,
                "author": author_dict.get("name") if isinstance(author_dict, dict) else None,
                "date": author_dict.get("date") if isinstance(author_dict, dict) else None,
            })
        has_more = len(values) >= clamped_per_page
        trimmed = values[: self.max_list_items]
        return {
            "values": trimmed,
            "page": clamped_page,
            "per_page": clamped_per_page,
            "has_more": has_more or len(values) > len(trimmed),
        }

    def get_commit(self, repo_slug: str | None, sha: str) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        if not is_commit_sha(sha):
            raise ToolError(f"Invalid commit SHA: {sha}")
        res = self.request("GET", f"/repos/{slug}/commits/{sha}")
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub get_commit")
        commit = res.get("commit")
        commit_obj: JsonObject | None = cast(JsonObject, commit) if isinstance(commit, dict) else None
        msg: str | None = commit_obj.get("message") if isinstance(commit_obj, dict) else None
        author_dict = commit_obj.get("author") if isinstance(commit_obj, dict) else None
        files = _coerce_dict_list(res.get("files", []))
        files_trimmed = files[: self.max_list_items]
        files_clean: list[JsonObject] = []
        for f in files_trimmed:
            files_clean.append({
                "filename": f.get("filename"),
                "status": f.get("status"),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
            })
        author_dict_obj: JsonObject | None = (
            cast(JsonObject, author_dict) if isinstance(author_dict, dict) else None
        )
        return {
            "sha": res.get("sha"),
            "message": msg,
            "author": author_dict_obj.get("name") if isinstance(author_dict_obj, dict) else None,
            "html_url": res.get("html_url"),
            "files": files_clean,
            "files_truncated": len(files) > len(files_clean),
        }

    def search_code(
        self, query: str, repo_slug: str | None = None, *, page: int = 1, per_page: int = 30
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clamped_page = max(1, page)
        clamped_per_page = max(1, min(per_page, MAX_PAGE_SIZE))
        scoped_q = f"{query} repo:{slug}"
        res = self.request(
            "GET",
            "/search/code",
            query={"q": scoped_q, "page": clamped_page, "per_page": clamped_per_page},
        )
        items = _coerce_dict_list(res.get("items", [])) if isinstance(res, dict) else []
        values: list[JsonObject] = []
        for i in items:
            repo_info = i.get("repository")
            repo_info_obj: JsonObject | None = (
                cast(JsonObject, repo_info) if isinstance(repo_info, dict) else None
            )
            repo_name: str | None = (
                repo_info_obj.get("full_name") if isinstance(repo_info_obj, dict) else None
            )
            values.append({"repo": repo_name, "path": i.get("path"), "url": i.get("html_url")})
        has_more = len(values) >= clamped_per_page
        trimmed = values[: self.max_list_items]
        return {
            "values": trimmed,
            "page": clamped_page,
            "per_page": clamped_per_page,
            "has_more": has_more or len(values) > len(trimmed),
        }

    def search_issues(
        self, query: str, repo_slug: str | None = None, *, page: int = 1, per_page: int = 30
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clamped_page = max(1, page)
        clamped_per_page = max(1, min(per_page, MAX_PAGE_SIZE))
        scoped_q = f"{query} repo:{slug}"
        res = self.request(
            "GET",
            "/search/issues",
            query={"q": scoped_q, "page": clamped_page, "per_page": clamped_per_page},
        )
        items = _coerce_dict_list(res.get("items", [])) if isinstance(res, dict) else []
        values: list[JsonObject] = []
        for i in items:
            values.append({
                "number": i.get("number"),
                "title": i.get("title"),
                "state": i.get("state"),
                "pull_request": "pull_request" in i,
                "html_url": i.get("html_url"),
            })
        has_more = len(values) >= clamped_per_page
        trimmed = values[: self.max_list_items]
        return {
            "values": trimmed,
            "page": clamped_page,
            "per_page": clamped_per_page,
            "has_more": has_more or len(values) > len(trimmed),
        }

    # Mutating methods (subject to sandbox policy)
    def write_file(
        self,
        repo_slug: str | None,
        path: str,
        message: str,
        content: str,
        *,
        branch: str | None = None,
        sha: str | None = None,
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        clean_path = path.strip("/")
        b64_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
        body: dict[str, Any] = {"message": message, "content": b64_content}
        if branch:
            body["branch"] = branch
        if sha:
            body["sha"] = sha

        res = self.request("PUT", f"/repos/{slug}/contents/{clean_path}", body=body)
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub write_file")
        content_info = res.get("content")
        commit_info = res.get("commit")
        content_info_obj: JsonObject | None = (
            cast(JsonObject, content_info) if isinstance(content_info, dict) else None
        )
        commit_info_obj: JsonObject | None = (
            cast(JsonObject, commit_info) if isinstance(commit_info, dict) else None
        )
        return {
            "path": path,
            "content_sha": content_info_obj.get("sha") if isinstance(content_info_obj, dict) else None,
            "commit_sha": commit_info_obj.get("sha") if isinstance(commit_info_obj, dict) else None,
            "html_url": content_info_obj.get("html_url") if isinstance(content_info_obj, dict) else None,
        }

    def create_issue(
        self,
        repo_slug: str | None,
        title: str,
        body: str | None = None,
        labels: list[str] | None = None,
        assignees: list[str] | None = None,
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        payload: dict[str, Any] = {"title": title}
        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees

        res = self.request("POST", f"/repos/{slug}/issues", body=payload)
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub create_issue")
        return {
            "number": res.get("number"),
            "title": res.get("title"),
            "html_url": res.get("html_url"),
            "state": res.get("state"),
        }

    def create_pull_request(
        self,
        repo_slug: str | None,
        title: str,
        head: str,
        base: str,
        body: str | None = None,
        draft: bool = False,
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        payload: dict[str, Any] = {"title": title, "head": head, "base": base, "draft": draft}
        if body:
            payload["body"] = body

        res = self.request("POST", f"/repos/{slug}/pulls", body=payload)
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub create_pull_request")
        return {
            "number": res.get("number"),
            "title": res.get("title"),
            "html_url": res.get("html_url"),
            "state": res.get("state"),
            "head": head,
            "base": base,
        }

    def create_branch(
        self,
        repo_slug: str | None,
        branch: str,
        *,
        from_sha: str | None = None,
        from_branch: str | None = None,
    ) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        if not is_git_ref_name(branch):
            raise ToolError(f"Invalid branch name '{branch}'")

        sha = from_sha
        if not sha:
            target_source = from_branch or "main"
            ref_info = self.request("GET", f"/repos/{slug}/git/ref/heads/{target_source}")
            if not isinstance(ref_info, dict) or "object" not in ref_info:
                raise ToolError(f"Could not resolve source branch '{target_source}' to commit SHA")
            obj = ref_info["object"]
            obj_obj: JsonObject | None = cast(JsonObject, obj) if isinstance(obj, dict) else None
            sha_obj: str | None = obj_obj.get("sha") if isinstance(obj_obj, dict) else None
            sha = sha_obj if isinstance(sha_obj, str) else None

        if not sha or not is_commit_sha(sha):
            raise ToolError(f"Could not resolve valid commit SHA to branch from: {sha}")

        sha_str: str = sha
        payload_body: dict[str, str] = {"ref": f"refs/heads/{branch}", "sha": sha_str}
        payload = cast(JsonObject, payload_body)
        res = self.request("POST", f"/repos/{slug}/git/refs", body=payload)
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub create_branch")
        return {"ref": res.get("ref"), "sha": sha, "branch": branch}

    def add_issue_comment(self, repo_slug: str | None, number: int, body: str) -> dict[str, Any]:
        slug = self.resolve_repo(repo_slug)
        res = self.request("POST", f"/repos/{slug}/issues/{number}/comments", body={"body": body})
        if not isinstance(res, dict):
            raise ToolError("Unexpected response shape from GitHub add_issue_comment")
        return {
            "id": res.get("id"),
            "html_url": res.get("html_url"),
            "created_at": res.get("created_at"),
        }


# Tool factories
def make_github_tools(provider: GitHubProvider) -> list[Tool]:
    """Create all GitHub model-facing tools bound to provider."""

    async def _get_repo(args: dict[str, Any]) -> dict[str, Any]:
        return provider.get_repo(args.get("repo"))

    async def _list_issues(args: dict[str, Any]) -> dict[str, Any]:
        return provider.list_issues(
            args.get("repo"),
            state=args.get("state", "open"),
            labels=args.get("labels"),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", 30)),
        )

    async def _get_issue(args: dict[str, Any]) -> dict[str, Any]:
        return provider.get_issue(args.get("repo"), int(args["number"]))

    async def _list_issue_comments(args: dict[str, Any]) -> dict[str, Any]:
        return provider.list_issue_comments(
            args.get("repo"),
            int(args["number"]),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", 30)),
        )

    async def _get_file(args: dict[str, Any]) -> dict[str, Any]:
        return provider.get_file(args.get("repo"), str(args["path"]), ref=args.get("ref"))

    async def _list_pull_requests(args: dict[str, Any]) -> dict[str, Any]:
        return provider.list_pull_requests(
            args.get("repo"),
            state=args.get("state", "open"),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", 30)),
        )

    async def _get_pull_request(args: dict[str, Any]) -> dict[str, Any]:
        return provider.get_pull_request(args.get("repo"), int(args["number"]))

    async def _list_pull_request_files(args: dict[str, Any]) -> dict[str, Any]:
        return provider.list_pull_request_files(
            args.get("repo"),
            int(args["number"]),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", 30)),
        )

    async def _list_branches(args: dict[str, Any]) -> dict[str, Any]:
        return provider.list_branches(
            args.get("repo"),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", 30)),
        )

    async def _list_commits(args: dict[str, Any]) -> dict[str, Any]:
        return provider.list_commits(
            args.get("repo"),
            branch=args.get("branch"),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", 30)),
        )

    async def _get_commit(args: dict[str, Any]) -> dict[str, Any]:
        return provider.get_commit(args.get("repo"), str(args["sha"]))

    async def _search_code(args: dict[str, Any]) -> dict[str, Any]:
        return provider.search_code(
            str(args["query"]),
            repo_slug=args.get("repo"),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", 30)),
        )

    async def _search_issues(args: dict[str, Any]) -> dict[str, Any]:
        return provider.search_issues(
            str(args["query"]),
            repo_slug=args.get("repo"),
            page=int(args.get("page", 1)),
            per_page=int(args.get("per_page", 30)),
        )

    # Mutating
    async def _write_file(args: dict[str, Any]) -> dict[str, Any]:
        return provider.write_file(
            args.get("repo"),
            str(args["path"]),
            str(args["message"]),
            str(args["content"]),
            branch=args.get("branch"),
            sha=args.get("sha"),
        )

    async def _create_issue(args: dict[str, Any]) -> dict[str, Any]:
        return provider.create_issue(
            args.get("repo"),
            str(args["title"]),
            body=args.get("body"),
            labels=args.get("labels"),
            assignees=args.get("assignees"),
        )

    async def _create_pull_request(args: dict[str, Any]) -> dict[str, Any]:
        return provider.create_pull_request(
            args.get("repo"),
            str(args["title"]),
            str(args["head"]),
            str(args["base"]),
            body=args.get("body"),
            draft=bool(args.get("draft", False)),
        )

    async def _create_branch(args: dict[str, Any]) -> dict[str, Any]:
        return provider.create_branch(
            args.get("repo"),
            str(args["branch"]),
            from_sha=args.get("from_sha"),
            from_branch=args.get("from_branch"),
        )

    async def _add_issue_comment(args: dict[str, Any]) -> dict[str, Any]:
        return provider.add_issue_comment(
            args.get("repo"),
            int(args["number"]),
            str(args["body"]),
        )

    return [
        Tool(
            name="github_get_repo",
            description="Get metadata for a GitHub repository (description, stars, default branch).",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/name; defaults to configured repo scope"}
                },
            },
            handler=_get_repo,
        ),
        Tool(
            name="github_list_issues",
            description="List issues (titles, numbers, state, authors). Excludes pull requests.",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "labels": {"type": "string", "description": "comma-separated label names"},
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
            handler=_list_issues,
        ),
        Tool(
            name="github_get_issue",
            description="Get the full body, labels, and comment count of one GitHub issue.",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "number": {"type": "integer", "description": "issue number"},
                },
                "required": ["number"],
            },
            handler=_get_issue,
        ),
        Tool(
            name="github_list_issue_comments",
            description="List comments on a GitHub issue (author, date, body).",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "number": {"type": "integer"},
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["number"],
            },
            handler=_list_issue_comments,
        ),
        Tool(
            name="github_get_file",
            description="Read a file from a repository, or list entries if path is a directory.",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                    "ref": {"type": "string", "description": "branch or commit SHA"},
                },
                "required": ["path"],
            },
            handler=_get_file,
        ),
        Tool(
            name="github_list_pull_requests",
            description="List pull requests for a repository.",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
            handler=_list_pull_requests,
        ),
        Tool(
            name="github_get_pull_request",
            description="Get full body, branches, diff URL, and mergeability of a pull request.",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "number": {"type": "integer"},
                },
                "required": ["number"],
            },
            handler=_get_pull_request,
        ),
        Tool(
            name="github_list_pull_request_files",
            description="List changed files and diff patch snippets for a pull request.",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "number": {"type": "integer"},
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["number"],
            },
            handler=_list_pull_request_files,
        ),
        Tool(
            name="github_list_branches",
            description="List repository branches with head commit SHAs.",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
            handler=_list_branches,
        ),
        Tool(
            name="github_list_commits",
            description="List recent commits (sha, message, author, date).",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "branch": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
            handler=_list_commits,
        ),
        Tool(
            name="github_get_commit",
            description="Get commit metadata and changed file list by SHA.",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "sha": {"type": "string"},
                },
                "required": ["sha"],
            },
            handler=_get_commit,
        ),
        Tool(
            name="github_search_code",
            description="Search code in the scoped GitHub repository.",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "repo": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["query"],
            },
            handler=_search_code,
        ),
        Tool(
            name="github_search_issues",
            description="Search issues and pull requests in the scoped GitHub repository.",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "repo": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["query"],
            },
            handler=_search_issues,
        ),
        # Mutating
        Tool(
            name="github_write_file",
            description="Create or update a file in the repository wrapped in a commit (mutating).",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                    "message": {"type": "string"},
                    "content": {"type": "string"},
                    "branch": {"type": "string"},
                    "sha": {"type": "string", "description": "current blob sha when updating"},
                },
                "required": ["path", "message", "content"],
            },
            handler=_write_file,
        ),
        Tool(
            name="github_create_issue",
            description="Create a new GitHub issue (mutating).",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "assignees": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title"],
            },
            handler=_create_issue,
        ),
        Tool(
            name="github_create_pull_request",
            description="Create a new GitHub pull request (mutating).",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "title": {"type": "string"},
                    "head": {"type": "string"},
                    "base": {"type": "string"},
                    "body": {"type": "string"},
                    "draft": {"type": "boolean"},
                },
                "required": ["title", "head", "base"],
            },
            handler=_create_pull_request,
        ),
        Tool(
            name="github_create_branch",
            description="Create a new git branch from a commit SHA or branch tip (mutating).",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "branch": {"type": "string"},
                    "from_sha": {"type": "string"},
                    "from_branch": {"type": "string"},
                },
                "required": ["branch"],
            },
            handler=_create_branch,
        ),
        Tool(
            name="github_add_issue_comment",
            description="Add a comment to a GitHub issue or pull request (mutating).",
            schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "number": {"type": "integer"},
                    "body": {"type": "string"},
                },
                "required": ["number", "body"],
            },
            handler=_add_issue_comment,
        ),
    ]


class GitHubPlugin:
    """Plugin that mounts GitHub provider and tools into the harness Context."""

    id = "github"

    def __init__(self, provider: GitHubProvider) -> None:
        self.provider = provider

    def apply(self, ctx: Context) -> None:
        ctx.provide("github", self.provider)
        tools = ctx.get("tools")
        if isinstance(tools, ToolRegistry):
            for tool in make_github_tools(self.provider):
                tools.register(tool)
