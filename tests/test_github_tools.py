"""Unit and integration tests for GitHub tool seam and provider."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

import pytest

from glmharness import (
    Context,
    HarnessConfig,
    ToolRegistry,
)
from glmharness.errors import ConfigError
from glmharness.github import (
    GitHubOptions,
    GitHubPlugin,
    GitHubProvider,
    is_commit_sha,
    is_git_ref_name,
    is_repo_slug,
    normalize_api_base,
)
from glmharness.sandbox import SandboxPlugin


def test_slug_and_ref_validators() -> None:
    assert is_repo_slug("octocat/Hello-World")
    assert is_repo_slug("org.name/repo-123_test")
    assert not is_repo_slug("justrepo")
    assert not is_repo_slug("../escaped")
    assert not is_repo_slug("owner/repo/extra")
    assert not is_repo_slug("./repo")

    assert is_git_ref_name("main")
    assert is_git_ref_name("feature/add-login")
    assert is_git_ref_name("v1.0.0")
    assert not is_git_ref_name("")
    assert not is_git_ref_name("/leading-slash")
    assert not is_git_ref_name("trailing-slash/")
    assert not is_git_ref_name("has..traversal")
    assert not is_git_ref_name("refs/heads/main")

    assert is_commit_sha("7fd1a60")
    assert is_commit_sha("0123456789abcdef0123456789abcdef01234567")
    assert not is_commit_sha("not-a-sha!")
    assert not is_commit_sha("123")  # too short (< 7)


def test_normalize_api_base() -> None:
    assert normalize_api_base("https://api.github.com") == "https://api.github.com"
    assert normalize_api_base("https://api.github.com/") == "https://api.github.com"
    assert normalize_api_base("http://127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert normalize_api_base("http://localhost:3000/") == "http://localhost:3000"

    with pytest.raises(ConfigError, match="must use HTTPS"):
        normalize_api_base("http://remote-insecure.com")

    with pytest.raises(ConfigError, match="must not contain credentials"):
        normalize_api_base("https://user:pass@api.github.com")


class ScriptedTransport:
    """Mock HTTP transport returning canned responses for tests without network."""

    def __init__(self, routes: dict[tuple[str, str], tuple[int, dict[str, str], Any]]) -> None:
        self.routes: dict[tuple[str, str], tuple[int, dict[str, str], Any]] = routes
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, req: urllib.request.Request, timeout_s: float
    ) -> tuple[int, dict[str, str], bytes]:
        url = req.full_url
        method = req.get_method()
        body = req.data
        self.calls.append({"method": method, "url": url, "body": body})

        parsed = urllib.parse.urlsplit(url)
        path: str = parsed.path

        # Match exact path first, then longest matching path
        for (m, p), (status, headers, payload) in sorted(
            self.routes.items(), key=lambda item: len(item[0][1]), reverse=True
        ):
            if m == method and (p == path or path.startswith(p)):
                payload_bytes = (
                    json.dumps(payload).encode("utf-8")
                    if not isinstance(payload, bytes)
                    else payload
                )
                return status, headers, payload_bytes

        # Fallback 404
        return 404, {"content-type": "application/json"}, b'{"message": "Not Found"}'


@pytest.fixture
def fake_github() -> GitHubProvider:
    routes: dict[tuple[str, str], tuple[int, dict[str, str], Any]] = {
        ("GET", "/repos/octocat/Hello-World"): (
            200,
            {},
            {
                "full_name": "octocat/Hello-World",
                "description": "My first repo",
                "default_branch": "main",
                "private": False,
                "html_url": "https://github.com/octocat/Hello-World",
                "stargazers_count": 42,
            },
        ),
        ("GET", "/repos/octocat/Hello-World/issues"): (
            200,
            {},
            [
                {
                    "number": 1,
                    "title": "Bug found",
                    "state": "open",
                    "user": {"login": "alice"},
                    "html_url": "https://github.com/octocat/Hello-World/issues/1",
                },
                {
                    "number": 2,
                    "title": "A PR issue",
                    "pull_request": {},
                    "state": "open",
                },
            ],
        ),
        ("GET", "/repos/octocat/Hello-World/issues/1"): (
            200,
            {},
            {
                "number": 1,
                "title": "Bug found",
                "state": "open",
                "user": {"login": "alice"},
                "body": "Something crashed",
                "labels": [{"name": "bug"}],
                "assignees": [{"login": "bob"}],
                "comments": 3,
                "html_url": "https://github.com/octocat/Hello-World/issues/1",
            },
        ),
        ("GET", "/repos/octocat/Hello-World/contents/README.md"): (
            200,
            {},
            {
                "sha": "blob123",
                "size": 13,
                "encoding": "base64",
                "content": "SGVsbG8gV29ybGQhCg==",  # "Hello World!\n"
            },
        ),
        ("GET", "/repos/octocat/Hello-World/contents/src"): (
            200,
            {},
            [
                {"name": "index.ts", "type": "file", "size": 120},
                {"name": "utils", "type": "dir", "size": 0},
            ],
        ),
        ("GET", "/repos/octocat/Hello-World/pulls"): (
            200,
            {},
            [
                {
                    "number": 10,
                    "title": "Add feature",
                    "state": "open",
                    "head": {"ref": "feature-branch"},
                    "base": {"ref": "main"},
                }
            ],
        ),
        ("GET", "/repos/octocat/Hello-World/branches"): (
            200,
            {},
            [{"name": "main", "commit": {"sha": "abc1234567890"}}],
        ),
        ("GET", "/repos/octocat/Hello-World/commits"): (
            200,
            {},
            [
                {
                    "sha": "c0ffee1234567890",
                    "commit": {
                        "message": "Initial commit",
                        "author": {"name": "Alice", "date": "2026-09-01T00:00:00Z"},
                    },
                }
            ],
        ),
        ("GET", "/search/code"): (
            200,
            {},
            {
                "items": [
                    {
                        "path": "src/main.py",
                        "repository": {"full_name": "octocat/Hello-World"},
                        "html_url": "https://github.com/octocat/Hello-World/blob/main/src/main.py",
                    }
                ]
            },
        ),
        ("PUT", "/repos/octocat/Hello-World/contents/file.txt"): (
            200,
            {},
            {
                "content": {"sha": "blobnew", "html_url": "https://github.com/..."},
                "commit": {"sha": "commitnew"},
            },
        ),
        ("POST", "/repos/octocat/Hello-World/issues"): (
            201,
            {},
            {
                "number": 99,
                "title": "New issue",
                "html_url": "https://github.com/octocat/Hello-World/issues/99",
                "state": "open",
            },
        ),
    }
    transport = ScriptedTransport(routes)
    opts = GitHubOptions(
        token="test-token",
        api_base="https://api.github.com",
        owner="octocat",
        repo="Hello-World",
        transport=transport,
    )
    return GitHubProvider(opts)


async def test_github_get_repo(fake_github: GitHubProvider) -> None:
    repo = fake_github.get_repo("octocat/Hello-World")
    assert repo["full_name"] == "octocat/Hello-World"
    assert repo["default_branch"] == "main"
    assert repo["stargazers_count"] == 42


async def test_github_list_issues_filters_pull_requests(fake_github: GitHubProvider) -> None:
    res = fake_github.list_issues("octocat/Hello-World")
    values = res["values"]
    assert len(values) == 1
    assert values[0]["number"] == 1
    assert values[0]["title"] == "Bug found"


async def test_github_get_file_and_directory(fake_github: GitHubProvider) -> None:
    # File read decoded from base64
    f = fake_github.get_file("octocat/Hello-World", "README.md")
    assert f["content"] == "Hello World!\n"
    assert f["sha"] == "blob123"

    # Directory listing
    d = fake_github.get_file("octocat/Hello-World", "src")
    assert d.get("listing") is True
    assert len(d["entries"]) == 2


async def test_github_search_code(fake_github: GitHubProvider) -> None:
    res = fake_github.search_code("main", repo_slug="octocat/Hello-World")
    assert len(res["values"]) == 1
    assert res["values"][0]["path"] == "src/main.py"


async def test_github_tools_pipeline_and_sandbox(fake_github: GitHubProvider) -> None:
    ctx = Context()
    registry = ToolRegistry(ctx)
    ctx.provide("tools", registry)

    plugin = GitHubPlugin(fake_github)
    plugin.apply(ctx)

    # Verify all tools registered
    registered_names = set(registry.tools.keys())
    assert "github_get_repo" in registered_names
    assert "github_write_file" in registered_names
    assert "github_create_issue" in registered_names

    # Test tool execution through registry
    repo_res = await registry.execute("github_get_repo", {"repo": "octocat/Hello-World"})
    assert repo_res["ok"] is True
    parsed_content = json.loads(repo_res["content"])
    assert parsed_content["full_name"] == "octocat/Hello-World"

    # Mutating tool under SandboxPlugin(mode="deny") fails closed
    sandbox = SandboxPlugin(mode="deny")
    sandbox.apply(ctx)

    write_res = await registry.execute(
        "github_write_file",
        {
            "repo": "octocat/Hello-World",
            "path": "file.txt",
            "message": "test commit",
            "content": "hello",
        },
    )
    assert write_res["ok"] is False
    assert write_res["error"] == "SANDBOX_DENIED"


def test_harness_config_github_options() -> None:
    cfg = HarnessConfig(
        github_token="ghp_12345",
        github_repo="octocat/Hello-World",
    )
    cfg.validate()
    assert cfg.github_repo == "octocat/Hello-World"
    assert cfg.github_token == "ghp_12345"

    bad_cfg = HarnessConfig(github_repo="invalid-slug")
    with pytest.raises(ConfigError, match="must be in 'owner/name' format"):
        bad_cfg.validate()
