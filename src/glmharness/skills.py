"""Dynamic Skill Trigger & Tool Gating System.

Provides deterministic trigger matching, multi-signal activation scoring,
risk interval gating, dynamic external SKILL.md bundle discovery, and
safe tool allowlist filtering.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

from glmharness.context import Context
from glmharness.logging import get_logger


class RiskLevel(IntEnum):
    """Execution risk boundary for tasks and skills."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def from_str(cls, val: str) -> RiskLevel:
        normalized = val.strip().upper()
        try:
            return cls[normalized]
        except KeyError:
            return cls.LOW


def _empty_str_set() -> frozenset[str]:
    return frozenset()


@dataclass(frozen=True)
class Skill:
    """Represents a domain capability that may unlock tools and inject guidance."""

    name: str
    description: str = ""
    tools: frozenset[str] = field(default_factory=_empty_str_set)
    triggers: frozenset[str] = field(default_factory=_empty_str_set)
    capabilities: frozenset[str] = field(default_factory=_empty_str_set)
    tags: frozenset[str] = field(default_factory=_empty_str_set)
    always: bool = False
    min_risk: RiskLevel = RiskLevel.LOW
    max_risk: RiskLevel = RiskLevel.CRITICAL
    guidance: str = ""


# --- Morphological and Lexical Stemming ---

_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ers", "es", "er", "ed", "s")

_IRREGULAR_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"run", "runs", "running", "ran"}),
    frozenset({"write", "writes", "writing", "wrote", "written"}),
    frozenset({"read", "reads", "reading"}),
    frozenset({"build", "builds", "building", "built"}),
    frozenset({"seek", "seeks", "seeking", "sought"}),
    frozenset({"make", "makes", "making", "made"}),
    frozenset({"send", "sends", "sending", "sent"}),
)

_IRREGULAR_MAP: dict[str, frozenset[str]] = {
    word: group for group in _IRREGULAR_GROUPS for word in group
}


def _tokenize(text: str) -> frozenset[str]:
    """Extract lowercase alphanumeric tokens."""
    return frozenset(re.findall(r"\b[a-z0-9_]+\b", text.lower()))


def _stem_forms(word: str) -> set[str]:
    """Generate morphological base forms and inflections."""
    word = word.lower()
    forms = {word}
    if word in _IRREGULAR_MAP:
        forms.update(_IRREGULAR_MAP[word])

    for sfx in _SUFFIXES:
        if word.endswith(sfx) and len(word) > len(sfx) + 2:
            base = word[: -len(sfx)]
            forms.add(base)
            # Restore doubled consonant (e.g., scanned -> scan, runner -> run)
            if len(base) >= 3 and base[-1] == base[-2] and base[-1] in "bcdgmnprstz":
                forms.add(base[:-1])
            # Restore silent -e (e.g., tuning -> tune, profiling -> profile)
            forms.add(base + "e")
            # Restore -y (e.g., dependencies -> dependency)
            if sfx == "ies":
                forms.add(base + "y")
    return forms


def _match_phrase(phrase: str, text: str) -> bool:
    """Matches a multi-word or hyphenated phrase with word-boundary constraints."""
    parts = [p for p in re.split(r"[\s\-_]+", phrase.strip().lower()) if p]
    if not parts:
        return False
    lead = [re.escape(p) for p in parts[:-1]]
    tail = re.escape(parts[-1])
    if lead:
        joined = r"[\s\-_]+".join([*lead, tail])
        regex = rf"\b{joined}(?:ings|ing|ers|er|es|ed|s)?\b"
    else:
        regex = rf"\b{tail}\b"
    return bool(re.search(regex, text.lower()))


def _calculate_trigger_hits(triggers: frozenset[str], text: str, tokens: frozenset[str]) -> set[str]:
    """Evaluates hits against objective text using token intersect, stemming, and phrase regex."""
    hits: set[str] = set()
    token_stems: dict[str, set[str]] = {t: _stem_forms(t) for t in tokens}

    for trigger in triggers:
        trig_lower = trigger.lower()
        if " " in trig_lower or "-" in trig_lower or "_" in trig_lower:
            if _match_phrase(trig_lower, text):
                hits.add(trigger)
            continue

        # Single word: Direct token intersect
        if trig_lower in tokens:
            hits.add(trigger)
            continue

        # Single word: Stem / irregular intersection
        trig_stems = _stem_forms(trig_lower)
        matched = False
        for stems in token_stems.values():
            if trig_stems & stems:
                hits.add(trigger)
                matched = True
                break
        if matched:
            continue

        # Fallback word-boundary check for punctuation-joined words (e.g. app.py, vuln-check)
        pattern = rf"\b{re.escape(trig_lower)}(?:ings|ing|ers|er|es|ed|s)?\b"
        if re.search(pattern, text.lower()):
            hits.add(trigger)

    return hits


# --- Frontmatter Parsing & Dynamic Directory Ingestion ---


def _parse_yaml_frontmatter(text: str) -> dict[str, Any]:
    """Extract and parse YAML frontmatter from between --- markers."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end < 0:
        return {}
    raw = text[3:end].strip()

    # Try standard PyYAML if present
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
    except Exception:
        pass

    # Resilient zero-dependency parser for common SKILL.md frontmatter formats
    parsed: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- ") and current_key is not None:
            val = line[2:].strip().strip("\"'")
            if current_list is None:
                current_list = []
                parsed[current_key] = current_list
            current_list.append(val)
            continue

        if ":" in line:
            current_list = None
            key, rest = line.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            current_key = key

            if not rest:
                continue

            if rest.startswith("[") and rest.endswith("]"):
                items = [item.strip().strip("\"'") for item in rest[1:-1].split(",") if item.strip()]
                parsed[key] = items
            elif rest.lower() == "true":
                parsed[key] = True
            elif rest.lower() == "false":
                parsed[key] = False
            else:
                parsed[key] = rest.strip("\"'")

    return parsed


def _body_after_frontmatter(text: str) -> str:
    """Return markdown content below the frontmatter block."""
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end < 0:
        return text
    return text[end + 3 :].strip()


def _extract_triggers(text: str) -> frozenset[str]:
    """Derive heuristic trigger keywords from description and markdown body."""
    words: set[str] = set()
    lower = text.lower()
    patterns = [
        r"\b(bug|defect|failing|repro|debug|trace|symptoms|isolate|localize|patch)\b",
        r"\b(readability|naming|cohesion|refactor|clean|duplication|extract|rename)\b",
        r"\b(review|audit|verification|gate|check|inspect|qa|quality)\b",
        r"\b(slop|clutter|generated|comment|cleanup|over-defensive)\b",
        r"\b(format|standardize|normalize|polyglot|lint|style|import|ordering)\b",
        r"\b(design|aesthetic|frontend|typography|ui|ux|css|layout|motion|animation)\b",
        r"\b(document|structure|markdown|doc|index|readme|table|headings)\b",
        r"\b(observability|llm|monitoring|langfuse|span|token|cost|latency)\b",
        r"\b(pr|pull.request|merge|branch|commit|push|github|git)\b",
        r"\b(shell|bash|command|exec|terminal|cli|script)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, lower):
            w = m.group(0).replace(".", " ")
            if len(w) >= 2:
                words.add(w)
    return frozenset(words)


def import_skills_from_dir(root: Path) -> list[tuple[Skill, bool]]:
    """Scan root directory for SKILL.md bundles, parse metadata, and return (Skill, is_valid)."""
    results: list[tuple[Skill, bool]] = []
    if not root.is_dir():
        return results

    for skill_file in sorted(root.rglob("SKILL.md")):
        if not skill_file.is_file():
            continue

        # Do not ingest hidden skill trees such as .git or .venv when a broad
        # skills root is supplied. The root itself is allowed to be hidden.
        relative_parts = skill_file.relative_to(root).parts[:-1]
        if any(part.startswith(".") for part in relative_parts):
            continue

        try:
            text = skill_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        meta = _parse_yaml_frontmatter(text)
        name = str(meta.get("name", skill_file.parent.name))
        description = str(meta.get("description", ""))
        valid = bool(name and description)

        body = _body_after_frontmatter(text)
        guidance = body if body else str(meta.get("guidance", ""))

        # Parse triggers
        triggers_raw = meta.get("triggers")
        if isinstance(triggers_raw, list):
            triggers_seq = cast(list[object], triggers_raw)
            triggers = frozenset(str(t).strip() for t in triggers_seq if t)
        elif isinstance(triggers_raw, str):
            triggers = frozenset(t.strip() for t in triggers_raw.split(",") if t.strip())
        else:
            triggers = _extract_triggers(f"{description} {body}")

        # Parse tools
        tools_raw = meta.get("tools")
        if isinstance(tools_raw, list):
            tools_seq = cast(list[object], tools_raw)
            tools = frozenset(str(t).strip() for t in tools_seq if t)
        elif isinstance(tools_raw, str):
            tools = frozenset(t.strip() for t in tools_raw.split(",") if t.strip())
        else:
            tools = frozenset[str]()

        # Parse capabilities
        category = str(meta.get("category", "other")).lower()
        cap_map: dict[str, frozenset[str]] = {
            "development": frozenset({"coding"}),
            "design": frozenset({"design"}),
            "operations": frozenset({"ops"}),
            "security": frozenset({"security"}),
            "review": frozenset({"review"}),
        }
        capabilities = cap_map.get(category, frozenset[str]())

        # Parse tags
        tags_raw = meta.get("tags")
        if isinstance(tags_raw, list):
            tags_seq = cast(list[object], tags_raw)
            tags = frozenset(str(t).strip() for t in tags_seq if t)
        elif isinstance(tags_raw, str):
            tags = frozenset(t.strip() for t in tags_raw.split(",") if t.strip())
        else:
            tags = frozenset[str]()

        always = bool(meta.get("always", False))
        min_risk = RiskLevel.from_str(str(meta.get("min_risk", "LOW")))
        max_risk = RiskLevel.from_str(str(meta.get("max_risk", "CRITICAL")))

        skill = Skill(
            name=name,
            description=description,
            tools=tools,
            triggers=triggers,
            capabilities=capabilities,
            tags=tags,
            always=always,
            min_risk=min_risk,
            max_risk=max_risk,
            guidance=guidance,
        )
        results.append((skill, valid))

    return results


# --- Skill Catalog ---


class SkillCatalog:
    """Catalog holding registered skills and computing matches & tool allowlists."""

    def __init__(
        self,
        skills: list[Skill] | None = None,
        *,
        gate_tools: bool = False,
        min_score: float = 0.5,
    ) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in (skills or [])}
        self.gate_tools = gate_tools
        self.min_score = min_score

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def all_managed_tools(self) -> frozenset[str]:
        """Union of all tools claimed by registered skills."""
        tools: set[str] = set()
        for s in self._skills.values():
            tools.update(s.tools)
        return frozenset(tools)

    def match(
        self,
        objective: str,
        capabilities: frozenset[str] = frozenset(),
        risk: RiskLevel = RiskLevel.LOW,
        metadata: Mapping[str, Any] | None = None,
        min_score: float | None = None,
    ) -> list[tuple[Skill, float]]:
        """Evaluate matching skills based on text triggers, capabilities, and tags."""
        cutoff = min_score if min_score is not None else self.min_score
        tokens = _tokenize(objective)
        meta = metadata or {}
        explicit_skills = set(meta.get("skills", []))
        explicit_tags = set(meta.get("skill_tags", []))

        matched: list[tuple[Skill, float]] = []

        for skill in self._skills.values():
            # Risk Gate: Reject if task risk falls outside the skill's interval
            if not (skill.min_risk <= risk <= skill.max_risk):
                continue

            scores: list[float] = []

            if skill.always:
                scores.append(1.0)

            if skill.name in explicit_skills:
                scores.append(1.0)

            if skill.capabilities and capabilities:
                overlap = len(skill.capabilities & capabilities)
                if overlap > 0:
                    scores.append(0.55 + 0.45 * (overlap / len(skill.capabilities)))

            if skill.triggers:
                hits = _calculate_trigger_hits(skill.triggers, objective, tokens)
                if hits:
                    scores.append(min(1.0, 0.45 + 0.20 * len(hits)))

            if skill.tags and (skill.tags & explicit_tags):
                scores.append(0.75)

            if not scores:
                continue

            final_score = max(scores)
            if final_score >= cutoff:
                matched.append((skill, round(final_score, 4)))

        matched.sort(key=lambda x: x[1], reverse=True)
        return matched

    def compute_allowed_tools(
        self,
        active_skills: list[Skill],
        all_system_tools: frozenset[str],
    ) -> frozenset[str]:
        """Calculate effective allowed tools: (all - managed) + active."""
        if not self.gate_tools:
            return all_system_tools

        managed = self.all_managed_tools()
        active_tools: set[str] = set()
        for s in active_skills:
            active_tools.update(s.tools)

        unmanaged = all_system_tools - managed
        return frozenset(unmanaged | active_tools)

    def import_from_dir(self, directory: Path | str) -> int:
        """Scan directory and register all valid skills found."""
        dir_path = Path(directory)
        imported = import_skills_from_dir(dir_path)
        count = 0
        for skill, valid in imported:
            if valid:
                self.register(skill)
                count += 1
        return count


def default_skills() -> list[Skill]:
    """Default baseline skill battery for coding and inspection tasks."""
    return [
        Skill(
            name="shell-command",
            description="Execute terminal commands and inspect subprocess outputs.",
            tools=frozenset({"bash"}),
            triggers=frozenset(
                {
                    "bash",
                    "terminal",
                    "shell",
                    "command",
                    "script",
                    "exec",
                    "subprocess",
                    "run",
                }
            ),
            capabilities=frozenset({"ops", "execution"}),
            tags=frozenset({"shell"}),
            guidance=(
                "Use the bash tool prudently. Verify commands before running, "
                "observe timeouts, and never run unbounded loops."
            ),
            min_risk=RiskLevel.LOW,
            max_risk=RiskLevel.CRITICAL,
        ),
        Skill(
            name="code-debugger",
            description="Diagnose crashes, race conditions, test failures, and exceptions.",
            tools=frozenset(),
            triggers=frozenset(
                {
                    "debug",
                    "debugging",
                    "crash",
                    "exception",
                    "trace",
                    "stacktrace",
                    "deadlock",
                    "defect",
                    "symptom",
                    "failing",
                }
            ),
            capabilities=frozenset({"diagnostics", "coding"}),
            tags=frozenset({"debug"}),
            guidance=(
                "Form hypotheses based on logs and stack traces. Inspect state before "
                "modifying code. Isolate the minimal reproduction case."
            ),
            min_risk=RiskLevel.LOW,
            max_risk=RiskLevel.HIGH,
        ),
        Skill(
            name="code-reviewer",
            description="Evaluate code quality, style, security invariants, and regressions.",
            tools=frozenset(),
            triggers=frozenset(
                {
                    "review",
                    "audit",
                    "inspect",
                    "pr",
                    "pull request",
                    "lint",
                    "code review",
                }
            ),
            capabilities=frozenset({"review"}),
            tags=frozenset({"review"}),
            guidance=(
                "Check for correctness, potential security vulnerabilities, edge cases, "
                "and adherence to project guidelines. Provide actionable feedback."
            ),
            min_risk=RiskLevel.LOW,
            max_risk=RiskLevel.CRITICAL,
        ),
        Skill(
            name="stop-slop",
            description="Remove AI writing patterns, throat-clearing, jargon, and crutches from prose.",
            tools=frozenset(
                {
                    "stop_slop_analyze",
                    "stop_slop_rewrite",
                    "stop_slop_rules",
                    "stop_slop_examples",
                }
            ),
            triggers=frozenset(
                {
                    "slop",
                    "de-slop",
                    "prose",
                    "writing",
                    "draft",
                    "ai patterns",
                    "clean prose",
                    "throat clearing",
                    "buzzword",
                    "jargon",
                    "adverb",
                }
            ),
            capabilities=frozenset({"review", "coding"}),
            tags=frozenset({"writing", "prose", "quality"}),
            guidance=(
                "Eliminate predictable AI writing patterns: cut throat-clearing openers, "
                "emphasis crutches, business jargon, and adverbs. Break binary contrasts "
                "and remove em dashes. Rate prose on 5 dimensions (Directness, Rhythm, "
                "Trust, Authenticity, Density); revise if below 35/50."
            ),
            min_risk=RiskLevel.LOW,
            max_risk=RiskLevel.CRITICAL,
        ),
    ]


# --- Skills Plugin ---


class SkillsPlugin:
    """Plugin mounting dynamic skill catalog, context guidance, and tool allowlist gating."""

    id = "harness-skills"

    def __init__(
        self,
        catalog: SkillCatalog | None = None,
        skills_dir: Path | None = None,
        gate_tools: bool = False,
        default_risk: RiskLevel = RiskLevel.LOW,
        min_score: float = 0.5,
    ) -> None:
        self.catalog = (
            catalog
            if catalog is not None
            else SkillCatalog(default_skills(), gate_tools=gate_tools, min_score=min_score)
        )
        self.catalog.gate_tools = gate_tools
        self.skills_dir = skills_dir
        self.default_risk = default_risk
        self.min_score = min_score
        self.active_skills: list[Skill] = []
        self.allowed_tools: frozenset[str] = frozenset()

    def apply(self, ctx: Context) -> None:
        if "skills" not in ctx.services:
            ctx.provide("skills", self.catalog)

        # Ingest external skills if directory configured or standard candidate directories exist
        search_dirs: list[Path] = []
        if self.skills_dir is not None:
            if self.skills_dir.is_dir():
                search_dirs.append(self.skills_dir)
        else:
            repo_skills = Path(__file__).resolve().parent.parent.parent / "skills"
            cwd_skills = Path.cwd() / "skills"
            if repo_skills.is_dir():
                search_dirs.append(repo_skills)
            elif cwd_skills.is_dir():
                search_dirs.append(cwd_skills)

        for s_dir in search_dirs:
            count = self.catalog.import_from_dir(s_dir)
            get_logger().info(
                "imported skills from directory",
                extra={"dir": str(s_dir), "count": count},
            )

        async def on_request(request: dict[str, Any], next_: Any) -> dict[str, Any]:
            messages: list[dict[str, Any]] = request.get("messages", [])
            # Extract latest user prompt
            user_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_text = str(msg.get("content", ""))
                    break

            matched = self.catalog.match(
                objective=user_text,
                risk=self.default_risk,
                min_score=self.min_score,
            )
            self.active_skills = [skill for skill, _ in matched]

            # Log match event into session if available
            sessions = ctx.services.get("sessions")
            if sessions is not None:
                sessions.append(
                    "skills/matched",
                    {
                        "skills": [s.name for s, _ in matched],
                        "scores": {s.name: score for s, score in matched},
                        "gate_tools": self.catalog.gate_tools,
                    },
                )

            # Inject guidance into request messages
            guidance_entries = [
                f"[Skill: {s.name}]\n{s.guidance.strip()}"
                for s in self.active_skills
                if s.guidance.strip()
            ]
            if guidance_entries:
                skills_context = "Active Skills Guidance:\n" + "\n\n".join(guidance_entries)
                # If first message is system, augment it; else prepend new system message
                messages_copy = copy.deepcopy(messages)
                if messages_copy and messages_copy[0].get("role") == "system":
                    orig = messages_copy[0].get("content", "")
                    messages_copy[0]["content"] = f"{orig}\n\n{skills_context}"
                else:
                    messages_copy.insert(0, {"role": "system", "content": skills_context})
                request["messages"] = messages_copy

            # Filter tools schema if gate_tools is active
            all_tools: set[str] = set()
            tools_schemas_raw = request.get("tools")
            if isinstance(tools_schemas_raw, list):
                ts_list = cast(list[dict[str, Any]], tools_schemas_raw)
                for ts in ts_list:
                    if "function" in ts and isinstance(ts["function"], dict):
                        fn_dict = cast(dict[str, Any], ts["function"])
                        fn_name = fn_dict.get("name")
                        if isinstance(fn_name, str):
                            all_tools.add(fn_name)

            self.allowed_tools = self.catalog.compute_allowed_tools(
                self.active_skills, frozenset(all_tools)
            )

            if self.catalog.gate_tools and isinstance(tools_schemas_raw, list):
                ts_list = cast(list[dict[str, Any]], tools_schemas_raw)
                filtered_schemas: list[dict[str, Any]] = []
                for ts in ts_list:
                    if "function" in ts and isinstance(ts["function"], dict):
                        fn_dict = cast(dict[str, Any], ts["function"])
                        fn_name = fn_dict.get("name")
                        if isinstance(fn_name, str) and fn_name in self.allowed_tools:
                            filtered_schemas.append(ts)
                request["tools"] = filtered_schemas

            return await next_(request)

        async def on_pre_execute(call: dict[str, Any], next_: Any) -> dict[str, Any]:
            name = call.get("name")
            if not isinstance(name, str):
                return await next_(call)

            # If tool gating is on, enforce that skill-managed tools must be allowed
            if self.catalog.gate_tools:
                managed = self.catalog.all_managed_tools()
                if name in managed and name not in self.allowed_tools:
                    return {
                        **call,
                        "denied": True,
                        "error": f"DENIED_BY_SKILL_POLICY: Tool '{name}' is not unlocked by any active skill",
                    }

            return await next_(call)

        ctx.on("agent/request", on_request)
        ctx.on("tools/pre-execute", on_pre_execute)
