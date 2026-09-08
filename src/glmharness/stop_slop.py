"""Stop-Slop Engine & Toolset.

Provides deterministic pattern detection, multi-dimensional scoring (Directness,
Rhythm, Trust, Authenticity, Density), rule-based rewriting, and model-facing tools
to eliminate predictable AI tells and slop from prose.
"""

from __future__ import annotations

import re
from typing import Any

from .tools import Tool

# --- Rules & Replacements ---

JARGON_REPLACEMENTS: dict[str, str] = {
    "navigate challenges": "handle challenges",
    "navigate uncertainty": "handle uncertainty",
    "navigate": "handle",
    "unpack": "explain",
    "lean into discomfort": "accept discomfort",
    "lean into": "accept",
    "fast-paced landscape": "competitive market",
    "landscape": "situation",
    "game-changer": "significant change",
    "game changer": "significant change",
    "double down": "commit",
    "deep dive": "analysis",
    "take a step back": "reconsider",
    "moving forward": "next",
    "circle back": "return to",
    "on the same page": "aligned",
}

THROAT_CLEARING: tuple[str, ...] = (
    "here's the thing:",
    "here's the thing",
    "the uncomfortable truth is",
    "it turns out that",
    "it turns out",
    "let me be clear",
    "the truth is,",
    "the truth is",
    "i'll say it again:",
    "i'll say it again",
    "i'm going to be honest",
    "can we talk about",
    "here's what i find interesting",
    "here's the problem though",
    "what makes this hard is",
    "here's what you need to know",
    "here's what",
    "here's why that matters",
    "here's why",
    "here's this",
    "here's that",
)

EMPHASIS_CRUTCHES: tuple[str, ...] = (
    "full stop.",
    "full stop",
    "period.",
    "let that sink in.",
    "let that sink in",
    "this matters because",
    "make no mistake",
    "think about it:",
    "think about it",
    "and that's okay.",
    "and that's okay",
)

FILLER_PHRASES: tuple[str, ...] = (
    "at its core",
    "in today's fast-paced",
    "in today's",
    "it's worth noting that",
    "it's worth noting",
    "at the end of the day",
    "when it comes to",
    "in a world where",
    "the reality is that",
    "the reality is",
)

META_COMMENTARY: tuple[str, ...] = (
    "hint:",
    "plot twist:",
    "spoiler:",
    "you already know this, but",
    "but that's another post",
    "is a feature, not a bug",
    "dressed up as",
    "the rest of this essay explains",
    "the rest of this essay",
    "let me walk you through",
    "in this section, we'll",
    "in this section",
    "as we'll see",
    "i want to explore",
)

OFFENDER_ADVERBS: tuple[str, ...] = (
    "really",
    "just",
    "literally",
    "genuinely",
    "honestly",
    "simply",
    "actually",
    "deeply",
    "truly",
    "fundamentally",
    "inherently",
    "inevitably",
    "interestingly",
    "importantly",
    "crucially",
)

BINARY_CONTRAST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bnot because\b.*?\b(?:but because|because)\b", re.IGNORECASE),
    re.compile(r"\bisn't the problem\b.*?\bis\b", re.IGNORECASE),
    re.compile(r"\bthe answer isn't\b.*?\bit's\b", re.IGNORECASE),
    re.compile(r"\bit feels like\b.*?\bit's actually\b", re.IGNORECASE),
    re.compile(r"\bthe question isn't\b.*?\bit's\b", re.IGNORECASE),
    re.compile(r"\bnot\s+[a-z0-9_\- ]+,\s*(?:but|it's)\s+[a-z0-9_\- ]+", re.IGNORECASE),
    re.compile(r"\bit's not this\b.*?\bit's that\b", re.IGNORECASE),
    re.compile(r"\bstops being\b.*?\band starts being\b", re.IGNORECASE),
    re.compile(r"\bnot just\b.*?\bbut also\b", re.IGNORECASE),
)

VAGUE_DECLARATIVES: tuple[str, ...] = (
    "the reasons are structural",
    "the implications are significant",
    "this is the deepest problem",
    "the stakes are high",
    "the consequences are real",
)

CANONICAL_EXAMPLES: list[dict[str, str]] = [
    {
        "title": "Example 1: Throat-Clearing + Binary Contrast",
        "before": (
            "Here's the thing: building products is hard. Not because the technology is complex. "
            "Because people are complex. Let that sink in."
        ),
        "after": "Building products is hard. Technology is manageable. People aren't.",
        "changes": "Removed opener, binary contrast structure, and emphasis crutch. Direct statements.",
    },
    {
        "title": "Example 2: Filler + Unnecessary Reassurance",
        "before": (
            "It turns out that most teams struggle with alignment. The uncomfortable truth is that "
            "nobody wants to admit they're confused. And that's okay."
        ),
        "after": "Teams struggle with alignment. Nobody admits confusion.",
        "changes": (
            "Cut hedging ('most'), removed throat-clearing phrases, deleted permission-granting ending."
        ),
    },
    {
        "title": "Example 3: Business Jargon Stack",
        "before": (
            "In today's fast-paced landscape, we need to lean into discomfort and navigate uncertainty "
            "with clarity. This matters because your competition isn't waiting."
        ),
        "after": "Move faster. Your competition is.",
        "changes": "Eliminated jargon entirely. Core message in six words.",
    },
    {
        "title": "Example 4: Dramatic Fragmentation",
        "before": "Speed. Quality. Cost. You can only pick two. That's it. That's the tradeoff.",
        "after": "Speed, quality, cost—pick two.",
        "changes": "Single sentence. No performative emphasis.",
    },
    {
        "title": "Example 5: Rhetorical Setup",
        "before": (
            "What if I told you that the best teams don't optimize for productivity? "
            "Here's what I mean: they optimize for learning. Think about it."
        ),
        "after": "The best teams optimize for learning, not productivity.",
        "changes": "Direct claim. No rhetorical scaffolding.",
    },
]


class StopSlopEngine:
    """Core analysis, scoring, and rewriting engine for prose slop reduction."""

    def analyze(self, text: str) -> dict[str, Any]:
        """Evaluate text across 5 dimensions and return violations + scores."""
        violations: list[dict[str, Any]] = []
        lower = text.lower()

        # 1. Throat-clearing openers
        tc_spans: list[tuple[int, int]] = []
        for phrase in sorted(THROAT_CLEARING, key=len, reverse=True):
            idx = lower.find(phrase)
            while idx != -1:
                end = idx + len(phrase)
                if not any(s <= idx < e or s < end <= e for s, e in tc_spans):
                    tc_spans.append((idx, end))
                    line_num = text[:idx].count("\n") + 1
                    violations.append({
                        "category": "throat_clearing",
                        "phrase": phrase,
                        "line": line_num,
                        "suggestion": "Delete announcement and state point directly.",
                    })
                idx = lower.find(phrase, idx + 1)

        # 2. Emphasis crutches
        ec_spans: list[tuple[int, int]] = []
        for phrase in sorted(EMPHASIS_CRUTCHES, key=len, reverse=True):
            idx = lower.find(phrase)
            while idx != -1:
                end = idx + len(phrase)
                if not any(s <= idx < e or s < end <= e for s, e in ec_spans):
                    ec_spans.append((idx, end))
                    line_num = text[:idx].count("\n") + 1
                    violations.append({
                        "category": "emphasis_crutch",
                        "phrase": phrase,
                        "line": line_num,
                        "suggestion": "Delete performative emphasis crutch.",
                    })
                idx = lower.find(phrase, idx + 1)

        # 3. Meta-commentary
        mc_spans: list[tuple[int, int]] = []
        for phrase in sorted(META_COMMENTARY, key=len, reverse=True):
            idx = lower.find(phrase)
            while idx != -1:
                end = idx + len(phrase)
                if not any(s <= idx < e or s < end <= e for s, e in mc_spans):
                    mc_spans.append((idx, end))
                    line_num = text[:idx].count("\n") + 1
                    violations.append({
                        "category": "meta_commentary",
                        "phrase": phrase,
                        "line": line_num,
                        "suggestion": "Cut meta-commentary; let the prose move forward directly.",
                    })
                idx = lower.find(phrase, idx + 1)

        # 4. Business jargon
        jargon_spans: list[tuple[int, int]] = []
        for jargon, replacement in sorted(
            JARGON_REPLACEMENTS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            pattern = rf"\b{re.escape(jargon)}\b"
            for m in re.finditer(pattern, text, re.IGNORECASE):
                s, e = m.span()
                if not any(js <= s < je or js < e <= je for js, je in jargon_spans):
                    jargon_spans.append((s, e))
                    line_num = text[:s].count("\n") + 1
                    violations.append({
                        "category": "business_jargon",
                        "phrase": m.group(0),
                        "line": line_num,
                        "suggestion": f"Use plain language: '{replacement}'.",
                    })

        # 5. Adverbs & filler phrases
        for phrase in FILLER_PHRASES:
            idx = lower.find(phrase)
            while idx != -1:
                line_num = text[:idx].count("\n") + 1
                violations.append({
                    "category": "filler_phrase",
                    "phrase": phrase,
                    "line": line_num,
                    "suggestion": "Delete filler phrase.",
                })
                idx = lower.find(phrase, idx + len(phrase))

        for adverb in OFFENDER_ADVERBS:
            pattern = rf"\b{re.escape(adverb)}\b"
            for m in re.finditer(pattern, text, re.IGNORECASE):
                line_num = text[: m.start()].count("\n") + 1
                violations.append({
                    "category": "adverb",
                    "phrase": m.group(0),
                    "line": line_num,
                    "suggestion": f"Remove adverb '{m.group(0)}'.",
                })

        # 6. Binary contrasts
        for pat in BINARY_CONTRAST_PATTERNS:
            for m in pat.finditer(text):
                line_num = text[: m.start()].count("\n") + 1
                violations.append({
                    "category": "binary_contrast",
                    "phrase": m.group(0),
                    "line": line_num,
                    "suggestion": "Drop negation runway; state positive point directly.",
                })

        # 7. Vague declaratives
        for phrase in VAGUE_DECLARATIVES:
            idx = lower.find(phrase)
            while idx != -1:
                line_num = text[:idx].count("\n") + 1
                violations.append({
                    "category": "vague_declarative",
                    "phrase": phrase,
                    "line": line_num,
                    "suggestion": "Name the specific thing instead of announcing significance.",
                })
                idx = lower.find(phrase, idx + len(phrase))

        # 8. Em-dashes
        em_dash_matches = list(re.finditer(r"—|--", text))
        for m in em_dash_matches:
            line_num = text[: m.start()].count("\n") + 1
            violations.append({
                "category": "em_dash",
                "phrase": m.group(0),
                "line": line_num,
                "suggestion": "Remove em dash; use a comma, period, or separate sentence.",
            })

        # Multi-dimensional scoring (1 to 10 each, total /50)
        throat_count = sum(1 for v in violations if v["category"] == "throat_clearing")
        meta_count = sum(1 for v in violations if v["category"] == "meta_commentary")
        crutch_count = sum(1 for v in violations if v["category"] == "emphasis_crutch")
        jargon_count = sum(1 for v in violations if v["category"] == "business_jargon")
        filler_count = sum(1 for v in violations if v["category"] in ("filler_phrase", "adverb"))
        binary_count = sum(1 for v in violations if v["category"] == "binary_contrast")
        vague_count = sum(1 for v in violations if v["category"] == "vague_declarative")
        em_dash_count = len(em_dash_matches)

        directness = max(1, 10 - 2 * throat_count - 2 * meta_count - 2 * vague_count)
        rhythm = max(1, 10 - 2 * binary_count - em_dash_count)
        trust = max(1, 10 - 2 * crutch_count)
        authenticity = max(1, 10 - 2 * jargon_count)
        density = max(1, 10 - filler_count)

        total_score = directness + rhythm + trust + authenticity + density
        passes = total_score >= 35

        summary_parts = [
            f"Score: {total_score}/50 ({'PASS' if passes else 'REVISE NEEDED'}).",
            (
                f"Directness: {directness}/10 | Rhythm: {rhythm}/10 | Trust: {trust}/10 "
                f"| Authenticity: {authenticity}/10 | Density: {density}/10."
            ),
            f"Violations detected: {len(violations)}.",
        ]

        return {
            "total_score": total_score,
            "passes_threshold": passes,
            "dimensions": {
                "directness": directness,
                "rhythm": rhythm,
                "trust": trust,
                "authenticity": authenticity,
                "density": density,
            },
            "violations_count": len(violations),
            "violations": violations,
            "summary": " ".join(summary_parts),
        }

    def rewrite(self, text: str) -> dict[str, Any]:
        """Apply deterministic cleaning to remove slop and substitute jargon."""
        changes: list[str] = []
        result = text

        # 1. Replace business jargon
        for jargon, replacement in sorted(
            JARGON_REPLACEMENTS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            pattern = re.compile(rf"\b{re.escape(jargon)}\b", re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub(replacement, result)
                changes.append(f"Replaced jargon '{jargon}' -> '{replacement}'")

        # 2. Strip emphasis crutches
        for crutch in EMPHASIS_CRUTCHES:
            pattern = re.compile(rf"\b{re.escape(crutch)}", re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub("", result)
                changes.append(f"Removed emphasis crutch: '{crutch}'")

        # 3. Strip throat-clearing openers
        for opener in THROAT_CLEARING:
            pattern = re.compile(rf"(^|[\.\?!]\s+){re.escape(opener)}\s*", re.IGNORECASE | re.MULTILINE)
            if pattern.search(result):
                result = pattern.sub(r"\1", result)
                changes.append(f"Removed throat-clearing opener: '{opener}'")

        # 4. Remove offender adverbs
        for adv in OFFENDER_ADVERBS:
            pattern = re.compile(rf"\b{re.escape(adv)}\s+", re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub("", result)
                changes.append(f"Removed adverb: '{adv}'")

        # 5. Remove filler phrases
        for filler in FILLER_PHRASES:
            pattern = re.compile(rf"\b{re.escape(filler)}\s*", re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub("", result)
                changes.append(f"Removed filler phrase: '{filler}'")

        # 6. Replace em-dashes with comma or period
        if re.search(r"—|--", result):
            result = re.sub(r"\s*(?:—|--)\s*", ", ", result)
            changes.append("Replaced em dashes with standard punctuation")

        # Normalize whitespace
        result = re.sub(r"[ \t]+", " ", result)
        result = re.sub(r"\s*([,\.\?!])", r"\1", result)
        result = re.sub(r"([,\.\?!])\s*([,\.\?!])+", r"\1", result)
        result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)
        result = result.strip()

        return {
            "original": text,
            "rewritten": result,
            "changes_count": len(changes),
            "changes": changes,
        }

    def get_rules(self, category: str | None = None) -> dict[str, Any]:
        """Return reference rules and scoring metrics."""
        all_rules: dict[str, Any] = {
            "throat_clearing": {
                "description": "Announcement phrases before the point. State content directly.",
                "forbidden_phrases": list(THROAT_CLEARING),
            },
            "emphasis_crutches": {
                "description": "Performative emphasis phrases that add no meaning.",
                "forbidden_phrases": list(EMPHASIS_CRUTCHES),
            },
            "business_jargon": {
                "description": "Buzzwords and corporate cliches to replace with plain language.",
                "replacements": JARGON_REPLACEMENTS,
            },
            "adverbs": {
                "description": "Kill all adverbs and softeners.",
                "offenders": list(OFFENDER_ADVERBS),
                "filler_phrases": list(FILLER_PHRASES),
            },
            "meta_commentary": {
                "description": "Self-referential asides about the text's structure.",
                "forbidden_phrases": list(META_COMMENTARY),
            },
            "structures": {
                "description": "Formulaic sentence architectures to break.",
                "avoid": [
                    "Binary contrasts ('Not X. Because Y.')",
                    "Negative listing ('Not a X... Not a Y... A Z.')",
                    "Dramatic fragmentation ('Noun. That's it. That's the thing.')",
                    "False agency ('a complaint becomes a fix')",
                    "Narrator from a distance ('Nobody designed this')",
                    "Em dashes (use commas or periods)",
                ],
            },
            "scoring": {
                "dimensions": [
                    {"name": "Directness", "max": 10, "question": "Statements or announcements?"},
                    {"name": "Rhythm", "max": 10, "question": "Varied or metronomic?"},
                    {"name": "Trust", "max": 10, "question": "Respects reader intelligence?"},
                    {"name": "Authenticity", "max": 10, "question": "Sounds human?"},
                    {"name": "Density", "max": 10, "question": "Anything cuttable?"},
                ],
                "passing_threshold": 35,
                "verdict": "Below 35/50: revise.",
            },
        }

        if category and category in all_rules:
            return {category: all_rules[category]}
        return all_rules

    def get_examples(self) -> list[dict[str, str]]:
        """Return canonical before/after transformations."""
        return CANONICAL_EXAMPLES


# --- Tool Factories ---

_DEFAULT_ENGINE = StopSlopEngine()


def make_stop_slop_analyze_tool(engine: StopSlopEngine | None = None) -> Tool:
    eng = engine or _DEFAULT_ENGINE
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "The prose text to inspect and score for AI patterns, throat-clearing, and jargon."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        text = str(args["text"])
        return eng.analyze(text)

    return Tool(
        name="stop_slop_analyze",
        description=(
            "Analyze prose for AI writing patterns, throat-clearing, jargon, adverbs, em-dashes, "
            "and score across 5 dimensions (Directness, Rhythm, Trust, Authenticity, Density)."
        ),
        schema=schema,
        handler=handler,
    )


def make_stop_slop_rewrite_tool(engine: StopSlopEngine | None = None) -> Tool:
    eng = engine or _DEFAULT_ENGINE
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The prose text to rewrite and clean of AI tells.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        text = str(args["text"])
        return eng.rewrite(text)

    return Tool(
        name="stop_slop_rewrite",
        description=(
            "Rewrite prose deterministically by removing filler, throat-clearing, emphasis crutches, "
            "em-dashes, and replacing business jargon."
        ),
        schema=schema,
        handler=handler,
    )


def make_stop_slop_rules_tool(engine: StopSlopEngine | None = None) -> Tool:
    eng = engine or _DEFAULT_ENGINE
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": (
                    "Optional category filter: throat_clearing, emphasis_crutches, "
                    "business_jargon, adverbs, meta_commentary, structures, scoring."
                ),
                "enum": [
                    "throat_clearing",
                    "emphasis_crutches",
                    "business_jargon",
                    "adverbs",
                    "meta_commentary",
                    "structures",
                    "scoring",
                ],
            },
        },
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        category = args.get("category")
        return eng.get_rules(str(category) if category else None)

    return Tool(
        name="stop_slop_rules",
        description=(
            "Retrieve the stop-slop style guide rules, forbidden patterns, and plain language alternatives."
        ),
        schema=schema,
        handler=handler,
    )


def make_stop_slop_examples_tool(engine: StopSlopEngine | None = None) -> Tool:
    eng = engine or _DEFAULT_ENGINE
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"examples": eng.get_examples()}

    return Tool(
        name="stop_slop_examples",
        description=(
            "Retrieve canonical before/after examples demonstrating how to eliminate AI writing tells."
        ),
        schema=schema,
        handler=handler,
    )
