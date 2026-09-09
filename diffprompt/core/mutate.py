"""
Fixed, hardcoded template transformations for `diffprompt evolve`.
Plain string/list operations — no LLM involved anywhere in this module.
"""
from __future__ import annotations
import random
import re

_WORD_LIMIT_RE = re.compile(r"(\d+)(\s+words\b)", re.IGNORECASE)

_SOFT_TO_EXPLICIT = {
    "briefly": "Keep responses to 3 sentences or fewer.",
    "concise": "Keep responses to 3 sentences or fewer.",
    "detailed": "Provide a thorough, multi-paragraph explanation.",
    "formal": "Use formal, professional language throughout.",
}
_FALLBACK_CONSTRAINT = "Follow every constraint above exactly as written."

_FORMAT_OPTIONS = [
    "Respond in valid JSON.",
    "Use bullet points for lists.",
    "Respond in plain text with no markdown formatting.",
]
_FORMAT_MARKERS = ("json", "bullet", "markdown")


def to_lines(prompt: str) -> list[str]:
    return [line.strip() for line in prompt.split("\n") if line.strip()]


def _reorder(lines: list[str], rng: random.Random) -> list[str]:
    if len(lines) <= 2:
        return list(lines)
    head, rest = lines[0], list(lines[1:])
    rng.shuffle(rest)
    return [head] + rest


def _tighten_word_limit(lines: list[str], rng: random.Random) -> list[str]:
    new_lines = list(lines)
    for i, line in enumerate(new_lines):
        m = _WORD_LIMIT_RE.search(line)
        if m:
            new_n = max(5, round(int(m.group(1)) * 0.7))
            new_lines[i] = _WORD_LIMIT_RE.sub(f"{new_n}\\2", line, count=1)
            return new_lines
    new_lines.append("Respond in under 50 words.")
    return new_lines


def _explicit_constraint(lines: list[str], rng: random.Random) -> list[str]:
    joined = " ".join(lines).lower()
    new_lines = list(lines)
    for soft, explicit in _SOFT_TO_EXPLICIT.items():
        if soft in joined and explicit not in new_lines:
            new_lines.append(explicit)
            return new_lines
    if _FALLBACK_CONSTRAINT not in new_lines:
        new_lines.append(_FALLBACK_CONSTRAINT)
    return new_lines


def _add_format(lines: list[str], rng: random.Random) -> list[str]:
    new_lines = list(lines)
    candidates = [f for f in _FORMAT_OPTIONS if f not in new_lines]
    if candidates:
        new_lines.append(rng.choice(candidates))
    return new_lines


def _remove_format(lines: list[str], rng: random.Random) -> list[str]:
    filtered = [line for line in lines if not any(m in line.lower() for m in _FORMAT_MARKERS)]
    return filtered if filtered else list(lines)


TRANSFORMS = {
    "reorder": _reorder,
    "tighten_word_limit": _tighten_word_limit,
    "explicit_constraint": _explicit_constraint,
    "add_format": _add_format,
    "remove_format": _remove_format,
}


def apply_transform(prompt: str, name: str, rng: random.Random) -> str:
    lines = TRANSFORMS[name](to_lines(prompt), rng)
    return "\n".join(lines)


def mutate(prompt: str, rng: random.Random) -> str:
    """Apply one randomly-chosen transform from the fixed menu."""
    name = rng.choice(list(TRANSFORMS))
    return apply_transform(prompt, name, rng)
