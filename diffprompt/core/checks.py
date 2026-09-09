"""
Deterministic check evaluation for golden tasks.
Every check returns 0.0 or 1.0 — pure functions, no LLM, no network.
"""
from __future__ import annotations
import json
import re

import jsonschema

from diffprompt.models import Check, CheckType

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def score_check(check: Check, output: str) -> float:
    if check.type == CheckType.REGEX:
        return _score_regex(check, output)
    if check.type == CheckType.KEYWORD:
        return _score_keyword(check, output)
    if check.type == CheckType.JSON_SCHEMA:
        return _score_json_schema(check, output)
    if check.type == CheckType.NUMERIC:
        return _score_numeric(check, output)
    raise ValueError(f"unknown check type: {check.type}")  # pragma: no cover


def _score_regex(check: Check, output: str) -> float:
    flags = re.IGNORECASE if check.ignore_case else 0
    return 1.0 if re.search(check.pattern, output, flags) else 0.0


def _score_keyword(check: Check, output: str) -> float:
    haystack = output.lower()
    present = [k for k in check.keywords if k.lower() in haystack]
    if check.match_any:
        return 1.0 if present else 0.0
    return 1.0 if len(present) == len(check.keywords) else 0.0


def _score_json_schema(check: Check, output: str) -> float:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return 0.0
    try:
        jsonschema.validate(data, check.schema_)
    except jsonschema.ValidationError:
        return 0.0
    return 1.0


def _extract_number(output: str, extract_pattern: str | None) -> float | None:
    if extract_pattern:
        m = re.search(extract_pattern, output)
        if not m:
            return None
        try:
            return float(m.group(1))
        except (IndexError, ValueError):
            return None
    m = _NUMBER_RE.search(output)
    return float(m.group()) if m else None


def _score_numeric(check: Check, output: str) -> float:
    value = _extract_number(output, check.extract_pattern)
    if value is None:
        return 0.0
    if check.comparator == "eq":
        return 1.0 if abs(value - check.expected) <= check.tolerance else 0.0
    if check.comparator == "gte":
        return 1.0 if value >= check.expected - check.tolerance else 0.0
    if check.comparator == "lte":
        return 1.0 if value <= check.expected + check.tolerance else 0.0
    raise ValueError(f"unknown comparator: {check.comparator}")  # pragma: no cover
