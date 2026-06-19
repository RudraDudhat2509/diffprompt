"""
Derived display insights for a DiffReport.

Pure functions, no LLM, no network. Both the terminal and HTML renderers
build the same Insights object so the two stay in sync. Everything here is
computed from data already on the report (diffs carry outputs, divergence,
category, latency).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from statistics import mean

from diffprompt.models import DiffReport, Verdict, TestCategory

_SPARK = "▁▂▃▄▅▆▇█"
_CATEGORY_ORDER = [
    TestCategory.TYPICAL,
    TestCategory.ADVERSARIAL,
    TestCategory.BOUNDARY,
    TestCategory.FORMAT,
]


@dataclass
class Insights:
    v1_avg_latency_ms: float | None = None
    v2_avg_latency_ms: float | None = None
    latency_ratio: float | None = None       # > 1 means v2 is faster
    v1_avg_words: float = 0.0
    v2_avg_words: float = 0.0
    length_pct: float = 0.0                   # negative = v2 shorter
    sparkline: str = ""
    scorecard: list[tuple[str, float, int]] = field(default_factory=list)
    gained: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    low_confidence: int = 0
    headline: str = ""


def build_insights(report: DiffReport) -> Insights:
    ins = Insights()
    diffs = report.diffs
    if not diffs:
        return ins

    _latency(ins, diffs)
    _length(ins, diffs)
    ins.sparkline = _sparkline([d.divergence for d in diffs])
    _scorecard(ins, diffs)
    ins.low_confidence = sum(1 for d in diffs if d.judge_confidence < 0.65)
    _trade_off(ins, report)
    ins.headline = _headline(report, ins)
    return ins


def _latency(ins: Insights, diffs) -> None:
    v1 = [d.v1_latency_ms for d in diffs if d.v1_latency_ms is not None]
    v2 = [d.v2_latency_ms for d in diffs if d.v2_latency_ms is not None]
    if not (v1 and v2):
        return
    ins.v1_avg_latency_ms = mean(v1)
    ins.v2_avg_latency_ms = mean(v2)
    if ins.v2_avg_latency_ms > 0:
        ins.latency_ratio = ins.v1_avg_latency_ms / ins.v2_avg_latency_ms


def _length(ins: Insights, diffs) -> None:
    ins.v1_avg_words = mean(len(d.v1_output.split()) for d in diffs)
    ins.v2_avg_words = mean(len(d.v2_output.split()) for d in diffs)
    if ins.v1_avg_words > 0:
        ins.length_pct = (ins.v2_avg_words - ins.v1_avg_words) / ins.v1_avg_words * 100


def _scorecard(ins: Insights, diffs) -> None:
    for cat in _CATEGORY_ORDER:
        rows = [d for d in diffs if d.test_case.category == cat]
        if rows:
            ins.scorecard.append((cat.value, mean(d.similarity for d in rows), len(rows)))


def _trade_off(ins: Insights, report: DiffReport) -> None:
    # speed + brevity show up on whichever side they fall
    if ins.latency_ratio and ins.latency_ratio >= 1.1:
        ins.gained.append(
            f"{ins.latency_ratio:.1f}x faster ({ins.v1_avg_latency_ms:.0f}->{ins.v2_avg_latency_ms:.0f}ms)"
        )
    elif ins.latency_ratio and ins.latency_ratio <= 0.9:
        ins.lost.append(
            f"{1 / ins.latency_ratio:.1f}x slower ({ins.v1_avg_latency_ms:.0f}->{ins.v2_avg_latency_ms:.0f}ms)"
        )

    if ins.length_pct <= -10:
        ins.gained.append(
            f"{abs(ins.length_pct):.0f}% shorter ({ins.v1_avg_words:.0f}->{ins.v2_avg_words:.0f} words)"
        )
    elif ins.length_pct >= 10:
        ins.lost.append(
            f"{ins.length_pct:.0f}% longer ({ins.v1_avg_words:.0f}->{ins.v2_avg_words:.0f} words)"
        )

    depth1 = [s for s in report.slices if s.depth == 1]
    wins = sorted(
        (s for s in depth1 if s.verdict == Verdict.IMPROVEMENT),
        key=lambda s: s.mean_similarity, reverse=True,
    )
    fails = sorted(
        (s for s in depth1 if s.verdict == Verdict.REGRESSION),
        key=lambda s: s.mean_similarity,
    )
    ins.gained.extend(f"{s.label} ({s.mean_similarity:.2f})" for s in wins[:3])
    ins.lost.extend(f"{s.label} ({s.mean_similarity:.2f})" for s in fails[:3])


def _headline(report: DiffReport, ins: Insights) -> str:
    perks = []
    if ins.latency_ratio and ins.latency_ratio >= 1.1:
        perks.append("faster")
    if ins.length_pct <= -10:
        perks.append("shorter")
    perk_text = " and ".join(perks)

    if report.verdict == Verdict.REGRESSION:
        if perk_text:
            return f"v2 is {perk_text} — but that's exactly what's breaking it."
        return "v2 changes behavior in ways that mostly regress."
    if report.verdict == Verdict.IMPROVEMENT:
        return "v2 improves across most of what matters."
    return "v2's changes are mixed — better in some places, worse in others."


def _sparkline(values: list[float], bins: int = 10) -> str:
    """Bucket divergences (0-1) into `bins` and render relative heights."""
    if not values:
        return ""
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int(max(0.0, min(1.0, v)) * bins))
        counts[idx] += 1
    peak = max(counts) or 1
    return "".join(
        _SPARK[min(len(_SPARK) - 1, int(c / peak * (len(_SPARK) - 1)))] for c in counts
    )
