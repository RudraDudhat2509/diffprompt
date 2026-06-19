"""
Tests for derived insights + the terminal renderer.
Run with: pytest tests/test_insights.py -v
"""
from diffprompt.models import (
    DiffReport, DiffResult, TestCase, TestCategory, Verdict,
    SliceResult, Cluster, KeyExample,
)
from diffprompt.output.insights import build_insights, _sparkline
from diffprompt.output import terminal
from diffprompt.output.exporter import render_html


def _diff(text, cat, sim, verdict, v1_words, v2_words, v1_ms, v2_ms):
    tc = TestCase(id=text.replace(" ", "_"), input=text, category=cat, tags={})
    return DiffResult(
        test_case=tc,
        v1_output=" ".join(["w"] * v1_words),
        v2_output=" ".join(["w"] * v2_words),
        similarity=sim,
        divergence=1 - sim,
        verdict=verdict,
        reason="dropped empathy and context",
        judge_confidence=0.9,
        v1_latency_ms=v1_ms,
        v2_latency_ms=v2_ms,
    )


def _report():
    diffs = [
        _diff("frustrated user a", TestCategory.TYPICAL, 0.20, Verdict.REGRESSION, 80, 20, 1200, 600),
        _diff("frustrated user b", TestCategory.TYPICAL, 0.25, Verdict.REGRESSION, 70, 18, 1100, 550),
        _diff("factual lookup a", TestCategory.ADVERSARIAL, 0.80, Verdict.IMPROVEMENT, 60, 22, 1000, 500),
        _diff("factual lookup b", TestCategory.ADVERSARIAL, 0.78, Verdict.IMPROVEMENT, 64, 24, 1050, 520),
    ]
    slices = [
        SliceResult(dimension="state", value="frustrated", label="state:frustrated",
                    n=2, mean_similarity=0.22, variance=0.01, typical_ratio=1.0,
                    confidence=0.8, verdict=Verdict.REGRESSION, depth=1),
        SliceResult(dimension="intent", value="lookup", label="intent:lookup",
                    n=2, mean_similarity=0.79, variance=0.01, typical_ratio=0.0,
                    confidence=0.8, verdict=Verdict.IMPROVEMENT, depth=1),
    ]
    clusters = [Cluster(label=0, name="CONTEXT_LOSS", description="drops context",
                        n=2, mean_similarity=0.22, test_ids=["frustrated_user_a"])]
    key_examples = [KeyExample(slot="most_important", diff=diffs[0],
                               why_it_matters="v2 strips the empathy that frustrated users need.")]
    return DiffReport(
        prompt_v1="socratic", prompt_v2="direct", model="groq/x", judge="local/y",
        test_cases=[d.test_case for d in diffs], diversity_score=0.74, diffs=diffs,
        slices=slices, clusters=clusters, unclustered=[], key_examples=key_examples,
        regression_score=22.0, n_improved=2, n_regressed=2, n_neutral=0,
        verdict=Verdict.REGRESSION, recommendation="Keep v1 for frustrated users.",
    )


def test_latency_and_length_deltas():
    ins = build_insights(_report())
    # v1 avg ~1087ms, v2 avg ~542ms -> v2 ~2x faster
    assert ins.latency_ratio > 1.8
    # v2 outputs are much shorter
    assert ins.length_pct < -50


def test_scorecard_covers_present_categories():
    ins = build_insights(_report())
    cats = {row[0] for row in ins.scorecard}
    assert cats == {"typical", "adversarial"}


def test_trade_off_has_gains_and_losses():
    ins = build_insights(_report())
    assert any("faster" in g for g in ins.gained)
    assert any("shorter" in g for g in ins.gained)
    assert any("frustrated" in l for l in ins.lost)


def test_headline_reflects_regression():
    ins = build_insights(_report())
    assert "faster" in ins.headline and "shorter" in ins.headline


def test_sparkline_length_matches_bins():
    assert len(_sparkline([0.1, 0.5, 0.9], bins=10)) == 10
    assert _sparkline([]) == ""


def test_terminal_render_smoke():
    """The renderer produces the verdict-first layout without crashing."""
    with terminal.console.capture() as cap:
        terminal.render(_report())
    out = cap.get()
    assert "DO NOT SHIP" in out
    assert "THE TRADE-OFF" in out
    assert "WHERE IT BREAKS" in out
    assert "THE CLEAREST EXAMPLE" in out


def test_html_puts_verdict_before_summary():
    """HTML export is verdict-first and carries the new trade-off section."""
    html = render_html(_report())
    assert html.index("verdict-block") < html.index(">Summary<")
    assert "The Trade-off" in html
