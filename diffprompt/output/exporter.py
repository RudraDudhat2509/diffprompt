"""
HTML and plain-text export for diffprompt reports.
Generates a self-contained single-file HTML report — no external dependencies.
"""
from __future__ import annotations
from diffprompt.models import DiffReport, Verdict

_VERDICT_COLOR = {
    Verdict.IMPROVEMENT: "#22c55e",
    Verdict.REGRESSION:  "#ef4444",
    Verdict.NEUTRAL:     "#94a3b8",
}

_VERDICT_LABEL = {
    Verdict.IMPROVEMENT: "SHIP IT",
    Verdict.NEUTRAL:     "CONDITIONAL",
    Verdict.REGRESSION:  "DO NOT SHIP",
}


def render_html(report: DiffReport) -> str:
    score        = report.regression_score
    score_color  = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    verdict_color = _VERDICT_COLOR[report.verdict]
    verdict_label = _VERDICT_LABEL[report.verdict]

    slices_html = _render_slices(report)
    examples_html = _render_examples(report)
    clusters_html = _render_clusters(report)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>diffprompt report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #0f172a; color: #e2e8f0; padding: 2rem; line-height: 1.6; }}
  h1 {{ font-size: 1.4rem; color: #f8fafc; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin: 2rem 0 0.75rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.4rem; }}
  .meta {{ color: #475569; font-size: 0.8rem; margin-bottom: 2rem; }}
  .score-block {{ display: flex; align-items: center; gap: 1rem; margin: 1rem 0; }}
  .score {{ font-size: 2.5rem; font-weight: bold; color: {score_color}; }}
  .bar-wrap {{ flex: 1; background: #1e293b; border-radius: 4px; height: 8px; }}
  .bar-fill {{ height: 8px; border-radius: 4px; background: {score_color}; width: {score}%; }}
  .counts {{ display: flex; gap: 1.5rem; font-size: 0.85rem; margin-top: 0.5rem; }}
  .improved {{ color: #22c55e; }} .regressed {{ color: #ef4444; }} .neutral {{ color: #94a3b8; }}
  .slice-row {{ display: flex; align-items: center; gap: 1rem; padding: 0.4rem 0; font-size: 0.85rem; border-bottom: 1px solid #1e293b; }}
  .slice-label {{ width: 220px; color: #cbd5e1; }}
  .slice-bar-wrap {{ flex: 1; background: #1e293b; border-radius: 2px; height: 6px; }}
  .example {{ background: #1e293b; border-radius: 8px; padding: 1rem; margin: 0.75rem 0; border-left: 3px solid #334155; }}
  .example-label {{ font-size: 0.75rem; text-transform: uppercase; color: #64748b; margin-bottom: 0.5rem; }}
  .example-row {{ font-size: 0.82rem; margin: 0.3rem 0; }}
  .example-row .key {{ color: #64748b; display: inline-block; width: 3rem; }}
  .verdict-block {{ background: #1e293b; border-radius: 8px; padding: 1.25rem; margin-top: 1rem; border-left: 4px solid {verdict_color}; }}
  .verdict-label {{ font-size: 1.2rem; font-weight: bold; color: {verdict_color}; }}
  .verdict-rec {{ color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem; }}
  .cluster {{ display: inline-block; background: #1e293b; border: 1px solid #334155; border-radius: 4px; padding: 0.25rem 0.6rem; margin: 0.25rem; font-size: 0.8rem; color: #94a3b8; }}
  .prompt-box {{ background: #1e293b; border-radius: 6px; padding: 0.75rem; font-size: 0.82rem; color: #cbd5e1; white-space: pre-wrap; word-break: break-word; max-height: 120px; overflow: auto; }}
</style>
</head>
<body>
<h1>diffprompt</h1>
<div class="meta">model: {report.model} &nbsp;|&nbsp; judge: {report.judge} &nbsp;|&nbsp; tests: {len(report.test_cases)} &nbsp;|&nbsp; diversity: {report.diversity_score:.2f}</div>

<h2>Prompts</h2>
<p style="font-size:0.8rem;color:#64748b;margin-bottom:0.3rem">v1 (baseline)</p>
<div class="prompt-box">{_esc(report.prompt_v1)}</div>
<p style="font-size:0.8rem;color:#64748b;margin:0.5rem 0 0.3rem">v2 (candidate)</p>
<div class="prompt-box">{_esc(report.prompt_v2)}</div>

<h2>Summary</h2>
<div class="score-block">
  <div class="score">{score}<span style="font-size:1rem;color:#64748b">/100</span></div>
  <div style="flex:1">
    <div class="bar-wrap"><div class="bar-fill"></div></div>
    <div class="counts">
      <span class="improved">▲ {report.n_improved} improved</span>
      <span class="regressed">▼ {report.n_regressed} regressed</span>
      <span class="neutral">→ {report.n_neutral} neutral</span>
    </div>
  </div>
</div>

{slices_html}
{examples_html}
{clusters_html}

<h2>Verdict</h2>
<div class="verdict-block">
  <div class="verdict-label">{verdict_label}</div>
  <div class="verdict-rec">{_esc(report.recommendation)}</div>
</div>
</body>
</html>"""


def _render_slices(report: DiffReport) -> str:
    if not report.slices:
        return ""

    rows = []
    for s in report.slices[:10]:
        color = _VERDICT_COLOR[s.verdict]
        pct = int(s.mean_similarity * 100)
        rows.append(f"""
    <div class="slice-row">
      <span class="slice-label" style="color:{color}">{_esc(s.label)}</span>
      <div class="slice-bar-wrap"><div style="height:6px;border-radius:2px;background:{color};width:{pct}%"></div></div>
      <span style="color:{color};font-size:0.8rem;width:3rem;text-align:right">{s.mean_similarity:.2f}</span>
      <span style="color:#475569;font-size:0.75rem;width:4rem;text-align:right">{s.n} tests</span>
    </div>""")

    return f"<h2>Behavioral Profile</h2>{''.join(rows)}"


def _render_examples(report: DiffReport) -> str:
    if not report.key_examples:
        return ""

    slot_label = {
        "most_important":   "Most Important",
        "best_improvement": "Best Improvement",
        "most_surprising":  "Most Surprising",
    }

    cards = []
    for ex in report.key_examples:
        d = ex.diff
        color = _VERDICT_COLOR[d.verdict]
        label = slot_label.get(ex.slot, ex.slot.replace("_", " ").title())
        tags  = "  ".join(f"{k}:{v}" for k, v in d.test_case.tags.items())
        cards.append(f"""
  <div class="example" style="border-left-color:{color}">
    <div class="example-label">{label} &nbsp;·&nbsp; {_esc(tags)} &nbsp;·&nbsp; divergence {d.divergence:.2f}</div>
    <div class="example-row"><span class="key">input</span>{_esc(d.test_case.input[:120])}</div>
    <div class="example-row"><span class="key">v1</span>{_esc(d.v1_output[:200])}</div>
    <div class="example-row" style="color:{color}"><span class="key">v2</span>{_esc(d.v2_output[:200])}</div>
    <div class="example-row"><span class="key">why</span><em style="color:#94a3b8">{_esc(ex.why_it_matters)}</em></div>
  </div>""")

    return f"<h2>Key Examples</h2>{''.join(cards)}"


def _render_clusters(report: DiffReport) -> str:
    if not report.clusters:
        return ""
    tags = "".join(
        f'<span class="cluster">{_esc(c.name)} ({c.n})</span>'
        for c in report.clusters
    )
    return f"<h2>Failure Mode Clusters</h2><div>{tags}</div>"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )