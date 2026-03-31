"""
HTML export for diffprompt reports.
Self-contained single-file output, no external dependencies.
"""
from __future__ import annotations
from collections import defaultdict
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
    score         = report.regression_score
    score_color   = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    verdict_color = _VERDICT_COLOR[report.verdict]
    verdict_label = _VERDICT_LABEL[report.verdict]

    slices_html   = _render_slices(report)
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
  body {{ font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace; background: #0f172a; color: #e2e8f0; padding: 2rem; line-height: 1.6; max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; color: #f8fafc; margin-bottom: 0.25rem; letter-spacing: -0.02em; }}
  h2 {{ font-size: 0.75rem; color: #475569; text-transform: uppercase; letter-spacing: 0.12em; margin: 2rem 0 0.75rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.4rem; }}
  .meta {{ color: #475569; font-size: 0.78rem; margin-bottom: 2rem; }}
  .meta span {{ margin-right: 1.5rem; }}
  .score-block {{ display: flex; align-items: center; gap: 1.5rem; margin: 1rem 0; }}
  .score {{ font-size: 2.8rem; font-weight: 700; color: {score_color}; letter-spacing: -0.03em; line-height: 1; }}
  .score sup {{ font-size: 1rem; color: #475569; font-weight: 400; }}
  .bar-wrap {{ flex: 1; background: #1e293b; border-radius: 4px; height: 6px; }}
  .bar-fill {{ height: 6px; border-radius: 4px; background: {score_color}; width: {score}%; transition: width 0.3s; }}
  .counts {{ display: flex; gap: 1.5rem; font-size: 0.82rem; margin-top: 0.6rem; }}
  .improved {{ color: #22c55e; }} .regressed {{ color: #ef4444; }} .neutral {{ color: #475569; }}
  .slice-table {{ width: 100%; border-collapse: collapse; }}
  .slice-table td {{ padding: 0.35rem 0; font-size: 0.82rem; vertical-align: middle; }}
  .slice-label {{ width: 200px; padding-right: 1rem; }}
  .slice-bar-cell {{ width: 200px; }}
  .slice-bar-wrap {{ background: #1e293b; border-radius: 2px; height: 5px; }}
  .slice-score {{ width: 3rem; text-align: right; padding: 0 0.75rem; font-variant-numeric: tabular-nums; }}
  .slice-n {{ color: #334155; font-size: 0.75rem; }}
  .example {{ background: #0d1829; border: 1px solid #1e293b; border-radius: 8px; padding: 1.1rem 1.25rem; margin: 0.75rem 0; }}
  .example + .example {{ margin-top: 0.5rem; }}
  .example-slot {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: #334155; margin-bottom: 0.6rem; }}
  .example-tags {{ font-size: 0.75rem; color: #475569; margin-bottom: 0.6rem; }}
  .example-tags .tag-key {{ color: #334155; }}
  .example-tags .tag-val {{ color: #64748b; }}
  .example-row {{ display: grid; grid-template-columns: 3rem 1fr; gap: 0.5rem; font-size: 0.82rem; margin: 0.25rem 0; align-items: baseline; }}
  .row-key {{ color: #334155; font-size: 0.75rem; }}
  .row-v1 {{ color: #64748b; }}
  .row-why {{ color: #475569; font-style: italic; font-size: 0.8rem; }}
  .divider {{ border: none; border-top: 1px solid #1e293b; margin: 2rem 0; }}
  .verdict-block {{ border-radius: 8px; padding: 1.25rem 1.5rem; margin-top: 1rem; border-left: 3px solid {verdict_color}; background: #0d1829; }}
  .verdict-label {{ font-size: 1.1rem; font-weight: 700; color: {verdict_color}; letter-spacing: 0.04em; }}
  .verdict-rec {{ color: #64748b; font-size: 0.82rem; margin-top: 0.5rem; line-height: 1.6; }}
  .cluster-wrap {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.25rem; }}
  .cluster {{ background: #1e293b; border: 1px solid #2d3f55; border-radius: 4px; padding: 0.2rem 0.65rem; font-size: 0.78rem; color: #64748b; }}
  .prompt-box {{ background: #0d1829; border: 1px solid #1e293b; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.8rem; color: #94a3b8; white-space: pre-wrap; word-break: break-word; max-height: 100px; overflow: auto; }}
  .prompt-label {{ font-size: 0.72rem; color: #334155; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.75rem 0 0.3rem; }}
</style>
</head>
<body>

<h1>diffprompt</h1>
<div class="meta">
  <span>model: {_esc(report.model)}</span>
  <span>judge: {_esc(report.judge)}</span>
  <span>tests: {len(report.test_cases)}</span>
  <span>diversity: {report.diversity_score:.2f}</span>
</div>

<h2>Prompts</h2>
<div class="prompt-label">v1 — baseline</div>
<div class="prompt-box">{_esc(report.prompt_v1)}</div>
<div class="prompt-label">v2 — candidate</div>
<div class="prompt-box">{_esc(report.prompt_v2)}</div>

<h2>Summary</h2>
<div class="score-block">
  <div class="score">{score}<sup>/100</sup></div>
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
    slices = [s for s in report.slices if s.depth == 1]
    if not slices:
        return ""

    # Normalize bar widths to the actual min/max range so differences are visible
    scores = [s.mean_similarity for s in slices]
    lo, hi = min(scores), max(scores)
    span   = (hi - lo) or 0.01

    def bar_pct(sim: float) -> int:
        # 15–95% range so bars are always visible and differences readable
        return int(15 + ((sim - lo) / span) * 80)

    wins  = [s for s in slices if s.verdict == Verdict.IMPROVEMENT]
    fails = [s for s in slices if s.verdict == Verdict.REGRESSION]

    rows = []
    if wins:
        rows.append('<tr><td colspan="4" style="padding-top:0.5rem;font-size:0.75rem;color:#22c55e;padding-bottom:0.2rem">v2 performs well when…</td></tr>')
        for s in wins[:4]:
            rows.append(_slice_row(s, "#22c55e", bar_pct(s.mean_similarity)))
    if fails:
        rows.append('<tr><td colspan="4" style="padding-top:0.75rem;font-size:0.75rem;color:#ef4444;padding-bottom:0.2rem">v2 struggles when…</td></tr>')
        for s in fails[:6]:
            rows.append(_slice_row(s, "#ef4444", bar_pct(s.mean_similarity)))

    return f"<h2>Behavioral Profile</h2><table class='slice-table'>{''.join(rows)}</table>"


def _slice_row(s, color: str, bar_pct: int) -> str:
    warn = ""
    if s.n < 3:
        warn = '<span style="color:#f59e0b;font-size:0.72rem"> ⚠ low n</span>'
    elif s.typical_ratio == 0:
        warn = '<span style="color:#f59e0b;font-size:0.72rem"> ⚠ no typical tests</span>'
    return f"""<tr>
      <td class="slice-label" style="color:{color}">{_esc(s.label)}</td>
      <td class="slice-bar-cell">
        <div class="slice-bar-wrap">
          <div style="height:5px;border-radius:2px;background:{color};width:{bar_pct}%"></div>
        </div>
      </td>
      <td class="slice-score" style="color:{color}">{s.mean_similarity:.2f}</td>
      <td class="slice-n">{s.n} tests{warn}</td>
    </tr>"""


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
        d     = ex.diff
        color = _VERDICT_COLOR[d.verdict]
        label = slot_label.get(ex.slot, ex.slot.replace("_", " ").title())
        tags  = "  ".join(
            f'<span class="tag-key">{_esc(k)}:</span><span class="tag-val">{_esc(v)}</span>'
            for k, v in d.test_case.tags.items()
        )
        div_color = _VERDICT_COLOR[d.verdict]
        cards.append(f"""
<div class="example" style="border-left: 3px solid {div_color}">
  <div class="example-slot">{label}</div>
  <div class="example-tags">{tags} &nbsp;·&nbsp; divergence <span style="color:{div_color}">{d.divergence:.2f}</span></div>
  <div class="example-row"><span class="row-key">input</span><span>{_esc(d.test_case.input[:140])}</span></div>
  <div class="example-row"><span class="row-key">v1</span><span class="row-v1">{_esc(d.v1_output[:220])}</span></div>
  <div class="example-row"><span class="row-key">v2</span><span style="color:{div_color}">{_esc(d.v2_output[:220])}</span></div>
  <div class="example-row"><span class="row-key">why</span><span class="row-why">{_esc(ex.why_it_matters)}</span></div>
</div>""")

    return f"<h2>Key Examples</h2>{''.join(cards)}"


def _render_clusters(report: DiffReport) -> str:
    if not report.clusters:
        return ""

    # Deduplicate: merge clusters with the same name, summing their counts
    merged: dict[str, int] = defaultdict(int)
    for c in report.clusters:
        merged[c.name] += c.n
    sorted_clusters = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    tags = "".join(
        f'<span class="cluster">{_esc(name)} <span style="color:#475569">({n})</span></span>'
        for name, n in sorted_clusters
    )
    return f"<h2>Failure Mode Clusters</h2><div class='cluster-wrap'>{tags}</div>"


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .lstrip("\ufeff")  # strip BOM if file was read with UTF-8-BOM
    )