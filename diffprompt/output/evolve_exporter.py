"""
HTML export for `diffprompt evolve` reports.
Self-contained single-file output, same dark visual language as exporter.py.
"""
from __future__ import annotations
import difflib

from diffprompt.models import EvolveReport

_SPARK = "▁▂▃▄▅▆▇█"


def render_html(report: EvolveReport) -> str:
    score = report.final_fitness
    score_color = "#22c55e" if score >= 0.75 else "#f59e0b" if score >= 0.5 else "#ef4444"
    spark = _sparkline([g.best_fitness for g in report.generations])
    diff_html = _render_diff(report)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>diffprompt evolve report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace; background: #0f172a; color: #e2e8f0; padding: 2rem; line-height: 1.6; max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; color: #f8fafc; margin-bottom: 0.25rem; letter-spacing: -0.02em; }}
  h2 {{ font-size: 0.75rem; color: #475569; text-transform: uppercase; letter-spacing: 0.12em; margin: 2rem 0 0.75rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.4rem; }}
  .meta {{ color: #475569; font-size: 0.78rem; margin-bottom: 2rem; }}
  .meta span {{ margin-right: 1.5rem; }}
  .score-block {{ display: flex; align-items: center; gap: 1.5rem; margin: 1rem 0; }}
  .score {{ font-size: 2.8rem; font-weight: 700; color: {score_color}; letter-spacing: -0.03em; line-height: 1; }}
  .bar-wrap {{ flex: 1; background: #1e293b; border-radius: 4px; height: 6px; }}
  .bar-fill {{ height: 6px; border-radius: 4px; background: {score_color}; width: {score * 100:.0f}%; }}
  .spark {{ font-size: 1.4rem; letter-spacing: 0.15em; color: {score_color}; }}
  .prompt-box {{ background: #0d1829; border: 1px solid #1e293b; border-radius: 6px; padding: 0.9rem 1.1rem; font-size: 0.82rem; color: #94a3b8; white-space: pre-wrap; word-break: break-word; }}
  .diff-box {{ background: #0d1829; border: 1px solid #1e293b; border-radius: 6px; padding: 0.9rem 1.1rem; font-size: 0.8rem; white-space: pre-wrap; word-break: break-word; }}
  .diff-add {{ color: #22c55e; }}
  .diff-del {{ color: #ef4444; }}
  .diff-hunk {{ color: #38bdf8; }}
  .diff-ctx {{ color: #475569; }}
  .footer {{ color: #334155; font-size: 0.78rem; margin-top: 2rem; }}
</style>
</head>
<body>

<h1>diffprompt evolve</h1>
<div class="meta">
  <span>model: {_esc(report.model)}</span>
  <span>population: {report.population_size}</span>
  <span>generations: {report.n_generations_run}</span>
  <span>tasks: {report.n_tasks}</span>
  <span>stopped: {_esc(report.stopped_reason)}</span>
</div>

<h2>Fitness</h2>
<div class="score-block">
  <div class="score">{score:.2f}</div>
  <div style="flex:1">
    <div class="bar-wrap"><div class="bar-fill"></div></div>
  </div>
</div>
<div class="spark">{_esc(spark)}</div>

<h2>Evolved prompt</h2>
<div class="prompt-box">{_esc(report.final_prompt)}</div>

<h2>What changed</h2>
{diff_html}

<div class="footer">golden tasks: {_esc(report.golden_tasks_path)}</div>

</body>
</html>"""


def _render_diff(report: EvolveReport) -> str:
    if report.original_prompt == report.final_prompt:
        return '<div class="diff-box diff-ctx">No change from the starting prompt.</div>'

    lines = difflib.unified_diff(
        report.original_prompt.splitlines(),
        report.final_prompt.splitlines(),
        fromfile="original", tofile="evolved", lineterm="",
    )
    rows = []
    for line in lines:
        css = "diff-ctx"
        if line.startswith("+++") or line.startswith("---"):
            css = "diff-ctx"
        elif line.startswith("+"):
            css = "diff-add"
        elif line.startswith("-"):
            css = "diff-del"
        elif line.startswith("@@"):
            css = "diff-hunk"
        rows.append(f'<div class="{css}">{_esc(line)}</div>')
    return f'<div class="diff-box">{"".join(rows)}</div>'


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    return "".join(
        _SPARK[min(len(_SPARK) - 1, int(max(0.0, min(1.0, v)) * (len(_SPARK) - 1)))]
        for v in values
    )


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .lstrip("﻿")  # strip BOM if file was read with UTF-8-BOM
    )
