"""
Rich terminal renderer — verdict-first, insight-led.

Layout (inverted pyramid): the answer, then why it's the answer, then the
evidence, then the raw numbers.

  1. VERDICT banner   — ship/don't-ship + score + one-line headline
  2. TRADE-OFF        — what v2 gained vs what it lost
  3. WHERE IT BREAKS  — behavioral slices, worst first, safe zone flagged
  4. WHY              — named failure modes + low-confidence caveat
  5. CLEAREST EXAMPLE — the single most important diff, full width
  6. DETAILS          — model/judge/tests/diversity footer
"""
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from diffprompt.models import DiffReport, Verdict, SliceResult
from diffprompt.output.insights import build_insights, Insights

console = Console(highlight=False)

VERDICT_COLOR = {
    Verdict.IMPROVEMENT: "green",
    Verdict.REGRESSION:  "red",
    Verdict.NEUTRAL:     "yellow",
}
VERDICT_ICON = {
    Verdict.IMPROVEMENT: "✓",
    Verdict.REGRESSION:  "✗",
    Verdict.NEUTRAL:     "→",
}
VERDICT_LABEL = {
    Verdict.IMPROVEMENT: "SHIP IT",
    Verdict.NEUTRAL:     "CONDITIONAL",
    Verdict.REGRESSION:  "DO NOT SHIP",
}


def render(report: DiffReport) -> None:
    ins = build_insights(report)
    console.print()
    _verdict_banner(report, ins)
    _trade_off(ins)
    _where_it_breaks(report)
    _why(report, ins)
    _clearest_example(report)
    _details(report, ins)
    console.print()


def _verdict_banner(report: DiffReport, ins: Insights) -> None:
    color = VERDICT_COLOR[report.verdict]
    icon  = VERDICT_ICON[report.verdict]
    label = VERDICT_LABEL[report.verdict]
    score = report.regression_score
    bar   = _bar(score, 24)

    head = Text()
    head.append(f"{icon}  {label}", style=f"bold {color}")
    head.append(f"        {score}/100  ", style="bold")
    head.append(bar, style=color)

    body = Text()
    body.append("\n")
    body.append(ins.headline or report.recommendation, style="dim")

    console.print(Panel(Text.assemble(head, body), border_style=color, padding=(0, 2)))


def _trade_off(ins: Insights) -> None:
    if not ins.gained and not ins.lost:
        return
    console.print("[bold]THE TRADE-OFF[/bold]")
    t = Table(show_header=True, header_style="dim", box=None, padding=(0, 4, 0, 2))
    t.add_column("GAINED", style="green")
    t.add_column("LOST", style="red")
    rows = max(len(ins.gained), len(ins.lost))
    for i in range(rows):
        g = f"+ {ins.gained[i]}" if i < len(ins.gained) else ""
        l = f"- {ins.lost[i]}"   if i < len(ins.lost)   else ""
        t.add_row(g, l)
    console.print(t)
    console.print()


def _where_it_breaks(report: DiffReport) -> None:
    depth1 = [s for s in report.slices if s.depth == 1]
    wins   = sorted((s for s in depth1 if s.verdict == Verdict.IMPROVEMENT),
                    key=lambda s: s.mean_similarity, reverse=True)
    fails  = sorted((s for s in depth1 if s.verdict == Verdict.REGRESSION),
                    key=lambda s: s.mean_similarity)
    if not wins and not fails:
        return

    total = len(report.diffs)
    console.print(f"[bold]WHERE IT BREAKS[/bold]  [dim]({report.n_regressed} of {total} regressed)[/dim]")
    for i, s in enumerate(fails[:4]):
        tag = "  [red]✗ worst hit[/red]" if i == 0 else ""
        _slice_row(s, "red", tag)
    if wins:
        best = wins[0]
        _slice_row(best, "green", "  [green]✓ safe zone[/green]")
    console.print()


def _slice_row(s: SliceResult, color: str, tag: str = "") -> None:
    bar  = _bar(s.mean_similarity * 100, 12)
    warn = _warn(s)
    console.print(
        f"  [{color}]{s.label:<34}[/{color}] [{color}]{bar}[/{color}] "
        f"[{color}]{s.mean_similarity:.2f}[/{color}] [dim]{s.n} tests[/dim]"
        + tag + (f"  {warn}" if warn else "")
    )


def _why(report: DiffReport, ins: Insights) -> None:
    if not report.clusters:
        return
    console.print(f"[bold]WHY[/bold]  [dim]({len(report.clusters)} failure modes)[/dim]")
    for c in report.clusters[:5]:
        console.print(
            f"  [bold]{c.name:<16}[/bold] [dim]×{c.n}[/dim]  [dim]{_clip(c.description, 70)}[/dim]"
        )
    if ins.low_confidence:
        console.print(
            f"  [yellow]⚠ {ins.low_confidence} verdict(s) below 0.65 confidence — eyeball these[/yellow]"
        )
    console.print()


def _clearest_example(report: DiffReport) -> None:
    if not report.key_examples:
        return
    ex = report.key_examples[0]
    d  = ex.diff
    color = VERDICT_COLOR[d.verdict]

    console.print(
        f"[bold]THE CLEAREST EXAMPLE[/bold]  "
        f"[dim]divergence[/dim] [{color}]{d.divergence:.2f}[/{color}]"
        f"  [dim]conf[/dim] {d.judge_confidence:.2f}"
    )
    console.print(f"  [dim]input[/dim]  {_clip(d.test_case.input, 92)}")
    console.print(f"  [dim]v1[/dim]  [green]✓[/green]  [dim]{_clip(d.v1_output, 96)}[/dim]")
    console.print(f"  [dim]v2[/dim]  [{color}]{VERDICT_ICON[d.verdict]}[/{color}]  [{color}]{_clip(d.v2_output, 96)}[/{color}]")
    console.print(f"  [dim]{'─' * 70}[/dim]")
    console.print(f"  [italic]{_clip(ex.why_it_matters, 200)}[/italic]")

    extra = len(report.key_examples) - 1
    if extra > 0:
        console.print(f"  [dim]▸ {extra} more example(s) in the full report (--output html/json)[/dim]")
    console.print()


def _details(report: DiffReport, ins: Insights) -> None:
    parts = [
        f"tests {len(report.test_cases)}",
        f"diversity {report.diversity_score:.2f}",
    ]
    if ins.sparkline:
        parts.append(f"spread {ins.sparkline}")
    console.print("[bold]DETAILS[/bold]  [dim]" + "  ·  ".join(parts) + "[/dim]")
    console.print(f"  [dim]model {report.model} · judge {report.judge}[/dim]")


def _bar(value_0_100: float, width: int) -> str:
    filled = int(max(0.0, min(100.0, value_0_100)) / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _warn(s: SliceResult) -> str:
    if s.confidence >= 0.7:
        return ""
    if s.n < 3:
        return "[yellow]⚠ low n[/yellow]"
    if s.typical_ratio == 0:
        return "[yellow]⚠ no typical tests[/yellow]"
    if s.variance > 0.15:
        return "[yellow]⚠ high variance[/yellow]"
    return ""


def _clip(text: str, n: int) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text
