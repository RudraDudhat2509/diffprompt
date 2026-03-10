"""
Rich terminal output renderer.
Designed to fit in one screen — the constraint that makes it shareable.
"""
from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich import box
from diffprompt.models import DiffReport, Verdict, SliceResult


console = Console()

VERDICT_COLOR = {
    Verdict.IMPROVEMENT: "green",
    Verdict.REGRESSION:  "red",
    Verdict.NEUTRAL:     "dim",
}

VERDICT_ICON = {
    Verdict.IMPROVEMENT: "✓",
    Verdict.REGRESSION:  "✗",
    Verdict.NEUTRAL:     "→",
}


def render(report: DiffReport) -> None:
    """Render the full diff report to terminal."""
    _render_header(report)
    _render_summary(report)
    _render_behavioral_profile(report)
    _render_key_examples(report)
    _render_verdict(report)


def _render_header(report: DiffReport) -> None:
    console.print()
    console.print(f"[bold]diffprompt[/bold]  [dim]v0.1.0[/dim]")
    console.print(f"[dim]model: {report.model}  judge: {report.judge}  tests: {len(report.test_cases)}[/dim]")
    console.print()


def _render_summary(report: DiffReport) -> None:
    # Score bar
    score = report.regression_score
    filled = int(score / 5)
    bar = "█" * filled + "░" * (20 - filled)
    color = "green" if score >= 80 else "yellow" if score >= 60 else "red"

    console.print(f"[bold]━━ SUMMARY[/bold]")
    console.print(
        f"  [{color}]{score}/100[/{color}]  [{color}]{bar}[/{color}]  "
        f"[green]{report.n_improved} improved[/green]  "
        f"[red]{report.n_regressed} regressed[/red]  "
        f"[dim]{report.n_neutral} neutral[/dim]"
    )
    console.print()


def _render_behavioral_profile(report: DiffReport) -> None:
    if not report.slices:
        return

    console.print("[bold]━━ BEHAVIORAL PROFILE[/bold]")

    wins  = [s for s in report.slices if s.verdict == Verdict.IMPROVEMENT and s.depth == 1]
    fails = [s for s in report.slices if s.verdict == Verdict.REGRESSION  and s.depth == 1]

    if wins:
        console.print("  [green]v2 performs well when...[/green]")
        for s in wins[:3]:
            conf = _confidence_label(s)
            console.print(
                f"  [green]✓[/green] {s.label:<35} "
                f"[dim]score {s.mean_similarity:.2f}  {s.n} tests  {conf}[/dim]"
            )

    if fails:
        console.print("  [red]v2 struggles when...[/red]")
        for s in fails[:3]:
            conf = _confidence_label(s)
            console.print(
                f"  [red]✗[/red] {s.label:<35} "
                f"[dim]score {s.mean_similarity:.2f}  {s.n} tests  {conf}[/dim]"
            )

    console.print()


def _render_key_examples(report: DiffReport) -> None:
    if not report.key_examples:
        return

    console.print("[bold]━━ KEY EXAMPLES[/bold]")

    slot_labels = {
        "most_important":  "MOST IMPORTANT",
        "best_improvement": "BEST IMPROVEMENT",
        "most_surprising":  "MOST SURPRISING",
    }

    for ex in report.key_examples:
        diff = ex.diff
        label = slot_labels.get(ex.slot, ex.slot.upper())
        tags = "  ".join(f"{k}:{v}" for k, v in diff.test_case.tags.items())
        color = VERDICT_COLOR[diff.verdict]

        console.print(
            f"  [bold]{label}[/bold]  [dim]{tags}  divergence {diff.divergence:.2f}[/dim]"
        )
        console.print(f"  [dim]input[/dim]   {diff.test_case.input[:80]}")
        console.print(f"  [dim]v1[/dim]      {diff.v1_output[:100]}")
        console.print(f"  [{color}]v2[/{color}]      {diff.v2_output[:100]}")
        console.print(f"  [dim]why[/dim]     {ex.why_it_matters}")
        console.print()


def _render_verdict(report: DiffReport) -> None:
    color = VERDICT_COLOR[report.verdict]
    icon = "✓" if report.verdict == Verdict.IMPROVEMENT else \
           "⚠" if report.verdict == Verdict.NEUTRAL else "✗"

    verdict_map = {
        Verdict.IMPROVEMENT: "SHIP IT",
        Verdict.NEUTRAL:     "CONDITIONAL",
        Verdict.REGRESSION:  "DO NOT SHIP",
    }

    console.print(f"[bold]━━ VERDICT[/bold]")
    console.print(f"  [{color}]{icon} {verdict_map[report.verdict]}[/{color}]")
    console.print(f"  [dim]{report.recommendation}[/dim]")
    console.print()


def _confidence_label(s: SliceResult) -> str:
    if s.confidence >= 0.7:
        return ""
    if s.typical_ratio == 0:
        return "[yellow]⚠ no typical tests[/yellow]"
    if s.n < 5:
        return "[yellow]⚠ low n[/yellow]"
    if s.variance > 0.15:
        return "[yellow]⚠ high variance[/yellow]"
    return ""
