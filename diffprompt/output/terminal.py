"""
Rich terminal output renderer.
Clean, minimal, fits in one screen.
"""
from __future__ import annotations
from rich.console import Console
from rich.text import Text
from rich.rule import Rule
from diffprompt.models import DiffReport, Verdict, SliceResult

console = Console(highlight=False)

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
VERDICT_LABEL = {
    Verdict.IMPROVEMENT: "SHIP IT",
    Verdict.NEUTRAL:     "CONDITIONAL",
    Verdict.REGRESSION:  "DO NOT SHIP",
}


def render(report: DiffReport) -> None:
    console.print()
    _header(report)
    _summary(report)
    _behavioral_profile(report)
    _key_examples(report)
    _verdict(report)
    console.print()


def _header(report: DiffReport) -> None:
    console.print(
        f"[bold white]diffprompt[/bold white]  [dim]v0.1.0[/dim]  "
        f"[dim]model:[/dim] [dim]{report.model}[/dim]  "
        f"[dim]judge:[/dim] [dim]{report.judge}[/dim]  "
        f"[dim]tests:[/dim] [dim]{len(report.test_cases)}[/dim]"
    )
    console.print()


def _summary(report: DiffReport) -> None:
    score  = report.regression_score
    filled = int(score / 5)
    bar    = "█" * filled + "░" * (20 - filled)
    color  = "green" if score >= 80 else "yellow" if score >= 60 else "red"

    console.print(f"[bold]━━ SUMMARY[/bold]")
    console.print(
        f"  [{color}]{score}/100[/{color}]  [{color}]{bar}[/{color}]  "
        f"[green]{report.n_improved} improved[/green]  "
        f"[red]{report.n_regressed} regressed[/red]  "
        f"[dim]{report.n_neutral} neutral[/dim]"
    )
    console.print()


def _behavioral_profile(report: DiffReport) -> None:
    if not report.slices:
        return

    depth1 = [s for s in report.slices if s.depth == 1]
    wins   = [s for s in depth1 if s.verdict == Verdict.IMPROVEMENT]
    fails  = [s for s in depth1 if s.verdict == Verdict.REGRESSION]

    if not wins and not fails:
        return

    console.print("[bold]━━ BEHAVIORAL PROFILE[/bold]")

    if wins:
        console.print("  [green]v2 performs well when...[/green]")
        for s in wins[:3]:
            _slice_row(s, "green")

    if fails:
        console.print("  [red]v2 struggles when...[/red]")
        for s in fails[:3]:
            _slice_row(s, "red")

    console.print()


def _slice_row(s: SliceResult, color: str) -> None:
    icon = "✓" if color == "green" else "✗"
    warn = _warn(s)
    console.print(
        f"  [{color}]{icon}[/{color}] [bold]{s.label:<38}[/bold] "
        f"[dim]score [/dim][{color}]{s.mean_similarity:.2f}[/{color}]"
        f"[dim]  {s.n} tests[/dim]"
        + (f"  {warn}" if warn else "")
    )


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


def _key_examples(report: DiffReport) -> None:
    if not report.key_examples:
        return

    console.print("[bold]━━ KEY EXAMPLES[/bold]")

    slot_labels = {
        "most_important":   "MOST IMPORTANT",
        "best_improvement": "BEST IMPROVEMENT",
        "most_surprising":  "MOST SURPRISING",
    }

    for ex in report.key_examples:
        d     = ex.diff
        label = slot_labels.get(ex.slot, ex.slot.upper())
        color = VERDICT_COLOR[d.verdict]
        tags  = "  ".join(f"[dim]{k}:[/dim][cyan]{v}[/cyan]" for k, v in d.test_case.tags.items())

        console.print(f"\n  [bold]{label}[/bold]  {tags}  [dim]divergence[/dim] [{color}]{d.divergence:.2f}[/{color}]")
        console.print(f"  [dim]input[/dim]  {_clip(d.test_case.input, 90)}")
        console.print(f"  [dim]v1   [/dim]  [dim]{_clip(d.v1_output, 100)}[/dim]")
        console.print(f"  [dim]v2   [/dim]  [{color}]{_clip(d.v2_output, 100)}[/{color}]")
        console.print(f"  [dim]why  [/dim]  [italic dim]{_clip(ex.why_it_matters, 160)}[/italic dim]")

    console.print()


def _verdict(report: DiffReport) -> None:
    color = VERDICT_COLOR[report.verdict]
    icon  = VERDICT_ICON[report.verdict]
    label = VERDICT_LABEL[report.verdict]

    console.print("[bold]━━ VERDICT[/bold]")
    console.print(f"  [{color}]{icon} {label}[/{color}]")
    console.print(f"  [dim]{report.recommendation}[/dim]")


def _clip(text: str, n: int) -> str:
    text = text.replace("\n", " ").strip()
    return text[:n] + "…" if len(text) > n else text