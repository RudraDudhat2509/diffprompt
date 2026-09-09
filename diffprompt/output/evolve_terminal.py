"""
Rich terminal renderer for `diffprompt evolve`.

Same inverted-pyramid instinct as output/terminal.py: score first, then the
generation-by-generation trend, then the winning prompt, then exactly what
changed. The diff at the bottom is a plain text/line diff (difflib) styled to
match — no LLM involved anywhere in this render path.
"""
from __future__ import annotations
import difflib

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from diffprompt.models import EvolveReport

console = Console(highlight=False)

_SPARK = "▁▂▃▄▅▆▇█"


def render(report: EvolveReport) -> None:
    console.print()
    _score_banner(report)
    _generation_chart(report)
    _final_prompt(report)
    _diff(report)
    _details(report)
    console.print()


def _score_banner(report: EvolveReport) -> None:
    score = report.final_fitness
    color = "green" if score >= 0.75 else "yellow" if score >= 0.5 else "red"
    bar = _bar(score * 100, 24)

    head = Text()
    head.append(f"{score:.2f}", style=f"bold {color}")
    head.append("  fitness        ", style="bold")
    head.append(bar, style=color)

    reason = "reached --generations" if report.stopped_reason == "max_generations" else "stopped early (--patience)"
    body = Text(f"\n{report.n_generations_run} generations · {reason}", style="dim")

    console.print(Panel(Text.assemble(head, body), border_style=color, padding=(0, 2)))


def _generation_chart(report: EvolveReport) -> None:
    if not report.generations:
        return
    console.print("[bold]SCORE BY GENERATION[/bold]")
    best_values = [g.best_fitness for g in report.generations]
    spark = _sparkline(best_values)
    first, last = best_values[0], best_values[-1]
    console.print(f"  {spark}  [dim]{first:.2f} → [/dim][bold]{last:.2f}[/bold]")
    console.print()


def _final_prompt(report: EvolveReport) -> None:
    console.print("[bold]EVOLVED PROMPT[/bold]")
    console.print(Panel(report.final_prompt, border_style="dim", padding=(0, 2)))
    console.print()


def _diff(report: EvolveReport) -> None:
    if report.original_prompt == report.final_prompt:
        console.print("[dim]No change from the starting prompt.[/dim]")
        console.print()
        return

    console.print("[bold]WHAT CHANGED[/bold]")
    lines = difflib.unified_diff(
        report.original_prompt.splitlines(),
        report.final_prompt.splitlines(),
        fromfile="original", tofile="evolved", lineterm="",
    )
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            console.print(f"  [dim]{line}[/dim]")
        elif line.startswith("+"):
            console.print(f"  [green]{line}[/green]")
        elif line.startswith("-"):
            console.print(f"  [red]{line}[/red]")
        elif line.startswith("@@"):
            console.print(f"  [cyan]{line}[/cyan]")
        else:
            console.print(f"  [dim]{line}[/dim]")
    console.print()


def _details(report: EvolveReport) -> None:
    parts = [
        f"population {report.population_size}",
        f"tasks {report.n_tasks}",
        f"model {report.model}",
    ]
    console.print("[bold]DETAILS[/bold]  [dim]" + "  ·  ".join(parts) + "[/dim]")


def _bar(value_0_100: float, width: int) -> str:
    filled = int(max(0.0, min(100.0, value_0_100)) / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    return "".join(
        _SPARK[min(len(_SPARK) - 1, int(max(0.0, min(1.0, v)) * (len(_SPARK) - 1)))]
        for v in values
    )
