"""
diffprompt CLI entry point.
"""
from __future__ import annotations
import asyncio
import json
from enum import Enum
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="diffprompt",
    help="git diff for your prompt's behavior",
    add_completion=False,
)
console = Console()


class OutputFormat(str, Enum):
    terminal = "terminal"
    json     = "json"
    html     = "html"
    txt      = "txt"


@app.command()
def diff(
    prompt_v1: str = typer.Argument(..., help="Baseline prompt (v1)"),
    prompt_v2: str = typer.Argument(..., help="Candidate prompt (v2)"),

    # Test generation
    auto_generate: bool    = typer.Option(False,  "--auto-generate", help="Auto-generate test cases"),
    n:             int     = typer.Option(40,     "--n",             help="Number of test cases"),
    test_file:     Optional[str] = typer.Option(None, "--test-file", help="Path to .jsonl test file"),
    logs:          Optional[str] = typer.Option(None, "--logs",      help="Production logs .jsonl"),
    augment:       int     = typer.Option(0,      "--augment",       help="Augment logs with N generated cases"),

    # Models
    model:         str     = typer.Option("groq/llama-3.3-70b-versatile", "--model"),
    judge:         str     = typer.Option("local/qwen2.5:7b",             "--judge"),
    local_only:    bool    = typer.Option(False,  "--local-only",    help="Never call any API"),
    no_judge:      bool    = typer.Option(False,  "--no-judge",      help="Skip judge, similarity only"),

    # Output
    format:        OutputFormat   = typer.Option(OutputFormat.terminal, "--format"),
    output:        Optional[str]  = typer.Option(None, "--output", help="Save report to file"),
    save:          bool           = typer.Option(False, "--save",   help="Save as diffprompt_report.txt"),
    top_n:         int            = typer.Option(3,    "--top-n",   help="Show top N regressions"),
    quiet:         bool           = typer.Option(False, "--quiet",  help="Score + verdict only"),
    verbose:       bool           = typer.Option(False, "--verbose"),

    # CI/CD
    ci:            bool    = typer.Option(False, "--ci",        help="Exit 1 if score below threshold"),
    threshold:     int     = typer.Option(75,    "--threshold", help="Regression score threshold for --ci"),
):
    """
    Diff the behavior of two prompts.

    Examples:\n
      diffprompt diff "helpful assistant" "concise assistant" --auto-generate\n
      diffprompt diff "v1" "v2" --logs prod.jsonl --augment 20\n
      diffprompt diff "v1" "v2" --auto-generate --ci --threshold 75
    """
    asyncio.run(_run_diff(
        prompt_v1=prompt_v1,
        prompt_v2=prompt_v2,
        auto_generate=auto_generate,
        n=n,
        test_file=test_file,
        logs=logs,
        augment=augment,
        model=model,
        judge=judge,
        local_only=local_only,
        no_judge=no_judge,
        format=format,
        output=output,
        save=save,
        top_n=top_n,
        quiet=quiet,
        verbose=verbose,
        ci=ci,
        threshold=threshold,
    ))


async def _run_diff(**kwargs):
    from diffprompt.core.ontology   import Ontology
    from diffprompt.core.generator  import generate_test_cases, diversity_score
    from diffprompt.core.runner     import run_both
    from diffprompt.core.embedder   import batch_similarity
    from diffprompt.core.judge      import judge_single
    from diffprompt.core.clusterer  import cluster_diffs
    from diffprompt.core.slicer     import compute_slices
    from diffprompt.core.scorer     import regression_score, select_key_examples
    from diffprompt.models          import DiffResult, DiffReport, Verdict
    from diffprompt.output.terminal import render

    prompt_v1   = kwargs["prompt_v1"]
    prompt_v2   = kwargs["prompt_v2"]
    local_only  = kwargs["local_only"]
    verbose     = kwargs["verbose"]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:

        # Step 1 — Ontology
        task = p.add_task("Inferring ontology from prompt...", total=None)
        ontology = Ontology()
        await ontology.infer(prompt_v1, local_only=local_only)
        await ontology.build_anchors(prompt_v1, local_only=local_only)
        p.update(task, description="[green]✓[/green] Ontology inferred")

        # Step 2 — Generate test cases
        p.update(task, description="Generating test cases...")
        test_cases = await generate_test_cases(
            prompt_v1,
            n=kwargs["n"],
            ontology=ontology,
            local_only=local_only,
        )
        div_score = diversity_score(test_cases, ontology.embedder)
        p.update(task, description=f"[green]✓[/green] Generated {len(test_cases)} test cases  diversity={div_score:.2f}")

        if div_score < 0.4:
            console.print(f"[yellow]⚠ Low diversity ({div_score:.2f}) — consider --n with a higher value[/yellow]")

        # Step 3 — Run both prompts
        p.update(task, description="Running both prompts...")
        v1_results, v2_results = await run_both(
            test_cases,
            prompt_v1,
            prompt_v2,
            local_only=local_only,
        )
        p.update(task, description=f"[green]✓[/green] Ran both prompts ({len(test_cases)*2} completions)")

        # Step 4 — Semantic diff
        p.update(task, description="Computing semantic diff...")
        pairs = [(v1_results[tc.id].output, v2_results[tc.id].output) for tc in test_cases]
        similarities = batch_similarity(pairs)

        diffs = []
        for i, tc in enumerate(test_cases):
            sim = similarities[i]
            v1_out = v1_results[tc.id].output
            v2_out = v2_results[tc.id].output

            if kwargs["no_judge"]:
                verdict, reason, confidence = Verdict.NEUTRAL, "judge skipped", 1.0
            else:
                verdict, reason, confidence = await judge_single(
                    tc, v1_out, v2_out, sim, local_only=local_only
                )

            diffs.append(DiffResult(
                test_case=tc,
                v1_output=v1_out,
                v2_output=v2_out,
                similarity=sim,
                divergence=1 - sim,
                verdict=verdict,
                reason=reason,
                judge_confidence=confidence,
            ))
        p.update(task, description="[green]✓[/green] Semantic diff computed")

        # Step 5 — Analysis
        p.update(task, description="Clustering failure modes...")
        clusters, unclustered = cluster_diffs(diffs)
        slices = compute_slices(diffs)
        score = regression_score(diffs)
        key_examples = await select_key_examples(diffs[:20], local_only=local_only)
        p.update(task, description="[green]✓[/green] Analysis complete")

    # Step 6 — Build report
    n_improved  = sum(1 for d in diffs if d.verdict == Verdict.IMPROVEMENT)
    n_regressed = sum(1 for d in diffs if d.verdict == Verdict.REGRESSION)
    n_neutral   = sum(1 for d in diffs if d.verdict == Verdict.NEUTRAL)

    overall_verdict = (
        Verdict.IMPROVEMENT if n_improved > n_regressed + n_neutral / 2
        else Verdict.REGRESSION if n_regressed > n_improved
        else Verdict.NEUTRAL
    )

    recommendation = _generate_recommendation(slices, clusters, overall_verdict)

    report = DiffReport(
        prompt_v1=prompt_v1,
        prompt_v2=prompt_v2,
        model=kwargs["model"],
        judge=kwargs["judge"],
        test_cases=test_cases,
        diversity_score=div_score,
        diffs=diffs,
        slices=slices,
        clusters=clusters,
        unclustered=unclustered,
        key_examples=key_examples,
        regression_score=score,
        n_improved=n_improved,
        n_regressed=n_regressed,
        n_neutral=n_neutral,
        verdict=overall_verdict,
        recommendation=recommendation,
    )

    # Step 7 — Output
    render(report)

    # Save if requested
    if kwargs["save"] or kwargs["output"]:
        path = kwargs["output"] or "diffprompt_report.txt"
        _save_report(report, path)
        console.print(f"[dim]Report saved → {path}[/dim]")

    # CI mode
    if kwargs["ci"] and score < kwargs["threshold"]:
        console.print(f"[red]✗ CI failed: score {score} < threshold {kwargs['threshold']}[/red]")
        raise typer.Exit(code=1)


def _generate_recommendation(slices, clusters, verdict) -> str:
    if not slices:
        return "Insufficient slice data to make a specific recommendation."

    worst_slices = [s for s in slices if s.verdict.value == "regression"][:2]
    best_slices  = [s for s in slices if s.verdict.value == "improvement"][:2]

    parts = []
    if best_slices:
        parts.append(f"Safe to deploy v2 for {', '.join(s.label for s in best_slices)}.")
    if worst_slices:
        parts.append(f"Keep v1 for {', '.join(s.label for s in worst_slices)}.")
    if clusters:
        top = clusters[0]
        parts.append(f"Primary failure mode: {top.name} ({top.n} cases).")

    return " ".join(parts) if parts else "Review key examples before shipping."


def _save_report(report: DiffReport, path: str) -> None:
    if path.endswith(".json"):
        with open(path, "w") as f:
            f.write(report.model_dump_json(indent=2))
    else:
        with open(path, "w") as f:
            f.write(f"diffprompt report\n")
            f.write(f"v1: {report.prompt_v1}\n")
            f.write(f"v2: {report.prompt_v2}\n")
            f.write(f"score: {report.regression_score}\n")
            f.write(f"verdict: {report.verdict.value}\n")
            f.write(f"recommendation: {report.recommendation}\n")


if __name__ == "__main__":
    app()
