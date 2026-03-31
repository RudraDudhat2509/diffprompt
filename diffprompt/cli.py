"""
diffprompt CLI — uses Click directly (no Typer) for cross-version compatibility.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def _load_prompt(value: str) -> str:
    p = Path(value)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return value.strip()


@click.group()
def app():
    """git diff for your prompt's behavior."""
    pass


@app.command()
@click.argument("prompt_v1")
@click.argument("prompt_v2")
@click.option("--auto-generate", is_flag=True, default=False, help="Auto-generate test cases")
@click.option("--n", default=40, show_default=True, help="Number of test cases")
@click.option("--test-file", default=None, help="Path to .jsonl test file")
@click.option("--model", default="groq/llama-3.3-70b-versatile", show_default=True)
@click.option("--judge", default="local/qwen2.5:7b", show_default=True)
@click.option("--local-only", is_flag=True, default=False, help="Never call any external API")
@click.option("--no-judge", is_flag=True, default=False, help="Skip judge, similarity only")
@click.option("--output", "output_format", default="terminal",
              type=click.Choice(["terminal", "json", "html"]), show_default=True)
@click.option("--save", default=None, help="Save report to this file path")
@click.option("--top-n", default=3, show_default=True)
@click.option("--quiet", is_flag=True, default=False, help="Score + verdict only")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--ci", is_flag=True, default=False, help="Exit 1 if score below threshold")
@click.option("--threshold", default=75, show_default=True)
def diff(prompt_v1, prompt_v2, auto_generate, n, test_file, model, judge,
         local_only, no_judge, output_format, save, top_n, quiet, verbose, ci, threshold):
    """Diff the behavioral impact of a prompt change.

    \b
    PROMPT_V1 and PROMPT_V2 can be file paths or inline strings.

    \b
    Examples:
      diffprompt diff v1.txt v2.txt --auto-generate
      diffprompt diff v1.txt v2.txt --auto-generate --n 20
      diffprompt diff v1.txt v2.txt --test-file inputs.jsonl
      diffprompt diff v1.txt v2.txt --auto-generate --ci --threshold 75
    """
    asyncio.run(_run_diff(
        prompt_v1=_load_prompt(prompt_v1),
        prompt_v2=_load_prompt(prompt_v2),
        auto_generate=auto_generate, n=n, test_file=test_file,
        model=model, judge=judge, local_only=local_only, no_judge=no_judge,
        output_format=output_format, save=save, top_n=top_n,
        quiet=quiet, verbose=verbose, ci=ci, threshold=threshold,
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

    prompt_v1  = kwargs["prompt_v1"]
    prompt_v2  = kwargs["prompt_v2"]
    local_only = kwargs["local_only"]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        task = p.add_task("Starting...", total=None)

        p.update(task, description="Inferring ontology...")
        ontology = Ontology()
        await ontology.infer(prompt_v1, local_only=local_only)
        await ontology.build_anchors(prompt_v1, local_only=local_only)
        p.update(task, description=f"[green]✓[/green] Ontology: {list(ontology.dimensions.keys())}")

        p.update(task, description="Generating test cases...")
        test_cases = []
        if kwargs["test_file"]:
            test_cases = _load_test_file(kwargs["test_file"])
        elif kwargs["auto_generate"]:
            test_cases = await generate_test_cases(
                prompt_v1, n=kwargs["n"], ontology=ontology, local_only=local_only,
            )
        else:
            console.print("[red]✗ Use --auto-generate or --test-file[/red]")
            raise SystemExit(1)

        if not test_cases:
            console.print("[red]✗ No test cases generated[/red]")
            raise SystemExit(1)

        for tc in test_cases:
            tc.tags = ontology.tag(tc.input)

        div_score = diversity_score(test_cases)
        p.update(task, description=f"[green]✓[/green] {len(test_cases)} test cases  diversity={div_score:.2f}")

        p.update(task, description="Running both prompts...")
        v1_results, v2_results = await run_both(test_cases, prompt_v1, prompt_v2, local_only=local_only)
        p.update(task, description=f"[green]✓[/green] {len(test_cases) * 2} completions done")

        p.update(task, description="Computing semantic diff...")
        pairs = [(v1_results[tc.id].output, v2_results[tc.id].output) for tc in test_cases]
        similarities = batch_similarity(pairs)

        diffs = []
        for i, tc in enumerate(test_cases):
            sim    = similarities[i]
            v1_out = v1_results[tc.id].output
            v2_out = v2_results[tc.id].output
            if kwargs["no_judge"]:
                verdict, reason, confidence = Verdict.NEUTRAL, "judge skipped", 1.0
            else:
                verdict, reason, confidence = await judge_single(tc, v1_out, v2_out, sim, local_only=local_only)
            diffs.append(DiffResult(
                test_case=tc, v1_output=v1_out, v2_output=v2_out,
                similarity=sim, divergence=1 - sim,
                verdict=verdict, reason=reason, judge_confidence=confidence,
            ))
        p.update(task, description="[green]✓[/green] Diff complete")

        p.update(task, description="Clustering and slicing...")
        clusters, unclustered = cluster_diffs(diffs)
        slices       = compute_slices(diffs)
        score        = regression_score(diffs)
        key_examples = await select_key_examples(
            sorted(diffs, key=lambda d: d.divergence, reverse=True)[:20],
            local_only=local_only,
        )
        p.update(task, description="[green]✓[/green] Analysis complete")

    n_improved  = sum(1 for d in diffs if d.verdict == Verdict.IMPROVEMENT)
    n_regressed = sum(1 for d in diffs if d.verdict == Verdict.REGRESSION)
    n_neutral   = sum(1 for d in diffs if d.verdict == Verdict.NEUTRAL)

    overall_verdict = _compute_verdict(n_improved, n_regressed, n_neutral)
    recommendation  = _generate_recommendation(slices, clusters, overall_verdict)

    report = DiffReport(
        prompt_v1=prompt_v1, prompt_v2=prompt_v2, model=kwargs["model"], judge=kwargs["judge"],
        test_cases=test_cases, diversity_score=div_score, diffs=diffs,
        slices=slices, clusters=clusters, unclustered=unclustered,
        key_examples=key_examples, regression_score=score,
        n_improved=n_improved, n_regressed=n_regressed, n_neutral=n_neutral,
        verdict=overall_verdict, recommendation=recommendation,
    )

    if not kwargs["quiet"]:
        render(report)
    else:
        console.print(f"score: {score}  verdict: {overall_verdict.value}")

    if kwargs["save"]:
        _save_report(report, kwargs["save"], kwargs["output_format"])
        console.print(f"[dim]Saved → {kwargs['save']}[/dim]")

    if kwargs["ci"] and score < kwargs["threshold"]:
        console.print(f"[red]✗ CI: score {score} < threshold {kwargs['threshold']}[/red]")
        raise SystemExit(1)


def _compute_verdict(n_improved, n_regressed, n_neutral):
    from diffprompt.models import Verdict
    if n_improved > n_regressed and n_improved > n_neutral * 0.5:
        return Verdict.IMPROVEMENT
    if n_regressed > n_improved:
        return Verdict.REGRESSION
    return Verdict.NEUTRAL


def _generate_recommendation(slices, clusters, verdict) -> str:
    from diffprompt.models import Verdict
    if not slices:
        return "Not enough data. Try --n 40 or higher."
    worst = [s for s in slices if s.verdict == Verdict.REGRESSION and s.depth == 1][:2]
    best  = [s for s in slices if s.verdict == Verdict.IMPROVEMENT and s.depth == 1][:2]
    parts = []
    if best:
        parts.append(f"Safe to ship v2 for {', '.join(s.label for s in best)}.")
    if worst:
        parts.append(f"Keep v1 for {', '.join(s.label for s in worst)}.")
    if clusters:
        parts.append(f"Primary failure mode: {clusters[0].name} ({clusters[0].n} cases).")
    return " ".join(parts) if parts else "Review key examples before shipping."


def _load_test_file(path: str) -> list:
    import uuid
    from diffprompt.models import TestCase, TestCategory
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            inp = data.get("input") or data.get("text") or data.get("prompt", "")
            if inp:
                cases.append(TestCase(id=str(uuid.uuid4()), input=inp, category=TestCategory.TYPICAL))
    return cases


def _save_report(report, path: str, fmt: str) -> None:
    if fmt == "json" or path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
    elif fmt == "html" or path.endswith(".html"):
        from diffprompt.output.exporter import render_html
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_html(report))
    else:
        lines = [
            "diffprompt report", "=" * 40,
            f"v1: {report.prompt_v1[:80]}", f"v2: {report.prompt_v2[:80]}",
            f"score: {report.regression_score}/100",
            f"verdict: {report.verdict.value.upper()}",
            f"improved: {report.n_improved}  regressed: {report.n_regressed}  neutral: {report.n_neutral}",
            "", f"recommendation: {report.recommendation}",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    app()