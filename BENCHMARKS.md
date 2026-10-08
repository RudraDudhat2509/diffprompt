# diffprompt benchmarks

Measured 2026-10-08. Raw outputs: `benchmarks/raw/`. Pairs and expected labels: `benchmarks/pairs.py` (labels fixed before any run).

## What was run, exactly

25 labeled prompt pairs for a customer-support agent, each diffed 3 times on the same 10 customer messages (75 diffs): 10 seeded regressions, 10 harmless rewordings of the same meaning (A/A), 5 clear improvements. The runs use diffprompt's own `run_both`, `batch_similarity`, `judge_all`, `regression_score` and `_compute_verdict`, unmodified.

**Deviations, read these:**
- The shipped defaults point at Ollama `qwen2.5:7b` and Groq Llama models, none of which this machine or key can reach. I redirected the model backends at runtime only (no source edited): runner and first-pass judge = OpenAI gpt-4o-mini, escalation = gpt-4o. The numbers describe diffprompt with these models, not with its defaults.
- The 75-diff run skips ontology anchors, clustering, slicing and key-example selection (they do not feed the overall verdict). The full CLI pipeline took 35 to 190 s per diff, too slow for 75 runs. 13 full-pipeline diffs were run as a cross-check (`benchmarks/raw/runs/`, not committed because of size).
- `--local-only` timing, and `--n 20` vs `--n 40` timing, were **not run**: Ollama has no `qwen2.5:7b` here.

| Metric | Value | Baseline | n | Command | Reproduced? |
|---|---|---|---|---|---|
| Tests | 95 pass, 70% line coverage | n/a | 95 | `pytest --cov=diffprompt` | yes |
| Seeded regressions flagged (recall) | 15 / 30 = 50% (per run: 60%, 50%, 40%) | n/a | 10 pairs x 3 | `python benchmarks/run_fast.py` | yes |
| Harmless rewordings wrongly flagged "regression" (false alarm rate) | 0 / 30 = 0% | n/a | 10 pairs x 3 | same | yes |
| Improvements wrongly flagged "regression" | 0 / 15 = 0% | n/a | 5 pairs x 3 | same | yes |
| Precision / recall / F1 (positive = seeded regression, flag = overall verdict "regression") | 1.00 / 0.50 / 0.67 | n/a | 75 diffs | same | yes |
| Confusion matrix (pooled) | TP 15, FN 15, FP 0, TN 45 | n/a | 75 diffs | same | yes |
| Verdict stability | 17 of 25 pairs changed verdict between the 3 runs | n/a | 25 pairs x 3 | same | yes |
| Default `--ci` gate (score < 75 fails) as a classifier | catches 23 / 30 regressions (77%); fails 3 / 45 non-regressions (7%): precision 0.88, F1 0.82 | the verdict rule above (F1 0.67) | 75 diffs | same | yes |
| Escalation to the larger judge | 24 of 593 judge calls = 4.0% | n/a | 593 calls | same | yes |
| Failure-mode clustering (full pipeline) | 10 diffs per run reduce to 2 to 4 named clusters (2 to 9 raw regression reasons per run) | n/a | 13 diffs | `python benchmarks/run_pairs.py` | yes |
| Published version on PyPI | 0.2.0 (2026-07-18); repo is 1.2.0 | n/a | n/a | PyPI JSON API | yes |
| Cost | 2,117 model calls (1,500 runner, 593 judge, 24 escalation) in 219 s; roughly $0.3 (estimated from list prices, not metered) | n/a | 75 diffs | same | n/a |

## Where it fails (the real findings)

1. **It misses half of the seeded regressions.** Never flagged in any run: R03 drop the safety rule (told the agent to reassure customers about reactions), R04 promise refunds, R05 JSON-only output, R09 invent tracking facts. R09 was judged "improvement" in all 3 runs, because made-up tracking details read as more helpful to a judge that sees only the answer.
2. **It calls rewordings "improvements."** Of the 30 harmless rewordings, 20 came back "improvement" and 10 "neutral". The verdict is overall "improvement" for text that means the same thing. A clean A/A test should be neutral.
3. **Verdicts are noisy.** 17 of 25 pairs gave different overall verdicts across identical reruns. One regression (R01 no empathy) came back regression, improvement, regression.
4. **The 75 default CI threshold does better than the verdict rule here**, because scores for real regressions are far lower (median 45.7) than for rewordings (median 87.6) or improvements (94.1). That threshold is the tool's default, not tuned on this data.
5. **Cluster names can be wrong or duplicated.** R10 (ask for a credit card number) was labeled BREVITY_LOSS and TONE_SHIFT, not a privacy problem. R02 produced two clusters both named BREVITY_LOSS.

## Caveats

- 25 pairs, 10 messages each, one domain, one model family. Small sample. Treat 50% recall as "about half", not a precise number. Many verdicts are close calls with noisy judging.
- The judge and the runner are the same model family (OpenAI). A different judge would give different numbers.
- Seeded regressions were my own; some (JSON-only, wrong language) may be judged as acceptable by a different reasonable reader. I fixed the labels beforehand and did not adjust them after seeing results.
- No source file was modified. New files are only under `benchmarks/` (the 14 MB `.agents/` folder was not touched).
- Not run: local-only runs, `--n 20/40` timing, PyPI download counts (pypistats returned HTTP 429).
