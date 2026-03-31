# Cookbook

Worked examples for common diffprompt use cases.

---

## 1. Testing a tone change

You have a customer support prompt and you want to make it more empathetic. You're not sure if adding warmth will hurt accuracy on factual questions.

**v1.txt**
```
You are a customer support assistant. Answer questions about our product clearly and accurately.
```

**v2.txt**
```
You are a warm and empathetic customer support assistant. Always acknowledge the customer's frustration before answering. Be clear and accurate.
```

```bash
diffprompt diff v1.txt v2.txt --auto-generate --n 40
```

What to look for in the output:
- `emotional_state:frustrated` slice should show improvement
- `topic_complexity:technical` slice should stay neutral (accuracy preserved)
- If `topic_complexity:technical` regresses, the empathy framing is interfering with factual answers

---

## 2. Testing a brevity change

You're trying to reduce output length. You added "Be concise." to your prompt. But does it hurt quality on complex questions?

```bash
diffprompt diff \
  "You are a helpful assistant. Answer clearly and completely." \
  "You are a helpful assistant. Answer clearly and completely. Be concise." \
  --auto-generate --n 40
```

Expected pattern: regressions on `topic_complexity:technical` and `request_type:open-ended`, improvements on `request_type:specific`. If you see regressions on emotional inputs, the brevity instruction is cutting empathetic framing too.

---

## 3. Bringing your own test cases

You already have a set of real user inputs from production logs. Use them instead of auto-generating.

Create a `.jsonl` file where each line is a JSON object with an `input` field:

```jsonl
{"input": "How do I reset my password?"}
{"input": "I've been charged twice and I'm really frustrated"}
{"input": "What's the difference between the Pro and Enterprise plans?"}
{"input": "Your product is broken and I want a refund"}
{"input": "Can you walk me through the API authentication flow?"}
```

```bash
diffprompt diff v1.txt v2.txt --test-file my_inputs.jsonl
```

This is the most valuable way to use diffprompt. Real user inputs expose failure modes that auto-generated inputs miss.

---

## 4. Running fully offline

You're working with a sensitive prompt and don't want inputs or outputs leaving your machine.

```bash
# Make sure Ollama is running
ollama serve

# Pull the required model
ollama pull qwen2.5:7b

# Run with local-only flag
diffprompt diff v1.txt v2.txt --auto-generate --local-only
```

All LLM calls go to Ollama. The embedding model (`all-MiniLM-L6-v2`) always runs locally regardless of this flag.

---

## 5. CI/CD integration

You want to block merges if a prompt change causes a regression score below 75.

**.github/workflows/prompt-check.yml**
```yaml
name: Prompt regression check

on:
  pull_request:
    paths:
      - 'prompts/**'

jobs:
  diffprompt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install diffprompt
        run: pip install diffprompt

      - name: Run prompt diff
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          diffprompt diff prompts/system_v1.txt prompts/system_v2.txt \
            --auto-generate \
            --n 20 \
            --ci \
            --threshold 75 \
            --output json \
            --save diff_report.json

      - name: Upload diff report
        uses: actions/upload-artifact@v4
        with:
          name: diff-report
          path: diff_report.json
```

The `--ci` flag makes diffprompt exit with code 1 if the regression score is below `--threshold`. The workflow fails and the PR is blocked.

Add `GROQ_API_KEY` as a repository secret in your GitHub settings.

---

## 6. Saving and sharing reports

```bash
# Save as JSON for downstream processing
diffprompt diff v1.txt v2.txt --auto-generate --output json --save report.json

# Save as HTML to share with non-technical stakeholders
diffprompt diff v1.txt v2.txt --auto-generate --output html --save report.html
```

The HTML report is a self-contained file. Open it in any browser, no server needed.

---

## 7. Using diffprompt as a Python library

You can use diffprompt programmatically without the CLI.

```python
import asyncio
from diffprompt.core.ontology import Ontology
from diffprompt.core.generator import generate_test_cases
from diffprompt.core.runner import run_both
from diffprompt.core.embedder import batch_similarity
from diffprompt.core.judge import judge_single
from diffprompt.core.scorer import regression_score
from diffprompt.models import DiffResult, Verdict

PROMPT_V1 = "You are a helpful assistant."
PROMPT_V2 = "You are a concise assistant. Be brief."

async def run_diff():
    # Build ontology
    ontology = Ontology()
    await ontology.infer(PROMPT_V1)
    await ontology.build_anchors(PROMPT_V1)

    # Generate test cases
    test_cases = await generate_test_cases(PROMPT_V1, n=20, ontology=ontology)

    # Run both prompts
    v1_results, v2_results = await run_both(test_cases, PROMPT_V1, PROMPT_V2)

    # Compute similarities
    pairs = [(v1_results[tc.id].output, v2_results[tc.id].output) for tc in test_cases]
    similarities = batch_similarity(pairs)

    # Judge each pair
    diffs = []
    for i, tc in enumerate(test_cases):
        sim = similarities[i]
        verdict, reason, confidence = await judge_single(
            tc, v1_results[tc.id].output, v2_results[tc.id].output, sim
        )
        diffs.append(DiffResult(
            test_case=tc,
            v1_output=v1_results[tc.id].output,
            v2_output=v2_results[tc.id].output,
            similarity=sim,
            divergence=1 - sim,
            verdict=verdict,
            reason=reason,
            judge_confidence=confidence,
        ))

    score = regression_score(diffs)
    regressions = [d for d in diffs if d.verdict == Verdict.REGRESSION]

    print(f"Score: {score}/100")
    print(f"Regressions: {len(regressions)}/{len(diffs)}")
    for d in sorted(regressions, key=lambda x: x.divergence, reverse=True)[:3]:
        print(f"  [{d.divergence:.2f}] {d.test_case.input[:60]}")
        print(f"         {d.reason}")

asyncio.run(run_diff())
```

---

## 8. Comparing the same prompt across two models

diffprompt can also tell you whether switching models changes behavior, not just switching prompts.

```python
from diffprompt.models.cascade import call_groq
import asyncio

PROMPT = "You are a helpful assistant."
INPUT = "Explain recursion in one sentence."

async def compare_models():
    out_70b, _ = await call_groq("llama-3.3-70b-versatile", INPUT, system=PROMPT)
    out_8b, _  = await call_groq("llama-3.1-8b-instant", INPUT, system=PROMPT)
    print("70B:", out_70b)
    print("8B: ", out_8b)

asyncio.run(compare_models())
```

For a full behavioral comparison across models, pass the same prompt as both v1 and v2 but override `--model` in two separate runs, then compare the saved JSON reports.

---

## 9. Debugging a specific regression

You got a low regression score and want to understand one specific failure. Pull it out of the report JSON and inspect it.

```python
import json

with open("report.json") as f:
    report = json.load(f)

# Find all regressions
regressions = [d for d in report["diffs"] if d["verdict"] == "regression"]

# Sort by divergence
regressions.sort(key=lambda d: d["divergence"], reverse=True)

# Print the worst one
worst = regressions[0]
print("Input:  ", worst["test_case"]["input"])
print("Tags:   ", worst["test_case"]["tags"])
print("V1:     ", worst["v1_output"][:200])
print("V2:     ", worst["v2_output"][:200])
print("Reason: ", worst["reason"])
print("Confidence:", worst["judge_confidence"])
```

If `judge_confidence` is below 0.65, the judge wasn't sure about this verdict. Treat it with skepticism and check the outputs manually.

---

## 10. Finding the right n

How many test cases do you actually need? It depends on how many behavioral dimensions your prompt has.

Start with `--n 20` for a quick sanity check. If the results look unstable across runs (different slices failing each time), increase to `--n 40`. For production prompts with many dimensions, use `--n 80`.

A practical rule: you want at least 5 test cases per tag value in your most important dimension. If `emotional_state` has 4 tags and you care most about catching emotional regressions, you need roughly 20 test cases just for that dimension to get reliable slice scores.

```bash
# Quick check
diffprompt diff v1.txt v2.txt --auto-generate --n 20

# Standard
diffprompt diff v1.txt v2.txt --auto-generate --n 40

# High confidence
diffprompt diff v1.txt v2.txt --auto-generate --n 80
```
