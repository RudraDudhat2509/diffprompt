# diffprompt

**git diff for your prompt's behavior.**

You changed one sentence in your prompt. Now you're wondering: *did that actually help?*

LangSmith tells you what happened after you shipped. LangFuse tells you what's happening right now. Neither tells you what will happen **before** you change anything.

diffprompt does.

---

## What it does

You give it two prompts. It generates test cases, runs both prompts on all of them, measures the semantic difference between every output pair, and tells you exactly where v2 works better, where it regresses, and why. With concrete examples.

```
$ diffprompt diff v1.txt v2.txt --auto-generate --n 40

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  diffprompt  ·  v1.txt → v2.txt  ·  40 tests  ·  14s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  REGRESSION SCORE   47 / 100   ⚠ CONDITIONAL

  ✓ user_intent:informational      score 0.91   18 tests
  ✓ topic_complexity:factual       score 0.88   12 tests
  ✗ emotional_state:anxious        score 0.21    6 tests   ← regresses badly
  ✗ user_intent:conversational     score 0.34    4 tests

━━ KEY EXAMPLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  #1  MOST IMPORTANT  ·  emotional_state:anxious  ·  divergence 0.79

  input   "I've been struggling with anxiety lately and I don't know
           what to do. Everything feels overwhelming."

  v1      "That sounds really difficult. Anxiety can feel all-
           consuming, especially when multiple stressors pile up.
           Let's think through this together..."

  v2      "Try deep breathing exercises and reduce caffeine."

  why     v2's brevity instruction strips the empathetic framing
          that makes anxious users feel heard before advice lands.

━━ VERDICT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ⚠  CONDITIONAL. Safe to ship for factual/lookup use cases.
     Do NOT ship for emotional or conversational inputs.
```

---

## Install

```bash
pip install diffprompt
```

Requires Python 3.10+. Works fully offline with Ollama. No OpenAI key needed.

---

## Quickstart

```bash
# Set up a free Groq key (runs the LLMs)
export GROQ_API_KEY=your_key_here   # free at console.groq.com

# Run a diff with auto-generated test cases
diffprompt diff "You are a helpful assistant." "You are a concise assistant. Be brief." --auto-generate

# Or bring your own test inputs
diffprompt diff v1.txt v2.txt --test-file inputs.jsonl

# Run fully offline with Ollama
diffprompt diff v1.txt v2.txt --auto-generate --local-only
```

---

## How it works

### 1. Ontology inference
diffprompt reads your prompt and infers what input dimensions matter for testing it. Tone, complexity, intent; whatever's relevant. No hardcoded dimensions. Every prompt gets its own.

### 2. Test generation
Test cases are generated across four buckets: **typical** (real usage), **adversarial** (designed to find failures), **boundary** (edge cases), and **format** (unusual input styles). Each test case is automatically tagged with its inferred dimensions.

### 3. Semantic diff
Both prompts run on all test cases concurrently. Outputs are compared using local embeddings (`all-MiniLM-L6-v2`) to produce a similarity score per pair. High similarity means the change didn't matter. Low similarity means something changed.

### 4. LLM judge
For every meaningfully different pair, a judge LLM evaluates direction: improvement, regression, or neutral. Confident verdicts stay local. Uncertain ones escalate to a larger model automatically.

### 5. Behavioral slicing
Results are grouped by dimension. Instead of one aggregate score, you get a score per behavioral slice. Not "47/100 overall" but "works for factual, breaks for emotional."

### 6. Failure mode clustering
HDBSCAN clusters the judge's reasons automatically. Instead of 40 individual explanations, you get named failure modes: `CONTEXT_LOSS`, `TONE_SHIFT`, `REFUSAL_SHIFT`. No cluster count to specify.

---

## Why not just use LangSmith / Langfuse?

Those tools monitor production. They tell you what happened.

diffprompt is a pre-flight check. It tells you what will happen before you touch production.

Different job. Different tool.

---

## Model cascade; zero cost default

| Layer | Task | Model | Cost |
|---|---|---|---|
| Test generation | Generate inputs | qwen2.5:7b via Ollama | Free local |
| Embedding | Similarity | all-MiniLM-L6-v2 | Free local |
| Runner | Execute prompts | llama-3.3-70b via Groq | Free tier |
| Judge | Verdict + reason | qwen2.5:7b via Ollama | Free local |
| Escalation | Low confidence | llama-3.3-70b via Groq | Free tier |

You can override any layer with `--model` and `--judge`.

---

## CI/CD integration

```yaml
- name: Prompt regression check
  run: |
    diffprompt diff prompts/v1.txt prompts/v2.txt \
      --auto-generate \
      --ci \
      --threshold 75
```

Exits with code 1 if regression score drops below threshold. Merge blocked.

---

## CLI reference

```
diffprompt diff <v1> <v2> [options]

  --auto-generate         Generate test cases automatically
  --n INT                 Number of test cases (default: 40)
  --test-file PATH        Use existing test inputs from .jsonl file
  --model STRING          Override runner model
  --judge STRING          Override judge model
  --local-only            Never call external APIs
  --output FORMAT         terminal (default), json, html
  --save PATH             Save report to file
  --ci                    CI mode; exit code 1 on regression
  --threshold INT         CI failure threshold 0-100 (default: 75)
  --concurrency INT       Max concurrent LLM calls (default: 5)
```

---

## Output formats

**Terminal.** Rich, color-coded, shareable as a GIF.

**JSON.** Full structured report for downstream processing.

**HTML.** Self-contained file, open in browser.

```bash
diffprompt diff v1.txt v2.txt --auto-generate --output html --save report.html
```

---

## Philosophy

Prompts have behavior, not just text.

When you change a prompt, you're not editing a document. You're changing how a system responds to thousands of possible inputs. Most of those inputs you've never seen. Some of them are emotional. Some are adversarial. Some are edge cases you didn't think to test.

diffprompt makes the invisible visible. It tells you which inputs your change helped, which it hurt, and why. Before any of it reaches a user.

---

## Stack

Python 3.10+ · sentence-transformers · HDBSCAN · UMAP · httpx · Typer · Rich · Pydantic · Groq API · Ollama

---

## Contributing

Issues and PRs welcome. The project is young; the core pipeline works, but there's a lot of room to improve test generation quality, tagging accuracy, and output design.

```bash
git clone https://github.com/RudraDudhat2509/diffprompt
cd diffprompt
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
