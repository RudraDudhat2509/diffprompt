# diffprompt

> git diff for your prompt's behavior

```bash
pip install diffprompt
diffprompt diff v1.txt v2.txt --auto-generate
```

---

You changed one sentence in your prompt. Now you're wondering: *did that actually help?*

LangSmith tells you what happened after you shipped. LangFuse tells you what's happening right now. Neither tells you what will happen **before** you change anything.

diffprompt does.

---

## What it does

You give it two prompts. It generates test cases, runs both prompts on all of them, measures the semantic difference between every output pair, and tells you exactly where v2 works better, where it regresses, and why.

```
$ diffprompt diff v1.txt v2.txt --auto-generate --n 20

diffprompt  v0.1.0  model: groq/llama-3.3-70b-versatile  judge: local/qwen2.5:7b  tests: 20
━━ SUMMARY
  18.2/100  ███░░░░░░░░░░░░░░░░░  4 improved  16 regressed  0 neutral
  mix:  9 typical  · 7 adversarial  · 2 boundary  · 2 format

━━ BEHAVIORAL PROFILE
  v2 performs well when...
  ✓ user_intent:informational        score 0.79  4 tests
  v2 struggles when...
  ✗ emotional_state:frustrated       score 0.43  5 tests
  ✗ request_type:specific_solutions  score 0.51  11 tests

━━ KEY EXAMPLES
  MOST IMPORTANT  emotional_state:frustrated  divergence 0.90  conf 0.91

  input  Can you help me with a math problem I'm stuck on
  v1     I'd be happy to help you with your math problem. What kind of problem are you working on?
  v2     What's the problem?
  why    v2's brevity instruction strips the empathetic framing that makes
         frustrated users feel heard before the question lands.

━━ VERDICT
  ✗ DO NOT SHIP
  Keep v1 for emotional_state:frustrated, request_type:specific_solutions.
  Primary failure mode: CONTEXT_LOSS (6 cases).
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
# Option A — use Groq (free at console.groq.com)
export GROQ_API_KEY=your_key_here

diffprompt diff v1.txt v2.txt --auto-generate

# Option B — run fully offline with Ollama
ollama pull qwen2.5:7b
diffprompt diff v1.txt v2.txt --auto-generate --local-only

# Option C — bring your own test inputs
diffprompt diff v1.txt v2.txt --test-file inputs.jsonl
```

---

## How it works

### 1. Ontology inference

diffprompt reads your prompt and infers what input dimensions matter for testing it — tone, complexity, intent, emotional state, whatever's relevant. No hardcoded dimensions. Every prompt gets its own.

### 2. Test generation

Test cases are generated across four buckets: **typical** (real usage), **adversarial** (designed to find failures), **boundary** (edge cases), and **format** (unusual input styles). Each case is automatically tagged with its inferred dimensions.

### 3. Semantic diff

Both prompts run on all test cases concurrently. Outputs are compared using local embeddings (`all-MiniLM-L6-v2`) to produce a similarity score per pair. High similarity means the change didn't matter. Low similarity means something changed.

### 4. LLM judge

For every meaningfully different pair, a judge LLM evaluates direction: improvement, regression, or neutral. Confident verdicts stay local. Uncertain ones escalate to a larger model automatically.

### 5. Behavioral slicing

Results are grouped by dimension. Instead of one aggregate score, you get a score per behavioral slice — not "47/100 overall" but "works for factual, breaks for emotional."

### 6. Failure mode clustering

HDBSCAN clusters the judge's reasons automatically. Instead of 20 individual explanations, you get named failure modes: `CONTEXT_LOSS`, `TONE_SHIFT`, `REFUSAL_SHIFT`.

---

## Why not LangSmith / Langfuse?

Those tools monitor production. They tell you what happened.

diffprompt is a pre-flight check. It tells you what will happen before you touch production.

Different job. Different tool.

---

## Model cascade — zero cost by default

| Layer | Task | Default | Cost |
|-------|------|---------|------|
| Test generation | Generate inputs | qwen2.5:7b via Ollama | Free local |
| Embedding | Similarity | all-MiniLM-L6-v2 | Free local |
| Runner | Execute prompts | llama-3.3-70b via Groq | Free tier |
| Judge | Verdict + reason | qwen2.5:7b via Ollama | Free local |
| Escalation | Low confidence | llama-3.3-70b via Groq | Free tier |

Override any layer with `--model` and `--judge`.

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
  --no-judge              Skip judge, similarity scores only
  --output FORMAT         terminal (default) | json | html
  --save PATH             Save report to file
  --top-n INT             Show top N key examples (default: 3)
  --verbose               Show all diffs ranked by divergence
  --quiet                 Score + verdict only
  --ci                    CI mode: exit 1 on regression
  --threshold INT         CI failure threshold 0-100 (default: 75)
```

---

## Output formats

**Terminal** — color-coded, fits in one screen.

**JSON** — full structured report for downstream processing.

**HTML** — self-contained file, open in browser.

```bash
diffprompt diff v1.txt v2.txt --auto-generate --output html --save report.html
```

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

## Philosophy

Prompts have behavior, not just text.

When you change a prompt, you're not editing a document. You're changing how a system responds to thousands of possible inputs. Most of those inputs you've never seen. Some of them are edge cases you didn't think to test.

diffprompt makes the invisible visible. It tells you which inputs your change helped, which it hurt, and why — before any of it reaches a user.

---

## Stack

Python 3.10+ · sentence-transformers · HDBSCAN · UMAP · httpx · Click · Rich · Pydantic · Groq API · Ollama

---

## Contributing

Issues and PRs welcome.

```bash
git clone https://github.com/RudraDudhat2509/diffprompt
cd diffprompt
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT