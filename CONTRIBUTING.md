# Contributing to diffprompt

Thanks for your interest. This document covers everything you need to get set up, understand the codebase, and submit a contribution.

---

## Setup

### Requirements

- Python 3.10 or higher
- Git
- Ollama (optional, for local model testing)
- A Groq API key (free at console.groq.com)

### Install in development mode

```bash
git clone https://github.com/RudraDudhat2509/diffprompt
cd diffprompt
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, pytest-asyncio, and other dev dependencies defined in `pyproject.toml`.

### Set up your API key

```bash
export GROQ_API_KEY=your_key_here
```

### Verify everything works

```bash
pytest tests/ -v
python examples/similarity_playground.py
python examples/ontology_inspect.py
```

All three should pass without errors before you make any changes.

---

## Project structure

```
diffprompt/
  diffprompt/
    cli.py              Entry point. Orchestrates the full pipeline.
    models/
      __init__.py       All Pydantic data models. Start here to understand data flow.
      cascade.py        LLM client. Ollama first, Groq fallback.
    core/
      ontology.py       Infers dimensions from prompt, tags test inputs.
      generator.py      Creates test cases across four taxonomy buckets.
      embedder.py       Local embeddings and similarity scoring.
      runner.py         Runs both prompts on all test cases concurrently.
      judge.py          LLM-as-judge; determines verdict per diff.
      clusterer.py      HDBSCAN clustering of failure modes.
      slicer.py         Behavioral slicing by input dimension.
      scorer.py         Regression score and key example selection.
    output/
      terminal.py       Rich terminal renderer.
      exporter.py       JSON/HTML export.
  tests/
    test_embedder.py
    test_generator.py
    test_slicer.py
  examples/
    basic_diff.py
    ontology_inspect.py
    similarity_playground.py
```

### Data flow

Every piece of data in diffprompt is a typed Pydantic model. The pipeline looks like this:

```
prompt v1 + prompt v2
      |
  Ontology          infers dimensions, tags every input
      |
  generator         produces list[TestCase]
      |
  runner            produces dict[test_id, RunResult] x2
      |
  embedder          produces list[float] similarity scores
      |
  judge             produces (Verdict, reason, confidence) per pair
      |
  DiffResult        assembled from all of the above
      |
  clusterer         groups DiffResults into Clusters
  slicer            groups DiffResults into SliceResults
  scorer            computes regression_score, selects KeyExamples
      |
  DiffReport        final container; passed to output layer
      |
  terminal/exporter rendered output
```

If you want to understand where to make a change, follow the data type. Adding a new field to `DiffResult`? It lives in `models/__init__.py`, gets populated somewhere in the pipeline, and consumed in `terminal.py` or `scorer.py`.

---

## Making changes

### Adding a new output format

1. Add the format to `OutputFormat` enum in `models/__init__.py`
2. Add the rendering logic in `output/exporter.py`
3. Handle the new format in `cli.py` where `--output` is processed
4. Add an example to `examples/`

### Adding a new taxonomy bucket

1. Add a new value to `TestCategory` in `models/__init__.py`
2. Add the generation prompt to `TAXONOMY_PROMPTS` in `core/generator.py`
3. Add the distribution fraction to `DISTRIBUTION` in `core/generator.py`
4. Make sure fractions still sum to 1.0

### Adding a new model provider

1. Add a new `call_<provider>()` function in `models/cascade.py` following the same pattern as `call_groq()`
2. Add it as a fallback in `call_cascade()` after Groq
3. Add the relevant API key check and error message
4. Document the new env variable in the README

### Changing the judge prompt

The judge prompt is in `core/judge.py` as `JUDGE_PROMPT`. Any change here affects every verdict in the pipeline. When testing a new judge prompt, run `examples/basic_diff.py` with both the old and new version and compare the verdict distributions. A good judge prompt should produce roughly 30-50% neutral verdicts on similar prompts, not collapse everything into improvement or regression.

### Changing the ontology prompt

The ontology prompt is in `core/ontology.py` as `INFER_PROMPT`. Changes here affect what dimensions get inferred, which affects tagging, which affects slicing. Run `examples/ontology_inspect.py` after any change to verify the dimensions and tagging quality.

---

## Tests

### Running tests

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_embedder.py -v

# With coverage
pytest tests/ --cov=diffprompt --cov-report=term-missing
```

### Writing tests

Tests live in `tests/`. Each core module has a corresponding test file. Follow the existing patterns.

For async functions, use `pytest-asyncio`. Mark async tests with `@pytest.mark.asyncio` or set `asyncio_mode = "auto"` in `pyproject.toml` (already done).

For functions that make LLM calls, do not call real APIs in tests. Either mock the call or test the function's behavior with pre-built inputs that don't require LLM calls. See `test_slicer.py` for an example of building `DiffResult` objects directly without running the full pipeline.

### What to test

Every public function in `core/` should have at least:
- A happy path test with typical inputs
- An edge case test (empty input, single item, etc.)
- A type/shape test verifying the return value structure

---

## Code style

- Type hints on every function signature
- Pydantic models for all data that crosses module boundaries
- Async for anything that makes network or LLM calls
- No print statements in library code. Use the `logger` or let the CLI handle output.
- Private helpers prefixed with underscore: `_parse_judge_response`, `_compute_confidence`
- Constants in ALL_CAPS at the top of the file
- Keep functions short. If a function is longer than 40 lines, it probably does too much.

---

## Submitting a PR

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `pytest tests/ -v` and make sure everything passes
4. Run `python examples/basic_diff.py` end to end at least once
5. Update documentation if you changed any public API or added a feature
6. Open a PR with a clear description of what changed and why

For significant changes (new features, architecture changes), open an issue first to discuss before writing code.

---

## Known gaps and good first contributions

These are areas where contributions would have high impact:

**Test generation quality.** The generator sometimes returns malformed JSON or truncated arrays. Better retry logic and input validation would make the tool more reliable.

**Tagging accuracy.** The zero-shot label embedding approach works but mislabels edge cases. A better approach for specific prompt domains (code, medical, legal) would be valuable.

**HTML output.** `output/exporter.py` is currently a stub. A proper HTML report with a sortable table and diff view per example would make the tool more shareable.

**More test coverage.** `core/runner.py`, `core/judge.py`, `core/clusterer.py`, and `core/scorer.py` have no tests yet. Adding tests for these would significantly improve confidence in the pipeline.

**Windows compatibility.** The tool was developed on Linux/Mac. There are known issues with environment variable handling on Windows PowerShell. Fixes and documentation for Windows users are welcome.
