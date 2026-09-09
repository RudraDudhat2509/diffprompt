# Changelog

## 1.2.0

### Added

- `diffprompt evolve` — deterministic, LLM-free genetic optimization of prompts against golden tasks.
  - New `--golden-tasks` file format (`.yaml`/`.yml` or `.jsonl`): checkable inputs with deterministic checks (regex, keyword, JSON schema, numeric) and/or a golden answer compared via embedding similarity.
  - Fitness is never LLM-judged — only deterministic checks and local embedding similarity (`all-MiniLM-L6-v2`, reused from `diff`).
  - Genetic loop: template-based population init, crossover, mutation from a fixed transform menu, top-K selection with elitism, and early stopping via `--patience`.
  - `--output terminal|json|html` and `--save PATH`, matching `diff`'s output options.
  - Does not change `diff`'s behavior or its LLM-judge pipeline.

### Dependencies

- Added `jsonschema>=4.0.0` (JSON schema validity checks).
- Added `pyyaml>=6.0` (golden task YAML files).

## 0.2.0

Prior release. See git history for changes before the changelog was introduced.
