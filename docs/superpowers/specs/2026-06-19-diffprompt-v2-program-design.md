# diffprompt v2 — Program Design Spec

> Status: DRAFT for review · Author: Rudra (w/ Claude) · Date: 2026-06-19
> Scope: the full v2 program — production hardening + three product modules + platform (service & web UI).

---

## 1. Vision

diffprompt today answers one question: *"I changed my prompt — did it help or hurt, before I ship?"* It does this well as a one-shot CLI.

v2 turns it into the **pre-production reliability layer for LLM apps**: not just a one-time diff, but a continuous, explainable system that tells you *which part* of a prompt drives behavior, *when* a model silently shifts under you, and *why* an agent made a decision. The through-line is **accuracy, speed, and explainability** — the AI-reliability + observability niche, made concrete.

Three product modules, one engine:

| Module | Question it answers | Maturity hook |
|---|---|---|
| **Attribution** (Idea 1) | *Which sentence in my prompt causes this behavior?* | deep algorithmic story (Shapley) |
| **Drift** (Idea 2) | *Did the model change under me?* | continuous monitoring, the observability hero |
| **Provenance** (Idea 3) | *Why did my agent do that — and was it hijacked?* | agentic security/reliability, most unique |

## 2. Goals & non-goals

**Goals** (the user's four words, made measurable):
- **Accurate** — judge agreement with human labels ≥ 0.8 on a labeled eval set; deterministic, reproducible analysis given a fixed seed; calibrated confidence.
- **Faster** — a 20-test diff completes in < 30s on warm cache against a fast provider; streaming results so first insight appears in < 3s.
- **Sleeker** — verdict-first output (shipped), a cohesive web dashboard, one consistent CLI grammar, zero `print()` noise.
- **Prod-ready** — provider abstraction, structured retry, structured logging, self-telemetry (OTel), ≥ 85% test coverage, CI gating lint + types + tests, graceful failure everywhere.

**Non-goals (v2)**:
- Not a training/fine-tuning tool.
- Not a general LLM gateway (we wrap providers for our use, not as a product).
- Not a replacement for prod observability (Langfuse/Phoenix) — we integrate with them, we are the *pre-prod and drift* layer.
- No multi-tenant SaaS auth/billing in v2 (single-tenant/self-host first; flagged as future).

## 3. Current state (post-hardening, as of this spec)

Already shipped this cycle (PRs #3/#5/#7/#9, #11 in flight):
- `--model`/`--top-n` honored; stale tests fixed.
- CI matrix py3.10–3.12 + badge.
- Pipeline parallelized (anchors/generation/judge) + token-bucket Groq rate limiter.
- Verdict-first output redesign + `output/insights.py` (latency/length/sparkline/scorecard/trade-off), terminal + HTML.
- Ontology hardened against non-string tags (#11).

Known debt this spec addresses: `print()` debug in `cascade.py`; ad-hoc retry; Groq/Ollama hardcoded; no streaming; no self-telemetry; thin coverage on `runner`/`clusterer`; judge over-leniency (observed 100/100); ontology wrapper-key schema; Windows cp1252 unicode crash on piped output.

## 4. Target architecture

```
diffprompt/
  providers/        provider abstraction (Groq, Ollama, OpenAI, Anthropic) + fallback + cost tables
  pipeline/         async orchestrator: emits a typed event stream (AsyncIterator[Event])
  core/             engine: ontology, generator, runner, embedder, judge, clusterer, slicer, scorer, insights
  attribution/      Idea 1 — Shapley attribution over prompt components
  drift/            Idea 2 — fingerprinting, drift detection, scheduler, workers
  provenance/       Idea 3 — source tagging, influence ablation, trust model
  store/            persistence (SQLite default, Postgres for service): runs, reports, baselines, traces, alerts
  service/          FastAPI app + scheduler + worker pool (drift), SSE/WebSocket for live runs
  web/              dashboard (frontend)
  telemetry/        OpenTelemetry tracing + metrics (token/cost/latency)
  obs/              structured logging, retry, ratelimit (existing)
  output/           terminal + html + json renderers
  cli/              command surface
```

**Key principle:** the `pipeline` orchestrator is an **async generator** that yields typed events (`OntologyInferred`, `TestCaseGenerated`, `RunCompleted`, `DiffJudged`, `ClusterFormed`, `ReportReady`). The CLI renders them as a live progress stream; the web service forwards them over SSE/WebSocket; tests consume them deterministically. One event model, three consumers — no divergence.

## 5. Cross-cutting production-readiness (Phase 0 — foundation)

### 5.1 Provider abstraction
- `Provider` protocol: `async def complete(messages, model, **opts) -> Completion` and `async def stream(messages, model, **opts) -> AsyncIterator[Token]`. `Completion` carries text, `usage` (prompt/completion tokens), `model`, `finish_reason`.
- Implementations: `GroqProvider`, `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider`. Capability flags (`supports_streaming`, `reports_usage`).
- `FallbackProvider` composes an ordered list (replaces today's cascade: Ollama → Groq) and applies retry + rate-limit policy per underlying provider.
- Selection via config + `provider/model` strings (extends existing `_split_model`).
- Cost tables (per-model $/1M tokens) → cost estimation surfaced in insights/telemetry.

### 5.2 Streaming (async generators)
- Providers expose token streams; the orchestrator exposes a stage-level event stream.
- CLI: live incremental render (test cases appearing, diffs resolving) instead of a spinner.
- Service: SSE endpoint `GET /runs/{id}/events`.

### 5.3 Retry (proper)
- Central `retry` policy: exponential backoff + full jitter, respects `Retry-After`, caps attempts, classifies retryable (429/5xx/timeout/connection) vs fatal (401/400/422).
- Per-provider **circuit breaker**: after N consecutive failures, open the breaker and fall through to the next provider; half-open probe to recover.
- Composes with the existing rate limiter (rate limiter paces; retry handles transient failure).

### 5.4 Logging
- Replace all `print("[DEBUG]...")` with the `logging` module + a Rich handler. Levels controlled by `-v/-vv`; `--log-format json` for prod. Library code never prints (enforced by a lint check + a test that asserts no stray prints).
- Log: provider calls (model, latency, tokens, cost) at DEBUG; retries/breaker transitions at WARNING; failures at ERROR with context.

### 5.5 Self-telemetry (new feature, on-brand)
- OpenTelemetry spans per pipeline stage and per provider call; metrics for tokens, cost, latency, cache hits.
- Optional OTLP export to Langfuse/Phoenix (the user's niche tools) via env config. This makes diffprompt *itself* observable — a live demo of the positioning.

### 5.6 Caching
- Content-addressed cache (`.diffprompt_cache/`) keyed on (prompt, input, model, params). `--replay` re-runs only downstream stages (e.g. swap judge model without re-paying for runner outputs). Cuts cost and turns iteration fast.

### 5.7 Test strategy (accurate regression tests)
- **Unit**: pure functions (scorer/slicer/insights/attribution math) — fast, no I/O.
- **Property tests** (hypothesis): invariants — e.g. regression_score ∈ [0,100]; all-improvement ⇒ 100; sparkline length == bins; Shapley contributions sum ≈ total effect (efficiency axiom) within tolerance.
- **Provider contract tests**: mocked HTTP (`respx`) — each provider parses success, 429, 401, 5xx, stream chunks correctly.
- **Golden/snapshot tests**: fixture `DiffReport` → terminal + HTML rendered → snapshot compare (catches output regressions).
- **Determinism**: fixed seeds for UMAP/HDBSCAN; deterministic clustering tests.
- **Judge eval harness**: small labeled dataset (human-verdict pairs) → measure judge agreement; gate that the default judge config clears a threshold (addresses observed over-leniency / 100-100).
- **Live smoke** (opt-in, marked `integration`, gated by env): tiny `n` against a real provider; not in default CI, runnable on demand.
- CI adds `ruff` + `mypy` gates (with ruff config for the intentional `E402` in `embedder.py`) alongside the test matrix. Coverage gate ≥ 85%.

## 6. Module specs

### 6.1 Attribution (Idea 1) — "which sentence causes this?"
- **Splitter**: break a prompt into components (sentences/instructions); stable IDs.
- **Method**: Shapley-value attribution via **KernelSHAP-style sampling**. Exact is 2^N coalitions — infeasible; sample M coalitions (component on/off), run each prompt variant over a fixed test set, score behavior vs a reference (reuse embedder similarity + judge), solve weighted least squares for per-component contributions.
- **Noise handling**: LLM nondeterminism → repeat + average, report confidence intervals; configurable sampling budget (`--budget`).
- **Output**: per-component contribution to overall behavior and per behavioral slice ("Sentence 3 drives 60% of the empathy behavior; removing it mainly hurts `emotional_state:frustrated`").
- **CLI**: `diffprompt attribute prompt.txt [--budget N]`. **Reuses** ontology/generator/runner/embedder/judge.
- **Risks**: cost (mitigate w/ budget + cache); attribution stability (report CIs, warn on low confidence).

### 6.2 Drift (Idea 2) — "did the model change under me?" (observability hero)
- **Fingerprint**: distribution-level signature of a (prompt, model) over a fixed probe set — output embeddings → HDBSCAN cluster structure + summary stats (centroids, dispersion, length/latency profile). Versioned, stored as a baseline.
- **Drift detection**: re-run probe set on a schedule; compare new fingerprint to baseline via embedding-distribution distance (MMD / energy distance) + cluster-structure change + centroid drift. Flag when above threshold. With prompt fixed, attribute drift to the model.
- **Service**: scheduler triggers probe runs; an `asyncio.Queue` + **worker pool** processes them (this is where a queue genuinely earns its place vs the fixed fan-out elsewhere); alerting via webhook/Slack/email on drift.
- **Storage**: time-series of fingerprints + alerts.
- **Web**: drift timeline, cluster maps over time, alert history.
- **CLI**: `diffprompt baseline create <prompt>`, `diffprompt drift check <baseline>`, `diffprompt watch <baseline> --interval`.
- **Risks**: false positives (calibrate thresholds, require sustained drift); probe-set design (document guidance).

### 6.3 Provenance (Idea 3) — "why did the agent do that?" (security flagship)
- **Source tagging**: wrap an agent's context; tag each segment by source (system / user / retrieved-doc / tool-output) and a trust level.
- **Influence attribution**: counterfactual ablation — mask/remove a source, re-run, measure change in the agent's action/output (reuses the diff + attribution machinery). Score each source's influence on the decision.
- **Trust composition** across multi-hop retrieval chains.
- **Injection detection**: a low-trust source exerting high influence on an action = red flag ("this decision was 70% driven by an untrusted document").
- **Integration**: framework-agnostic core + a LangGraph adapter (the user's stack). 
- **Output**: provenance trace per agent action; web view of the influence breakdown.
- **Risks**: influence without attention access is approximate (counterfactual is the honest method; state limits clearly); agent-framework coupling (keep core agnostic).

## 7. Web UI / service
- **Backend**: FastAPI (existing user expertise). REST for reports/baselines/traces; SSE/WebSocket for live runs and drift updates. Single-tenant/self-host in v2.
- **Frontend**: **OPEN DECISION (see §10)** — proposed Vite + React + Tailwind, aesthetic matched to the portfolio theme (verify: memory conflicts cinematic-gold vs beige-editorial). Views: diff report viewer, attribution explorer, drift timeline, provenance trace.
- **Packaging**: `diffprompt serve` boots API + static dashboard; Docker image for self-host.

## 8. Data model (store)
Entities: `Run`, `Report`, `Baseline`, `Fingerprint`, `Attribution`, `ProvenanceTrace`, `Alert`. SQLite default (single file, zero-config); Postgres for the service. Pydantic models already define the report shape — extend, version with a schema migration tool (Alembic for Postgres path).

## 9. CLI grammar (v2)
```
diffprompt diff <v1> <v2>          # existing
diffprompt attribute <prompt>      # Idea 1
diffprompt baseline create <p>     # Idea 2
diffprompt drift check <baseline>
diffprompt watch <baseline>
diffprompt provenance <trace.json> # Idea 3
diffprompt serve                   # web + API
diffprompt replay <report.json>    # cache replay
```
Consistent flags across commands: `--model`, `--judge`, `--provider`, `--local-only`, `--output`, `-v/-vv`.

## 10. Open decisions (need your call)
1. **Frontend stack & theme** — Vite+React+Tailwind vs Next.js; and which portfolio aesthetic to match (cinematic-gold vs beige-editorial — memory conflicts, must verify against the live site).
2. **Judge default** — keep Groq 8b fallback, or require a stronger judge by default given observed over-leniency? Affects accuracy goal.
3. **Self-host only vs hosted later** — confirm v2 is single-tenant/self-host (no auth/billing).

## 11. Phasing (build order; all speced deep, sequenced for delivery)
- **Phase 0 — Foundation**: providers, retry, logging, streaming orchestrator, telemetry, caching, test infra + CI gates. *Everything else depends on this.*
- **Phase 1 — Attribution**: pure analysis on the new engine; fastest standalone win after foundation.
- **Phase 2 — Drift**: store + scheduler + worker pool + alerting + first web views (the observability hero).
- **Phase 3 — Provenance**: source tagging + ablation + trust model + LangGraph adapter.
- **Phase 4 — Web maturation**: unify all modules in the dashboard, Docker self-host, polish.

## 12. Success metrics
- Speed: n=20 diff < 30s warm; first event < 3s.
- Reliability: 0 unhandled exceptions across a 100-run live soak; retry recovers ≥ 95% of transient failures.
- Accuracy: judge agreement ≥ 0.8 on the eval set; attribution efficiency-axiom error < 5%.
- Quality: coverage ≥ 85%; ruff + mypy clean in CI.
- Traction: stars, and at least one external user running drift monitoring.

## 13. Risks (program-level)
- **Scope** — four phases is large; Phase 0 must stay disciplined or everything slips. Mitigate: ship each phase behind its own milestone, each independently useful.
- **Cost/limits** — Shapley + drift probes multiply LLM calls. Mitigate: cache, budgets, rate limiter, local-first.
- **Judge quality** — the accuracy goal hinges on it; the eval harness is a Phase-0 gate, not an afterthought.
- **Web maintenance** — a dashboard is ongoing surface area; keep it thin and generated from the same event model.

---

## Appendix A — mapping to `future-ideas/IDEAS.md`
- Idea 1 → §6.1 Attribution. Idea 2 → §6.2 Drift. Idea 3 → §6.3 Provenance. The repo's own notes (Idea 2 fastest to users, Idea 1 deepest interview story, Idea 3 most unique) are reflected in the phasing rationale.
