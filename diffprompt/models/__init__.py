"""
Core data models for diffprompt.
These types flow through the entire pipeline — generator → runner → diff → analysis → output.
"""
from __future__ import annotations
import uuid
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class TestCategory(str, Enum):
    TYPICAL     = "typical"
    BOUNDARY    = "boundary"
    ADVERSARIAL = "adversarial"
    FORMAT      = "format"


class Verdict(str, Enum):
    IMPROVEMENT = "improvement"
    REGRESSION  = "regression"
    NEUTRAL     = "neutral"


class OutputFormat(str, Enum):
    TERMINAL = "terminal"
    JSON     = "json"
    HTML     = "html"
    TXT      = "txt"


class TestCase(BaseModel):
    id: str
    input: str
    category: TestCategory
    tags: dict[str, str] = Field(default_factory=dict)


class RunResult(BaseModel):
    test_id: str
    prompt_version: str
    output: str
    model_used: str
    latency_ms: Optional[float] = None


class DiffResult(BaseModel):
    test_case: TestCase
    v1_output: str
    v2_output: str
    similarity: float
    divergence: float
    verdict: Verdict
    reason: str
    judge_confidence: float
    importance_score: float = 0.0
    cluster_label: int = -1
    cluster_centrality: float = 0.0
    v1_latency_ms: Optional[float] = None
    v2_latency_ms: Optional[float] = None


class SliceResult(BaseModel):
    dimension: str
    value: str
    label: str
    n: int
    mean_similarity: float
    variance: float
    typical_ratio: float
    confidence: float
    verdict: Verdict
    depth: int = 1


class Cluster(BaseModel):
    label: int
    name: str
    description: str
    n: int
    mean_similarity: float
    test_ids: list[str]


class KeyExample(BaseModel):
    slot: str
    diff: DiffResult
    why_it_matters: str


class DiffReport(BaseModel):
    prompt_v1: str
    prompt_v2: str
    model: str
    judge: str
    test_cases: list[TestCase]
    diversity_score: float
    diffs: list[DiffResult]
    slices: list[SliceResult]
    clusters: list[Cluster]
    unclustered: list[DiffResult]
    key_examples: list[KeyExample]
    regression_score: float
    n_improved: int
    n_regressed: int
    n_neutral: int
    verdict: Verdict
    recommendation: str


# ── evolve ──────────────────────────────────────────────────────────────
# Golden-task scoring is deterministic-only: regex/keyword/json_schema/numeric
# checks, plus embedding similarity against a golden answer. Never an LLM judge.

class CheckType(str, Enum):
    REGEX       = "regex"
    JSON_SCHEMA = "json_schema"
    KEYWORD     = "keyword"
    NUMERIC     = "numeric"


class Check(BaseModel):
    type: CheckType
    weight: float = 1.0

    # regex
    pattern: Optional[str] = None
    ignore_case: bool = False

    # json_schema
    schema_: Optional[dict] = Field(None, alias="schema")

    # keyword
    keywords: Optional[list[str]] = None
    match_any: bool = False  # False = ALL keywords must be present

    # numeric
    expected: Optional[float] = None
    tolerance: float = 0.0
    comparator: Literal["eq", "gte", "lte"] = "eq"
    extract_pattern: Optional[str] = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _require_fields_for_type(self) -> "Check":
        if self.type == CheckType.REGEX and not self.pattern:
            raise ValueError("regex check requires 'pattern'")
        if self.type == CheckType.JSON_SCHEMA and self.schema_ is None:
            raise ValueError("json_schema check requires 'schema'")
        if self.type == CheckType.KEYWORD and not self.keywords:
            raise ValueError("keyword check requires 'keywords'")
        if self.type == CheckType.NUMERIC and self.expected is None:
            raise ValueError("numeric check requires 'expected'")
        return self


class GoldenTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    input: str
    weight: float = 1.0
    golden_answer: Optional[str] = None
    golden_answer_weight: float = 1.0
    checks: list[Check] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_scoring_signal(self) -> "GoldenTask":
        if not self.checks and not self.golden_answer:
            raise ValueError(
                f"task {self.id!r} has no checks and no golden_answer — nothing to score it on"
            )
        return self


class PromptVariant(BaseModel):
    prompt: str
    generation: int
    fitness: Optional[float] = None


class GenerationRecord(BaseModel):
    generation: int
    best_fitness: float
    mean_fitness: float
    best_prompt: str


class EvolveReport(BaseModel):
    original_prompt: str
    final_prompt: str
    final_fitness: float
    model: str
    population_size: int
    n_generations_run: int
    stopped_reason: Literal["max_generations", "patience"]
    generations: list[GenerationRecord]
    golden_tasks_path: str
    n_tasks: int
