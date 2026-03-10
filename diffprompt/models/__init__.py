"""
Core data models for diffprompt.
These types flow through the entire pipeline — generator → runner → diff → analysis → output.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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
