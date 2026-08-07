from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


EvaluationJobStatus = Literal[
    "queued",
    "running",
    "completed",
    "cancelled",
    "failed",
]
EvaluationCaseStatus = Literal[
    "pending",
    "running",
    "passed",
    "failed",
    "error",
]


class EvaluationJudgment(BaseModel):
    score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1000)
    strengths: list[str] = Field(default_factory=list, max_length=5)
    gaps: list[str] = Field(default_factory=list, max_length=5)


class EvaluationCaseResult(BaseModel):
    row_number: int = Field(ge=2)
    question: str
    expected: str
    actual_answer: str | None = None
    status: EvaluationCaseStatus = "pending"
    passed: bool | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    error: str | None = None
    elapsed_ms: int | None = Field(default=None, ge=0)


class EvaluationSummary(BaseModel):
    total: int = Field(ge=0)
    completed: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)


class EvaluationJobResponse(BaseModel):
    job_id: str
    filename: str
    sheet_name: str
    kb_version: str
    judge_model: str
    pass_threshold: float = Field(ge=0, le=1)
    status: EvaluationJobStatus
    summary: EvaluationSummary
    cases: list[EvaluationCaseResult]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class EvaluationHistoryItem(BaseModel):
    job_id: str
    filename: str
    sheet_name: str
    kb_version: str
    judge_model: str
    pass_threshold: float = Field(ge=0, le=1)
    status: EvaluationJobStatus
    summary: EvaluationSummary
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class EvaluationHistoryResponse(BaseModel):
    evaluations: list[EvaluationHistoryItem]


class EvaluationDeleteResponse(BaseModel):
    deleted: bool
