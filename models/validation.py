"""Pydantic models for truthfulness validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClaimReview(BaseModel):
    """One claim extracted from the draft and checked against the resume."""

    claim: str
    status: Literal["supported", "unsupported", "unclear"]
    supporting_resume_ids: list[str] = Field(default_factory=list)
    reason: str


class ClaimReviewBatch(BaseModel):
    """Structured-output wrapper for a batch of LLM claim reviews."""

    reviews: list[ClaimReview] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """The combined deterministic + LLM truthfulness result."""

    valid_source_ids: bool
    unsupported_claims: list[ClaimReview] = Field(default_factory=list)
    unclear_claims: list[ClaimReview] = Field(default_factory=list)
    deterministic_checks: list[dict] = Field(default_factory=list)
    passed: bool

    def failed_checks(self) -> list[dict]:
        """Return only the deterministic checks that failed."""
        return [check for check in self.deterministic_checks if not check.get("passed")]
