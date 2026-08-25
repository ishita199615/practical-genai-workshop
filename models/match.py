"""Pydantic models for the deterministic Demo Job Match Score."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchResult(BaseModel):
    """The explainable Demo Job Match Score for one job posting.

    Every numeric field is calculated in Python. The optional explanation is
    the only field a language model may write, and it may not change a number.
    """

    job_id: str
    total_score: int
    skill_score: int
    similarity_score: int
    role_score: int
    experience_score: int
    preference_score: int
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    explanation: str | None = None
