"""Pydantic models for the tailored application package."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RevisedBullet(BaseModel):
    """One revised experience bullet traced back to a real source bullet."""

    source_bullet_id: str
    original_text: str
    revised_text: str


class TailoredDraft(BaseModel):
    """The structured-output schema Gemini fills in when drafting.

    Deliberately excludes ``job_id``: identifiers are assigned in Python so the
    model cannot attach a draft to the wrong posting.
    """

    revised_summary: str
    revised_bullets: list[RevisedBullet] = Field(default_factory=list)
    reordered_skills: list[str] = Field(default_factory=list)
    keywords_used: list[str] = Field(default_factory=list)
    applied_ats_recommendation_ids: list[str] = Field(default_factory=list)
    unsupported_ats_gaps_not_applied: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    cover_letter: str = ""


class TailoredApplication(BaseModel):
    """The small, truthful application package drafted for one job."""

    job_id: str
    revised_summary: str
    revised_bullets: list[RevisedBullet] = Field(default_factory=list)
    reordered_skills: list[str] = Field(default_factory=list)
    keywords_used: list[str] = Field(default_factory=list)
    applied_ats_recommendation_ids: list[str] = Field(default_factory=list)
    unsupported_ats_gaps_not_applied: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    cover_letter: str
