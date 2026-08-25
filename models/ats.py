"""Pydantic models for the deterministic Demo ATS Readiness Score."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ATS_DISCLAIMER = (
    "Estimated using this demo rubric; not an official employer ATS score."
)

ATS_LONG_DISCLAIMER = (
    "Estimated using this demo rubric; not an official employer ATS score. "
    "Different employers and applicant-tracking systems use different "
    "proprietary rules."
)

AtsBand = Literal["strong", "needs_targeted_changes", "low"]
AtsPriority = Literal["high", "medium", "low"]
AtsRecommendationCategory = Literal[
    "keyword_alignment",
    "summary",
    "skills",
    "experience",
    "education",
    "section_completeness",
    "format_and_parseability",
    "unsupported_gap",
]

BAND_LABELS: dict[str, str] = {
    "strong": "Strong",
    "needs_targeted_changes": "Needs Targeted Changes",
    "low": "Low",
}

PRIORITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


class AtsRecommendation(BaseModel):
    """One prioritized, section-specific change proposal.

    ``safe_to_apply`` is the guardrail: only safe recommendations may reach the
    drafting step, and unsupported job gaps must stay gaps.
    """

    recommendation_id: str
    priority: AtsPriority
    category: AtsRecommendationCategory
    target_section: str
    current_text: str | None = None
    recommended_change: str
    reason: str
    evidence_resume_ids: list[str] = Field(default_factory=list)
    safe_to_apply: bool
    projected_effect: Literal["high", "medium", "low"]


class AtsAssessment(BaseModel):
    """A complete six-component ATS readiness assessment for one resume."""

    job_id: str
    resume_version: Literal["original", "proposed"]
    total_score: int
    band: AtsBand
    keyword_score: int
    qualification_score: int
    evidence_score: int
    section_score: int
    structure_score: int
    contact_score: int
    matched_required_keywords: list[str] = Field(default_factory=list)
    supported_but_missing_keywords: list[str] = Field(default_factory=list)
    unsupported_job_gaps: list[str] = Field(default_factory=list)
    recommendations: list[AtsRecommendation] = Field(default_factory=list)
    disclaimer: str = ATS_DISCLAIMER

    def band_label(self) -> str:
        """Return the human-readable score band."""
        return BAND_LABELS[self.band]
