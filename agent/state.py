"""Typed state for the Cougar Career Agent graph.

List fields that several nodes contribute to use additive reducers so the
activity log, warnings, and errors accumulate instead of overwriting one
another.
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, TypedDict

from models.application import TailoredApplication
from models.ats import AtsAssessment, AtsRecommendation
from models.job import (
    ExperienceLevel,
    FreshnessWindow,
    JobPosting,
    RawJobResult,
    SourceCategory,
)
from models.match import MatchResult
from models.resume import ResumeProfile
from models.validation import ValidationReport


class CareerAgentState(TypedDict, total=False):
    """Everything the graph reads and writes for one workshop run."""

    thread_id: str

    # Search preferences
    role: str
    location: str
    work_mode: str
    freshness_window: FreshnessWindow
    freshness_tbs: str
    freshness_cutoff_utc: datetime
    query_category: SourceCategory
    # The seniority the user asked for. "unknown" — the default everywhere —
    # means no level filter at all. It is never copied onto a result: what a
    # posting states about its own level is a separate, detected value.
    experience_level: ExperienceLevel
    source_domains: list[str]
    search_query: str

    # Retrieval and ranking
    resume: ResumeProfile
    raw_jobs: list[RawJobResult]
    normalized_jobs: list[JobPosting]
    filtered_jobs: list[JobPosting]
    ranked_matches: list[MatchResult]

    # Selection and assessment
    selected_job_id: str | None
    selected_job: JobPosting | None
    ats_assessment: AtsAssessment | None
    ats_recommendations: list[AtsRecommendation]
    projected_ats_assessment: AtsAssessment | None
    safe_ats_recommendations_applied: bool

    # Drafting, validation, approval
    tailored_application: TailoredApplication | None
    validation_report: ValidationReport | None
    revision_count: int
    approval_decision: str | None
    approval_feedback: str | None

    # Run metadata
    data_mode: str
    # True for the shipped fictional cache, False for a real captured run.
    cache_is_synthetic: bool
    retrieval_timestamp: datetime | None
    approved_at: datetime | None
    output_files: list[str]
    progress_events: Annotated[list[dict], operator.add]
    warnings: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def event(label: str, status: str = "ok", detail: str | None = None) -> dict[str, Any]:
    """Build one observable activity event.

    Events describe what the agent did — retrieval, filtering, scoring — and
    never expose model reasoning.
    """
    record: dict[str, Any] = {"label": label, "status": status}
    if detail:
        record["detail"] = detail
    return record
