"""Pydantic models for retrieved and normalized public job postings."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

SourceCategory = Literal[
    "all",
    "linkedin",
    "indeed",
    "google_jobs",
    "company_careers",
    "other",
]

FreshnessWindow = Literal[
    "last_hour",
    "last_24_hours",
    "last_3_days",
    "last_7_days",
]

FreshnessEvidence = Literal[
    "exact_timestamp",
    "date_only",
    "search_filter_only",
    "unavailable",
]

FreshnessStatus = Literal[
    "verified_recent",
    "date_unavailable",
    "possibly_stale",
]

WorkMode = Literal["remote", "hybrid", "onsite", "unknown"]

# Seniority the user is searching for, and the seniority detected on a posting.
# "unknown" is a real answer: many postings never state a level, and inventing
# one would be exactly the kind of unsupported claim this project refuses to make.
ExperienceLevel = Literal[
    "internship",
    "entry",
    "mid",
    "senior",
    "staff_principal",
    "manager",
    "unknown",
]

EXPERIENCE_LEVEL_LABELS: dict[str, str] = {
    "internship": "Internship",
    "entry": "Entry level / Junior",
    "mid": "Mid-level",
    "senior": "Senior",
    "staff_principal": "Staff / Principal / Lead",
    "manager": "Manager / Director",
    "unknown": "Level not stated",
}

# What each level typically asks for, used for honest expectation setting only.
# These are never treated as facts about a specific posting.
EXPERIENCE_LEVEL_YEARS: dict[str, tuple[float, float | None]] = {
    "internship": (0.0, 1.0),
    "entry": (0.0, 2.0),
    "mid": (2.0, 5.0),
    "senior": (5.0, 8.0),
    "staff_principal": (8.0, None),
    "manager": (5.0, None),
}


class RawJobResult(BaseModel):
    """One unprocessed search result returned by the retrieval adapter."""

    title: str | None = None
    description: str | None = None
    url: str
    final_url: str | None = None
    markdown: str | None = None
    metadata: dict = Field(default_factory=dict)
    query_category: SourceCategory = "all"
    freshness_window: FreshnessWindow = "last_24_hours"
    detected_source_category: SourceCategory = "other"
    detected_source_label: str = "Other"
    retrieved_at: datetime
    # False when the search that produced this result carried no time filter,
    # so an undated posting must not claim to be search-filtered.
    time_filter_applied: bool = True


class ExtractedJobFields(BaseModel):
    """Schema Gemini fills in when reading one public job page.

    Deliberately narrow: it holds only what a language model may read off the
    page. Scores, IDs, URLs, freshness labels, and source classification are
    computed in Python.
    """

    title: str | None = None
    company: str | None = None
    location: str | None = None
    work_mode: WorkMode = "unknown"
    description: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    minimum_experience_years: float | None = None
    education_requirement: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    posted_at_text: str | None = None
    apply_url: str | None = None
    is_closed: bool = False
    is_specific_opening: bool = True


class JobPosting(BaseModel):
    """A cleaned, validated public job posting ready for scoring and display."""

    job_id: str
    title: str
    company: str
    location: str | None = None
    work_mode: WorkMode = "unknown"
    query_category: SourceCategory
    source_category: SourceCategory
    source_label: str
    description: str
    description_excerpt: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    minimum_experience_years: float | None = None
    education_requirement: str | None = None
    responsibilities: list[str] = Field(default_factory=list)
    source_url: str
    original_source_url: str | None = None
    apply_url: str | None = None
    source_domain: str
    freshness_window: FreshnessWindow
    posted_at: datetime | None = None
    posting_date: date | None = None
    posting_age_hours: float | None = None
    freshness_evidence: FreshnessEvidence = "unavailable"
    retrieved_at: datetime
    is_closed: bool = False
    freshness_status: FreshnessStatus = "date_unavailable"
    data_mode: Literal["live", "cached"] = "live"
    requested_experience_level: ExperienceLevel = "unknown"
    experience_level: ExperienceLevel = "unknown"
    experience_level_evidence: str | None = None

    def experience_level_label(self) -> str:
        """Return the human-readable seniority detected on this posting."""
        return EXPERIENCE_LEVEL_LABELS[self.experience_level]

    def freshness_label(self) -> str:
        """Return an honest, human-readable freshness statement.

        The label never claims verification the source evidence does not
        support.
        """
        if self.freshness_evidence == "exact_timestamp":
            window_text = FRESHNESS_LABELS[self.freshness_window]
            return f"Verified in {window_text.lower()}"
        if self.freshness_evidence == "date_only":
            return "Date shown; exact time unavailable"
        if self.freshness_evidence == "search_filter_only":
            return "Search-filtered; source timestamp unavailable"
        return "Posting time unavailable"


FRESHNESS_LABELS: dict[str, str] = {
    "last_hour": "Last 1 hour",
    "last_24_hours": "Last 24 hours",
    "last_3_days": "Last 3 days",
    "last_7_days": "Last 7 days",
}

SOURCE_CATEGORY_LABELS: dict[str, str] = {
    "all": "All Public Sources",
    "linkedin": "LinkedIn",
    "indeed": "Indeed",
    "google_jobs": "Google Jobs / Web",
    "company_careers": "Direct Company Careers",
    "other": "Other",
}
