"""Shared fixtures. No test in the default suite makes a network call."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings  # noqa: E402
from models.job import JobPosting, RawJobResult  # noqa: E402
from models.resume import ResumeProfile  # noqa: E402
from services.llm_interface import NullLLMClient  # noqa: E402

FIXED_NOW = datetime(2026, 8, 20, 15, 0, 0, tzinfo=timezone.utc)

SAMPLE_DESCRIPTION = (
    "Data Analyst Intern\nLakeside Analytics — Houston, TX (Hybrid)\n\n"
    "About the role\n\nWe are looking for a Data Analyst Intern to support "
    "recurring reporting, data cleaning, and dashboard development. You will "
    "write SQL queries against our reporting warehouse, use Python and pandas "
    "for repeatable data cleaning, and build dashboards for business partners. "
    "Minimum qualifications include enrollment in a bachelor's degree program, "
    "working knowledge of SQL including joins and aggregation, comfort with "
    "Excel including pivot tables, and some exposure to Python for data "
    "analysis. Preferred qualifications include experience building dashboards "
    "in Tableau and coursework in statistics. This is a paid internship with a "
    "hybrid schedule in our Houston office."
)


@pytest.fixture(autouse=True)
def _no_quota_backoff(monkeypatch):
    """Remove the provider backoff sleep so the suite stays fast.

    The backoff is real behaviour, but waiting for it in tests only buys
    slower feedback.
    """
    monkeypatch.setattr("services.gemini_client.QUOTA_BACKOFF_SECONDS", 0.0)


@pytest.fixture
def resume() -> ResumeProfile:
    """The fictional master resume."""
    payload = json.loads(
        (PROJECT_ROOT / "data" / "sample_resume.json").read_text(encoding="utf-8")
    )
    return ResumeProfile.model_validate(payload)


@pytest.fixture
def now() -> datetime:
    """A fixed reference time so every score is reproducible."""
    return FIXED_NOW


@pytest.fixture
def settings() -> Settings:
    """Cached-mode settings pointing at the repository's own data files."""
    return Settings(demo_mode="cached")


@pytest.fixture
def null_llm() -> NullLLMClient:
    """An LLM client that always declines, forcing deterministic paths."""
    return NullLLMClient()


def make_job(**overrides: Any) -> JobPosting:
    """Build a valid :class:`JobPosting` with sensible demo defaults."""
    defaults: dict[str, Any] = {
        "job_id": "job_test000001",
        "title": "Data Analyst Intern",
        "company": "Lakeside Analytics",
        "location": "Houston, TX",
        "work_mode": "hybrid",
        "query_category": "company_careers",
        "source_category": "company_careers",
        "source_label": "Greenhouse",
        "description": SAMPLE_DESCRIPTION,
        "description_excerpt": SAMPLE_DESCRIPTION[:200],
        "required_skills": ["SQL", "Excel", "Python"],
        "preferred_skills": ["Tableau", "Statistics"],
        "minimum_experience_years": None,
        "education_requirement": "Currently enrolled in a bachelor's degree program",
        "responsibilities": ["Prepare recurring weekly reporting"],
        "source_url": "https://job-boards.greenhouse.io/lakeside/jobs/4410021",
        "original_source_url": "https://job-boards.greenhouse.io/lakeside/jobs/4410021",
        "apply_url": None,
        "source_domain": "job-boards.greenhouse.io",
        "freshness_window": "last_24_hours",
        "posted_at": FIXED_NOW,
        "posting_date": FIXED_NOW.date(),
        "posting_age_hours": 3.0,
        "freshness_evidence": "exact_timestamp",
        "retrieved_at": FIXED_NOW,
        "is_closed": False,
        "freshness_status": "verified_recent",
        "data_mode": "live",
    }
    defaults.update(overrides)
    return JobPosting(**defaults)


def make_raw(**overrides: Any) -> RawJobResult:
    """Build a valid :class:`RawJobResult` for normalization tests."""
    defaults: dict[str, Any] = {
        "title": "Data Analyst Intern — Lakeside Analytics",
        "description": "Short search snippet.",
        "url": "https://job-boards.greenhouse.io/lakeside/jobs/4410021",
        "final_url": "https://job-boards.greenhouse.io/lakeside/jobs/4410021",
        "markdown": SAMPLE_DESCRIPTION,
        "metadata": {},
        "query_category": "company_careers",
        "freshness_window": "last_24_hours",
        "detected_source_category": "company_careers",
        "detected_source_label": "Greenhouse",
        "retrieved_at": FIXED_NOW,
    }
    defaults.update(overrides)
    return RawJobResult(**defaults)
