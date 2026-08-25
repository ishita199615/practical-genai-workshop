"""Build the data a teaching step runs against.

The lab must open and work in a lecture hall with flaky wifi and an exhausted
API quota, so the sample data comes from the cached demonstration file and the
normalization runs with no language model at all. Every step therefore has real
job text to teach with before a single network call is made.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from config import Settings
from lessons.base import LessonContext
from models.job import JobPosting, RawJobResult
from models.resume import ResumeProfile
from services.llm_interface import NullLLMClient
from tools.firecrawl_search import cached_raw_results, load_cache
from tools.job_normalizer import normalize_jobs

logger = logging.getLogger(__name__)


def load_sample_resume(settings: Settings) -> ResumeProfile:
    """Load and validate the master resume the settings point at."""
    with settings.resume_path.open("r", encoding="utf-8") as handle:
        return ResumeProfile.model_validate(json.load(handle))


def load_cached_postings(settings: Settings) -> list[JobPosting]:
    """Return normalized postings from the cached demonstration data.

    Uses :class:`NullLLMClient` deliberately: extraction falls back to the
    deterministic path, so the result is identical on every machine and needs no
    API key. Returns an empty list rather than raising if the cache is missing,
    because a broken cache must not take the whole lab down.
    """
    try:
        payload = load_cache(settings.cache_path)
    except Exception as exc:  # noqa: BLE001 - the lab must still open
        logger.warning("Lesson cache load failed: %s", type(exc).__name__)
        return []

    entries = cached_raw_results(
        payload, query_category="all", freshness_window="last_24_hours"
    )
    retrieved_at = _parse_iso(payload.get("originally_retrieved_at")) or datetime.now(
        timezone.utc
    )

    raw: list[RawJobResult] = []
    for entry in entries:
        record = dict(entry)
        record["query_category"] = "all"
        record["freshness_window"] = "last_24_hours"
        record["retrieved_at"] = retrieved_at
        try:
            raw.append(RawJobResult.model_validate(record))
        except Exception:  # noqa: BLE001 - cache is data, not code
            logger.warning("Skipping an invalid cached lesson record.")

    postings, _ = normalize_jobs(
        raw,
        NullLLMClient("Lessons normalize offline so they always work."),
        max_description_chars=settings.max_job_description_chars,
        data_mode="cached",
        limit=settings.max_job_results,
    )
    return postings


def build_lesson_context(settings: Settings, llm: object | None = None) -> LessonContext:
    """Assemble everything the teaching steps need.

    ``llm`` is optional. When it is absent or unreachable, every step still runs
    its deterministic path.
    """
    return LessonContext(
        settings=settings,
        llm=llm,
        resume=load_sample_resume(settings),
        jobs=load_cached_postings(settings),
    )


def _parse_iso(value: object) -> datetime | None:
    """Parse an ISO timestamp from cached data, tolerating a trailing Z."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
