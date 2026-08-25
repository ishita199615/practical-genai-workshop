"""Turn a board's structured posting into the agent's :class:`JobPosting`.

The other retrieval route hands a scraped page to a language model and asks it
to read out the title, location, work mode, and posting date. A board states all
four as fields, so this conversion is a rename, not an extraction — no model is
called, nothing is inferred, and the same board returns the same posting twice.

That also makes the freshness claim stronger. A search result usually carries no
publish date, so the agent can only say the *search* was filtered to a window. A
board that published a timestamp lets the agent say the posting itself was
verified inside that window, and this records which of the two happened.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from models.job import ExperienceLevel, FreshnessWindow, JobPosting
from tools.ats_boards import AtsPosting
from tools.employment_type import detect_employment_type
from tools.experience_level import detect_experience_level
from tools.firecrawl_search import (
    canonicalize_job_url,
    freshness_cutoff,
    registrable_domain,
)
from tools.job_normalizer import (
    clean_description,
    detect_closed,
    make_excerpt,
    make_job_id,
    strip_card_header,
)
from tools.places import parse_place

ATS_LABELS = {
    "workday": "Workday",
    "greenhouse": "Greenhouse",
    "lever": "Lever",
    "ashby": "Ashby",
}

# A date with no time of day. Good to the day, and no better.
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def to_job_posting(
    posting: AtsPosting,
    *,
    freshness_window: FreshnessWindow,
    retrieved_at: datetime,
    requested_experience_level: ExperienceLevel = "unknown",
    max_description_chars: int = 20000,
) -> JobPosting:
    """Convert one board posting, keeping every fact the board stated."""
    canonical = canonicalize_job_url(posting.url) if posting.url else posting.url
    description = clean_description(posting.description, max_description_chars)
    place = parse_place(posting.location)

    posted_at = posting.posted_at
    posting_date = posted_at.date() if posted_at else None
    age_hours = None
    if posted_at is not None:
        age_hours = max(
            0.0, (retrieved_at - posted_at).total_seconds() / 3600
        )

    # Only a board that published a time of day earns "exact timestamp". A bare
    # date ("2026-08-24") and an age phrase ("Posted Yesterday") are both good
    # to the day and no better, and must not claim otherwise.
    if posted_at is None:
        evidence = "unavailable"
    elif _DATE_ONLY_RE.match(posting.posted_text.strip()):
        evidence = "date_only"
    elif posting.posted_text.strip().lower().startswith("posted"):
        evidence = "date_only"
    else:
        evidence = "exact_timestamp"

    cutoff = freshness_cutoff(freshness_window, retrieved_at)
    if posted_at is None:
        freshness_status = "date_unavailable"
    elif posted_at < cutoff:
        freshness_status = "possibly_stale"
    else:
        freshness_status = "verified_recent"

    level = detect_experience_level(posting.title, description, None)
    engagement = detect_employment_type(
        posting.title, description, stated=posting.commitment or None
    )

    return JobPosting(
        job_id=make_job_id(canonical or posting.title),
        title=posting.title or "Untitled posting",
        company=posting.company or "Unknown company",
        location=posting.location or None,
        work_mode=_work_mode(posting, place),
        query_category="employer_boards",
        source_category="employer_boards",
        source_label=ATS_LABELS.get(posting.ats, posting.ats.title()),
        description=description,
        description_excerpt=make_excerpt(
            strip_card_header(
                description, title=posting.title, location=posting.location
            )
        ),
        source_url=canonical or posting.url,
        original_source_url=posting.url,
        apply_url=posting.url,
        source_domain=registrable_domain(posting.url) if posting.url else "",
        freshness_window=freshness_window,
        posted_at=posted_at,
        posting_date=posting_date,
        posting_age_hours=age_hours,
        freshness_evidence=evidence,
        retrieved_at=retrieved_at,
        is_closed=detect_closed(description),
        freshness_status=freshness_status,
        data_mode="live",
        requested_experience_level=requested_experience_level,
        experience_level=level.level,
        experience_level_evidence=level.evidence,
        employment_type=engagement.employment_type,
        employment_type_evidence=engagement.evidence,
    )


def _work_mode(posting: AtsPosting, place) -> str:
    """Use the mode the board stated; fall back to what the location implies."""
    if posting.work_mode in {"remote", "hybrid", "onsite"}:
        return posting.work_mode
    if place.is_remote:
        return "remote"
    return "unknown"


def to_job_postings(
    postings: list[AtsPosting],
    *,
    freshness_window: FreshnessWindow,
    retrieved_at: datetime | None = None,
    requested_experience_level: ExperienceLevel = "unknown",
    max_description_chars: int = 20000,
) -> list[JobPosting]:
    """Convert a whole board result set."""
    when = retrieved_at or datetime.now(timezone.utc)
    return [
        to_job_posting(
            posting,
            freshness_window=freshness_window,
            retrieved_at=when,
            requested_experience_level=requested_experience_level,
            max_description_chars=max_description_chars,
        )
        for posting in postings
    ]
