"""Deterministic filtering and deduplication of normalized postings.

No language model participates in these decisions. A posting is kept only when
its link, description, and posting evidence survive explicit Python rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.job import EXPERIENCE_LEVEL_LABELS, ExperienceLevel, JobPosting
from tools.experience_level import levels_conflict
from tools.firecrawl_search import canonicalize_job_url, looks_like_generic_listing
from tools.job_normalizer import MIN_DESCRIPTION_CHARS

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_COMPANY_SUFFIXES = (
    " inc",
    " inc.",
    " llc",
    " l.l.c.",
    " ltd",
    " limited",
    " corp",
    " corporation",
    " co",
    " company",
    " plc",
    " gmbh",
)

PLACEHOLDER_TITLES = {"", "untitled posting"}
PLACEHOLDER_COMPANIES = {"", "unknown company"}


@dataclass
class FilterOutcome:
    """The result of filtering, with an auditable reason per removal."""

    kept: list[JobPosting] = field(default_factory=list)
    removed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def removed_count(self) -> int:
        """How many postings were rejected."""
        return len(self.removed)

    def reasons(self) -> list[str]:
        """Human-readable removal reasons for the activity log."""
        return [reason for _, reason in self.removed]


def normalize_company(name: str) -> str:
    """Normalize a company name for duplicate detection."""
    lowered = (name or "").strip().lower()
    for suffix in _COMPANY_SUFFIXES:
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)]
            break
    return _NON_ALNUM_RE.sub(" ", lowered).strip()


def normalize_title(title: str) -> str:
    """Normalize a job title for duplicate detection."""
    return _NON_ALNUM_RE.sub(" ", (title or "").strip().lower()).strip()


def normalize_location(location: str | None) -> str:
    """Normalize a location for duplicate detection."""
    return _NON_ALNUM_RE.sub(" ", (location or "").strip().lower()).strip()


def dedup_key(job: JobPosting) -> str:
    """Return the normalized company + title + location duplicate key."""
    return "|".join(
        (
            normalize_company(job.company),
            normalize_title(job.title),
            normalize_location(job.location),
        )
    )


def rejection_reason(
    job: JobPosting,
    *,
    min_description_chars: int,
    requested_experience_level: ExperienceLevel = "unknown",
) -> str | None:
    """Return why a posting must be rejected, or ``None`` when it is usable.

    ``requested_experience_level`` rejects a posting only when the posting
    itself states a level and that level differs from the one asked for. A
    posting that never states a level is kept: there is no evidence to drop it
    on, and dropping it would hide real openings. The default disables the rule.
    """
    if job.is_closed:
        return f"{job.title} at {job.company} is closed or no longer accepting applications."
    if normalize_title(job.title) in PLACEHOLDER_TITLES:
        return "A posting was removed because no job title could be read from the page."
    if normalize_company(job.company) in {normalize_company(c) for c in PLACEHOLDER_COMPANIES}:
        return f"'{job.title}' was removed because no company could be confirmed on the page."
    if looks_like_generic_listing(job.source_url):
        return f"A general listing page from {job.source_label} was removed."
    if len(job.description.strip()) < min_description_chars:
        return (
            f"'{job.title}' was removed because only a search snippet was available, "
            "not a full public job description."
        )
    if job.freshness_status == "possibly_stale":
        return (
            f"'{job.title}' was removed because its posting date is older than the "
            "requested freshness window."
        )
    if levels_conflict(requested_experience_level, job.experience_level):
        return (
            f"{EXPERIENCE_LEVEL_LABELS[job.experience_level]} posting removed: "
            f"you searched {EXPERIENCE_LEVEL_LABELS[requested_experience_level]}."
        )
    return None


def filter_and_deduplicate(
    jobs: list[JobPosting],
    *,
    min_description_chars: int = MIN_DESCRIPTION_CHARS,
    requested_experience_level: ExperienceLevel = "unknown",
) -> FilterOutcome:
    """Apply every rejection rule, then remove duplicates.

    Deduplication uses both the canonicalized URL and the normalized
    company + title + location key, keeping the first occurrence.

    ``requested_experience_level`` defaults to ``"unknown"``, which filters no
    posting by seniority at all.
    """
    outcome = FilterOutcome()
    seen_urls: set[str] = set()
    seen_keys: set[str] = set()

    for job in jobs:
        reason = rejection_reason(
            job,
            min_description_chars=min_description_chars,
            requested_experience_level=requested_experience_level,
        )
        if reason:
            outcome.removed.append((job.job_id, reason))
            continue

        canonical = canonicalize_job_url(job.source_url)
        key = dedup_key(job)
        if canonical in seen_urls or key in seen_keys:
            outcome.removed.append(
                (job.job_id, f"Duplicate posting removed: {job.title} at {job.company}.")
            )
            continue

        seen_urls.add(canonical)
        seen_keys.add(key)
        outcome.kept.append(job)

    return outcome
