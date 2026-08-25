"""Deterministic filtering and deduplication of normalized postings.

No language model participates in these decisions. A posting is kept only when
its link, description, and posting evidence survive explicit Python rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from models.job import (
    EMPLOYMENT_TYPE_LABELS,
    EXPERIENCE_LEVEL_LABELS,
    ExperienceLevel,
    JobPosting,
)
from tools.employment_type import types_conflict
from tools.experience_level import levels_conflict
from tools.firecrawl_search import (
    US_STATE_NAMES,
    canonicalize_job_url,
    looks_like_generic_listing,
)
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

# A posting tied to no particular place can be worked from anywhere, including
# the requested city.
LOCATION_ANYWHERE = {
    "remote",
    "fully remote",
    "anywhere",
    "remote anywhere",
    "global",
    "worldwide",
}

# A posting scoped to the whole country still covers a city inside it.
NATIONWIDE_US = {
    "united states",
    "united states of america",
    "usa",
    "us",
    "u s",
    "nationwide",
    "remote us",
    "remote united states",
    "us remote",
    "anywhere in the us",
}


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


# Towns that share a commuting area with a big city. A job in Sugar Land is a
# Houston job to anyone driving to it; a job in Austin is not, though both are
# in Texas. Extend this for your own metro rather than widening to the state.
METRO_NEIGHBOURS: dict[str, tuple[str, ...]] = {
    "houston": (
        "sugar land",
        "the woodlands",
        "katy",
        "pearland",
        "pasadena",
        "spring",
        "cypress",
        "bellaire",
        "stafford",
        "missouri city",
        "league city",
        "friendswood",
        "humble",
        "baytown",
        "conroe",
        "greater houston",
    ),
}


def requested_location_terms(requested_location: str) -> list[str]:
    """Return the place names a posting may name to satisfy the request.

    ``"Houston, TX"`` yields Houston and its commuting towns. The state is
    deliberately *not* included: "TX" alone would make Austin, Dallas, and El
    Paso all count as Houston, which is not what someone searching a city means.
    """
    parts = [part.strip() for part in (requested_location or "").split(",")]
    if not parts or not parts[0]:
        return []
    city = normalize_location(parts[0])
    if not city:
        return []
    return [city, *METRO_NEIGHBOURS.get(city, ())]


def location_conflict(requested_location: str, job: JobPosting) -> bool:
    """True when a posting names a place that is clearly not the one requested.

    Absence of evidence is never treated as a conflict: a posting that states no
    location, or one tied to no particular place, is kept. "Remote" alone is not
    a pass — a role advertised as remote *from Hong Kong* still names a region
    the requested city is not in, so it is the stated place that decides.
    """
    terms = requested_location_terms(requested_location)
    if not terms:
        return False

    job_location = normalize_location(job.location)
    if not job_location or job_location in LOCATION_ANYWHERE:
        return False
    if job_location in NATIONWIDE_US and requested_region_is_us(requested_location):
        return False

    words = set(job_location.split())
    return not any(
        term in job_location if " " in term else term in words for term in terms
    )


def requested_region_is_us(requested_location: str) -> bool:
    """True when the requested location names a US state."""
    return any(
        US_STATE_NAMES.get(part.strip().upper())
        for part in (requested_location or "").split(",")
    )


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
    requested_location: str = "",
    requested_employment_type: str = "any",
) -> str | None:
    """Return why a posting must be rejected, or ``None`` when it is usable.

    ``requested_experience_level`` rejects a posting only when the posting
    itself states a level and that level differs from the one asked for. A
    posting that never states a level is kept: there is no evidence to drop it
    on, and dropping it would hide real openings. The default disables the rule.

    ``requested_location`` and ``requested_employment_type`` follow the same
    shape: only a posting that states something other than what was asked for is
    dropped. Every default disables its own rule.
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
    if types_conflict(requested_employment_type, job.employment_type):
        return (
            f"{EMPLOYMENT_TYPE_LABELS[job.employment_type]} posting removed: "
            f"you searched {EMPLOYMENT_TYPE_LABELS.get(requested_employment_type, requested_employment_type)}."
        )
    if location_conflict(requested_location, job):
        return (
            f"'{job.title}' in {job.location} removed: "
            f"you searched {requested_location.strip()}."
        )
    return None


def filter_and_deduplicate(
    jobs: list[JobPosting],
    *,
    min_description_chars: int = MIN_DESCRIPTION_CHARS,
    requested_experience_level: ExperienceLevel = "unknown",
    requested_location: str = "",
    requested_employment_type: str = "any",
) -> FilterOutcome:
    """Apply every rejection rule, then remove duplicates.

    Deduplication uses both the canonicalized URL and the normalized
    company + title + location key, keeping the first occurrence.

    ``requested_experience_level`` defaults to ``"unknown"`` and
    ``requested_location`` to ``""``, which filter no posting by seniority or
    place at all.
    """
    outcome = FilterOutcome()
    seen_urls: set[str] = set()
    seen_keys: set[str] = set()

    for job in jobs:
        reason = rejection_reason(
            job,
            min_description_chars=min_description_chars,
            requested_experience_level=requested_experience_level,
            requested_location=requested_location,
            requested_employment_type=requested_employment_type,
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
