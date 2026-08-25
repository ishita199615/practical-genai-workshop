"""Turn raw retrieved pages into validated :class:`JobPosting` records.

Gemini reads the page and fills in :class:`ExtractedJobFields`. Everything that
must be trustworthy — the job ID, the canonical URL, the detected source, the
posting-age arithmetic, the freshness evidence label, and the seniority read off
the posting — is computed here in Python.
"""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Any

from models.ats import ATS_DISCLAIMER  # noqa: F401  (re-exported for convenience)
from models.job import (
    ExperienceLevel,
    ExtractedJobFields,
    FreshnessEvidence,
    FreshnessWindow,
    JobPosting,
    RawJobResult,
)
from prompts import render_prompt
from tools.experience_level import detect_experience_level
from tools.firecrawl_search import (
    canonicalize_job_url,
    classify_source,
    company_from_url,
    freshness_cutoff,
    is_safe_public_job_url,
    looks_like_generic_listing,
    registrable_domain,
)

logger = logging.getLogger(__name__)

MIN_DESCRIPTION_CHARS = 400
EXCERPT_CHARS = 260

# Markdown noise that carries no job information.
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_WHITESPACE_RE = re.compile(r"[ \t]{2,}")

# Page chrome that some applicant-tracking systems emit above the posting.
_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"Sorry,?\s*Internet Explorer.*?supported web browsers here\.?",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:We use cookies|This site uses cookies).{0,400}?"
        r"(?:Accept(?: All)?(?: Cookies)?|Got it|Continue)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"^\s*Skip to (?:main )?content\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"Please enable JavaScript.{0,200}?(?:to continue|in your browser)\.?",
        re.IGNORECASE | re.DOTALL,
    ),
)

CLOSED_MARKERS: tuple[str, ...] = (
    "no longer accepting applications",
    "this position has been filled",
    "position has been filled",
    "this job is no longer available",
    "job posting has expired",
    "applications are closed",
    "no longer available",
    "posting closed",
)


def clean_description(markdown: str | None, max_chars: int = 20000) -> str:
    """Strip markdown noise and page chrome, then clamp to a safe length.

    Only well-known boilerplate is removed — browser-support notices, cookie
    banners, skip links. Job content is never edited or summarized.
    """
    if not markdown:
        return ""
    text = markdown
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0].rstrip() + "\n\n[Description truncated]"
    return text


def make_excerpt(description: str, limit: int = EXCERPT_CHARS) -> str:
    """Build a short, deterministic 2–3 line card excerpt.

    Computed in Python: spending an LLM call on a preview would be waste and
    would risk paraphrasing the source.
    """
    flat = " ".join(description.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(",.;:") + "…"


# --------------------------------------------------------------------------
# Posting time parsing
# --------------------------------------------------------------------------

_RELATIVE_RE = re.compile(
    r"(\d+)\s*\+?\s*(minute|min|hour|hr|day|week|month)s?\s*ago", re.IGNORECASE
)
_ISO_DATETIME_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?"
)
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_US_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_NAME_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_posted_at(
    text: str | None, retrieved_at: datetime
) -> tuple[datetime | None, date | None, FreshnessEvidence]:
    """Parse a posting time string into evidence the UI can state honestly.

    Returns ``(posted_at, posting_date, evidence)``. Minute/hour resolution
    counts as an exact timestamp; day resolution or a bare date is date-only.
    A date is never inferred when the page shows none.
    """
    if not text or not str(text).strip():
        return None, None, "unavailable"
    raw = str(text).strip()
    lowered = raw.lower()

    if lowered in {"just posted", "just now", "posted today", "today", "new"}:
        return retrieved_at, retrieved_at.date(), "date_only"
    if lowered in {"yesterday", "posted yesterday"}:
        stamp = retrieved_at - timedelta(days=1)
        return stamp, stamp.date(), "date_only"

    relative = _RELATIVE_RE.search(lowered)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        deltas = {
            "minute": timedelta(minutes=amount),
            "min": timedelta(minutes=amount),
            "hour": timedelta(hours=amount),
            "hr": timedelta(hours=amount),
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
            "month": timedelta(days=30 * amount),
        }
        stamp = retrieved_at - deltas[unit]
        evidence: FreshnessEvidence = (
            "exact_timestamp" if unit in {"minute", "min", "hour", "hr"} else "date_only"
        )
        return stamp, stamp.date(), evidence

    iso_dt = _ISO_DATETIME_RE.search(raw)
    if iso_dt:
        year, month, day, hour, minute = (int(iso_dt.group(i)) for i in range(1, 6))
        second = int(iso_dt.group(6) or 0)
        stamp = datetime(
            year, month, day, hour, minute, second, tzinfo=timezone.utc
        )
        return stamp, stamp.date(), "exact_timestamp"

    iso_date = _ISO_DATE_RE.search(raw)
    if iso_date:
        year, month, day = (int(iso_date.group(i)) for i in range(1, 4))
        return None, date(year, month, day), "date_only"

    month_name = _MONTH_NAME_RE.search(raw)
    if month_name:
        month = _MONTHS[month_name.group(1).lower()]
        return None, date(int(month_name.group(3)), month, int(month_name.group(2))), "date_only"

    us_date = _US_DATE_RE.search(raw)
    if us_date:
        month, day, year = (int(us_date.group(i)) for i in range(1, 4))
        return None, date(year, month, day), "date_only"

    return None, None, "unavailable"


def compute_freshness(
    posted_at: datetime | None,
    posting_date: date | None,
    evidence: FreshnessEvidence,
    window: FreshnessWindow,
    retrieved_at: datetime,
    time_filter_applied: bool = True,
) -> tuple[float | None, FreshnessEvidence, str]:
    """Decide posting age and freshness status from source evidence only.

    Returns ``(posting_age_hours, evidence, freshness_status)``. A search-time
    filter alone never produces ``verified_recent``, and when no time filter was
    actually sent, an undated posting is reported as simply unavailable rather
    than as search-filtered.
    """
    cutoff = freshness_cutoff(window, retrieved_at)

    if evidence == "exact_timestamp" and posted_at is not None:
        age_hours = (retrieved_at - posted_at).total_seconds() / 3600.0
        if posted_at >= cutoff:
            return round(age_hours, 2), "exact_timestamp", "verified_recent"
        return round(age_hours, 2), "exact_timestamp", "possibly_stale"

    if evidence == "date_only" and posting_date is not None:
        age_hours = None
        if posted_at is not None:
            age_hours = round((retrieved_at - posted_at).total_seconds() / 3600.0, 2)
        if posting_date >= cutoff.date():
            return age_hours, "date_only", "verified_recent"
        return age_hours, "date_only", "possibly_stale"

    if time_filter_applied:
        return None, "search_filter_only", "date_unavailable"
    return None, "unavailable", "date_unavailable"


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def make_job_id(canonical_url: str) -> str:
    """Return a stable job ID derived from the canonical URL."""
    digest = hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()
    return f"job_{digest[:12]}"


def detect_closed(text: str) -> bool:
    """True when the page text says the opening is no longer available."""
    lowered = text.lower()
    return any(marker in lowered for marker in CLOSED_MARKERS)


def split_title_and_company(page_title: str) -> tuple[str, str | None]:
    """Split a page title such as "Data Analyst @ Joko" into its two parts.

    Recognises the separators used by applicant-tracking systems. Returns the
    title and, when a separator is present, the company; never a guess.
    """
    text = " ".join((page_title or "").split())
    for separator in (" @ ", " | ", " — ", " – ", " - ", " at "):
        if separator in text:
            left, right = text.split(separator, 1)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    return text, None


def _fallback_extraction(raw: RawJobResult, description: str) -> ExtractedJobFields:
    """Build minimal fields without an LLM.

    Used for cached rehearsal data, offline runs, and whenever the provider is
    unavailable or rate-limited. It copies text and reads the URL; it never
    guesses requirements, and it never invents a posting date.
    """
    title, title_company = split_title_and_company(raw.title or "")
    company = ""
    metadata = raw.metadata or {}
    for key in ("company", "ogSiteName", "og:site_name", "siteName", "author"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            company = value.strip()
            break
    # Applicant-tracking URLs name the employer, which keeps a readable posting
    # from being discarded just because no model was available to read it.
    if not company:
        company = company_from_url(raw.final_url or raw.url) or ""
    if not company and title_company:
        company = title_company
    return ExtractedJobFields(
        title=title or None,
        company=company or None,
        description=description or None,
        is_closed=detect_closed(description),
        is_specific_opening=not looks_like_generic_listing(raw.final_url or raw.url),
    )


def extract_job_fields(
    raw: RawJobResult,
    llm: Any,
    *,
    max_description_chars: int = 20000,
) -> ExtractedJobFields | None:
    """Extract structured fields for one page.

    Cached records carry a previously extracted payload so the rehearsal path
    needs no API call. Live records go to the LLM; on failure the caller is
    told, and a snippet-only record is later rejected rather than scored.
    """
    cached = (raw.metadata or {}).get("cached_extraction")
    if isinstance(cached, dict):
        try:
            return ExtractedJobFields.model_validate(cached)
        except Exception:  # noqa: BLE001 - cache is data, not code
            logger.warning("Cached extraction failed validation; ignoring it.")

    source_content = clean_description(
        raw.markdown or raw.description, max_description_chars
    )
    if llm is not None and getattr(llm, "available", False) and source_content:
        prompt = render_prompt(
            "extract_job",
            SOURCE_URL=raw.final_url or raw.url,
            SOURCE_TITLE=raw.title or "",
            SOURCE_CONTENT=source_content,
        )
        extracted = llm.generate_structured(prompt, ExtractedJobFields, temperature=0.0)
        if extracted is not None:
            return extracted
        logger.warning("Structured extraction failed for %s", raw.final_url or raw.url)

    if not source_content:
        return None
    return _fallback_extraction(raw, source_content)


def normalize_job(
    raw: RawJobResult,
    llm: Any,
    *,
    max_description_chars: int = 20000,
    data_mode: str = "live",
    requested_experience_level: ExperienceLevel = "unknown",
) -> tuple[JobPosting | None, str | None]:
    """Convert one raw result into a validated posting.

    Returns ``(posting, warning)``. A ``None`` posting means the record could
    not be trusted; the warning explains why in user-facing language.

    ``requested_experience_level`` is recorded on the posting so the UI can say
    what was asked for, and is never used to decide what the posting *is*: the
    detected level comes only from the page's own title, stated years, and body.
    """
    final_url = raw.final_url or raw.url
    if not is_safe_public_job_url(final_url):
        return None, "Removed a result whose link was not a public web address."

    canonical = canonicalize_job_url(final_url)
    extracted = extract_job_fields(
        raw, llm, max_description_chars=max_description_chars
    )
    if extracted is None:
        return None, f"No readable job description was available at {registrable_domain(canonical)}."

    cleaned_markdown = clean_description(raw.markdown, max_description_chars)
    description = clean_description(extracted.description, max_description_chars)
    if len(description) < MIN_DESCRIPTION_CHARS and len(cleaned_markdown) > len(description):
        description = cleaned_markdown
    if not description:
        description = clean_description(raw.description, max_description_chars)

    title = (extracted.title or raw.title or "").strip()
    company = (extracted.company or "").strip()
    category, label = classify_source(canonical)

    posted_at, posting_date, evidence = parse_posted_at(
        extracted.posted_at_text, raw.retrieved_at
    )
    age_hours, evidence, status = compute_freshness(
        posted_at,
        posting_date,
        evidence,
        raw.freshness_window,
        raw.retrieved_at,
        raw.time_filter_applied,
    )

    apply_url = extracted.apply_url if is_safe_public_job_url(extracted.apply_url) else None
    is_closed = bool(extracted.is_closed) or detect_closed(description)

    # Read the seniority back off the page. The request is stored beside it but
    # never copied into it: a posting that states no level stays "unknown".
    level = detect_experience_level(
        title, description, extracted.minimum_experience_years
    )

    posting = JobPosting(
        job_id=make_job_id(canonical),
        title=title or "Untitled posting",
        company=company or "Unknown company",
        location=extracted.location,
        work_mode=extracted.work_mode,
        query_category=raw.query_category,
        source_category=category,
        source_label=label,
        description=description,
        description_excerpt=make_excerpt(description),
        required_skills=[skill.strip() for skill in extracted.required_skills if skill.strip()],
        preferred_skills=[skill.strip() for skill in extracted.preferred_skills if skill.strip()],
        minimum_experience_years=extracted.minimum_experience_years,
        education_requirement=extracted.education_requirement,
        responsibilities=[item.strip() for item in extracted.responsibilities if item.strip()],
        source_url=canonical,
        original_source_url=raw.url,
        apply_url=apply_url,
        source_domain=registrable_domain(canonical),
        freshness_window=raw.freshness_window,
        posted_at=posted_at,
        posting_date=posting_date,
        posting_age_hours=age_hours,
        freshness_evidence=evidence,
        retrieved_at=raw.retrieved_at,
        is_closed=is_closed,
        freshness_status=status,
        data_mode="cached" if data_mode == "cached" else "live",
        requested_experience_level=requested_experience_level,
        experience_level=level.level,
        experience_level_evidence=level.evidence,
    )
    if not extracted.is_specific_opening or looks_like_generic_listing(canonical):
        return None, f"Removed a general listing page from {posting.source_label}."
    return posting, None


def normalize_jobs(
    raw_results: list[RawJobResult],
    llm: Any,
    *,
    max_description_chars: int = 20000,
    data_mode: str = "live",
    limit: int = 8,
    max_workers: int = 8,
    requested_experience_level: ExperienceLevel = "unknown",
) -> tuple[list[JobPosting], list[str]]:
    """Normalize at most ``limit`` pages and collect user-facing warnings.

    Pages are independent, so extraction runs concurrently: eight sequential
    model calls would on their own exceed the workshop's sixty-second budget.
    Results keep the input order so ranking stays reproducible.

    ``requested_experience_level`` is recorded on every posting for display; it
    does not affect what level is detected on any of them.
    """
    batch = raw_results[:limit]
    if not batch:
        return [], []

    def normalize_one(raw: RawJobResult) -> tuple[JobPosting | None, str | None]:
        try:
            return normalize_job(
                raw,
                llm,
                max_description_chars=max_description_chars,
                data_mode=data_mode,
                requested_experience_level=requested_experience_level,
            )
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
            logger.warning(
                "Normalization failed for %s: %s",
                raw.final_url or raw.url,
                type(exc).__name__,
            )
            return None, "A retrieved page could not be read and was skipped."

    if len(batch) == 1 or max_workers <= 1:
        outcomes = [normalize_one(raw) for raw in batch]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
            outcomes = list(pool.map(normalize_one, batch))

    postings: list[JobPosting] = []
    warnings: list[str] = []
    for posting, warning in outcomes:
        if posting is not None:
            postings.append(posting)
        if warning:
            warnings.append(warning)
    return postings, warnings
