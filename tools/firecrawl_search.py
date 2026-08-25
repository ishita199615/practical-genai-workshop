"""Firecrawl Search API v2 adapter, query building, and URL rules.

Everything that decides *what* to ask the web for, and *whether an answer is
trustworthy*, lives here:

* query construction per job-source category and requested experience level,
* freshness window to Firecrawl ``tbs`` translation,
* public URL safety validation and canonicalization,
* detection of the actual source from the final URL,
* live retrieval with a timeout and a clearly labelled cache fallback.

The network call is isolated in :class:`FirecrawlSearchAdapter` so tests can
inject a fake client and the workshop never needs a live API key.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from models.job import (
    ExperienceLevel,
    FreshnessWindow,
    RawJobResult,
    SourceCategory,
)
from tools.experience_level import level_query_terms

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Domain rules
# --------------------------------------------------------------------------

SOURCE_DOMAIN_RULES: dict[str, tuple[str, str]] = {
    "linkedin.com": ("linkedin", "LinkedIn"),
    "indeed.com": ("indeed", "Indeed"),
    "google.com": ("google_jobs", "Google Jobs / Web"),
    "boards.greenhouse.io": ("company_careers", "Greenhouse"),
    "job-boards.greenhouse.io": ("company_careers", "Greenhouse"),
    "jobs.lever.co": ("company_careers", "Lever"),
    "jobs.ashbyhq.com": ("company_careers", "Ashby"),
    "jobs.smartrecruiters.com": ("company_careers", "SmartRecruiters"),
    "careers.workday.com": ("company_careers", "Workday Careers"),
    "myworkdayjobs.com": ("company_careers", "Workday Careers"),
}

COMPANY_CAREER_DOMAINS: list[str] = [
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.lever.co",
    "jobs.ashbyhq.com",
    "jobs.smartrecruiters.com",
    "careers.workday.com",
]

CATEGORY_DOMAIN_FILTERS: dict[str, list[str]] = {
    "all": [],
    "linkedin": ["linkedin.com"],
    "indeed": ["indeed.com"],
    "google_jobs": [],
    "company_careers": COMPANY_CAREER_DOMAINS,
}

CATEGORY_SITE_PREFIX: dict[str, str] = {
    "all": "",
    "linkedin": "site:linkedin.com/jobs/view",
    "indeed": "site:indeed.com/viewjob",
    "google_jobs": "",
    "company_careers": "",
}

# Query-string keys that only carry tracking noise and must not change identity.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gh_src",
        "gh_jid_src",
        "source",
        "src",
        "ref",
        "referrer",
        "trk",
        "trackingid",
        "refid",
        "originaltopref",
        "position",
        "pagenum",
        "eboriginal",
        "lipi",
    }
)

# Paths that are search or index pages rather than one specific opening.
GENERIC_PATH_MARKERS: tuple[str, ...] = (
    "/jobs/search",
    "/jobs/collections",
    "/job-search",
    "/search",
    "/browse",
    "/q-",
)

GENERIC_EXACT_PATHS: frozenset[str] = frozenset(
    {"", "/", "/jobs", "/careers", "/career", "/en/careers", "/openings", "/join-us"}
)

# Budget for the time-filtered first attempt, which is usually empty and slow.
FAIL_FAST_TIMEOUT_SECONDS = 12

FRESHNESS_HOURS: dict[str, float] = {
    "last_hour": 1.0,
    "last_24_hours": 24.0,
    "last_3_days": 72.0,
    "last_7_days": 168.0,
}


# --------------------------------------------------------------------------
# Query building
# --------------------------------------------------------------------------


def role_variants(role: str) -> list[str]:
    """Return the quoted role phrases used in the search query.

    Deterministic and conservative: the target role itself, plus one
    entry-level synonym when the role is an internship title.
    """
    cleaned = " ".join(role.strip().lower().split())
    if not cleaned:
        return []
    variants = [cleaned]
    for suffix in (" intern", " internship"):
        if cleaned.endswith(suffix):
            base = cleaned[: -len(suffix)].strip()
            if base:
                variants.append(f"entry level {base}")
            break
    return variants


def location_clause(location: str, work_mode: str) -> str:
    """Build the location half of the query from the location and work mode."""
    city = location.split(",")[0].strip()
    mode = (work_mode or "any").strip().lower()
    if mode == "remote":
        return "(remote)"
    if not city:
        return "(remote)" if mode != "onsite" else ""
    if mode == "onsite":
        return f"({city})"
    if mode == "hybrid":
        return f"({city} OR hybrid)"
    return f"({city} OR remote)"


def role_clause_variants(role: str, experience_level: ExperienceLevel) -> list[str]:
    """Return the quoted role phrases for a role at a requested seniority.

    ``"unknown"`` means the user did not filter by level, so the historical
    :func:`role_variants` behaviour is used unchanged. When a level *is*
    selected, the phrases come from :func:`level_query_terms`, which strips any
    seniority word already present in the free-text role so the two inputs can
    never contradict each other.
    """
    if experience_level == "unknown":
        return role_variants(role)
    return level_query_terms(role, experience_level)


def build_search_query(
    role: str,
    location: str,
    work_mode: str,
    query_category: SourceCategory,
    experience_level: ExperienceLevel = "unknown",
) -> str:
    """Build the Firecrawl query string for one category and seniority.

    The category only shapes the query; it is never treated as proof of where a
    result actually came from. The same holds for the experience level: asking
    for senior roles narrows what is *fetched* and is never evidence that a
    returned posting is senior.
    """
    variants = role_clause_variants(role, experience_level)
    role_clause = (
        "(" + " OR ".join(f'"{variant}"' for variant in variants) + ")"
        if variants
        else ""
    )
    parts = [
        CATEGORY_SITE_PREFIX.get(query_category, ""),
        role_clause,
        location_clause(location, work_mode),
    ]
    if query_category == "google_jobs":
        parts.append("jobs")
    return " ".join(part for part in parts if part).strip()


def domain_filter_for(query_category: SourceCategory) -> list[str]:
    """Return the ``includeDomains`` filter for one category."""
    return list(CATEGORY_DOMAIN_FILTERS.get(query_category, []))


def freshness_to_tbs(window: FreshnessWindow, now: datetime | None = None) -> str:
    """Translate a freshness window into a Firecrawl ``tbs`` value.

    ``last_3_days`` uses a custom date range generated at runtime in UTC.
    """
    if window == "last_hour":
        return "sbd:1,qdr:h"
    if window == "last_24_hours":
        return "sbd:1,qdr:d"
    if window == "last_7_days":
        return "sbd:1,qdr:w"
    if window == "last_3_days":
        current = now or datetime.now(timezone.utc)
        start = current - timedelta(days=3)
        return (
            "sbd:1,cdr:1,"
            f"cd_min:{start.strftime('%m/%d/%Y')},"
            f"cd_max:{current.strftime('%m/%d/%Y')}"
        )
    raise ValueError(f"Unknown freshness window: {window}")


def freshness_cutoff(window: FreshnessWindow, now: datetime) -> datetime:
    """Return the local verification cutoff for a freshness window."""
    hours = FRESHNESS_HOURS.get(window)
    if hours is None:
        raise ValueError(f"Unknown freshness window: {window}")
    return now - timedelta(hours=hours)


# --------------------------------------------------------------------------
# URL rules
# --------------------------------------------------------------------------


def is_safe_public_job_url(url: str | None) -> bool:
    """True when the URL is a public ``http``/``https`` address we may open."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and not parsed.username
        and not parsed.password
        and " " not in url.strip()
    )


def canonicalize_job_url(url: str) -> str:
    """Return the canonical direct opening URL.

    Lowercases the scheme and host, drops the fragment, removes tracking-only
    query parameters, and trims a trailing slash. Identity-bearing parameters
    such as ``currentJobId`` or ``jk`` are preserved.
    """
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key.lower() not in TRACKING_PARAMS
    ]
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", urlencode(query_pairs), ""))


def registrable_domain(url: str) -> str:
    """Return the host of a URL without a leading ``www.``."""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.split(":")[0]


def classify_source(url: str) -> tuple[SourceCategory, str]:
    """Detect the actual source category and label from the final URL.

    The selected query category is never used here: a result found through an
    ``all`` search is classified by where it truly lives.
    """
    host = registrable_domain(url)
    if not host:
        return "other", "Other"
    for domain, (category, label) in SOURCE_DOMAIN_RULES.items():
        if host == domain or host.endswith("." + domain):
            return category, label  # type: ignore[return-value]
    return "other", _company_label(host)


def _company_label(host: str) -> str:
    """Build a readable label for an unrecognised employer domain."""
    if not host:
        return "Other"
    parts = [part for part in host.split(".") if part not in {"com", "org", "net", "io"}]
    if not parts:
        return host
    name = parts[0] if parts[0] not in {"careers", "jobs", "apply"} else parts[-1]
    return name.replace("-", " ").title()


# Applicant-tracking URLs carry the employer in a known path position, which
# makes the company recoverable without a model.
_ATS_COMPANY_SLOT: dict[str, int] = {
    "boards.greenhouse.io": 0,
    "job-boards.greenhouse.io": 0,
    "jobs.lever.co": 0,
    "jobs.ashbyhq.com": 0,
    "jobs.smartrecruiters.com": 0,
}

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def company_from_url(url: str) -> str | None:
    """Recover the employer name from an applicant-tracking URL.

    ``https://jobs.lever.co/portcast/<id>`` yields "Portcast". Used when
    structured extraction is unavailable — for instance under a provider quota
    limit — so a readable posting is not thrown away for want of a company.
    Returns ``None`` when the URL shape is unknown, never a guess.
    """
    host = registrable_domain(url)
    slot = None
    for domain, index in _ATS_COMPANY_SLOT.items():
        if host == domain or host.endswith("." + domain):
            slot = index
            break
    if slot is None:
        if host.endswith("myworkdayjobs.com"):
            subdomain = host.split(".")[0]
            return _humanize_company(subdomain) if subdomain else None
        return None

    segments = [seg for seg in urlparse(url).path.split("/") if seg]
    if len(segments) <= slot:
        return None
    return _humanize_company(segments[slot])


def _humanize_company(slug: str) -> str | None:
    """Turn a URL slug into a readable company name."""
    cleaned = slug.replace("_", "-").strip("-")
    if not cleaned or cleaned.isdigit() or len(cleaned) < 2:
        return None
    spaced = _CAMEL_BOUNDARY_RE.sub(" ", cleaned).replace("-", " ")
    words = [word for word in spaced.split() if word]
    if not words:
        return None
    return " ".join(word if word.isupper() else word.capitalize() for word in words)


def looks_like_generic_listing(url: str) -> bool:
    """True when the URL is a search or index page rather than one opening."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").lower()
    if path in GENERIC_EXACT_PATHS:
        return True
    if any(marker in path for marker in GENERIC_PATH_MARKERS):
        return True
    return False


# --------------------------------------------------------------------------
# Firecrawl adapter
# --------------------------------------------------------------------------


class SearchClient(Protocol):
    """Minimal protocol the adapter needs from a Firecrawl client."""

    def search(self, query: str, **kwargs: Any) -> Any:  # pragma: no cover
        ...


@dataclass
class SearchRequest:
    """The fully resolved request the adapter will send."""

    query: str
    query_category: SourceCategory
    freshness_window: FreshnessWindow
    include_domains: list[str]
    tbs: str
    location: str
    limit: int
    timeout_seconds: int
    experience_level: ExperienceLevel = "unknown"

    def as_payload(self) -> dict[str, Any]:
        """Return a log-safe dictionary of the request (no secrets).

        ``queryCategory``, ``freshnessWindow``, and ``experienceLevel`` are
        application metadata that describe how the query was shaped. They are
        recorded here for the activity log and never sent to Firecrawl.
        """
        return {
            "query": self.query,
            "queryCategory": self.query_category,
            "freshnessWindow": self.freshness_window,
            "experienceLevel": self.experience_level,
            "limit": self.limit,
            "sources": ["web"],
            "includeDomains": self.include_domains,
            "tbs": self.tbs,
            "location": self.location,
            "timeout": self.timeout_seconds * 1000,
            "ignoreInvalidURLs": True,
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
        }


class FirecrawlError(RuntimeError):
    """Raised when live retrieval fails and the caller should fall back."""


def build_search_request(
    *,
    role: str,
    location: str,
    work_mode: str,
    query_category: SourceCategory,
    freshness_window: FreshnessWindow,
    experience_level: ExperienceLevel = "unknown",
    limit: int = 8,
    timeout_seconds: int = 10,
    now: datetime | None = None,
) -> SearchRequest:
    """Combine category, freshness, and seniority into one search request.

    The three controls are independent: the category decides where to look, the
    freshness window how recent, and the experience level which seniority to
    fetch. ``"unknown"`` leaves the level unfiltered.
    """
    current = now or datetime.now(timezone.utc)
    return SearchRequest(
        query=build_search_query(
            role, location, work_mode, query_category, experience_level
        ),
        query_category=query_category,
        freshness_window=freshness_window,
        experience_level=experience_level,
        include_domains=domain_filter_for(query_category),
        tbs=freshness_to_tbs(freshness_window, current),
        location=_firecrawl_location(location),
        limit=max(1, min(limit, 8)),
        timeout_seconds=timeout_seconds,
    )


def _firecrawl_location(location: str) -> str:
    """Format the user's location for the Firecrawl ``location`` field."""
    text = location.strip()
    if not text:
        return "United States"
    if "," in text:
        city, region = (part.strip() for part in text.split(",", 1))
        region_full = US_STATE_NAMES.get(region.upper(), region)
        return f"{city},{region_full},United States"
    return f"{text},United States"


US_STATE_NAMES: dict[str, str] = {
    "TX": "Texas",
    "CA": "California",
    "NY": "New York",
    "FL": "Florida",
    "IL": "Illinois",
    "WA": "Washington",
    "MA": "Massachusetts",
    "GA": "Georgia",
    "CO": "Colorado",
    "AZ": "Arizona",
}


class FirecrawlSearchAdapter:
    """Thin adapter over Firecrawl Search API v2.

    A client factory is injected so tests (and offline rehearsals) can supply a
    fake client. A self-hosted base URL can be added later without touching the
    graph.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.firecrawl.dev",
        client_factory: Callable[[str, str], SearchClient] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client_factory = client_factory or _default_client_factory
        self._client: SearchClient | None = None

    @property
    def available(self) -> bool:
        """True when the adapter has what it needs to attempt a live call."""
        return bool(self._api_key) or self._client_factory is not _default_client_factory

    def _get_client(self) -> SearchClient:
        if self._client is None:
            self._client = self._client_factory(self._api_key, self._base_url)
        return self._client

    def search(self, request: SearchRequest) -> list[dict[str, Any]]:
        """Run one live search and return raw result dictionaries.

        Raises :class:`FirecrawlError` on any failure so the caller can decide
        whether to fall back to the cache.
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "sources": ["web"],
            "limit": request.limit,
            "location": request.location,
            "ignore_invalid_urls": True,
            "timeout": request.timeout_seconds * 1000,
            "scrape_options": _scrape_options(),
        }
        # An empty tbs means the caller deliberately dropped the time filter.
        if request.tbs:
            kwargs["tbs"] = request.tbs
        if request.include_domains:
            kwargs["include_domains"] = request.include_domains
        try:
            response = client.search(request.query, **kwargs)
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            logger.warning("Firecrawl search failed: %s", type(exc).__name__)
            raise FirecrawlError(str(exc)) from exc
        return _extract_web_results(response)


@dataclass
class SearchOutcome:
    """What one retrieval attempt produced, and which hints survived it."""

    results: list[dict[str, Any]]
    notes: list[str]
    time_filter_applied: bool = True


def search_with_domain_retry(
    adapter: FirecrawlSearchAdapter, request: SearchRequest
) -> SearchOutcome:
    """Search, relaxing server-side hints in order until results come back.

    Measured against the live API, Firecrawl's ``tbs`` time filter is what
    empties a result set: the same domain-restricted query returns eight
    results in about a second with no ``tbs``, and usually nothing with one.

    The ladder therefore is:

    1. the request as specified (domains + ``tbs``),
    2. the same query without ``tbs``,
    3. the same query without the domain filter.

    Dropping ``tbs`` costs nothing real. It is only a search-time hint, and it
    is never what makes a result trustworthy — the caller still verifies each
    posting's age against its own source evidence and rejects anything outside
    the requested window. Dropping the domain filter is likewise safe because
    the caller keeps only results whose *actual* source matches the selected
    category. Neither step puts something on screen that the user did not ask
    for; each is reported in the activity log.
    """
    # The time-filtered attempt is the one that usually comes back empty, and it
    # is slow when it does (16-23s measured). Cap it so a likely-empty attempt
    # cannot eat the demo's time budget; a genuinely fast success still lands
    # well inside this window.
    first = request
    if request.tbs:
        first = replace(
            request,
            timeout_seconds=min(request.timeout_seconds, FAIL_FAST_TIMEOUT_SECONDS),
        )

    results = adapter.search(first)
    if results:
        return SearchOutcome(results, [], time_filter_applied=bool(request.tbs))

    if request.tbs:
        results = adapter.search(replace(request, tbs=""))
        if results:
            return SearchOutcome(
                results,
                [
                    "The time-filtered search returned nothing, so the same query "
                    "ran without Firecrawl's time filter. Posting age is verified "
                    "locally against each source's own timestamp instead."
                ],
                time_filter_applied=False,
            )

    if request.include_domains:
        results = adapter.search(replace(request, include_domains=[], tbs=""))
        if results:
            return SearchOutcome(
                results,
                [
                    "The domain-restricted search returned nothing, so the same "
                    "query ran without the server-side domain filter. Results are "
                    "still filtered to the selected category by their actual source."
                ],
                time_filter_applied=False,
            )

    return SearchOutcome([], [], time_filter_applied=bool(request.tbs))


def filter_to_category(
    results: list[RawJobResult], query_category: SourceCategory
) -> tuple[list[RawJobResult], int]:
    """Keep only results whose detected source matches the selected category.

    ``all`` and ``google_jobs`` are broad by definition and pass through
    unchanged. Returns the kept results and how many were dropped.
    """
    if query_category in {"all", "google_jobs"}:
        return results, 0
    kept = [
        result
        for result in results
        if result.detected_source_category == query_category
    ]
    return kept, len(results) - len(kept)


def _default_client_factory(api_key: str, base_url: str) -> SearchClient:
    """Build the real Firecrawl client. Imported lazily to keep tests light."""
    from firecrawl import Firecrawl  # noqa: PLC0415 - optional dependency

    if not api_key:
        raise FirecrawlError("FIRECRAWL_API_KEY is not configured.")
    return Firecrawl(api_key=api_key, api_url=base_url)


def _scrape_options() -> Any:
    """Return Firecrawl scrape options, or a plain dict if the SDK is absent."""
    try:
        from firecrawl.v2.types import ScrapeOptions  # noqa: PLC0415

        return ScrapeOptions(formats=["markdown"], only_main_content=True)
    except Exception:  # noqa: BLE001 - fall back to a dict payload
        return {"formats": ["markdown"], "onlyMainContent": True}


def _extract_web_results(response: Any) -> list[dict[str, Any]]:
    """Normalize the SDK response into plain dictionaries.

    Handles the documented shapes: a ``SearchData`` object, a dict payload, and
    a bare list of results.
    """
    if response is None:
        return []
    web = None
    if isinstance(response, dict):
        web = response.get("web") or response.get("data", {}).get("web")
    else:
        web = getattr(response, "web", None)
    if web is None and isinstance(response, list):
        web = response
    if not web:
        return []

    results: list[dict[str, Any]] = []
    for item in web:
        if isinstance(item, dict):
            results.append(item)
            continue
        dumped = getattr(item, "model_dump", None)
        results.append(dumped(exclude_none=False) if dumped else vars(item))
    return results


def raw_results_to_models(
    results: Iterable[dict[str, Any]],
    *,
    query_category: SourceCategory,
    freshness_window: FreshnessWindow,
    retrieved_at: datetime,
    time_filter_applied: bool = True,
) -> tuple[list[RawJobResult], list[str]]:
    """Convert raw dictionaries into validated :class:`RawJobResult` models.

    Unsafe or missing URLs are dropped here with a warning, before any content
    is shown or scored. ``time_filter_applied`` records whether the search that
    produced these results actually carried a time filter, so an undated
    posting is never labelled "search-filtered" when no filter was used.
    """
    models: list[RawJobResult] = []
    warnings: list[str] = []
    for item in results:
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        original_url = (
            item.get("url")
            or metadata.get("url")
            or metadata.get("sourceURL")
            or metadata.get("source_url")
            or ""
        )
        final_url = (
            item.get("final_url")
            or metadata.get("sourceURL")
            or metadata.get("source_url")
            or metadata.get("ogUrl")
            or original_url
        )
        if not is_safe_public_job_url(final_url):
            warnings.append("Dropped a result with a missing or unsafe URL.")
            continue
        canonical = canonicalize_job_url(final_url)
        category, label = classify_source(canonical)
        models.append(
            RawJobResult(
                title=item.get("title") or metadata.get("title"),
                description=item.get("description") or metadata.get("description"),
                url=original_url or canonical,
                final_url=canonical,
                markdown=item.get("markdown"),
                metadata=metadata,
                query_category=query_category,
                freshness_window=freshness_window,
                detected_source_category=category,
                detected_source_label=label,
                retrieved_at=retrieved_at,
                time_filter_applied=time_filter_applied,
            )
        )
    return models, warnings


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def load_cache(path: str | Path) -> dict[str, Any]:
    """Load the clearly-labelled cached demonstration payload."""
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    with cache_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cached_raw_results(
    payload: dict[str, Any],
    *,
    query_category: SourceCategory,
    freshness_window: FreshnessWindow,
) -> list[dict[str, Any]]:
    """Return cached entries, preferring ones matching the selected category.

    The selected category is honoured when the cache can satisfy it. When it
    cannot, every entry is returned and the caller reports that the cache is
    broader than the request — the category is never silently rewritten on the
    individual records.
    """
    jobs = payload.get("jobs", [])
    if query_category in {"all", "google_jobs"}:
        return list(jobs)
    matching = [
        job for job in jobs if job.get("detected_source_category") == query_category
    ]
    return matching or list(jobs)
