"""Read postings straight from applicant-tracking systems' own public APIs.

Searching the open web for jobs fights the whole internet: LinkedIn, Indeed, and
Glassdoor serve a scraper a login wall, and what survives is aggregator pages
whose location and posting date have to be guessed at from prose.

Every major ATS already publishes the same data as JSON, because employers want
their boards syndicated. Those endpoints need no key, block nobody, and return
the two fields the search path can only estimate:

* **where the job is** — a real location field, so Austin is not mistaken for
  Houston, and
* **when it was posted** — a real publish date, so "recent" is evidence rather
  than a search filter that was merely *asked* for.

Because the fields arrive structured, nothing here needs a language model. The
results are deterministic and cost nothing to fetch.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from tools.places import Place, parse_places

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 20
BOARD_CONCURRENCY = 8
USER_AGENT = "Mozilla/5.0 (compatible; CougarCareerAgent/1.0)"

# "Posted Today", "Posted 8 Days Ago" — Workday reports age, not a timestamp.
_WORKDAY_AGE_RE = re.compile(r"(\d+)\+?\s*days?\s*ago", re.IGNORECASE)


@dataclass(frozen=True)
class Employer:
    """One employer's board, and how to read it."""

    name: str
    ats: str
    slug: str
    site: str = ""
    host: str = "wd5"


@dataclass
class AtsPosting:
    """One posting, with the fields the ATS stated rather than ones inferred."""

    title: str
    company: str
    location: str
    url: str
    ats: str
    description: str = ""
    posted_at: datetime | None = None
    posted_text: str = ""
    work_mode: str = "unknown"
    # What the board called the engagement, verbatim: "Full-time: Remote",
    # "Contract", "Intern". Empty when the board did not say.
    commitment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> float | None:
        """Days since publication, or ``None`` when the board stated no date."""
        if self.posted_at is None:
            return None
        delta = datetime.now(timezone.utc) - self.posted_at
        return max(0.0, delta.total_seconds() / 86400)

    @property
    def places(self) -> list[Place]:
        """Every place the location field named — boards often list several."""
        return parse_places(self.location)

    @property
    def place(self) -> Place:
        """The first place named, used when one label is needed."""
        return self.places[0]

    @property
    def country(self) -> str:
        """The country stated, or ``""`` when the board named none."""
        return self.place.country

    @property
    def city_label(self) -> str:
        """The city heading this posting groups under."""
        return self.place.city_label


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _request(url: str, payload: dict[str, Any] | None = None) -> Any:
    """GET, or POST when a payload is given. Returns parsed JSON or ``None``."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
        logger.info("ATS board unreachable (%s): %s", type(exc).__name__, url)
        return None


# --------------------------------------------------------------------------
# Per-ATS readers
# --------------------------------------------------------------------------


def read_greenhouse(employer: Employer) -> list[AtsPosting]:
    """Read a Greenhouse board. Location and first_published are stated."""
    payload = _request(
        f"https://boards-api.greenhouse.io/v1/boards/{employer.slug}/jobs?content=true"
    )
    if not isinstance(payload, dict):
        return []
    postings = []
    for job in payload.get("jobs") or []:
        published = _parse_iso(job.get("first_published") or job.get("updated_at"))
        postings.append(
            AtsPosting(
                title=job.get("title") or "",
                company=job.get("company_name") or employer.name,
                location=((job.get("location") or {}).get("name") or "").strip(),
                url=job.get("absolute_url") or "",
                ats="greenhouse",
                description=_strip_html(job.get("content") or ""),
                posted_at=published,
                posted_text=job.get("first_published") or "",
            )
        )
    return postings


def read_lever(employer: Employer) -> list[AtsPosting]:
    """Read a Lever board. Lever states workplaceType and every location."""
    payload = _request(f"https://api.lever.co/v0/postings/{employer.slug}?mode=json")
    if not isinstance(payload, list):
        return []
    postings = []
    for job in payload:
        categories = job.get("categories") or {}
        locations = [categories.get("location") or ""]
        locations += list(categories.get("allLocations") or [])
        created = job.get("createdAt")
        postings.append(
            AtsPosting(
                title=job.get("text") or "",
                company=employer.name,
                location=" / ".join(sorted({loc for loc in locations if loc})),
                url=job.get("hostedUrl") or job.get("applyUrl") or "",
                ats="lever",
                description=job.get("descriptionPlain") or job.get("additionalPlain") or "",
                posted_at=_parse_epoch_ms(created),
                posted_text=str(created or ""),
                work_mode=_lever_work_mode(categories.get("commitment") or "",
                                           job.get("workplaceType") or ""),
                commitment=categories.get("commitment") or "",
            )
        )
    return postings


def read_ashby(employer: Employer) -> list[AtsPosting]:
    """Read an Ashby board."""
    payload = _request(
        "https://api.ashbyhq.com/posting-api/job-board/"
        f"{employer.slug}?includeCompensation=false"
    )
    if not isinstance(payload, dict):
        return []
    postings = []
    for job in payload.get("jobs") or []:
        postings.append(
            AtsPosting(
                title=job.get("title") or "",
                company=employer.name,
                location=(job.get("location") or "").strip(),
                url=job.get("jobUrl") or "",
                ats="ashby",
                description=job.get("descriptionPlain") or "",
                posted_at=_parse_iso(job.get("publishedAt")),
                posted_text=job.get("publishedAt") or "",
                work_mode="remote" if job.get("isRemote") else "unknown",
                commitment=job.get("employmentType") or "",
            )
        )
    return postings


def read_workday(employer: Employer, search_text: str = "") -> list[AtsPosting]:
    """Read a Workday site — the ATS most large employers run.

    The list endpoint returns only a requisition number in place of a
    description, and states age ("Posted Yesterday") rather than a date. Each
    posting's own page carries the full description and a real ``startDate``, so
    those are fetched too: without them a posting cannot be scored against a
    resume, and would be discarded for looking like a search snippet.
    """
    base = f"https://{employer.slug}.{employer.host}.myworkdayjobs.com"
    payload = _request(
        f"{base}/wday/cxs/{employer.slug}/{employer.site}/jobs",
        {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": search_text},
    )
    if not isinstance(payload, dict):
        return []

    listed = payload.get("jobPostings") or []
    paths = [job.get("externalPath") or "" for job in listed]
    with ThreadPoolExecutor(max_workers=max(1, min(BOARD_CONCURRENCY, len(paths) or 1))) as pool:
        details = list(pool.map(lambda p: _workday_detail(employer, base, p), paths))

    postings = []
    for job, detail in zip(listed, details):
        listed_text = job.get("postedOn") or ""
        path = job.get("externalPath") or ""
        description = _strip_html(detail.get("jobDescription") or "")
        # The posting's own page states a date; the list only states an age.
        start_date = detail.get("startDate") or ""
        posted_at = _parse_iso(start_date) or _parse_workday_age(listed_text)
        postings.append(
            AtsPosting(
                title=job.get("title") or detail.get("title") or "",
                company=employer.name,
                location=(
                    job.get("locationsText") or detail.get("location") or ""
                ).strip(),
                url=f"{base}/{employer.site}{path}" if path else base,
                ats="workday",
                description=description,
                posted_at=posted_at,
                posted_text=start_date or listed_text,
                commitment=detail.get("timeType") or "",
            )
        )
    return postings


def _workday_detail(employer: Employer, base: str, path: str) -> dict[str, Any]:
    """Fetch one Workday posting's page for its description and date."""
    if not path:
        return {}
    payload = _request(f"{base}/wday/cxs/{employer.slug}/{employer.site}{path}")
    if not isinstance(payload, dict):
        return {}
    info = payload.get("jobPostingInfo")
    return info if isinstance(info, dict) else {}


READERS: dict[str, Callable[..., list[AtsPosting]]] = {
    "greenhouse": read_greenhouse,
    "lever": read_lever,
    "ashby": read_ashby,
    "workday": read_workday,
}


# --------------------------------------------------------------------------
# Date handling
# --------------------------------------------------------------------------


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_epoch_ms(value: Any) -> datetime | None:
    """Parse Lever's millisecond epoch."""
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _parse_workday_age(text: str) -> datetime | None:
    """Turn "Posted Yesterday" / "Posted 8 Days Ago" into a date.

    Workday states age, not a timestamp, so this is an approximation and the
    original phrase is preserved on the posting for display alongside it.
    """
    lowered = (text or "").lower()
    now = datetime.now(timezone.utc)
    if "today" in lowered or "just posted" in lowered:
        return now
    if "yesterday" in lowered:
        return now - timedelta(days=1)
    match = _WORKDAY_AGE_RE.search(lowered)
    if match:
        return now - timedelta(days=int(match.group(1)))
    return None


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&nbsp;": " ", "&rsquo;": "’", "&ndash;": "–",
}


def _strip_html(text: str) -> str:
    """Greenhouse returns escaped HTML; reduce it to readable plain text."""
    cleaned = urllib.parse.unquote(text or "")
    for entity, char in _HTML_ENTITIES.items():
        cleaned = cleaned.replace(entity, char)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _lever_work_mode(commitment: str, workplace_type: str) -> str:
    """Read Lever's stated work mode; never guess when it says nothing."""
    blob = f"{commitment} {workplace_type}".lower()
    if "remote" in blob:
        return "remote"
    if "hybrid" in blob:
        return "hybrid"
    if "on-site" in blob or "onsite" in blob:
        return "onsite"
    return "unknown"


# --------------------------------------------------------------------------
# Registry and search
# --------------------------------------------------------------------------


def load_employers(path: str | Path) -> list[Employer]:
    """Load the employer registry, skipping malformed rows rather than failing."""
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Employer registry unreadable: %s", exc)
        return []
    employers = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("ats") not in READERS:
            continue
        if not row.get("slug") or not row.get("name"):
            continue
        employers.append(
            Employer(
                name=str(row["name"]),
                ats=str(row["ats"]),
                slug=str(row["slug"]),
                site=str(row.get("site") or ""),
                host=str(row.get("host") or "wd5"),
            )
        )
    return employers


def fetch_board(employer: Employer, search_text: str = "") -> list[AtsPosting]:
    """Read one employer's board, returning [] rather than raising."""
    reader = READERS.get(employer.ats)
    if reader is None:
        return []
    try:
        if employer.ats == "workday":
            return reader(employer, search_text)
        return reader(employer)
    except Exception:  # noqa: BLE001 - one bad board must not stop the rest
        logger.exception("Reader failed for %s", employer.name)
        return []


def fetch_all(
    employers: Iterable[Employer],
    *,
    search_text: str = "",
    max_workers: int = BOARD_CONCURRENCY,
) -> list[AtsPosting]:
    """Read every board concurrently. Boards are independent, so one 404 is not fatal."""
    employers = list(employers)
    if not employers:
        return []
    postings: list[AtsPosting] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(employers)))) as pool:
        for result in pool.map(lambda e: fetch_board(e, search_text), employers):
            postings.extend(result)
    return postings


def matches_location(posting: AtsPosting, terms: Iterable[str]) -> bool:
    """True when the posting's stated location contains one of ``terms``.

    The ATS stated this location, so unlike the search path there is nothing to
    infer — a posting whose location is empty is not claimed to match.
    """
    haystack = posting.location.lower()
    if not haystack:
        return False
    return any(term and term.lower() in haystack for term in terms)


def matches_role(posting: AtsPosting, pattern: re.Pattern[str]) -> bool:
    """True when the posting's title matches the requested role."""
    return bool(pattern.search(posting.title or ""))


def matches_country(posting: AtsPosting, countries: Iterable[str]) -> bool:
    """True when any place the posting names sits in one of ``countries``.

    A posting whose board stated no country cannot be confirmed to be in one, so
    it does not match — the caller sees it under "Country not stated" instead of
    being told it is somewhere it might not be.
    """
    wanted = {country.strip().lower() for country in countries if country.strip()}
    if not wanted:
        return True
    return any(place.country.lower() in wanted for place in posting.places if place.country)


def group_by_city(
    postings: Iterable[AtsPosting], *, within_country: str | None = None
) -> dict[str, list[AtsPosting]]:
    """Group postings under a city heading, busiest city first.

    A posting listing several cities appears under each of them: it is genuinely
    open in all of them, and hiding it under only the first would lose that.

    ``within_country`` restricts the cities considered to those in one country,
    so a posting open in both Dublin and London is not listed under London when
    grouped beneath Ireland.
    """
    # One board writes "New York", another "New York, NY". They are the same
    # city, so they are keyed together and shown under the fuller of the two.
    grouped: dict[tuple[str, str], list[AtsPosting]] = {}
    labels: dict[tuple[str, str], str] = {}
    for posting in postings:
        places = posting.places
        if within_country is not None:
            places = [
                place for place in places if place.country_label == within_country
            ] or places
        seen_keys: set[tuple[str, str]] = set()
        for place in places:
            key = (place.city.strip().lower(), place.country_label)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            grouped.setdefault(key, []).append(posting)
            best = labels.get(key, "")
            if len(place.city_label) > len(best):
                labels[key] = place.city_label
    named = {labels[key]: members for key, members in grouped.items()}
    return _sorted_groups(named)


def group_by_country(
    postings: Iterable[AtsPosting],
) -> dict[str, dict[str, list[AtsPosting]]]:
    """Group postings by country, then by city within that country."""
    by_country: dict[str, list[AtsPosting]] = {}
    for posting in postings:
        for label in dict.fromkeys(place.country_label for place in posting.places):
            by_country.setdefault(label, []).append(posting)
    return {
        country: group_by_city(members, within_country=country)
        for country, members in _sorted_groups(by_country).items()
    }


def _sorted_groups(grouped: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Order groups by size, then name; unknown buckets always last."""

    def key(item: tuple[str, list[Any]]) -> tuple[int, int, str]:
        label, members = item
        unstated = "not stated" in label.lower()
        return (1 if unstated else 0, -len(members), label)

    return dict(sorted(grouped.items(), key=key))


def search(
    employers: Iterable[Employer],
    *,
    location_terms: Iterable[str],
    role_pattern: re.Pattern[str],
    max_age_days: float | None = None,
    search_text: str = "",
    countries: Iterable[str] = (),
) -> list[AtsPosting]:
    """Read every board, then keep postings matching role, place, and age.

    A posting with no stated date is kept when ``max_age_days`` is set: the
    board not saying is not the same as the posting being old, and dropping it
    would hide real openings. Its missing date stays visible on the result.
    """
    return _filter(
        fetch_all(employers, search_text=search_text),
        list(location_terms),
        role_pattern,
        max_age_days,
        list(countries),
    )


def _filter(
    postings: Iterable[AtsPosting],
    location_terms: list[str],
    role_pattern: re.Pattern[str],
    max_age_days: float | None,
    countries: list[str] | None = None,
) -> list[AtsPosting]:
    """Keep postings matching role, place, country, and age; freshest first."""
    results = []
    for posting in postings:
        if not matches_role(posting, role_pattern):
            continue
        if location_terms and not matches_location(posting, location_terms):
            continue
        if countries and not matches_country(posting, countries):
            continue
        if max_age_days is not None:
            age = posting.age_days
            if age is not None and age > max_age_days:
                continue
        results.append(posting)
    results.sort(key=lambda p: (p.age_days if p.age_days is not None else 1e9, p.title))
    return results
