"""The one search the Cougar Career Agent is running.

Both routes the agent can take — searching public pages, and reading employers'
own job boards — answer the same question: this role, in this place, posted this
recently. Holding that question in one object keeps the two from drifting apart,
so a location typed on one screen means the same thing on the other.

Each route reads the fields it can honour. The boards, for instance, state a
publish date, so they filter on age directly; the search path can only ask a
search engine to prefer recent pages and then verify what comes back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from models.job import normalize_work_mode

FRESHNESS_DAYS: dict[str, float] = {
    "last_hour": 1.0,
    "last_24_hours": 1.0,
    "last_3_days": 3.0,
    "last_7_days": 7.0,
    "last_14_days": 14.0,
    "last_30_days": 30.0,
}

DEFAULT_ROLE = "Data Analyst Intern"
DEFAULT_LOCATION = "Houston, TX"


@dataclass(frozen=True)
class SearchQuery:
    """What the user asked for, in the words they asked for it."""

    role: str = DEFAULT_ROLE
    location: str = DEFAULT_LOCATION
    work_mode: str = "any"
    query_category: str = "company_careers"
    freshness_window: str = "last_24_hours"
    experience_level: str = "internship"
    employment_type: str = "any"

    # ---- derived views, one per route -----------------------------------

    @property
    def max_age_days(self) -> float | None:
        """The freshness window in days, or ``None`` when it is not a window."""
        return FRESHNESS_DAYS.get(self.freshness_window)

    @property
    def city(self) -> str:
        """The city on its own, without the state or country."""
        return self.location.split(",")[0].strip()

    def role_pattern(self) -> re.Pattern[str]:
        """Match a job title against the requested role.

        Every word is an alternative, so "Data Analyst Intern" also finds a
        "Data Analyst" and an "Analyst Intern". Words too short or too generic
        to narrow anything are dropped.
        """
        stop = {"a", "an", "the", "of", "and", "or", "for", "in", "at", "jobs", "job"}
        words = [
            re.escape(word)
            for word in re.split(r"[^A-Za-z0-9+#]+", self.role)
            if len(word) > 2 and word.lower() not in stop
        ]
        if not words:
            return re.compile(".")
        return re.compile("|".join(words), re.IGNORECASE)

    def location_terms(self, extra: list[str] | None = None) -> list[str]:
        """Place names a posting may state to count as being here."""
        terms = [self.city] if self.city else []
        return terms + list(extra or [])

    def wants_remote(self) -> bool:
        """True when a remote posting satisfies this search."""
        return normalize_work_mode(self.work_mode) in {"any", "remote"}

    # ---- plumbing --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The shape the agent graph expects."""
        return {
            "role": self.role,
            "location": self.location,
            "work_mode": self.work_mode,
            "query_category": self.query_category,
            "freshness_window": self.freshness_window,
            "experience_level": self.experience_level,
            "employment_type": self.employment_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SearchQuery":
        """Read a query back, falling back to the defaults field by field.

        The value crosses a Streamlit session boundary, so nothing here trusts
        the input to be complete or to be the right type.
        """
        if not isinstance(data, dict):
            return cls()
        base = cls()
        known = {
            field: data.get(field)
            for field in (
                "role", "location", "work_mode", "query_category",
                "freshness_window", "experience_level", "employment_type",
            )
        }
        usable = {
            field: str(value)
            for field, value in known.items()
            if isinstance(value, (str, int, float)) and str(value).strip()
        }
        return replace(base, **usable) if usable else base
