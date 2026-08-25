"""Detect whether a posting is full-time, a contract, an internship, and so on.

This follows the same rule as seniority detection: read it off the posting, and
when the posting does not say, answer ``"unknown"`` rather than guessing. A
board that states the type outright is believed over any reading of the prose,
because the employer wrote it as a field.

"Unknown" appears on both sides and means different things. On the request side
it means the user did not filter by type at all. On a posting it means the
employer never stated one — which is not evidence that it is full-time, however
common that is, so such a posting is never dropped for failing a type filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from models.job import EmploymentType

# Ordered: the first pattern to match wins, so the more specific ones lead.
# Internship before full-time, because a "Full-time Summer Internship" is an
# internship that happens to be full-time.
_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "internship",
        re.compile(r"\b(intern|internship|co[-\s]?op|placement\s+year|summer\s+analyst)\b", re.I),
        "an internship",
    ),
    (
        "contract",
        re.compile(r"\b(contract|contractor|freelance|consultant|c2c|corp[-\s]to[-\s]corp|fixed[-\s]term|\d+\s*month[s]?\s+contract)\b", re.I),
        "a contract",
    ),
    (
        "temporary",
        re.compile(r"\b(temporary|temp|seasonal|casual|interim|locum)\b", re.I),
        "temporary",
    ),
    (
        "part_time",
        re.compile(r"\bpart[-\s]?time\b", re.I),
        "part-time",
    ),
    (
        "full_time",
        re.compile(r"\b(full[-\s]?time|permanent|regular\s+full)\b", re.I),
        "full-time",
    ),
)


@dataclass(frozen=True)
class TypeDetection:
    """A detected employment type plus the evidence that produced it."""

    employment_type: EmploymentType
    evidence: str | None

    @property
    def is_known(self) -> bool:
        """True when the posting actually stated a type."""
        return self.employment_type != "unknown"


UNKNOWN = TypeDetection("unknown", None)


def detect_from_text(text: str, where: str) -> TypeDetection:
    """Read an employment type out of one piece of text."""
    if not text:
        return UNKNOWN
    for employment_type, pattern, phrase in _TYPE_PATTERNS:
        match = pattern.search(text)
        if match:
            return TypeDetection(
                employment_type,
                f'{where} says "{match.group(0)}", so it reads as {phrase}',
            )
    return UNKNOWN


def detect_employment_type(
    title: str,
    description: str = "",
    stated: str | None = None,
) -> TypeDetection:
    """Detect a posting's employment type from the best evidence available.

    A value the board stated as a field wins outright. Otherwise the title is
    read, then the description — and when neither says, the answer is
    ``"unknown"``.
    """
    if stated:
        from_field = detect_from_text(stated, "the board")
        if from_field.is_known:
            return TypeDetection(
                from_field.employment_type,
                f'the employer listed it as "{stated.strip()}"',
            )
    from_title = detect_from_text(title, "the title")
    if from_title.is_known:
        return from_title
    # Only the opening of a description is read: a full-time posting often
    # mentions contractors somewhere far down, and that is not what it is.
    return detect_from_text((description or "")[:600], "the description")


def types_conflict(requested: str, detected: EmploymentType) -> bool:
    """True when a detected type clearly contradicts what the user asked for.

    An unknown type never conflicts, on either side: the posting simply did not
    say, and dropping it would hide real openings on no evidence.
    """
    if requested in ("unknown", "any", "", None) or detected == "unknown":
        return False
    return requested != detected
