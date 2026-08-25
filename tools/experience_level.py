"""Seniority: what the user asked for, and what a posting actually offers.

Two jobs, kept deliberately separate:

* :func:`level_query_terms` shapes the *search* so the right level is fetched.
* :func:`detect_experience_level` reads the level back off a retrieved posting.

The second never trusts the first. Asking for senior roles is not evidence that
a result is senior, in exactly the same way that a freshness filter is not
evidence a posting is recent. When a posting does not state a level, the answer
is ``"unknown"`` — never a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from models.job import EXPERIENCE_LEVEL_YEARS, ExperienceLevel

# Ordered from most to least senior. Order matters: a title reading
# "Senior Staff Engineer" must resolve to the higher of the two.
LEVEL_ORDER: tuple[ExperienceLevel, ...] = (
    "manager",
    "staff_principal",
    "senior",
    "mid",
    "entry",
    "internship",
)

# The order markers are *read* in, which is not the seniority order above.
#
# An internship word ("intern", "co-op", "apprentice") is a decisive statement
# about what a posting is. Several entry markers are not: "graduate", "campus",
# and "associate" appear on internship titles constantly — "Graduate Intern"
# and "Summer Analyst Intern, Campus Program" are internships, not junior
# roles. Scanning entry first labelled both of those "entry", so a user asking
# for Internship had real internships filtered off the screen. Internship is
# therefore read before entry; every other level keeps its most-senior-first
# order, so "Senior Data Analyst Intern" still resolves to senior.
DETECTION_ORDER: tuple[ExperienceLevel, ...] = (
    "manager",
    "staff_principal",
    "senior",
    "mid",
    "internship",
    "entry",
)

# Title markers, checked as whole words so "internal" never reads as "intern"
# and "Seniority" never reads as "Senior".
LEVEL_TITLE_MARKERS: dict[ExperienceLevel, tuple[str, ...]] = {
    "internship": (
        "intern",
        "interns",
        "internship",
        "co-op",
        "coop",
        "trainee",
        "apprentice",
        "apprenticeship",
    ),
    "entry": (
        "junior",
        "jr",
        "entry level",
        "entry-level",
        "associate",
        "graduate",
        "new grad",
        "new graduate",
        "campus",
        "early career",
        "i",
    ),
    "mid": ("mid level", "mid-level", "intermediate", "ii", "2"),
    "senior": ("senior", "sr", "snr", "iii", "3"),
    # "architect" is deliberately absent: it names the role, not its seniority.
    # There are junior, senior, and principal architects, so reading "Data
    # Architect" as staff level states a seniority the posting never did.
    "staff_principal": (
        "staff",
        "principal",
        "lead",
        "distinguished",
        "iv",
        "4",
    ),
    "manager": (
        "manager",
        "director",
        "head of",
        "vp",
        "vice president",
        "chief",
    ),
}

# Roman numerals and bare digits are only a level signal inside a job title,
# never inside body text, where they mean something else entirely.
_TITLE_ONLY_MARKERS: frozenset[str] = frozenset(
    {"i", "ii", "iii", "iv", "2", "3", "4"}
)

# Phrases that state a level in the body of a posting.
LEVEL_BODY_MARKERS: dict[ExperienceLevel, tuple[str, ...]] = {
    "internship": ("this internship", "as an intern", "summer internship"),
    "entry": ("entry level position", "entry-level role", "new graduate program"),
    "senior": ("senior-level role", "senior level position"),
    "staff_principal": ("staff-level", "principal-level"),
    "manager": ("people management", "manage a team", "direct reports"),
}

# Words that turn a numeral into something other than a seniority level. A
# "Tier 3 Support Analyst" states a support tier and a "Grade 2 Technician" a
# pay grade; neither is a level III or level II role.
_NUMERAL_CONTEXT_WORDS: frozenset[str] = frozenset(
    {
        "tier",
        "grade",
        "level",
        "phase",
        "shift",
        "class",
        "band",
        "step",
        "group",
        "type",
        "zone",
        "stage",
        "round",
        "unit",
        "building",
        "team",
        "site",
    }
)

# Collocations where a marker word belongs to the role rather than to its
# seniority: "Lead Generation Specialist" is a marketing role, not a lead
# engineer.
_MARKER_BLOCKING_FOLLOWERS: dict[str, frozenset[str]] = {
    "lead": frozenset({"generation", "gen", "generations", "generator", "generators"}),
}

# A body phrase inside a negation states the opposite of the level it names:
# "no direct reports" and "people management experience is not required" are
# evidence *against* a management role, so neither may label one.
_NEGATIONS: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "never",
        "without",
        "none",
        "non",
        "neither",
        "nor",
        "cannot",
        "isn't",
        "aren't",
        "doesn't",
        "don't",
        "won't",
    }
)
_NEGATION_WINDOW = 3

# Words that carry no role meaning on their own. A role that strips down to
# only these ("Chief of Staff" -> "of") has no base left to search for.
_CONNECTOR_WORDS: frozenset[str] = frozenset(
    {"of", "the", "a", "an", "and", "or", "for", "to", "in", "at", "on", "with"}
)

_WORD_RE = re.compile(r"[a-z0-9+#.'\-]+")
_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+|plus)?\s*(?:-|to|–)?\s*(\d+(?:\.\d+)?)?\s*"
    r"(?:\+|plus)?\s*years?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LevelDetection:
    """A detected seniority plus the evidence that produced it."""

    level: ExperienceLevel
    evidence: str | None

    @property
    def is_known(self) -> bool:
        """True when the posting actually stated a level."""
        return self.level != "unknown"


def _words(text: str) -> list[str]:
    """Lowercase word tokens, punctuation stripped.

    A trailing period is removed so the very common ``"Sr."`` and ``"Jr."``
    abbreviations match their markers, while an internal dot is preserved so
    tokens like ``node.js`` survive intact. An internal apostrophe is kept for
    the same reason in reverse: the possessive in "Principal's Office" stays one
    token, so it never matches the seniority marker "principal".
    """
    lowered = (text or "").lower().replace("’", "'")
    return [
        stripped
        for token in _WORD_RE.findall(lowered)
        if (stripped := token.strip(".-'"))
    ]


def _marker_spans(tokens: list[str], marker: str) -> list[tuple[int, int]]:
    """Return every ``[start, end)`` token span where ``marker`` appears in full.

    Markers are matched as whole token sequences, never as raw substrings:
    "Head Office Analyst" contains the letters of "head of" and must not read as
    a management title, exactly as "Internal" must not read as "intern".
    """
    parts = marker.split()
    width = len(parts)
    if not width or width > len(tokens):
        return []
    return [
        (index, index + width)
        for index in range(len(tokens) - width + 1)
        if tokens[index : index + width] == parts
    ]


def _states_a_level(tokens: list[str], marker: str, start: int, end: int) -> bool:
    """True when this occurrence of ``marker`` really states a seniority.

    Guards the two ways a marker word appears without meaning a level: a
    numeral owned by a preceding noun ("Tier 3"), and a marker that is the first
    half of a role name ("Lead Generation").
    """
    if (
        marker in _TITLE_ONLY_MARKERS
        and start
        and tokens[start - 1] in _NUMERAL_CONTEXT_WORDS
    ):
        return False
    blocked = _MARKER_BLOCKING_FOLLOWERS.get(marker)
    if blocked and end < len(tokens) and tokens[end] in blocked:
        return False
    return True


def _is_negated(tokens: list[str], start: int, end: int) -> bool:
    """True when a negation sits close enough to reverse the phrase's meaning."""
    before = tokens[max(0, start - _NEGATION_WINDOW) : start]
    after = tokens[end : end + _NEGATION_WINDOW]
    return any(word in _NEGATIONS for word in (*before, *after))


def _contains_marker(tokens: list[str], marker: str) -> bool:
    """True when a marker appears as a whole word or whole phrase."""
    return any(
        _states_a_level(tokens, marker, start, end)
        for start, end in _marker_spans(tokens, marker)
    )


# Single words removed from a typed role before a level is applied to it.
_STRIPPABLE_LEVEL_WORDS: frozenset[str] = frozenset(
    marker
    for markers in LEVEL_TITLE_MARKERS.values()
    for marker in markers
    if " " not in marker
)


def strip_level_words(role: str) -> str:
    """Remove seniority words from a role title.

    ``"Senior Data Analyst"`` becomes ``"data analyst"``. This runs before a
    level is applied to a query, so choosing *Senior* for a role already typed
    as "Data Analyst Intern" cannot produce "senior data analyst intern".
    """
    tokens = _words(role)
    if not tokens:
        return ""
    # Whole phrases go first, by token span. A raw string replace here turned
    # "Head Office Analyst" into "fice analyst", which then went into the live
    # search query.
    for markers in LEVEL_TITLE_MARKERS.values():
        for marker in markers:
            if " " not in marker:
                continue
            for start, end in reversed(_marker_spans(tokens, marker)):
                del tokens[start:end]
    kept = [
        token
        for index, token in enumerate(tokens)
        if not (
            token in _STRIPPABLE_LEVEL_WORDS
            and _states_a_level(tokens, token, index, index + 1)
        )
    ]
    if all(token in _CONNECTOR_WORDS for token in kept):
        return ""
    return " ".join(kept).strip()


# What to search for when the typed role is nothing but seniority words.
#
# Typing "Intern" or "Manager" into the role box is ordinary, and stripping the
# seniority out of it leaves nothing at all. Returning an empty list there
# dropped the role clause from the query entirely, so the search became
# "(Houston OR remote)" — every page in a city, no job filter. These terms keep
# the query about the level the user actually asked for instead.
BARE_LEVEL_TERMS: dict[ExperienceLevel, tuple[str, ...]] = {
    "internship": ("intern", "internship"),
    "entry": ("entry level", "junior", "associate"),
    "mid": ("mid level",),
    "senior": ("senior", "sr"),
    "staff_principal": ("staff", "principal", "lead"),
    "manager": ("manager", "director"),
}


def level_query_terms(role: str, level: ExperienceLevel) -> list[str]:
    """Return the quoted phrases that search for ``role`` at ``level``.

    The base role is stripped of any seniority word first, so the caller's free
    text and the selected level cannot contradict each other. When that leaves
    nothing — the user typed only "Intern" or only "Manager" — the level's own
    terms are searched rather than no role at all.
    """
    base = strip_level_words(role)
    if not base:
        return [] if not (role or "").strip() else list(BARE_LEVEL_TERMS.get(level, ()))
    if level == "internship":
        return [f"{base} intern", f"{base} internship"]
    if level == "entry":
        return [f"entry level {base}", f"junior {base}", f"associate {base}"]
    if level == "mid":
        return [base, f"mid level {base}"]
    if level == "senior":
        return [f"senior {base}", f"sr {base}"]
    if level == "staff_principal":
        return [f"staff {base}", f"principal {base}", f"lead {base}"]
    if level == "manager":
        return [f"{base} manager", f"director of {base}", f"head of {base}"]
    return [base]


def detect_from_title(title: str) -> LevelDetection:
    """Detect seniority from a job title alone.

    Scans in :data:`DETECTION_ORDER`, so a compound title resolves to its
    highest stated level — except that an explicit internship word outranks the
    softer entry markers, because "Graduate Intern" is an internship.
    """
    tokens = _words(title)
    if not tokens:
        return LevelDetection("unknown", None)
    for level in DETECTION_ORDER:
        for marker in LEVEL_TITLE_MARKERS[level]:
            if _contains_marker(tokens, marker):
                return LevelDetection(level, f'title contains "{marker}"')
    return LevelDetection("unknown", None)


def detect_from_years(minimum_years: float | None) -> LevelDetection:
    """Map an explicitly stated minimum years-of-experience to a level."""
    if minimum_years is None:
        return LevelDetection("unknown", None)
    evidence = f"posting states {minimum_years:g}+ years of experience"
    if minimum_years <= 0:
        return LevelDetection("entry", evidence)
    if minimum_years < 2:
        return LevelDetection("entry", evidence)
    if minimum_years < 5:
        return LevelDetection("mid", evidence)
    if minimum_years < 8:
        return LevelDetection("senior", evidence)
    return LevelDetection("staff_principal", evidence)


def detect_from_body(description: str) -> LevelDetection:
    """Detect seniority from explicit statements in the posting body.

    A phrase sitting inside a negation is skipped. "This role has no direct
    reports" says the opposite of what the marker names, and labelling that
    posting *Manager* would both mislabel it and drop it from a search for any
    other level.
    """
    tokens = _words(description)
    if not tokens:
        return LevelDetection("unknown", None)
    for level in DETECTION_ORDER:
        for marker in LEVEL_BODY_MARKERS.get(level, ()):  # body markers are sparse
            if marker in _TITLE_ONLY_MARKERS:
                continue
            for start, end in _marker_spans(tokens, marker):
                if not _states_a_level(tokens, marker, start, end):
                    continue
                if _is_negated(tokens, start, end):
                    continue
                return LevelDetection(level, f'description says "{marker}"')
    return LevelDetection("unknown", None)


def detect_experience_level(
    title: str,
    description: str = "",
    minimum_years: float | None = None,
) -> LevelDetection:
    """Detect a posting's seniority from the strongest evidence available.

    Title wins, because that is where employers state seniority. An explicit
    years-of-experience requirement is the next best evidence, then a direct
    statement in the body. When nothing states a level the answer is
    ``"unknown"`` and the UI says so rather than guessing.
    """
    from_title = detect_from_title(title)
    if from_title.is_known:
        return from_title
    from_years = detect_from_years(minimum_years)
    if from_years.is_known:
        return from_years
    from_body = detect_from_body(description)
    if from_body.is_known:
        return from_body
    return LevelDetection("unknown", None)


def levels_conflict(requested: ExperienceLevel, detected: ExperienceLevel) -> bool:
    """True when a detected level clearly contradicts what the user asked for.

    An unknown level never conflicts: the posting simply did not say, and
    dropping it would hide real openings on no evidence. ``unknown`` on the
    request side means the user did not filter at all.
    """
    if requested in ("unknown", None) or detected == "unknown":
        return False
    return requested != detected


def expected_years(level: ExperienceLevel) -> tuple[float, float | None]:
    """Return the typical (minimum, maximum) years associated with a level."""
    return EXPERIENCE_LEVEL_YEARS.get(level, (0.0, None))
