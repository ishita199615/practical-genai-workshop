"""Deterministic Demo Job Match Score.

The language model never produces or edits a number here. Given the same
resume, the same job text, and the same reference time, this module always
returns the same score.

Weighting (100 points):

* 45 — required-skill coverage
* 20 — resume/job text similarity
* 15 — role-title alignment
* 10 — experience alignment
* 10 — location and work-mode alignment
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.job import JobPosting
from models.match import MatchResult
from models.resume import ResumeProfile
from tools.experience_level import expected_years

# --------------------------------------------------------------------------
# Skill canonicalization
# --------------------------------------------------------------------------

SKILL_ALIASES: dict[str, str] = {
    "ms excel": "excel",
    "microsoft excel": "excel",
    "advanced excel": "excel",
    "excel (pivot tables)": "excel",
    "spreadsheets": "excel",
    "postgresql": "sql",
    "postgres": "sql",
    "mysql": "sql",
    "t-sql": "sql",
    "structured query language": "sql",
    "sql queries": "sql",
    "data viz": "data visualization",
    "data visualisation": "data visualization",
    "visualization": "data visualization",
    "powerbi": "power bi",
    "power-bi": "power bi",
    "microsoft power bi": "power bi",
    "python 3": "python",
    "python programming": "python",
    "pandas library": "pandas",
    "statistical analysis": "statistics",
    "stats": "statistics",
    "tableau desktop": "tableau",
    "google sheets": "google sheets",
    "a/b testing": "a/b testing",
    "ab testing": "a/b testing",
    "etl": "etl",
}

# Skills the demo can recognise inside free job text when a posting does not
# expose an explicit requirements list.
SKILL_VOCABULARY: tuple[str, ...] = (
    "python",
    "sql",
    "excel",
    "tableau",
    "power bi",
    "pandas",
    "statistics",
    "data visualization",
    "dashboards",
    "reporting",
    "data cleaning",
    "data analysis",
    "r",
    "sas",
    "spss",
    "looker",
    "etl",
    "a/b testing",
    "machine learning",
    "google sheets",
    "snowflake",
    "airflow",
)

# Resume evidence terms for concepts the resume proves without naming directly.
CONCEPT_EVIDENCE: dict[str, tuple[str, ...]] = {
    "data visualization": ("tableau", "dashboard"),
    "dashboards": ("tableau", "dashboard"),
    "reporting": ("reporting", "dashboard", "weekly"),
    "data cleaning": ("clean", "pandas"),
    "data analysis": ("analyz", "survey data", "summarize"),
    "excel": ("excel",),
    "statistics": ("statistics", "survey"),
    "pandas": ("pandas",),
    "python": ("python",),
    "sql": ("sql",),
    "tableau": ("tableau",),
}

_SPLIT_RE = re.compile(r"[,/;|]| and | or ")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9+#. ]+")

# A cosine similarity at or above this level earns the full similarity score.
SIMILARITY_FULL_CREDIT = 0.40

FUZZY_SKILL_THRESHOLD = 90

# Shortest skill name allowed to match inside a longer requirement phrase.
# Word-boundary matching keeps "sql" from matching "nosql"; one- and two-letter
# names such as "r" stay excluded because substring matching would be noise.
MIN_TOKEN_MATCH_LENGTH = 3
MIN_FUZZY_LENGTH = 4


def canonical_skill(name: str) -> str:
    """Normalize one skill name to its canonical lowercase form."""
    cleaned = _NON_ALNUM_RE.sub(" ", (name or "").strip().lower())
    cleaned = " ".join(cleaned.split()).strip(" .")
    return SKILL_ALIASES.get(cleaned, cleaned)


def canonical_skill_set(names: list[str]) -> set[str]:
    """Normalize a list of skill names, splitting compound entries."""
    result: set[str] = set()
    for name in names or []:
        for part in _SPLIT_RE.split(name or ""):
            canonical = canonical_skill(part)
            if canonical:
                result.add(canonical)
    return result


def skills_from_text(text: str) -> list[str]:
    """Detect known skills mentioned in free job text.

    Used only when a posting exposes no explicit requirements list. The
    vocabulary is fixed, so detection stays deterministic.
    """
    lowered = " " + " ".join((text or "").lower().split()) + " "
    found: list[str] = []
    for skill in SKILL_VOCABULARY:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.append(skill)
    return found


def required_skills_for(job: JobPosting) -> list[str]:
    """Return the canonical required skills used for scoring one job."""
    explicit = canonical_skill_set(job.required_skills)
    if explicit:
        return sorted(explicit)
    detected = canonical_skill_set(skills_from_text(job.description))
    preferred = canonical_skill_set(job.preferred_skills)
    return sorted(detected - preferred)


def skill_is_matched(required: str, resume_skills: set[str]) -> bool:
    """True when the resume evidences one required skill.

    Exact canonical matching first, then a conservative fuzzy pass. Distinct
    tools never collapse into one another: Tableau is not Power BI.
    """
    if required in resume_skills:
        return True
    for owned in resume_skills:
        if owned == required:
            return True
        if len(owned) >= MIN_TOKEN_MATCH_LENGTH and re.search(
            r"(?<![a-z0-9])" + re.escape(owned) + r"(?![a-z0-9])", required
        ):
            return True
        if len(owned) >= MIN_FUZZY_LENGTH and len(required) >= MIN_FUZZY_LENGTH:
            if fuzz.token_set_ratio(owned, required) >= FUZZY_SKILL_THRESHOLD:
                return True
    return False


def score_skill_coverage(
    resume: ResumeProfile, job: JobPosting
) -> tuple[int, list[str], list[str]]:
    """Score required-skill coverage from 0 to 100.

    A posting with no readable requirements scores a neutral 50 rather than a
    zero: an unknown requirement is not a failure.
    """
    required = required_skills_for(job)
    resume_skills = canonical_skill_set(resume.skills)
    if not required:
        return 50, [], []

    matched = [skill for skill in required if skill_is_matched(skill, resume_skills)]
    missing = [skill for skill in required if skill not in matched]
    score = round(100 * len(matched) / len(required))
    return int(score), matched, missing


def score_text_similarity(resume_text: str, job_text: str) -> int:
    """Score TF-IDF cosine similarity from 0 to 100.

    A cosine of ``SIMILARITY_FULL_CREDIT`` or above earns full credit; below
    that the score scales linearly, which keeps the number explainable.
    """
    if not resume_text.strip() or not job_text.strip():
        return 0
    try:
        vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
        matrix = vectorizer.fit_transform([resume_text, job_text])
        cosine = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except ValueError:
        return 0
    scaled = min(1.0, cosine / SIMILARITY_FULL_CREDIT)
    return int(round(max(0.0, scaled) * 100))


def score_role_alignment(target_roles: list[str], job_title: str) -> int:
    """Score title alignment from 0 to 100 using token overlap and fuzz."""
    title = " ".join(_NON_ALNUM_RE.sub(" ", (job_title or "").lower()).split())
    if not title or not target_roles:
        return 0
    best = 0
    title_tokens = set(title.split())
    for role in target_roles:
        role_clean = " ".join(_NON_ALNUM_RE.sub(" ", role.lower()).split())
        role_tokens = set(role_clean.split())
        if not role_tokens:
            continue
        overlap = len(role_tokens & title_tokens) / len(role_tokens)
        fuzzy = fuzz.token_set_ratio(role_clean, title) / 100.0
        best = max(best, round(100 * (0.5 * overlap + 0.5 * fuzzy)))
    return int(best)


def combined_target_roles(
    target_roles: list[str], extra_target_roles: list[str] | None = None
) -> list[str]:
    """Merge the resume's target roles with roles the user actually searched for.

    The searched role matters because the resume is fixed demo data while the
    search box is not: someone looking for a Senior Data Scientist should see
    that title rewarded. Order is preserved and duplicates are dropped
    case-insensitively, so passing no extra roles reproduces the resume-only
    comparison exactly.
    """
    merged: list[str] = []
    seen: set[str] = set()
    for role in list(target_roles or []) + list(extra_target_roles or []):
        cleaned = " ".join((role or "").split())
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        merged.append(cleaned)
    return merged


def estimate_experience_years(
    resume: ResumeProfile, now: datetime | None = None
) -> float:
    """Estimate related experience in years from the resume's own dates.

    Never fabricates a number: entries without a parseable start date
    contribute nothing.
    """
    current = now or datetime.now(timezone.utc)
    months = 0.0
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
        "december": 12,
    }
    for entry in resume.experience:
        dates = (entry.dates or "").lower()
        match = re.search(r"([a-z]+)\s+(\d{4})", dates)
        if not match or match.group(1) not in month_names:
            continue
        start_month = month_names[match.group(1)]
        start_year = int(match.group(2))
        end_month, end_year = current.month, current.year
        end_match = re.search(r"[–\-]\s*([a-z]+)\s+(\d{4})", dates)
        if end_match and end_match.group(1) in month_names:
            end_month = month_names[end_match.group(1)]
            end_year = int(end_match.group(2))
        months += max(0, (end_year - start_year) * 12 + (end_month - start_month))
    return round(months / 12.0, 2)


def score_experience_alignment(
    job: JobPosting, resume: ResumeProfile, now: datetime | None = None
) -> tuple[int, list[str]]:
    """Score experience alignment from 0 to 100 with any concern noted."""
    minimum = job.minimum_experience_years
    if minimum is None:
        return 80, []
    years = estimate_experience_years(resume, now)
    if years >= minimum:
        return 100, []
    concern = (
        f"{minimum:g} year(s) of related experience is requested; the resume "
        f"evidences about {years:g}."
    )
    if years >= minimum - 1:
        return 70, [concern]
    return 40, [concern]


def seniority_concern(
    job: JobPosting, resume: ResumeProfile, now: datetime | None = None
) -> str | None:
    """Note when a posting's stated level outranks the resume's own experience.

    This is a concern, never a score change: the 45/20/15/10/10 weighting is
    fixed. A posting that never states a level stays ``"unknown"`` and produces
    no concern, because asking for a level is not evidence about a result.
    """
    if job.experience_level == "unknown":
        return None
    minimum, _maximum = expected_years(job.experience_level)
    years = estimate_experience_years(resume, now)
    if minimum <= years:
        return None
    return (
        f"This is a {job.experience_level_label()} posting (typically "
        f"{minimum:g}+ years); the resume evidences about {years:g}."
    )


def score_preference_alignment(
    job: JobPosting, location: str, work_mode: str
) -> tuple[int, str]:
    """Score location and work-mode fit from 0 to 100 with a short summary."""
    city = location.split(",")[0].strip().lower()
    job_location = (job.location or "").lower()
    mode = (work_mode or "any").strip().lower()
    location_match = bool(city) and city in job_location
    job_mode = job.work_mode

    if mode == "remote":
        score = {"remote": 100, "hybrid": 50, "onsite": 20, "unknown": 60}[job_mode]
    elif mode == "onsite":
        if job_mode == "remote":
            score = 70
        else:
            score = 100 if location_match else 40
    elif mode == "hybrid":
        if job_mode == "hybrid":
            score = 100
        elif location_match:
            score = 70
        else:
            score = 40
    else:
        if job_mode == "remote" or location_match:
            score = 100
        elif job_mode == "unknown" and not job_location:
            score = 60
        else:
            score = 40

    summary = (
        f"Preference {mode} vs posting {job_mode}"
        + (f" in {job.location}" if job.location else "")
    )
    return score, summary


def score_job(
    job: JobPosting,
    resume: ResumeProfile,
    *,
    location: str,
    work_mode: str,
    now: datetime | None = None,
    extra_target_roles: list[str] | None = None,
) -> MatchResult:
    """Calculate the complete Demo Job Match Score for one posting.

    ``extra_target_roles`` adds the role the user actually searched for to the
    title comparison. Omitting it scores exactly as the resume's own target
    roles always have.
    """
    skill_score, matched, missing = score_skill_coverage(resume, job)
    similarity_score = score_text_similarity(resume.as_plain_text(), job.description)
    role_score = score_role_alignment(
        combined_target_roles(resume.target_roles, extra_target_roles), job.title
    )
    experience_score, concerns = score_experience_alignment(job, resume, now)
    preference_score, preference_summary = score_preference_alignment(
        job, location, work_mode
    )

    if not job.required_skills and not matched and not missing:
        concerns.append("The posting does not list explicit required skills.")
    level_concern = seniority_concern(job, resume, now)
    if level_concern:
        concerns.append(level_concern)
    if preference_score <= 40:
        concerns.append(preference_summary + ".")

    total = round(
        0.45 * skill_score
        + 0.20 * similarity_score
        + 0.15 * role_score
        + 0.10 * experience_score
        + 0.10 * preference_score
    )

    return MatchResult(
        job_id=job.job_id,
        total_score=int(max(0, min(100, total))),
        skill_score=skill_score,
        similarity_score=similarity_score,
        role_score=role_score,
        experience_score=experience_score,
        preference_score=preference_score,
        matched_skills=matched,
        missing_skills=missing,
        concerns=concerns,
    )


def rank_jobs(
    jobs: list[JobPosting],
    resume: ResumeProfile,
    *,
    location: str,
    work_mode: str,
    now: datetime | None = None,
    top_n: int = 3,
    extra_target_roles: list[str] | None = None,
) -> list[MatchResult]:
    """Score every posting and return the highest-scoring ``top_n``.

    Ties break on ``job_id`` so repeated runs produce an identical ordering.
    ``extra_target_roles`` is passed straight through to :func:`score_job`.
    """
    results = [
        score_job(
            job,
            resume,
            location=location,
            work_mode=work_mode,
            now=now,
            extra_target_roles=extra_target_roles,
        )
        for job in jobs
    ]
    results.sort(key=lambda result: (-result.total_score, result.job_id))
    return results[:top_n]
