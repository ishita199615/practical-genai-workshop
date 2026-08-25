"""Truthfulness validation for the tailored application.

Two layers run in order:

1. Deterministic Python checks that cannot be talked out of a verdict.
2. An LLM claim review that classifies each claim as supported, unsupported, or
   unclear, with a deterministic fallback when no model is available.

The validator is the reason the demo can promise that Power BI is never added
to a resume that cannot evidence it.
"""

from __future__ import annotations

import re
from typing import Any

from models.application import TailoredApplication
from models.ats import AtsRecommendation
from models.resume import ResumeProfile
from models.validation import ClaimReview, ClaimReviewBatch, ValidationReport
from prompts import render_prompt
from tools.ats_scorer import ResumeSnapshot, find_evidence, snapshot_from_resume
from tools.job_scorer import SKILL_VOCABULARY, canonical_skill, canonical_skill_set

_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Company names commonly contain lowercase connectors ("Jobs for Humanity",
# "Bank of America"), so the pattern spans them rather than stopping short and
# reporting a fragment as an unknown employer.
_EMPLOYER_RE = re.compile(
    r"\bat ([A-Z][\w&.\-]*(?:\s+(?:for|of|and|the|de|la|von|van)\s+[A-Z][\w&.\-]*"
    r"|\s+[A-Z][\w&.\-]*){0,4})"
)
_DEGREE_RE = re.compile(
    r"\b(bachelor[s']*|master[s']*|mba|ph\.?d|doctorate|associate[s']*)\b",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Words that follow "at" without naming an employer.
_EMPLOYER_STOPWORDS = {
    "the",
    "your",
    "a",
    "an",
    "this",
    "example",
    "least",
    "scale",
    "work",
}


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    """Build one deterministic check record."""
    return {"name": name, "passed": passed, "detail": detail}


def _revised_texts(application: TailoredApplication) -> list[str]:
    """Return every piece of generated prose in one list."""
    return [
        application.revised_summary,
        *[bullet.revised_text for bullet in application.revised_bullets],
        application.cover_letter,
    ]


def run_deterministic_checks(
    application: TailoredApplication,
    resume: ResumeProfile,
    applied_recommendations: list[AtsRecommendation] | None = None,
    unsupported_gaps: list[str] | None = None,
    allowed_organizations: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run every Python-side truthfulness rule.

    Each rule returns a record so the UI can show exactly what was verified.
    ``allowed_organizations`` names companies the draft may address without
    claiming employment — normally the company being applied to.
    """
    checks: list[dict[str, Any]] = []
    bullet_index = resume.bullet_index()
    resume_text = resume.as_plain_text()
    resume_lower = resume_text.lower()
    generated = _revised_texts(application)
    generated_text = "\n".join(generated)

    # 1. Source bullet IDs exist.
    unknown_ids = [
        bullet.source_bullet_id
        for bullet in application.revised_bullets
        if bullet.source_bullet_id not in bullet_index
    ]
    checks.append(
        _check(
            "Source IDs verified",
            not unknown_ids and bool(application.revised_bullets),
            "Every revised bullet points at a real resume bullet."
            if not unknown_ids
            else f"Unknown source bullet IDs: {', '.join(unknown_ids)}.",
        )
    )

    # 2. Original text matches the referenced source.
    mismatched = [
        bullet.source_bullet_id
        for bullet in application.revised_bullets
        if bullet.source_bullet_id in bullet_index
        and bullet.original_text.strip() != bullet_index[bullet.source_bullet_id].strip()
    ]
    checks.append(
        _check(
            "Original bullet text matches the master resume",
            not mismatched,
            "Quoted original text is identical to the master resume."
            if not mismatched
            else f"Original text was altered for: {', '.join(mismatched)}.",
        )
    )

    # 3. No new employer.
    known_orgs = {entry.organization.lower() for entry in resume.experience}
    known_orgs |= {name.lower() for name in (allowed_organizations or []) if name}
    new_employers = []
    for match in _EMPLOYER_RE.finditer(generated_text):
        candidate = match.group(1).strip()
        if candidate.lower() in _EMPLOYER_STOPWORDS:
            continue
        lowered = candidate.lower()
        if lowered in known_orgs or lowered in resume_lower:
            continue
        # A partial capture of an allowed name is not a new employer, so the
        # prefix test runs in both directions.
        if any(
            org and (lowered.startswith(org) or org.startswith(lowered))
            for org in known_orgs
        ):
            continue
        new_employers.append(candidate)
    checks.append(
        _check(
            "No unsupported employer added",
            not new_employers,
            "No employer outside the master resume appears."
            if not new_employers
            else f"Unsupported employer reference: {', '.join(sorted(set(new_employers)))}.",
        )
    )

    # 4. No new degree or institution.
    resume_degrees = {match.group(0).lower() for match in _DEGREE_RE.finditer(resume_text)}
    new_degrees = {
        match.group(0).lower()
        for match in _DEGREE_RE.finditer(generated_text)
        if match.group(0).lower() not in resume_degrees
    }
    checks.append(
        _check(
            "No unsupported degree added",
            not new_degrees,
            "No degree outside the master resume appears."
            if not new_degrees
            else f"Unsupported degree reference: {', '.join(sorted(new_degrees))}.",
        )
    )

    # 5. No new date.
    resume_years = {match.group(0) for match in _YEAR_RE.finditer(resume_text)}
    new_years = {
        match.group(0)
        for match in _YEAR_RE.finditer(generated_text)
        if match.group(0) not in resume_years
    }
    checks.append(
        _check(
            "No new dates introduced",
            not new_years,
            "All years trace back to the master resume."
            if not new_years
            else f"Unsupported year(s): {', '.join(sorted(new_years))}.",
        )
    )

    # 6. No invented numeric metric.
    resume_numbers = {match.group(0) for match in _NUMBER_RE.finditer(resume_text)}
    new_numbers = {
        match.group(0)
        for match in _NUMBER_RE.finditer(generated_text)
        if match.group(0) not in resume_numbers
    }
    checks.append(
        _check(
            "No invented metrics",
            not new_numbers,
            "No number appears that the master resume does not contain."
            if not new_numbers
            else f"Unsupported number(s): {', '.join(sorted(new_numbers))}.",
        )
    )

    # 7. Reordered skills are a subset of the original skills.
    original_skills = {skill.strip().lower() for skill in resume.skills}
    added_skills = [
        skill
        for skill in application.reordered_skills
        if skill.strip().lower() not in original_skills
    ]
    checks.append(
        _check(
            "Skills remain grounded in the master resume",
            not added_skills,
            "The skills list was reordered, not expanded."
            if not added_skills
            else f"Skills not present in the master resume: {', '.join(added_skills)}.",
        )
    )

    # 8. Applied recommendations were safe and cite evidence.
    applied_recommendations = applied_recommendations or []
    applied_ids = set(application.applied_ats_recommendation_ids)
    by_id = {rec.recommendation_id: rec for rec in applied_recommendations}
    unsafe_applied = [
        rec_id
        for rec_id in applied_ids
        if rec_id not in by_id
        or not by_id[rec_id].safe_to_apply
        or not by_id[rec_id].evidence_resume_ids
    ]
    checks.append(
        _check(
            "Only safe ATS recommendations were applied",
            not unsafe_applied,
            "Every applied recommendation was marked safe and cites resume evidence."
            if not unsafe_applied
            else f"Recommendation(s) applied without safe evidence: {', '.join(sorted(unsafe_applied))}.",
        )
    )

    # 9. Unsupported gaps never entered the document.
    gaps = unsupported_gaps or application.unsupported_ats_gaps_not_applied
    leaked = [
        gap
        for gap in gaps
        if re.search(
            r"(?<![a-z0-9])" + re.escape(gap.lower()) + r"(?![a-z0-9])",
            generated_text.lower(),
        )
    ]
    checks.append(
        _check(
            "Unsupported job gaps were not added",
            not leaked,
            "No unsupported requirement was written into the application."
            if not leaked
            else f"Unsupported requirement(s) present in the draft: {', '.join(leaked)}.",
        )
    )

    # 10. The cover letter claims no unsupported skill.
    snapshot = snapshot_from_resume(resume)
    unsupported_in_letter = _unsupported_skills_in(application.cover_letter, snapshot)
    checks.append(
        _check(
            "Cover letter claims no unsupported skill",
            not unsupported_in_letter,
            "Every tool named in the cover letter exists in the master resume."
            if not unsupported_in_letter
            else f"Cover letter names unsupported skill(s): {', '.join(unsupported_in_letter)}.",
        )
    )

    return checks


def _unsupported_skills_in(text: str, snapshot: ResumeSnapshot) -> list[str]:
    """Return recognised skills named in ``text`` that the resume cannot back."""
    lowered = (text or "").lower()
    resume_skills = canonical_skill_set(snapshot.skills)
    found: list[str] = []
    for skill in SKILL_VOCABULARY:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if not re.search(pattern, lowered):
            continue
        canonical = canonical_skill(skill)
        if canonical in resume_skills:
            continue
        if find_evidence(canonical, snapshot):
            continue
        found.append(skill)
    return found


def extract_claims(application: TailoredApplication) -> list[str]:
    """Split the generated content into individually checkable claims."""
    claims: list[str] = []
    if application.revised_summary.strip():
        claims.append(application.revised_summary.strip())
    for bullet in application.revised_bullets:
        if bullet.revised_text.strip():
            claims.append(bullet.revised_text.strip())
    for sentence in _SENTENCE_RE.split(application.cover_letter.strip()):
        cleaned = sentence.strip()
        if len(cleaned.split()) >= 4:
            claims.append(cleaned)
    return claims


def review_claims(
    claims: list[str],
    resume: ResumeProfile,
    llm: Any | None = None,
) -> list[ClaimReview]:
    """Classify each claim, using the LLM when available.

    The deterministic fallback is intentionally strict: a claim naming a tool
    the resume cannot evidence is unsupported, regardless of phrasing.
    """
    if not claims:
        return []

    if llm is not None and getattr(llm, "available", False):
        prompt = render_prompt(
            "validate_claims",
            MASTER_RESUME=resume.as_plain_text(),
            RESUME_IDS=", ".join(sorted(resume.bullet_index())),
            DRAFT_CLAIMS="\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims)),
        )
        batch = llm.generate_structured(prompt, ClaimReviewBatch, temperature=0.0)
        if batch is not None and batch.reviews:
            return batch.reviews

    return _fallback_reviews(claims, resume)


def _fallback_reviews(claims: list[str], resume: ResumeProfile) -> list[ClaimReview]:
    """Deterministic claim classification used when no model is available."""
    snapshot = snapshot_from_resume(resume)
    reviews: list[ClaimReview] = []
    for claim in claims:
        unsupported = _unsupported_skills_in(claim, snapshot)
        if unsupported:
            reviews.append(
                ClaimReview(
                    claim=claim,
                    status="unsupported",
                    supporting_resume_ids=[],
                    reason=(
                        "Names "
                        + ", ".join(unsupported)
                        + ", which the master resume does not evidence."
                    ),
                )
            )
            continue
        evidence: list[str] = []
        for skill in resume.skills:
            canonical = canonical_skill(skill)
            pattern = r"(?<![a-z0-9])" + re.escape(canonical) + r"(?![a-z0-9])"
            if re.search(pattern, claim.lower()):
                evidence.extend(find_evidence(canonical, snapshot))
        reviews.append(
            ClaimReview(
                claim=claim,
                status="supported",
                supporting_resume_ids=sorted(set(evidence)),
                reason="Every named fact traces back to the master resume.",
            )
        )
    return reviews


def validate_application(
    application: TailoredApplication,
    resume: ResumeProfile,
    *,
    applied_recommendations: list[AtsRecommendation] | None = None,
    unsupported_gaps: list[str] | None = None,
    allowed_organizations: list[str] | None = None,
    llm: Any | None = None,
) -> ValidationReport:
    """Run deterministic checks and the claim review, then combine the verdict."""
    checks = run_deterministic_checks(
        application,
        resume,
        applied_recommendations,
        unsupported_gaps,
        allowed_organizations,
    )
    reviews = review_claims(extract_claims(application), resume, llm)

    unsupported = [review for review in reviews if review.status == "unsupported"]
    unclear = [review for review in reviews if review.status == "unclear"]
    source_ids_ok = next(
        (check["passed"] for check in checks if check["name"] == "Source IDs verified"),
        False,
    )
    passed = all(check["passed"] for check in checks) and not unsupported

    return ValidationReport(
        valid_source_ids=bool(source_ids_ok),
        unsupported_claims=unsupported,
        unclear_claims=unclear,
        deterministic_checks=checks,
        passed=passed,
    )
