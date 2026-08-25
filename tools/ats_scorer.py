"""Deterministic Demo ATS Readiness Score and improvement guidance.

This is an educational estimate produced by a transparent rubric, not the score
of any employer's proprietary applicant-tracking system. The language model
never calculates, adjusts, or overrides a number here.

Weighting (100 points):

* 40 — required keyword and skill coverage
* 20 — required qualification alignment
* 15 — evidence and specificity in experience/project bullets
* 10 — standard section completeness
* 10 — ATS-safe text structure and parseability
*  5 — contact and application essentials
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from models.application import TailoredApplication
from models.ats import (
    ATS_DISCLAIMER,
    PRIORITY_ORDER,
    AtsAssessment,
    AtsBand,
    AtsRecommendation,
)
from models.job import JobPosting
from models.resume import ResumeProfile, render_resume_text
from tools.job_scorer import (
    CONCEPT_EVIDENCE,
    canonical_skill,
    canonical_skill_set,
    estimate_experience_years,
    skill_is_matched,
    skills_from_text,
)

ACTION_VERBS: tuple[str, ...] = (
    "analyzed",
    "analysed",
    "built",
    "created",
    "designed",
    "developed",
    "delivered",
    "automated",
    "cleaned",
    "summarized",
    "summarised",
    "reported",
    "produced",
    "maintained",
    "used",
    "prepared",
    "supported",
    "documented",
)

DEGREE_LEVELS: dict[str, int] = {
    "high school": 1,
    "associate": 2,
    "bachelor": 3,
    "bachelors": 3,
    "b.s": 3,
    "bs": 3,
    "ba": 3,
    "master": 4,
    "masters": 4,
    "m.s": 4,
    "mba": 4,
    "phd": 5,
    "doctorate": 5,
}

MAX_RECOMMENDATIONS_LOW = 5
MAX_RECOMMENDATIONS_MID = 3
MAX_RECOMMENDATIONS_STRONG = 2


# --------------------------------------------------------------------------
# Resume snapshots
# --------------------------------------------------------------------------


@dataclass
class ResumeSnapshot:
    """A scoreable view of a resume: either the original or the proposal.

    Both versions go through the same rubric, so the projected score is
    comparable to the original by construction.
    """

    name: str
    email: str
    phone: str | None
    location: str
    summary: str
    skills: list[str]
    bullets: dict[str, str]
    education: list[dict]
    experience: list[dict]
    projects: list[dict]
    document_features: dict = field(default_factory=dict)

    @property
    def plain_text(self) -> str:
        """Canonical plain-text rendering used for keyword search."""
        return render_resume_text(
            name=self.name,
            email=self.email,
            phone=self.phone,
            location=self.location,
            summary=self.summary,
            skills=self.skills,
            education=self.education,
            experience=self.experience,
            projects=self.projects,
        )

    def searchable_units(self) -> list[tuple[str, str]]:
        """Return ``(resume_id, text)`` pairs used to locate evidence."""
        units: list[tuple[str, str]] = [
            ("professional_summary", self.summary),
            ("skills", ", ".join(self.skills)),
        ]
        units.extend(self.bullets.items())
        return units


def snapshot_from_resume(resume: ResumeProfile) -> ResumeSnapshot:
    """Build a snapshot of the untouched master resume."""
    return ResumeSnapshot(
        name=resume.name,
        email=resume.email,
        phone=resume.phone,
        location=resume.location,
        summary=resume.professional_summary,
        skills=list(resume.skills),
        bullets=resume.bullet_index(),
        education=list(resume.education),
        experience=[
            {
                "title": entry.title,
                "organization": entry.organization,
                "dates": entry.dates,
                "bullets": [bullet.text for bullet in entry.bullets],
            }
            for entry in resume.experience
        ],
        projects=[
            {
                "name": project.get("name", ""),
                "bullets": [b.get("text", "") for b in project.get("bullets", []) or []],
            }
            for project in resume.projects
        ],
        document_features=dict(resume.document_features),
    )


def snapshot_with_patch(
    resume: ResumeProfile, application: TailoredApplication
) -> ResumeSnapshot:
    """Build a snapshot of the proposed resume from the drafted patch.

    Only the summary, the two revised bullets, and skill ordering change. No
    new section, employer, or credential is introduced.
    """
    replacements = {
        bullet.source_bullet_id: bullet.revised_text
        for bullet in application.revised_bullets
    }
    bullets = dict(resume.bullet_index())
    for bullet_id, text in replacements.items():
        if bullet_id in bullets:
            bullets[bullet_id] = text

    experience = []
    for entry in resume.experience:
        experience.append(
            {
                "title": entry.title,
                "organization": entry.organization,
                "dates": entry.dates,
                "bullets": [
                    replacements.get(bullet.id, bullet.text) for bullet in entry.bullets
                ],
            }
        )
    projects = []
    for project in resume.projects:
        projects.append(
            {
                "name": project.get("name", ""),
                "bullets": [
                    replacements.get(b.get("id", ""), b.get("text", ""))
                    for b in project.get("bullets", []) or []
                ],
            }
        )

    skills = application.reordered_skills or list(resume.skills)
    return ResumeSnapshot(
        name=resume.name,
        email=resume.email,
        phone=resume.phone,
        location=resume.location,
        summary=application.revised_summary or resume.professional_summary,
        skills=list(skills),
        bullets=bullets,
        education=list(resume.education),
        experience=experience,
        projects=projects,
        document_features=dict(resume.document_features),
    )


# --------------------------------------------------------------------------
# Keyword analysis
# --------------------------------------------------------------------------


def required_keywords(job: JobPosting) -> list[str]:
    """Return the canonical keywords the rubric checks for one posting.

    Explicit required skills plus recognised skill terms that appear in the
    job description text.
    """
    keywords = canonical_skill_set(job.required_skills)
    keywords |= canonical_skill_set(skills_from_text(job.description))
    return sorted(keywords)


def find_evidence(keyword: str, snapshot: ResumeSnapshot) -> list[str]:
    """Return resume IDs that evidence a keyword.

    A keyword counts as evidenced when the resume names it directly, or when a
    fixed concept map ties it to wording already in the resume — for example
    Tableau dashboard work evidencing "data visualization". This is the test
    for whether a missing keyword is *safe to surface*.
    """
    canonical = canonical_skill(keyword)
    terms = {canonical, *CONCEPT_EVIDENCE.get(canonical, ())}
    evidence: list[str] = []
    for resume_id, text in snapshot.searchable_units():
        lowered = (text or "").lower()
        if any(term and term in lowered for term in terms):
            evidence.append(resume_id)
    return evidence


def find_direct_evidence(keyword: str, snapshot: ResumeSnapshot) -> list[str]:
    """Return resume IDs that name a keyword outright.

    Stricter than :func:`find_evidence`: no concept map. Used for the evidence
    component, where the question is whether the resume actually says the thing
    rather than whether a reader could infer it.
    """
    canonical = canonical_skill(keyword)
    pattern = r"(?<![a-z0-9])" + re.escape(canonical) + r"(?![a-z0-9])"
    return [
        resume_id
        for resume_id, text in snapshot.searchable_units()
        if re.search(pattern, (text or "").lower())
    ]


def classify_keywords(
    job: JobPosting, snapshot: ResumeSnapshot
) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
    """Split job keywords into matched, supported-but-missing, and gaps.

    Returns ``(matched, supported_but_missing, unsupported_gaps, evidence)``.
    A supported-but-missing keyword is safe to surface because the resume
    already proves the underlying work. An unsupported gap must stay a gap.
    """
    keywords = required_keywords(job)
    resume_skills = canonical_skill_set(snapshot.skills)
    resume_text = snapshot.plain_text.lower()

    matched: list[str] = []
    supported_missing: list[str] = []
    gaps: list[str] = []
    evidence_map: dict[str, list[str]] = {}

    for keyword in keywords:
        pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
        in_text = bool(re.search(pattern, resume_text))
        if in_text or skill_is_matched(keyword, resume_skills):
            matched.append(keyword)
            evidence_map[keyword] = find_evidence(keyword, snapshot)
            continue
        evidence = find_evidence(keyword, snapshot)
        if evidence:
            supported_missing.append(keyword)
            evidence_map[keyword] = evidence
        else:
            gaps.append(keyword)
            evidence_map[keyword] = []
    return matched, supported_missing, gaps, evidence_map


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def score_keyword_coverage(matched: list[str], total: int) -> int:
    """Score keyword coverage from 0 to 100 without rewarding repetition."""
    if total == 0:
        return 50
    return int(round(100 * len(matched) / total))


def score_qualification_alignment(
    job: JobPosting, resume: ResumeProfile, now: datetime | None = None
) -> tuple[int, list[str]]:
    """Score explicit required qualifications from 0 to 100.

    Unknown requirements are not failures, and no degree, certification, or
    year of experience is ever fabricated.
    """
    notes: list[str] = []
    score = 100

    if job.minimum_experience_years is not None:
        years = estimate_experience_years(resume, now)
        if years < job.minimum_experience_years:
            shortfall = job.minimum_experience_years - years
            penalty = 25 if shortfall <= 1 else 45
            score -= penalty
            notes.append(
                f"The posting requests {job.minimum_experience_years:g} year(s) of "
                f"experience; the resume evidences about {years:g}."
            )

    requirement = (job.education_requirement or "").lower()
    if requirement:
        needed = _degree_level(requirement)
        held = max(
            (_degree_level(str(entry.get("degree", ""))) for entry in resume.education),
            default=0,
        )
        if needed and held and held < needed:
            score -= 30
            notes.append(
                "The posting requests a higher degree level than the resume shows."
            )
        elif needed and not held:
            score -= 20
            notes.append("The resume does not state a degree the posting requires.")

    return int(max(0, min(100, score))), notes


def _degree_level(text: str) -> int:
    """Return a comparable degree level, or 0 when none is stated."""
    lowered = text.lower()
    best = 0
    for name, level in DEGREE_LEVELS.items():
        if re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", lowered):
            best = max(best, level)
    return best


CONTEXT_MARKERS: tuple[str, ...] = (
    " for ",
    " to ",
    " across ",
    " using ",
    " with ",
    " that ",
    " which ",
    " so ",
    " into ",
)


def score_evidence_quality(
    matched: list[str], snapshot: ResumeSnapshot, evidence_map: dict[str, list[str]]
) -> int:
    """Score how well matched skills are backed by specific bullets.

    Two halves, both drawn from the rubric:

    * 60% — matched skills named inside an experience or project bullet rather
      than only listed in the skills section.
    * 40% — bullet specificity: an action verb, enough words to carry a task,
      and a context clause. Numeric metrics are never required, and inventing
      one is never recommended.
    """
    bullet_ids = set(snapshot.bullets)
    if matched:
        proven = sum(
            1
            for keyword in matched
            if set(find_direct_evidence(keyword, snapshot)) & bullet_ids
        )
        evidence_ratio = proven / len(matched)
    else:
        evidence_ratio = 0.0

    texts = list(snapshot.bullets.values())
    if texts:
        specificity = sum(bullet_specificity(text) for text in texts) / len(texts)
    else:
        specificity = 0.0

    return int(round(100 * (0.6 * evidence_ratio + 0.4 * specificity)))


def bullet_specificity(text: str) -> float:
    """Score one bullet's action + task + context wording between 0 and 1."""
    if not text or not text.strip():
        return 0.0
    words = text.split()
    padded = " " + " ".join(words).lower() + " "
    score = 0.0
    if _starts_with_action_verb(text):
        score += 1 / 3
    if len(words) >= 10:
        score += 1 / 3
    if len(words) >= 8 and any(marker in padded for marker in CONTEXT_MARKERS):
        score += 1 / 3
    return round(score, 4)


def _starts_with_action_verb(text: str) -> bool:
    """True when a bullet opens with a recognised action verb."""
    first = re.sub(r"[^a-z]", "", (text or "").strip().lower().split(" ")[0] if text else "")
    return first in ACTION_VERBS


def score_section_completeness(snapshot: ResumeSnapshot) -> tuple[int, list[str]]:
    """Score standard section presence from 0 to 100.

    Only the five standard sections are required; Projects is optional and its
    absence is never penalised.
    """
    checks = {
        "Contact information": bool(snapshot.name and snapshot.email),
        "Professional summary": bool(snapshot.summary.strip()),
        "Skills": bool(snapshot.skills),
        "Education": bool(snapshot.education),
        "Experience": bool(snapshot.experience),
    }
    missing = [name for name, present in checks.items() if not present]
    score = int(round(100 * sum(checks.values()) / len(checks)))
    return score, missing


def score_structure_parseability(snapshot: ResumeSnapshot) -> tuple[int, list[str]]:
    """Score simulated ATS-safe structure from the document features.

    The demo resume is structured data rather than an uploaded file, so this
    component simulates parseability rather than measuring a real document.
    """
    features = snapshot.document_features or {}
    checks = {
        "text parses cleanly": bool(features.get("parse_success", True)),
        "standard headings": bool(features.get("standard_headings", True)),
        "single-column layout": bool(features.get("single_column", True)),
        "no layout tables": not features.get("uses_tables_for_layout", False),
        "no text boxes": not features.get("uses_text_boxes", False),
        "no images holding critical text": not features.get(
            "uses_images_for_critical_text", False
        ),
    }
    issues = [name for name, ok in checks.items() if not ok]
    score = int(round(100 * sum(checks.values()) / len(checks)))
    return score, issues


def score_contact_completeness(snapshot: ResumeSnapshot) -> tuple[int, list[str]]:
    """Score basic contact essentials only; no demographic data is scored."""
    checks = {
        "name": bool(snapshot.name.strip()),
        "email": bool(snapshot.email.strip()),
        "phone": bool((snapshot.phone or "").strip()),
        "location": bool(snapshot.location.strip()),
    }
    missing = [name for name, present in checks.items() if not present]
    return int(round(100 * sum(checks.values()) / len(checks))), missing


def band_for_score(score: int) -> AtsBand:
    """Map a total score onto its neutral score band."""
    if score >= 80:
        return "strong"
    if score >= 65:
        return "needs_targeted_changes"
    return "low"


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------


def assess_ats(
    job: JobPosting,
    resume: ResumeProfile,
    *,
    snapshot: ResumeSnapshot | None = None,
    resume_version: Literal["original", "proposed"] = "original",
    now: datetime | None = None,
    threshold: int = 80,
    include_recommendations: bool = True,
) -> AtsAssessment:
    """Run the full six-component rubric for one resume version."""
    view = snapshot or snapshot_from_resume(resume)
    matched, supported_missing, gaps, evidence_map = classify_keywords(job, view)
    total_keywords = len(matched) + len(supported_missing) + len(gaps)

    keyword_score = score_keyword_coverage(matched, total_keywords)
    qualification_score, qualification_notes = score_qualification_alignment(
        job, resume, now
    )
    evidence_score = score_evidence_quality(matched, view, evidence_map)
    section_score, missing_sections = score_section_completeness(view)
    structure_score, structure_issues = score_structure_parseability(view)
    contact_score, missing_contact = score_contact_completeness(view)

    total = int(
        round(
            0.40 * keyword_score
            + 0.20 * qualification_score
            + 0.15 * evidence_score
            + 0.10 * section_score
            + 0.10 * structure_score
            + 0.05 * contact_score
        )
    )
    total = max(0, min(100, total))

    recommendations: list[AtsRecommendation] = []
    if include_recommendations:
        recommendations = build_recommendations(
            total_score=total,
            job=job,
            snapshot=view,
            supported_missing=supported_missing,
            gaps=gaps,
            evidence_map=evidence_map,
            qualification_notes=qualification_notes,
            missing_sections=missing_sections,
            structure_issues=structure_issues,
            missing_contact=missing_contact,
            threshold=threshold,
        )

    return AtsAssessment(
        job_id=job.job_id,
        resume_version=resume_version,
        total_score=total,
        band=band_for_score(total),
        keyword_score=keyword_score,
        qualification_score=qualification_score,
        evidence_score=evidence_score,
        section_score=section_score,
        structure_score=structure_score,
        contact_score=contact_score,
        matched_required_keywords=matched,
        supported_but_missing_keywords=supported_missing,
        unsupported_job_gaps=gaps,
        recommendations=recommendations,
        disclaimer=ATS_DISCLAIMER,
    )


def build_recommendations(
    *,
    total_score: int,
    job: JobPosting,
    snapshot: ResumeSnapshot,
    supported_missing: list[str],
    gaps: list[str],
    evidence_map: dict[str, list[str]],
    qualification_notes: list[str],
    missing_sections: list[str],
    structure_issues: list[str],
    missing_contact: list[str],
    threshold: int = 80,
) -> list[AtsRecommendation]:
    """Build prioritized, section-specific change proposals.

    Safe recommendations are backed by resume evidence IDs. Unsupported job
    gaps are returned as ``safe_to_apply=False`` so they can be shown and
    refused rather than quietly added.
    """
    candidates: list[AtsRecommendation] = []
    counter = 1

    explicit_required = {
        canonical_skill(skill) for skill in job.required_skills
    }
    for keyword in supported_missing:
        evidence = evidence_map.get(keyword, [])
        # Target the bullet that already proves the concept. The skills list is
        # reorder-only, so a keyword is added where its evidence lives.
        anchor = next(
            (ref for ref in evidence if ref in snapshot.bullets),
            "professional_summary",
        )
        anchor_text = snapshot.bullets.get(anchor, snapshot.summary)
        section = (
            f"Experience bullet {anchor}"
            if anchor in snapshot.bullets
            else "Professional summary"
        )
        candidates.append(
            AtsRecommendation(
                recommendation_id=f"ats_rec_{counter:02d}",
                priority="high" if keyword in explicit_required else "medium",
                category="keyword_alignment",
                target_section=section,
                current_text=anchor_text,
                recommended_change=(
                    f"State “{keyword}” in this bullet, where the resume already "
                    "evidences the work. Do not add it to the skills list."
                ),
                reason=(
                    f"The selected job uses “{keyword}”, and the resume already "
                    "proves this work."
                ),
                evidence_resume_ids=evidence,
                safe_to_apply=True,
                projected_effect="high",
            )
        )
        counter += 1

    for gap in gaps:
        candidates.append(
            AtsRecommendation(
                recommendation_id=f"ats_rec_{counter:02d}",
                priority="high",
                category="unsupported_gap",
                target_section="Missing qualification",
                current_text=None,
                recommended_change=(
                    f"Do not add {gap}. Keep it in the learning-gap list."
                ),
                reason=(
                    f"The job requests {gap}, but the master resume contains no "
                    f"{gap} evidence."
                ),
                evidence_resume_ids=[],
                safe_to_apply=False,
                projected_effect="high",
            )
        )
        counter += 1

    title_terms = [
        term
        for term in re.sub(r"[^a-z ]", " ", job.title.lower()).split()
        if len(term) > 3
    ]
    summary_lower = snapshot.summary.lower()
    if title_terms and not all(term in summary_lower for term in title_terms[:2]):
        candidates.append(
            AtsRecommendation(
                recommendation_id=f"ats_rec_{counter:02d}",
                priority="medium",
                category="summary",
                target_section="Professional summary",
                current_text=snapshot.summary,
                recommended_change=(
                    f"Open the summary with the target role wording “{job.title}” "
                    "while keeping every existing fact unchanged."
                ),
                reason=(
                    "Resume screens weight the summary heavily, and the posting "
                    f"uses the title “{job.title}”."
                ),
                evidence_resume_ids=["professional_summary"],
                safe_to_apply=True,
                projected_effect="medium",
            )
        )
        counter += 1

    weak_bullets = [
        (bullet_id, text)
        for bullet_id, text in snapshot.bullets.items()
        if not _starts_with_action_verb(text) or len(text.split()) < 8
    ]
    for bullet_id, text in weak_bullets[:2]:
        candidates.append(
            AtsRecommendation(
                recommendation_id=f"ats_rec_{counter:02d}",
                priority="medium",
                category="experience",
                target_section=f"Experience bullet {bullet_id}",
                current_text=text,
                recommended_change=(
                    "Rewrite this bullet as action + task + context using the "
                    "terminology in the posting. Add no new facts and no invented "
                    "numbers."
                ),
                reason=(
                    "Short bullets give a screen little to match; the same true "
                    "work can be stated with more specific, job-relevant wording."
                ),
                evidence_resume_ids=[bullet_id],
                safe_to_apply=True,
                projected_effect="medium",
            )
        )
        counter += 1

    for note in qualification_notes:
        candidates.append(
            AtsRecommendation(
                recommendation_id=f"ats_rec_{counter:02d}",
                priority="medium",
                category="unsupported_gap",
                target_section="Qualifications",
                current_text=None,
                recommended_change=(
                    "Do not change the resume for this. Note it as a gap when "
                    "deciding whether to apply."
                ),
                reason=note,
                evidence_resume_ids=[],
                safe_to_apply=False,
                projected_effect="low",
            )
        )
        counter += 1

    for section in missing_sections:
        candidates.append(
            AtsRecommendation(
                recommendation_id=f"ats_rec_{counter:02d}",
                priority="high",
                category="section_completeness",
                target_section=section,
                current_text=None,
                recommended_change=f"Add a clearly headed {section} section.",
                reason="Standard headings help a parser assign content correctly.",
                evidence_resume_ids=[],
                safe_to_apply=True,
                projected_effect="high",
            )
        )
        counter += 1

    for issue in structure_issues:
        candidates.append(
            AtsRecommendation(
                recommendation_id=f"ats_rec_{counter:02d}",
                priority="medium",
                category="format_and_parseability",
                target_section="Document structure",
                current_text=None,
                recommended_change=f"Fix the document so it has {issue}.",
                reason="Simulated parseability check on the document features.",
                evidence_resume_ids=[],
                safe_to_apply=True,
                projected_effect="medium",
            )
        )
        counter += 1

    for item in missing_contact:
        candidates.append(
            AtsRecommendation(
                recommendation_id=f"ats_rec_{counter:02d}",
                priority="low",
                category="section_completeness",
                target_section="Contact information",
                current_text=None,
                recommended_change=f"Add a {item} to the contact block.",
                reason="Basic contact essentials should be machine-readable.",
                evidence_resume_ids=[],
                safe_to_apply=True,
                projected_effect="low",
            )
        )
        counter += 1

    # The panel answers "what to change first", so a change the candidate can
    # actually make outranks a "do not add this" note of the same priority.
    candidates.sort(
        key=lambda rec: (
            0 if rec.safe_to_apply else 1,
            PRIORITY_ORDER[rec.priority],
            PRIORITY_ORDER[rec.projected_effect],
            rec.recommendation_id,
        )
    )

    if total_score >= threshold:
        limit = MAX_RECOMMENDATIONS_STRONG
    elif total_score >= 65:
        limit = MAX_RECOMMENDATIONS_MID
    else:
        limit = MAX_RECOMMENDATIONS_LOW

    selected = candidates[:limit]
    selected = _guarantee_present(
        selected, candidates, lambda rec: not rec.safe_to_apply
    )
    selected = _guarantee_present(selected, candidates, lambda rec: rec.safe_to_apply)
    return selected


def _guarantee_present(
    selected: list[AtsRecommendation],
    candidates: list[AtsRecommendation],
    predicate: Any,
) -> list[AtsRecommendation]:
    """Ensure the capped panel keeps at least one recommendation of a kind.

    The display cap must never hide the unsupported-gap guardrail, and it must
    never leave the panel with nothing the candidate can act on.
    """
    if any(predicate(rec) for rec in selected):
        return selected
    replacement = next((rec for rec in candidates if predicate(rec)), None)
    if replacement is None:
        return selected
    if not selected:
        return [replacement]
    selected[-1] = replacement
    return selected


def safe_recommendations(assessment: AtsAssessment) -> list[AtsRecommendation]:
    """Return only the recommendations Gemini is permitted to apply."""
    return [rec for rec in assessment.recommendations if rec.safe_to_apply]


def unsupported_gap_names(assessment: AtsAssessment) -> list[str]:
    """Return the job requirements that must never be added to the resume."""
    return list(assessment.unsupported_job_gaps)


def rescore_proposed(
    job: JobPosting,
    resume: ResumeProfile,
    application: TailoredApplication,
    *,
    now: datetime | None = None,
    threshold: int = 80,
) -> AtsAssessment:
    """Re-score the proposed resume under the identical rubric."""
    snapshot = snapshot_with_patch(resume, application)
    return assess_ats(
        job,
        resume,
        snapshot=snapshot,
        resume_version="proposed",
        now=now,
        threshold=threshold,
        include_recommendations=False,
    )
