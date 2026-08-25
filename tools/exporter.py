"""Markdown and JSON export of the approved application package.

Export is the only side effect in this project, and it happens only after a
human approves. Nothing is submitted anywhere. API keys, raw prompts, and model
reasoning never appear in an exported file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.application import TailoredApplication
from models.ats import ATS_LONG_DISCLAIMER, BAND_LABELS, AtsAssessment
from models.job import (
    EXPERIENCE_LEVEL_LABELS,
    FRESHNESS_LABELS,
    SOURCE_CATEGORY_LABELS,
    JobPosting,
)
from models.match import MatchResult
from models.resume import ResumeProfile
from models.validation import ValidationReport

NO_SUBMISSION_NOTICE = "No application was submitted."

EXPERIENCE_LEVEL_NOTICE = (
    "The requested level filtered the search. It is not evidence about this "
    "posting: a posting that never states a level is recorded as unknown."
)

FRESHNESS_EVIDENCE_LABELS: dict[str, str] = {
    "exact_timestamp": "Exact timestamp from the source page",
    "date_only": "Date shown; exact time unavailable",
    "search_filter_only": "Search-filtered; source timestamp unavailable",
    "unavailable": "Posting time unavailable",
}

# On the request side "unknown" means the user did not filter by seniority at
# all, which is a different statement from a posting that never stated one.
REQUESTED_LEVEL_LABELS: dict[str, str] = {
    **EXPERIENCE_LEVEL_LABELS,
    "unknown": "Any level",
}


def _timestamp_slug(moment: datetime) -> str:
    """Return a filesystem-safe timestamp for export filenames."""
    return moment.strftime("%Y%m%d_%H%M%S")


def _format_datetime(value: datetime | None) -> str:
    """Format an optional datetime for display."""
    if value is None:
        return "not available"
    return value.strftime("%Y-%m-%d %H:%M:%S %Z").strip() or value.isoformat()


def _experience_level_evidence(job: JobPosting) -> str:
    """Say why a posting carries its detected level, without inventing one.

    The requested level never appears here. Asking for senior roles is not
    evidence that a result is senior, exactly as a freshness filter is not
    evidence of a posting time.
    """
    if job.experience_level == "unknown":
        return "the posting does not state a level"
    return job.experience_level_evidence or "stated on the posting"


def build_markdown(
    *,
    resume: ResumeProfile,
    job: JobPosting,
    match: MatchResult,
    original_ats: AtsAssessment,
    projected_ats: AtsAssessment | None,
    application: TailoredApplication,
    validation: ValidationReport,
    approved_at: datetime,
    data_mode: str,
    resume_descriptor: str = "fictional demonstration profile",
) -> str:
    """Render the human-readable application package."""
    lines: list[str] = []
    add = lines.append

    add("# Application Package")
    add("")
    add(f"**Candidate:** {resume.name} — {resume_descriptor}")
    add(f"**Approved at:** {_format_datetime(approved_at)}")
    add(f"**Data mode:** {'LIVE PUBLIC RESULTS' if data_mode == 'live' else 'CACHED DEMONSTRATION RESULTS'}")
    add("")
    add(f"> {NO_SUBMISSION_NOTICE}")
    add("")

    add("## 1. Selected job")
    add("")
    add(f"- **Title:** {job.title}")
    add(f"- **Company:** {job.company}")
    add(f"- **Location:** {job.location or 'not stated'} · Work mode: {job.work_mode}")
    add(f"- **Selected query category:** {SOURCE_CATEGORY_LABELS.get(job.query_category, job.query_category)}")
    add(f"- **Detected source category:** {SOURCE_CATEGORY_LABELS.get(job.source_category, job.source_category)} ({job.source_label})")
    add(f"- **Requested experience level:** {REQUESTED_LEVEL_LABELS[job.requested_experience_level]}")
    add(f"- **Detected experience level:** {job.experience_level_label()}")
    add(f"- **Experience level evidence:** {_experience_level_evidence(job)}")
    add(f"- **Requested freshness window:** {FRESHNESS_LABELS.get(job.freshness_window, job.freshness_window)}")
    add(f"- **Source URL:** [{job.source_url}]({job.source_url})")
    if job.original_source_url and job.original_source_url != job.source_url:
        add(f"- **Original result URL:** {job.original_source_url}")
    if job.apply_url:
        add(f"- **Apply URL published on the page:** [{job.apply_url}]({job.apply_url})")
    add("")
    add(f"_{EXPERIENCE_LEVEL_NOTICE}_")
    add("")

    add("## 2. Retrieval and posting evidence")
    add("")
    add(f"- **Retrieved at:** {_format_datetime(job.retrieved_at)}")
    add(f"- **Posting timestamp:** {_format_datetime(job.posted_at)}")
    add(f"- **Posting date:** {job.posting_date.isoformat() if job.posting_date else 'not available'}")
    age = f"{job.posting_age_hours:.1f} hours" if job.posting_age_hours is not None else "not calculable"
    add(f"- **Posting age:** {age}")
    add(f"- **Freshness evidence:** {FRESHNESS_EVIDENCE_LABELS[job.freshness_evidence]}")
    add(f"- **Freshness status:** {job.freshness_status.replace('_', ' ')}")
    add("")

    add("## 3. Job description as retrieved")
    add("")
    add(job.description)
    add("")

    add("## 4. Demo Job Match Score")
    add("")
    add(f"**{match.total_score}/100** (calculated in Python)")
    add("")
    add("| Component | Score |")
    add("| --- | --- |")
    add(f"| Required-skill coverage (45%) | {match.skill_score}/100 |")
    add(f"| Resume/job text similarity (20%) | {match.similarity_score}/100 |")
    add(f"| Role-title alignment (15%) | {match.role_score}/100 |")
    add(f"| Experience alignment (10%) | {match.experience_score}/100 |")
    add(f"| Location and work-mode alignment (10%) | {match.preference_score}/100 |")
    add("")
    add(f"- **Matched skills:** {', '.join(match.matched_skills) or 'none'}")
    add(f"- **Missing or unclear:** {', '.join(match.missing_skills) or 'none'}")
    if match.concerns:
        add(f"- **Concerns:** {'; '.join(match.concerns)}")
    if match.explanation:
        add("")
        add(f"_{match.explanation}_")
    add("")

    add("## 5. Demo ATS Readiness Score — original resume")
    add("")
    add(f"**{original_ats.total_score}/100 — {BAND_LABELS[original_ats.band]}**")
    add("")
    add(f"> {ATS_LONG_DISCLAIMER}")
    add("")
    add(_ats_component_table(original_ats))
    add("")
    add(f"- **Matched required keywords:** {', '.join(original_ats.matched_required_keywords) or 'none'}")
    add(f"- **Supported but missing keywords:** {', '.join(original_ats.supported_but_missing_keywords) or 'none'}")
    add(f"- **Unsupported job gaps:** {', '.join(original_ats.unsupported_job_gaps) or 'none'}")
    add("")

    add("## 6. Prioritized ATS recommendations")
    add("")
    if not original_ats.recommendations:
        add("No recommendations were generated.")
    for rec in original_ats.recommendations:
        safe = "Yes" if rec.safe_to_apply else "No"
        add(f"### {rec.priority.upper()} · {rec.target_section} ({rec.recommendation_id})")
        add("")
        if rec.current_text:
            add(f"- **Current text:** {rec.current_text}")
        add(f"- **Recommended change:** {rec.recommended_change}")
        add(f"- **Why it matters:** {rec.reason}")
        add(f"- **Supporting resume IDs:** {', '.join(rec.evidence_resume_ids) or 'none'}")
        add(f"- **Safe to apply:** {safe}")
        add(f"- **Expected effect:** {rec.projected_effect.title()}")
        add("")

    add("## 7. Revised professional summary")
    add("")
    add(f"- **Original:** {resume.professional_summary}")
    add(f"- **Revised:** {application.revised_summary}")
    add("")

    add("## 8. Revised experience bullets")
    add("")
    for bullet in application.revised_bullets:
        add(f"- **Source ID:** `{bullet.source_bullet_id}`")
        add(f"  - Original: {bullet.original_text}")
        add(f"  - Revised: {bullet.revised_text}")
    add("")
    add(f"**Applied recommendation IDs:** {', '.join(application.applied_ats_recommendation_ids) or 'none'}")
    add("")

    add("## 9. Reordered skills")
    add("")
    add(", ".join(application.reordered_skills) or "unchanged")
    add("")

    add("## 10. Missing requirements and unsupported gaps")
    add("")
    add(f"- **Missing requirements:** {', '.join(application.missing_requirements) or 'none'}")
    add(f"- **Unsupported gaps deliberately not applied:** {', '.join(application.unsupported_ats_gaps_not_applied) or 'none'}")
    add("")

    add("## 11. Projected Demo ATS Readiness Score — proposed resume")
    add("")
    if projected_ats is None:
        add("A projected score was not calculated.")
    else:
        add(
            f"- Original estimated ATS readiness: {original_ats.total_score}/100 — "
            f"{BAND_LABELS[original_ats.band]}"
        )
        add(
            f"- Projected after proposed safe changes: {projected_ats.total_score}/100 — "
            f"{BAND_LABELS[projected_ats.band]}"
        )
        add("")
        add(_ats_component_table(projected_ats))
        add("")
        add("Projected under the same demo rubric. It is not a promise that a real ATS score will improve.")
        if projected_ats.total_score < original_ats.total_score:
            add("")
            add("**Warning:** the projected score is lower than the original score; these changes are not an improvement under this rubric.")
    add("")

    add("## 12. Cover letter")
    add("")
    add(application.cover_letter)
    add("")

    add("## 13. Validation report")
    add("")
    add(f"- **Passed:** {'yes' if validation.passed else 'no'}")
    add(f"- **Source IDs verified:** {'yes' if validation.valid_source_ids else 'no'}")
    add("")
    for check in validation.deterministic_checks:
        mark = "✓" if check.get("passed") else "✗"
        add(f"- {mark} {check.get('name')} — {check.get('detail')}")
    if validation.unsupported_claims:
        add("")
        add("**Unsupported claims:**")
        for review in validation.unsupported_claims:
            add(f"- {review.claim} — {review.reason}")
    if validation.unclear_claims:
        add("")
        add("**Unclear claims:**")
        for review in validation.unclear_claims:
            add(f"- {review.claim} — {review.reason}")
    add("")

    add("## 14. Approval")
    add("")
    add(f"- **Approved at:** {_format_datetime(approved_at)}")
    add(f"- **{NO_SUBMISSION_NOTICE}**")
    add("")

    return "\n".join(lines).rstrip() + "\n"


def _ats_component_table(assessment: AtsAssessment) -> str:
    """Render the six-component ATS breakdown as a Markdown table."""
    rows = [
        ("Required keyword and skill coverage (40%)", assessment.keyword_score),
        ("Required qualification alignment (20%)", assessment.qualification_score),
        ("Evidence and specificity (15%)", assessment.evidence_score),
        ("Standard section completeness (10%)", assessment.section_score),
        ("ATS-safe structure and parseability (10%)", assessment.structure_score),
        ("Contact and application essentials (5%)", assessment.contact_score),
    ]
    lines = ["| Component | Score |", "| --- | --- |"]
    lines.extend(f"| {name} | {score}/100 |" for name, score in rows)
    return "\n".join(lines)


def build_json(
    *,
    resume: ResumeProfile,
    job: JobPosting,
    match: MatchResult,
    original_ats: AtsAssessment,
    projected_ats: AtsAssessment | None,
    application: TailoredApplication,
    validation: ValidationReport,
    approved_at: datetime,
    data_mode: str,
    resume_descriptor: str = "fictional demonstration profile",
) -> dict[str, Any]:
    """Build the machine-readable application package."""
    return {
        "package_version": "1.0",
        "candidate": {
            "name": resume.name,
            "candidate_id": resume.candidate_id,
            "notice": f"{resume_descriptor[:1].upper()}{resume_descriptor[1:]}.",
        },
        "data_mode": data_mode,
        "approved_at": approved_at.isoformat(),
        "no_submission_notice": NO_SUBMISSION_NOTICE,
        "selected_job": job.model_dump(mode="json"),
        "experience_level": {
            "requested": job.requested_experience_level,
            "requested_label": REQUESTED_LEVEL_LABELS[job.requested_experience_level],
            "detected": job.experience_level,
            "detected_label": job.experience_level_label(),
            "evidence": job.experience_level_evidence,
            "notice": EXPERIENCE_LEVEL_NOTICE,
        },
        "job_match": match.model_dump(mode="json"),
        "ats_original": original_ats.model_dump(mode="json"),
        "ats_projected": projected_ats.model_dump(mode="json") if projected_ats else None,
        "tailored_application": application.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "disclaimer": ATS_LONG_DISCLAIMER,
    }


def export_package(
    *,
    resume: ResumeProfile,
    job: JobPosting,
    match: MatchResult,
    original_ats: AtsAssessment,
    projected_ats: AtsAssessment | None,
    application: TailoredApplication,
    validation: ValidationReport,
    output_dir: str | Path,
    approved_at: datetime | None = None,
    data_mode: str = "live",
    resume_descriptor: str = "fictional demonstration profile",
) -> list[str]:
    """Write the Markdown and JSON package and return the file paths.

    Raises ``OSError`` when the output directory cannot be written; the caller
    reports that as a concise, actionable message.
    """
    moment = approved_at or datetime.now(timezone.utc)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    slug = _timestamp_slug(moment)
    markdown_path = directory / f"application_package_{slug}.md"
    json_path = directory / f"application_package_{slug}.json"

    payload = {
        "resume": resume,
        "job": job,
        "match": match,
        "original_ats": original_ats,
        "projected_ats": projected_ats,
        "application": application,
        "validation": validation,
        "approved_at": moment,
        "data_mode": data_mode,
        "resume_descriptor": resume_descriptor,
    }

    markdown_path.write_text(build_markdown(**payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(build_json(**payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return [str(markdown_path), str(json_path)]
