"""What the exported package says about seniority, and what it refuses to say.

The export is the record of the run, so it carries the same separation the
screen does: the requested level shaped the search, the detected level came off
the page, and a posting that never stated one is exported as "Level not
stated" — never as the level the user happened to ask for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from models.application import RevisedBullet, TailoredApplication
from models.ats import AtsAssessment
from models.job import JobPosting
from models.match import MatchResult
from models.resume import ResumeProfile
from models.validation import ValidationReport
from tests.conftest import FIXED_NOW, make_job
from tools.exporter import (
    EXPERIENCE_LEVEL_NOTICE,
    build_json,
    build_markdown,
    export_package,
)


def make_match(job_id: str) -> MatchResult:
    """Build a calculated Demo Job Match Score for one posting."""
    return MatchResult(
        job_id=job_id,
        total_score=82,
        skill_score=80,
        similarity_score=61,
        role_score=95,
        experience_score=100,
        preference_score=100,
        matched_skills=["python", "sql", "excel"],
        missing_skills=["power bi"],
        concerns=[],
    )


def make_ats(
    job_id: str,
    *,
    resume_version: str = "original",
    total_score: int = 62,
    band: str = "low",
) -> AtsAssessment:
    """Build a six-component ATS assessment for one posting."""
    return AtsAssessment(
        job_id=job_id,
        resume_version=resume_version,
        total_score=total_score,
        band=band,
        keyword_score=55,
        qualification_score=70,
        evidence_score=60,
        section_score=100,
        structure_score=100,
        contact_score=100,
        matched_required_keywords=["sql", "excel"],
        supported_but_missing_keywords=["data visualization"],
        unsupported_job_gaps=["power bi"],
    )


def make_application(job_id: str, resume: ResumeProfile) -> TailoredApplication:
    """Build a small tailored package grounded in the master resume."""
    bullet = resume.experience[0].bullets[0]
    return TailoredApplication(
        job_id=job_id,
        revised_summary="Information systems student who analyzes survey data.",
        revised_bullets=[
            RevisedBullet(
                source_bullet_id=bullet.id,
                original_text=bullet.text,
                revised_text="Analyzed survey data using Python and Excel.",
            )
        ],
        reordered_skills=list(resume.skills),
        keywords_used=["sql"],
        applied_ats_recommendation_ids=["ats_rec_1"],
        unsupported_ats_gaps_not_applied=["power bi"],
        missing_requirements=["Power BI"],
        cover_letter="Dear hiring team, I would like to apply for this role.",
    )


def make_validation() -> ValidationReport:
    """Build a passing validation report."""
    return ValidationReport(
        valid_source_ids=True,
        passed=True,
        deterministic_checks=[
            {"name": "Source IDs verified", "detail": "1 bullet checked", "passed": True}
        ],
    )


def package(
    resume: ResumeProfile, job: JobPosting | None = None, **overrides: Any
) -> dict[str, Any]:
    """Build the keyword arguments every export entry point takes."""
    job = job or make_job()
    payload: dict[str, Any] = {
        "resume": resume,
        "job": job,
        "match": make_match(job.job_id),
        "original_ats": make_ats(job.job_id),
        "projected_ats": make_ats(
            job.job_id,
            resume_version="proposed",
            total_score=76,
            band="needs_targeted_changes",
        ),
        "application": make_application(job.job_id, resume),
        "validation": make_validation(),
        "approved_at": FIXED_NOW,
        "data_mode": "live",
    }
    payload.update(overrides)
    return payload


def field_line(markdown: str, label: str) -> str:
    """Return the value of exactly one labelled Markdown bullet."""
    prefix = f"- **{label}:** "
    matches = [line for line in markdown.splitlines() if line.startswith(prefix)]
    assert len(matches) == 1, f"expected one {label!r} line, found {matches}"
    return matches[0][len(prefix) :]


class TestMarkdownExperienceLevel:
    """The Markdown package records both levels and the evidence for one."""

    def test_the_requested_level_is_recorded_with_its_label(self, resume):
        job = make_job(requested_experience_level="senior", experience_level="senior")
        markdown = build_markdown(**package(resume, job))
        assert field_line(markdown, "Requested experience level") == "Senior"

    def test_no_level_filter_is_recorded_as_any_level(self, resume):
        job = make_job(requested_experience_level="unknown")
        markdown = build_markdown(**package(resume, job))
        assert field_line(markdown, "Requested experience level") == "Any level"

    def test_the_detected_level_is_the_one_read_off_the_posting(self, resume):
        job = make_job(
            requested_experience_level="internship",
            experience_level="internship",
            experience_level_evidence='title contains "intern"',
        )
        markdown = build_markdown(**package(resume, job))
        assert field_line(markdown, "Detected experience level") == "Internship"
        assert (
            field_line(markdown, "Experience level evidence")
            == 'title contains "intern"'
        )

    def test_a_detected_level_without_recorded_evidence_still_says_so(self, resume):
        job = make_job(experience_level="senior", experience_level_evidence=None)
        markdown = build_markdown(**package(resume, job))
        assert (
            field_line(markdown, "Experience level evidence") == "stated on the posting"
        )

    def test_a_silent_posting_is_exported_as_level_not_stated(self, resume):
        """The request must never be exported as though it were detected."""
        job = make_job(
            requested_experience_level="senior",
            experience_level="unknown",
            experience_level_evidence=None,
        )
        markdown = build_markdown(**package(resume, job))
        assert field_line(markdown, "Requested experience level") == "Senior"
        assert field_line(markdown, "Detected experience level") == "Level not stated"
        assert (
            field_line(markdown, "Experience level evidence")
            == "the posting does not state a level"
        )

    def test_the_notice_separates_the_search_filter_from_the_evidence(self, resume):
        markdown = build_markdown(**package(resume))
        assert EXPERIENCE_LEVEL_NOTICE in markdown

    def test_a_job_carrying_no_level_fields_exports_unchanged_defaults(self, resume):
        """A run from before levels existed still exports cleanly."""
        markdown = build_markdown(**package(resume, make_job()))
        assert field_line(markdown, "Requested experience level") == "Any level"
        assert field_line(markdown, "Detected experience level") == "Level not stated"

    @pytest.mark.parametrize(
        ("level", "label"),
        [
            ("internship", "Internship"),
            ("entry", "Entry level / Junior"),
            ("mid", "Mid-level"),
            ("senior", "Senior"),
            ("staff_principal", "Staff / Principal / Lead"),
            ("manager", "Manager / Director"),
        ],
    )
    def test_every_real_level_exports_its_human_label(self, resume, level, label):
        job = make_job(requested_experience_level=level, experience_level=level)
        markdown = build_markdown(**package(resume, job))
        assert field_line(markdown, "Requested experience level") == label
        assert field_line(markdown, "Detected experience level") == label


class TestJsonExperienceLevel:
    """The JSON package carries the same pair, plus machine-readable values."""

    def test_the_block_carries_requested_detected_and_evidence(self, resume):
        job = make_job(
            requested_experience_level="internship",
            experience_level="internship",
            experience_level_evidence='title contains "intern"',
        )
        block = build_json(**package(resume, job))["experience_level"]
        assert block["requested"] == "internship"
        assert block["requested_label"] == "Internship"
        assert block["detected"] == "internship"
        assert block["detected_label"] == "Internship"
        assert block["evidence"] == 'title contains "intern"'
        assert block["notice"] == EXPERIENCE_LEVEL_NOTICE

    def test_the_detected_value_is_the_posting_not_the_request(self, resume):
        job = make_job(requested_experience_level="senior", experience_level="unknown")
        block = build_json(**package(resume, job))["experience_level"]
        assert block["requested"] == "senior"
        assert block["detected"] == "unknown"
        assert block["detected_label"] == "Level not stated"
        assert block["evidence"] is None

    def test_no_level_filter_is_recorded_as_any_level(self, resume):
        block = build_json(**package(resume, make_job()))["experience_level"]
        assert block["requested"] == "unknown"
        assert block["requested_label"] == "Any level"

    def test_the_selected_job_dump_keeps_the_raw_level_fields(self, resume):
        job = make_job(
            requested_experience_level="senior",
            experience_level="staff_principal",
            experience_level_evidence='title contains "principal"',
        )
        dumped = build_json(**package(resume, job))["selected_job"]
        assert dumped["requested_experience_level"] == "senior"
        assert dumped["experience_level"] == "staff_principal"
        assert dumped["experience_level_evidence"] == 'title contains "principal"'

    def test_the_payload_stays_json_serializable(self, resume):
        job = make_job(requested_experience_level="mid", experience_level="mid")
        assert json.dumps(build_json(**package(resume, job)))


class TestExportedFilesCarryTheLevel:
    """Both written files record the level, not only the in-memory payload."""

    def test_the_markdown_and_json_files_both_record_the_pair(self, resume, tmp_path):
        job = make_job(requested_experience_level="senior", experience_level="unknown")
        paths = export_package(**package(resume, job), output_dir=tmp_path)

        markdown = Path(paths[0]).read_text(encoding="utf-8")
        assert field_line(markdown, "Requested experience level") == "Senior"
        assert field_line(markdown, "Detected experience level") == "Level not stated"

        payload = json.loads(Path(paths[1]).read_text(encoding="utf-8"))
        assert payload["experience_level"]["requested"] == "senior"
        assert payload["experience_level"]["detected"] == "unknown"
