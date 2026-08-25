"""Tests for pointing the app at a resume other than the shipped sample.

The behaviour under test is mostly about honesty: when a real person's resume is
loaded, nothing in the interface or the export may still describe it as
fictional.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from config import DEFAULT_RESUME_FILE, load_settings
from models.resume import ResumeProfile


@pytest.fixture()
def sample_payload() -> dict:
    """The shipped fictional profile, as raw JSON."""
    path = Path(__file__).resolve().parent.parent / "data" / "sample_resume.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _settings(monkeypatch, tmp_path, **env):
    """Load settings with a clean environment plus the given overrides."""
    for key in ("RESUME_FILE", "GEMINI_API_KEY", "FIRECRAWL_API_KEY", "DEMO_MODE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # Prevent the real .env from bleeding into the test.
    return load_settings(env_file=tmp_path / "absent.env")


def test_defaults_to_the_shipped_sample(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    assert settings.resume_file == DEFAULT_RESUME_FILE
    assert settings.using_custom_resume is False
    assert settings.resume_descriptor == "fictional demonstration profile"


def test_custom_resume_is_detected(monkeypatch, tmp_path, sample_payload):
    custom = tmp_path / "mine.json"
    custom.write_text(json.dumps(sample_payload), encoding="utf-8")

    settings = _settings(monkeypatch, tmp_path, RESUME_FILE=str(custom))

    assert settings.using_custom_resume is True
    assert settings.resume_descriptor == "your own resume"
    assert settings.resume_path == custom


def test_missing_resume_file_falls_back_with_a_warning(monkeypatch, tmp_path):
    settings = _settings(
        monkeypatch, tmp_path, RESUME_FILE=str(tmp_path / "nope.json")
    )

    assert settings.resume_file == DEFAULT_RESUME_FILE
    assert settings.using_custom_resume is False
    assert any("does not exist" in w for w in settings.startup_warnings)


def test_custom_resume_with_a_key_warns_that_text_leaves_the_machine(
    monkeypatch, tmp_path, sample_payload
):
    custom = tmp_path / "mine.json"
    custom.write_text(json.dumps(sample_payload), encoding="utf-8")

    settings = _settings(
        monkeypatch, tmp_path, RESUME_FILE=str(custom), GEMINI_API_KEY="test-key"
    )

    assert any(
        "sent to Google" in w for w in settings.startup_warnings
    ), "a personal resume plus a live key must warn where the text goes"


def test_custom_resume_without_a_key_does_not_warn(
    monkeypatch, tmp_path, sample_payload
):
    custom = tmp_path / "mine.json"
    custom.write_text(json.dumps(sample_payload), encoding="utf-8")

    settings = _settings(monkeypatch, tmp_path, RESUME_FILE=str(custom))

    assert not any("sent to Google" in w for w in settings.startup_warnings)


def test_export_does_not_call_a_personal_resume_fictional(sample_payload):
    """The exported package must not stamp 'fictional' on real data."""
    from models.ats import AtsAssessment
    from models.application import TailoredApplication
    from models.job import JobPosting
    from models.match import MatchResult
    from models.validation import ValidationReport
    from tools.exporter import build_markdown

    resume = ResumeProfile.model_validate(sample_payload)
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    job = JobPosting(
        job_id="j1",
        title="Data Analyst",
        company="Example Co",
        query_category="all",
        source_category="other",
        source_label="Example",
        description="A description long enough to be scored properly.",
        description_excerpt="A description",
        source_url="https://example.com/jobs/1",
        source_domain="example.com",
        freshness_window="last_24_hours",
        retrieved_at=now,
        freshness_status="date_unavailable",
    )
    match = MatchResult(
        job_id="j1",
        total_score=50,
        skill_score=50,
        similarity_score=50,
        role_score=50,
        experience_score=50,
        preference_score=50,
    )
    ats = AtsAssessment(
        job_id="j1",
        resume_version="original",
        total_score=70,
        band="needs_targeted_changes",
        keyword_score=70,
        qualification_score=70,
        evidence_score=70,
        section_score=70,
        structure_score=70,
        contact_score=70,
    )
    application = TailoredApplication(
        job_id="j1", revised_summary="A summary.", cover_letter="A letter."
    )
    validation = ValidationReport(valid_source_ids=True, passed=True)

    markdown = build_markdown(
        resume=resume,
        job=job,
        match=match,
        original_ats=ats,
        projected_ats=None,
        application=application,
        validation=validation,
        approved_at=now,
        data_mode="cached",
        resume_descriptor="your own resume",
    )

    assert "your own resume" in markdown
    assert "fictional" not in markdown.lower()


def test_export_still_labels_the_sample_profile_fictional(sample_payload):
    """The default must keep its honest 'fictional' marking."""
    from tools.exporter import build_markdown

    signature = build_markdown.__defaults__ or ()
    kwdefaults = build_markdown.__kwdefaults__ or {}
    assert kwdefaults.get("resume_descriptor") == "fictional demonstration profile"
