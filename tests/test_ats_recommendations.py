"""Prioritized “what to change first” recommendations and their guardrails."""

from __future__ import annotations

from models.ats import PRIORITY_ORDER
from tests.conftest import make_job
from tools.ats_scorer import (
    MAX_RECOMMENDATIONS_LOW,
    MAX_RECOMMENDATIONS_MID,
    MAX_RECOMMENDATIONS_STRONG,
    assess_ats,
    safe_recommendations,
    snapshot_from_resume,
    unsupported_gap_names,
)

POWER_BI_JOB = {
    "required_skills": ["SQL", "Excel", "Power BI", "Data visualization"],
    "preferred_skills": ["Python"],
}


class TestRecommendationCounts:
    """The display cap follows the score band."""

    def test_a_strong_score_shows_at_most_two_refinements(self, resume, now):
        assessment = assess_ats(make_job(), resume, now=now)
        assert assessment.band == "strong"
        assert len(assessment.recommendations) <= MAX_RECOMMENDATIONS_STRONG

    def test_a_mid_band_score_shows_at_most_three(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        if assessment.band == "needs_targeted_changes":
            assert len(assessment.recommendations) <= MAX_RECOMMENDATIONS_MID

    def test_a_low_score_shows_at_most_five(self, resume, now):
        job = make_job(
            required_skills=["Power BI", "Looker", "SAS", "ETL", "Machine learning"],
            education_requirement="PhD required",
            minimum_experience_years=8,
        )
        assessment = assess_ats(job, resume, now=now)
        assert assessment.band == "low"
        assert len(assessment.recommendations) <= MAX_RECOMMENDATIONS_LOW


class TestRecommendationContent:
    """Every recommendation carries the fields the interface must show."""

    def test_required_fields_are_populated(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        for rec in assessment.recommendations:
            assert rec.recommendation_id
            assert rec.priority in PRIORITY_ORDER
            assert rec.target_section
            assert rec.recommended_change
            assert rec.reason
            assert rec.projected_effect in {"high", "medium", "low"}

    def test_safe_recommendations_cite_resume_evidence(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        for rec in safe_recommendations(assessment):
            if rec.category in {"keyword_alignment", "summary", "experience"}:
                assert rec.evidence_resume_ids

    def test_recommendations_are_ordered_actionable_first(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        safety_order = [0 if rec.safe_to_apply else 1 for rec in assessment.recommendations]
        assert safety_order == sorted(safety_order)

    def test_priority_is_ordered_within_the_safe_group(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        priorities = [
            PRIORITY_ORDER[rec.priority]
            for rec in assessment.recommendations
            if rec.safe_to_apply
        ]
        assert priorities == sorted(priorities)


class TestUnsupportedGapGuardrail:
    """A gap is shown, labelled unsafe, and never proposed as a resume claim."""

    def test_power_bi_gap_is_reported(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        assert "power bi" in unsupported_gap_names(assessment)

    def test_gap_recommendation_is_marked_unsafe(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        gap_recs = [
            rec for rec in assessment.recommendations if rec.category == "unsupported_gap"
        ]
        assert gap_recs
        for rec in gap_recs:
            assert rec.safe_to_apply is False
            assert rec.evidence_resume_ids == []
            assert "do not add" in rec.recommended_change.lower()

    def test_a_gap_survives_the_display_cap(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        assert any(
            rec.category == "unsupported_gap" for rec in assessment.recommendations
        )

    def test_a_safe_change_survives_the_display_cap(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        assert any(rec.safe_to_apply for rec in assessment.recommendations)

    def test_gaps_are_never_offered_to_the_drafting_step(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        offered = " ".join(rec.recommended_change for rec in safe_recommendations(assessment))
        assert "power bi" not in offered.lower()

    def test_safe_keyword_changes_never_target_the_skills_list(self, resume, now):
        """The skills list is reorder-only, so keywords are added where the
        evidence lives instead."""
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        for rec in safe_recommendations(assessment):
            if rec.category == "keyword_alignment":
                assert rec.target_section != "Skills"
                assert rec.evidence_resume_ids

    def test_supported_keyword_recommendations_point_at_real_resume_ids(
        self, resume, now
    ):
        snapshot = snapshot_from_resume(resume)
        valid_ids = {"professional_summary", "skills", *snapshot.bullets}
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        for rec in assessment.recommendations:
            for resume_id in rec.evidence_resume_ids:
                assert resume_id in valid_ids


class TestNoUnsafeAdvice:
    """The rubric never suggests stuffing, hiding text, or inventing facts."""

    def test_no_recommendation_suggests_a_forbidden_tactic(self, resume, now):
        forbidden = (
            "white text",
            "hidden",
            "keyword stuffing",
            "copy the entire job description",
            "invent",
            "make up",
        )
        for job in (make_job(), make_job(**POWER_BI_JOB)):
            assessment = assess_ats(job, resume, now=now)
            text = " ".join(
                f"{rec.recommended_change} {rec.reason}"
                for rec in assessment.recommendations
            ).lower()
            for phrase in forbidden:
                assert phrase not in text

    def test_no_recommendation_asks_for_a_numeric_metric(self, resume, now):
        assessment = assess_ats(make_job(**POWER_BI_JOB), resume, now=now)
        text = " ".join(rec.recommended_change for rec in assessment.recommendations)
        assert "no invented numbers" in text.lower() or "%" not in text
