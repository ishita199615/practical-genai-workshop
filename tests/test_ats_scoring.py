"""Deterministic Demo ATS Readiness Score and projected re-score."""

from __future__ import annotations

import pytest

from models.application import RevisedBullet, TailoredApplication
from models.ats import ATS_DISCLAIMER
from tests.conftest import make_job
from tools.ats_scorer import (
    assess_ats,
    band_for_score,
    bullet_specificity,
    classify_keywords,
    find_direct_evidence,
    find_evidence,
    rescore_proposed,
    score_contact_completeness,
    score_keyword_coverage,
    score_qualification_alignment,
    score_section_completeness,
    score_structure_parseability,
    snapshot_from_resume,
    snapshot_with_patch,
)


class TestScoreBands:
    """Bands are neutral and use the documented boundaries."""

    @pytest.mark.parametrize(
        ("score", "band"),
        [
            (100, "strong"),
            (80, "strong"),
            (79, "needs_targeted_changes"),
            (65, "needs_targeted_changes"),
            (64, "low"),
            (0, "low"),
        ],
    )
    def test_boundaries(self, score, band):
        assert band_for_score(score) == band


class TestKeywordClassification:
    """Supported-but-missing keywords are separated from real gaps."""

    def test_power_bi_is_an_unsupported_gap(self, resume):
        job = make_job(required_skills=["SQL", "Power BI"])
        _, _, gaps, _ = classify_keywords(job, snapshot_from_resume(resume))
        assert "power bi" in gaps

    def test_data_visualization_is_supported_but_missing(self, resume):
        job = make_job(required_skills=["Data visualization"])
        _, supported, gaps, evidence = classify_keywords(
            job, snapshot_from_resume(resume)
        )
        assert "data visualization" in supported
        assert "data visualization" not in gaps
        assert "experience_1_bullet_2" in evidence["data visualization"]

    def test_matched_keywords_are_recognised(self, resume):
        job = make_job(required_skills=["SQL", "Excel", "Python"])
        matched, _, _, _ = classify_keywords(job, snapshot_from_resume(resume))
        assert {"sql", "excel", "python"} <= set(matched)

    def test_concept_evidence_is_broader_than_direct_evidence(self, resume):
        snapshot = snapshot_from_resume(resume)
        assert find_evidence("data visualization", snapshot)
        assert not find_direct_evidence("data visualization", snapshot)


class TestComponents:
    """Each component is bounded and follows its documented rule."""

    def test_keyword_coverage_is_a_ratio(self):
        assert score_keyword_coverage(["a", "b"], 4) == 50
        assert score_keyword_coverage([], 0) == 50

    def test_unknown_requirements_are_not_failures(self, resume, now):
        job = make_job(minimum_experience_years=None, education_requirement=None)
        score, notes = score_qualification_alignment(job, resume, now)
        assert score == 100
        assert notes == []

    def test_experience_shortfall_reduces_qualification_alignment(self, resume, now):
        job = make_job(minimum_experience_years=5)
        score, notes = score_qualification_alignment(job, resume, now)
        assert score < 100
        assert notes

    def test_higher_degree_requirement_reduces_alignment(self, resume, now):
        job = make_job(education_requirement="Master's degree required")
        score, _ = score_qualification_alignment(job, resume, now)
        assert score < 100

    def test_preferred_skills_are_not_scored_as_required(self, resume, now):
        with_preferred = make_job(preferred_skills=["Power BI", "Looker", "SAS"])
        without = make_job(preferred_skills=[])
        assert score_qualification_alignment(with_preferred, resume, now)[
            0
        ] == score_qualification_alignment(without, resume, now)[0]

    def test_section_completeness_of_the_demo_resume(self, resume):
        score, missing = score_section_completeness(snapshot_from_resume(resume))
        assert score == 100
        assert missing == []

    def test_missing_sections_are_reported(self, resume):
        snapshot = snapshot_from_resume(resume)
        snapshot.skills = []
        snapshot.summary = ""
        score, missing = score_section_completeness(snapshot)
        assert score == 60
        assert set(missing) == {"Skills", "Professional summary"}

    def test_structure_penalizes_unparseable_features(self, resume):
        snapshot = snapshot_from_resume(resume)
        snapshot.document_features["uses_tables_for_layout"] = True
        snapshot.document_features["uses_text_boxes"] = True
        score, issues = score_structure_parseability(snapshot)
        assert score < 100
        assert len(issues) == 2

    def test_contact_completeness_reports_gaps(self, resume):
        snapshot = snapshot_from_resume(resume)
        snapshot.phone = None
        score, missing = score_contact_completeness(snapshot)
        assert score == 75
        assert missing == ["phone"]

    def test_bullet_specificity_rewards_context_and_length(self):
        thin = bullet_specificity("Analyzed survey data using Python and Excel.")
        rich = bullet_specificity(
            "Analyzed survey data using Python and Excel to produce the recurring "
            "reporting used by the research team."
        )
        assert 0.0 <= thin < rich <= 1.0

    def test_metrics_are_never_required(self):
        without_numbers = bullet_specificity(
            "Created Tableau dashboards for weekly reporting used across the lab team."
        )
        assert without_numbers == pytest.approx(1.0)


class TestAssessment:
    """The total is the documented weighted sum and cannot be changed by an LLM."""

    def test_total_matches_the_weighted_formula(self, resume, now):
        assessment = assess_ats(make_job(), resume, now=now)
        expected = round(
            0.40 * assessment.keyword_score
            + 0.20 * assessment.qualification_score
            + 0.15 * assessment.evidence_score
            + 0.10 * assessment.section_score
            + 0.10 * assessment.structure_score
            + 0.05 * assessment.contact_score
        )
        assert assessment.total_score == expected

    def test_score_is_bounded(self, resume, now):
        assessment = assess_ats(make_job(), resume, now=now)
        assert 0 <= assessment.total_score <= 100
        for component in (
            assessment.keyword_score,
            assessment.qualification_score,
            assessment.evidence_score,
            assessment.section_score,
            assessment.structure_score,
            assessment.contact_score,
        ):
            assert 0 <= component <= 100

    def test_the_same_inputs_produce_the_same_score(self, resume, now):
        first = assess_ats(make_job(), resume, now=now)
        second = assess_ats(make_job(), resume, now=now)
        assert first.model_dump() == second.model_dump()

    def test_disclaimer_is_always_present(self, resume, now):
        assert assess_ats(make_job(), resume, now=now).disclaimer == ATS_DISCLAIMER

    def test_band_matches_the_total(self, resume, now):
        assessment = assess_ats(make_job(), resume, now=now)
        assert assessment.band == band_for_score(assessment.total_score)

    def test_original_version_is_labelled(self, resume, now):
        assert assess_ats(make_job(), resume, now=now).resume_version == "original"


class TestProjectedRescore:
    """The proposal is re-scored under the identical rubric."""

    def _application(self, resume, revised_bullet_text: str) -> TailoredApplication:
        bullets = list(resume.bullet_index().items())
        return TailoredApplication(
            job_id="job_test000001",
            revised_summary=resume.professional_summary,
            revised_bullets=[
                RevisedBullet(
                    source_bullet_id=bullets[1][0],
                    original_text=bullets[1][1],
                    revised_text=revised_bullet_text,
                ),
                RevisedBullet(
                    source_bullet_id=bullets[0][0],
                    original_text=bullets[0][1],
                    revised_text=bullets[0][1],
                ),
            ],
            reordered_skills=list(reversed(resume.skills)),
            cover_letter="Cover letter.",
        )

    def test_projected_version_is_labelled(self, resume, now):
        job = make_job(required_skills=["Data visualization"])
        projected = rescore_proposed(
            job, resume, self._application(resume, "unchanged"), now=now
        )
        assert projected.resume_version == "proposed"

    def test_stating_a_supported_keyword_raises_the_projected_score(self, resume, now):
        job = make_job(required_skills=["SQL", "Excel", "Data visualization"])
        original = assess_ats(job, resume, now=now)
        projected = rescore_proposed(
            job,
            resume,
            self._application(
                resume,
                "Created Tableau dashboards for weekly reporting, delivering data "
                "visualization the research team reviewed every week.",
            ),
            now=now,
        )
        assert projected.total_score > original.total_score

    def test_the_projected_score_uses_the_same_formula(self, resume, now):
        job = make_job()
        projected = rescore_proposed(
            job, resume, self._application(resume, "unchanged"), now=now
        )
        expected = round(
            0.40 * projected.keyword_score
            + 0.20 * projected.qualification_score
            + 0.15 * projected.evidence_score
            + 0.10 * projected.section_score
            + 0.10 * projected.structure_score
            + 0.05 * projected.contact_score
        )
        assert projected.total_score == expected

    def test_a_weaker_proposal_lowers_the_projected_score(self, resume, now):
        job = make_job(required_skills=["SQL", "Excel", "Python"])
        original = assess_ats(job, resume, now=now)
        weakened = rescore_proposed(
            job, resume, self._application(resume, "Did stuff."), now=now
        )
        assert weakened.total_score < original.total_score

    def test_the_patch_only_changes_summary_bullets_and_order(self, resume):
        application = self._application(resume, "Rewritten bullet text.")
        snapshot = snapshot_with_patch(resume, application)
        assert set(snapshot.skills) == set(resume.skills)
        assert snapshot.education == resume.education
        assert len(snapshot.bullets) == len(resume.bullet_index())
