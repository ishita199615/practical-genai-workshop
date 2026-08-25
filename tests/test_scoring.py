"""Deterministic Demo Job Match Score."""

from __future__ import annotations

import pytest

from tests.conftest import make_job
from tools.job_scorer import (
    canonical_skill,
    canonical_skill_set,
    combined_target_roles,
    estimate_experience_years,
    rank_jobs,
    score_job,
    score_preference_alignment,
    score_role_alignment,
    score_skill_coverage,
    score_text_similarity,
    seniority_concern,
    skill_is_matched,
    skills_from_text,
)


class TestSkillAliases:
    """Alias normalization is conservative and never merges distinct tools."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("MS Excel", "excel"),
            ("Microsoft Excel", "excel"),
            ("PostgreSQL", "sql"),
            ("Structured Query Language", "sql"),
            ("Data Viz", "data visualization"),
            ("PowerBI", "power bi"),
            ("  Python  ", "python"),
        ],
    )
    def test_aliases_map_to_canonical_forms(self, raw, expected):
        assert canonical_skill(raw) == expected

    def test_compound_entries_are_split(self):
        assert canonical_skill_set(["Python, SQL and Excel"]) == {
            "python",
            "sql",
            "excel",
        }

    def test_tableau_is_not_power_bi(self, resume):
        owned = canonical_skill_set(resume.skills)
        assert not skill_is_matched("power bi", owned)

    def test_python_is_not_every_python_library(self):
        assert not skill_is_matched("scikit-learn", {"python"})

    def test_a_resume_skill_inside_a_requirement_phrase_matches(self):
        assert skill_is_matched("sql queries", {"sql"})

    def test_known_skills_are_found_in_free_text(self):
        found = skills_from_text("You will use SQL and Power BI to build dashboards.")
        assert "sql" in found
        assert "power bi" in found


class TestComponentScores:
    """Every component is bounded from 0 to 100."""

    def test_skill_coverage_of_a_full_match(self, resume):
        score, matched, missing = score_skill_coverage(resume, make_job())
        assert score == 100
        assert missing == []
        assert set(matched) == {"sql", "excel", "python"}

    def test_skill_coverage_reports_gaps(self, resume):
        job = make_job(required_skills=["SQL", "Power BI"])
        score, matched, missing = score_skill_coverage(resume, job)
        assert score == 50
        assert missing == ["power bi"]

    def test_unknown_requirements_are_not_a_failure(self, resume):
        job = make_job(required_skills=[], description="We are hiring. Apply today.")
        score, matched, missing = score_skill_coverage(resume, job)
        assert score == 50
        assert (matched, missing) == ([], [])

    def test_similarity_is_bounded_and_deterministic(self, resume):
        job = make_job()
        first = score_text_similarity(resume.as_plain_text(), job.description)
        assert first == score_text_similarity(resume.as_plain_text(), job.description)
        assert 0 <= first <= 100

    def test_similarity_of_empty_text_is_zero(self):
        assert score_text_similarity("", "text") == 0

    def test_role_alignment_rewards_the_target_title(self, resume):
        assert score_role_alignment(resume.target_roles, "Data Analyst Intern") == 100

    def test_role_alignment_penalizes_an_unrelated_title(self, resume):
        assert score_role_alignment(resume.target_roles, "Line Cook") < 40

    def test_experience_years_come_from_resume_dates(self, resume, now):
        assert estimate_experience_years(resume, now) == pytest.approx(1.0)

    def test_preference_remote_matches_a_remote_posting(self):
        score, _ = score_preference_alignment(
            make_job(work_mode="remote", location="Remote — United States"),
            "Houston, TX",
            "Remote",
        )
        assert score == 100

    def test_preference_remote_penalizes_an_onsite_posting(self):
        score, _ = score_preference_alignment(
            make_job(work_mode="onsite", location="Austin, TX"), "Houston, TX", "Remote"
        )
        assert score == 20


class TestTotalScore:
    """The total is the documented weighted sum, and the LLM cannot move it."""

    def test_total_matches_the_weighted_formula(self, resume, now):
        result = score_job(
            make_job(), resume, location="Houston, TX", work_mode="Any", now=now
        )
        expected = round(
            0.45 * result.skill_score
            + 0.20 * result.similarity_score
            + 0.15 * result.role_score
            + 0.10 * result.experience_score
            + 0.10 * result.preference_score
        )
        assert result.total_score == expected

    def test_score_is_bounded(self, resume, now):
        result = score_job(
            make_job(), resume, location="Houston, TX", work_mode="Any", now=now
        )
        assert 0 <= result.total_score <= 100

    def test_the_same_inputs_produce_the_same_score(self, resume, now):
        first = score_job(
            make_job(), resume, location="Houston, TX", work_mode="Any", now=now
        )
        second = score_job(
            make_job(), resume, location="Houston, TX", work_mode="Any", now=now
        )
        assert first.model_dump() == second.model_dump()

    def test_experience_shortfall_is_recorded_as_a_concern(self, resume, now):
        result = score_job(
            make_job(minimum_experience_years=5),
            resume,
            location="Houston, TX",
            work_mode="Any",
            now=now,
        )
        assert result.experience_score == 40
        assert any("experience" in concern for concern in result.concerns)

    def test_explanation_starts_empty(self, resume, now):
        result = score_job(
            make_job(), resume, location="Houston, TX", work_mode="Any", now=now
        )
        assert result.explanation is None


class TestRanking:
    """Ranking returns exactly three jobs and breaks ties deterministically."""

    def test_top_three_are_returned(self, resume, now):
        jobs = [
            make_job(job_id=f"job_{index}", source_url=f"https://jobs.lever.co/a/{index}")
            for index in range(5)
        ]
        ranked = rank_jobs(
            jobs, resume, location="Houston, TX", work_mode="Any", now=now
        )
        assert len(ranked) == 3

    def test_ranking_is_ordered_by_score(self, resume, now):
        jobs = [
            make_job(job_id="job_strong"),
            make_job(
                job_id="job_weak",
                title="Line Cook",
                required_skills=["Power BI", "Looker"],
                source_url="https://jobs.lever.co/a/2",
            ),
        ]
        ranked = rank_jobs(
            jobs, resume, location="Houston, TX", work_mode="Any", now=now
        )
        assert ranked[0].job_id == "job_strong"
        assert ranked[0].total_score > ranked[1].total_score

    def test_ties_break_on_job_id(self, resume, now):
        jobs = [
            make_job(job_id="job_b", source_url="https://jobs.lever.co/a/b"),
            make_job(job_id="job_a", source_url="https://jobs.lever.co/a/a"),
        ]
        ranked = rank_jobs(
            jobs, resume, location="Houston, TX", work_mode="Any", now=now
        )
        assert [match.job_id for match in ranked] == ["job_a", "job_b"]


class TestSearchedRoleAlignment:
    """The role the user searched for counts, and omitting it changes nothing."""

    def test_extra_roles_are_merged_after_the_resume_roles(self):
        merged = combined_target_roles(
            ["Data Analyst Intern"], ["Senior Data Scientist"]
        )
        assert merged == ["Data Analyst Intern", "Senior Data Scientist"]

    def test_duplicates_are_dropped_case_insensitively(self):
        merged = combined_target_roles(
            ["Data Analyst Intern"], ["  data analyst intern  ", ""]
        )
        assert merged == ["Data Analyst Intern"]

    def test_no_extra_roles_reproduces_the_resume_roles(self, resume):
        assert combined_target_roles(resume.target_roles) == resume.target_roles
        assert combined_target_roles(resume.target_roles, []) == resume.target_roles

    def test_default_reproduces_todays_score_exactly(self, resume, now):
        job = make_job(title="Senior Data Scientist")
        without = score_job(
            job, resume, location="Houston, TX", work_mode="Any", now=now
        )
        explicit_none = score_job(
            job,
            resume,
            location="Houston, TX",
            work_mode="Any",
            now=now,
            extra_target_roles=None,
        )
        empty = score_job(
            job,
            resume,
            location="Houston, TX",
            work_mode="Any",
            now=now,
            extra_target_roles=[],
        )
        assert without.model_dump() == explicit_none.model_dump()
        assert without.model_dump() == empty.model_dump()

    def test_the_searched_role_improves_alignment_for_a_matching_title(
        self, resume, now
    ):
        job = make_job(title="Senior Data Scientist")
        baseline = score_job(
            job, resume, location="Houston, TX", work_mode="Any", now=now
        )
        searched = score_job(
            job,
            resume,
            location="Houston, TX",
            work_mode="Any",
            now=now,
            extra_target_roles=["Senior Data Scientist"],
        )
        assert searched.role_score == 100
        assert searched.role_score > baseline.role_score
        assert searched.total_score > baseline.total_score

    def test_the_searched_role_does_not_rescue_an_unrelated_title(self, resume, now):
        job = make_job(title="Line Cook")
        searched = score_job(
            job,
            resume,
            location="Houston, TX",
            work_mode="Any",
            now=now,
            extra_target_roles=["Senior Data Scientist"],
        )
        assert searched.role_score < 40

    def test_ranking_accepts_the_searched_role(self, resume, now):
        jobs = [make_job(job_id="job_a", title="Senior Data Scientist")]
        ranked = rank_jobs(
            jobs,
            resume,
            location="Houston, TX",
            work_mode="Any",
            now=now,
            extra_target_roles=["Senior Data Scientist"],
        )
        assert ranked[0].role_score == 100


class TestSeniorityConcern:
    """A posting above the resume's own experience is flagged, never rescored."""

    def test_a_senior_posting_is_flagged_against_a_student_resume(self, resume, now):
        job = make_job(experience_level="senior")
        assert seniority_concern(job, resume, now) == (
            "This is a Senior posting (typically 5+ years); "
            "the resume evidences about 1."
        )

    @pytest.mark.parametrize("level", ["mid", "senior", "staff_principal", "manager"])
    def test_levels_above_the_resume_raise_a_concern(self, resume, now, level):
        job = make_job(experience_level=level)
        result = score_job(
            job, resume, location="Houston, TX", work_mode="Any", now=now
        )
        assert any("the resume evidences about" in c for c in result.concerns)

    @pytest.mark.parametrize("level", ["internship", "entry", "unknown"])
    def test_levels_within_reach_raise_no_concern(self, resume, now, level):
        job = make_job(experience_level=level)
        assert seniority_concern(job, resume, now) is None
        result = score_job(
            job, resume, location="Houston, TX", work_mode="Any", now=now
        )
        assert not any("the resume evidences about" in c for c in result.concerns)

    def test_an_unstated_level_is_never_guessed_from_the_request(self, resume, now):
        job = make_job(requested_experience_level="senior", experience_level="unknown")
        assert seniority_concern(job, resume, now) is None

    def test_the_concern_does_not_change_any_score(self, resume, now):
        plain = score_job(
            make_job(), resume, location="Houston, TX", work_mode="Any", now=now
        )
        senior = score_job(
            make_job(experience_level="senior"),
            resume,
            location="Houston, TX",
            work_mode="Any",
            now=now,
        )
        assert senior.total_score == plain.total_score
        assert senior.experience_score == plain.experience_score
        assert len(senior.concerns) == len(plain.concerns) + 1
