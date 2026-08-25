"""Truthfulness validation: the Power BI guardrail and its neighbours."""

from __future__ import annotations

import pytest

from models.application import RevisedBullet, TailoredApplication
from models.ats import AtsRecommendation
from models.validation import ClaimReview, ClaimReviewBatch
from tests.conftest import make_job
from tools.claim_validator import (
    extract_claims,
    review_claims,
    run_deterministic_checks,
    validate_application,
)

SAFE_REC = AtsRecommendation(
    recommendation_id="ats_rec_01",
    priority="high",
    category="keyword_alignment",
    target_section="Experience bullet experience_1_bullet_2",
    current_text="Created Tableau dashboards for weekly reporting.",
    recommended_change="State “data visualization” in this bullet.",
    reason="The job uses the phrase and the resume proves the work.",
    evidence_resume_ids=["experience_1_bullet_2"],
    safe_to_apply=True,
    projected_effect="high",
)

UNSAFE_REC = AtsRecommendation(
    recommendation_id="ats_rec_09",
    priority="high",
    category="unsupported_gap",
    target_section="Missing qualification",
    recommended_change="Do not add power bi. Keep it in the learning-gap list.",
    reason="The resume contains no Power BI evidence.",
    evidence_resume_ids=[],
    safe_to_apply=False,
    projected_effect="high",
)


def build_application(resume, **overrides) -> TailoredApplication:
    """Build a truthful baseline application, then apply overrides."""
    bullets = list(resume.bullet_index().items())
    defaults = {
        "job_id": "job_test000001",
        "revised_summary": (
            "Data Analyst Intern candidate and information systems student who "
            "analyzes survey data and builds recurring dashboards."
        ),
        "revised_bullets": [
            RevisedBullet(
                source_bullet_id=bullets[0][0],
                original_text=bullets[0][1],
                revised_text="Analyzed survey data using Python and Excel.",
            ),
            RevisedBullet(
                source_bullet_id=bullets[1][0],
                original_text=bullets[1][1],
                revised_text=(
                    "Created Tableau dashboards for weekly reporting, delivering "
                    "data visualization for the research team."
                ),
            ),
        ],
        "reordered_skills": ["SQL", "Excel", "Python", "Tableau", "Pandas", "Statistics"],
        "keywords_used": ["sql", "excel", "data visualization"],
        "applied_ats_recommendation_ids": ["ats_rec_01"],
        "unsupported_ats_gaps_not_applied": ["power bi"],
        "missing_requirements": ["power bi"],
        "cover_letter": (
            "Dear Hiring Team, I am applying for the Data Analyst Intern position "
            "at Lakeside Analytics. I analyze survey data with Python and Excel and "
            "build Tableau dashboards for weekly reporting. Thank you for your "
            "consideration."
        ),
    }
    defaults.update(overrides)
    return TailoredApplication(**defaults)


def check_named(checks, name: str) -> dict:
    """Return one deterministic check record by name."""
    return next(check for check in checks if check["name"] == name)


class TestDeterministicChecks:
    """Python rules that no phrasing can talk its way past."""

    def test_a_truthful_draft_passes_every_check(self, resume):
        checks = run_deterministic_checks(
            build_application(resume), resume, [SAFE_REC, UNSAFE_REC], ["power bi"],
            ["Lakeside Analytics"],
        )
        assert all(check["passed"] for check in checks), [
            check for check in checks if not check["passed"]
        ]

    def test_an_unknown_source_bullet_id_fails(self, resume):
        application = build_application(
            resume,
            revised_bullets=[
                RevisedBullet(
                    source_bullet_id="experience_9_bullet_9",
                    original_text="Invented bullet.",
                    revised_text="Invented bullet.",
                )
            ],
        )
        checks = run_deterministic_checks(application, resume)
        assert not check_named(checks, "Source IDs verified")["passed"]

    def test_altered_original_text_fails(self, resume):
        bullets = list(resume.bullet_index().items())
        application = build_application(
            resume,
            revised_bullets=[
                RevisedBullet(
                    source_bullet_id=bullets[0][0],
                    original_text="Something the resume never said.",
                    revised_text="Analyzed survey data.",
                )
            ],
        )
        checks = run_deterministic_checks(application, resume)
        assert not check_named(
            checks, "Original bullet text matches the master resume"
        )["passed"]

    def test_a_new_employer_fails(self, resume):
        application = build_application(
            resume,
            cover_letter="I spent two summers at Globex Corporation building reports.",
        )
        checks = run_deterministic_checks(application, resume)
        assert not check_named(checks, "No unsupported employer added")["passed"]

    def test_addressing_the_target_company_is_allowed(self, resume):
        checks = run_deterministic_checks(
            build_application(resume), resume, allowed_organizations=["Lakeside Analytics"]
        )
        assert check_named(checks, "No unsupported employer added")["passed"]

    @pytest.mark.parametrize(
        "company",
        ["Jobs for Humanity", "Bank of America", "Smith and Sons", "The Home Depot"],
    )
    def test_company_names_with_lowercase_connectors_are_allowed(self, resume, company):
        """A multi-word employer must not be reported as an unknown fragment."""
        application = build_application(
            resume,
            cover_letter=(
                f"Dear Hiring Team, I am applying for the Data Analyst Intern "
                f"position at {company}. I analyze survey data with Python and "
                f"Excel. Thank you for your consideration."
            ),
        )
        checks = run_deterministic_checks(
            application, resume, allowed_organizations=[company]
        )
        check = check_named(checks, "No unsupported employer added")
        assert check["passed"], check["detail"]

    def test_a_genuinely_new_employer_is_still_caught(self, resume):
        application = build_application(
            resume,
            cover_letter="I previously worked at Globex Corporation on reporting.",
        )
        checks = run_deterministic_checks(
            application, resume, allowed_organizations=["Jobs for Humanity"]
        )
        assert not check_named(checks, "No unsupported employer added")["passed"]

    def test_a_new_degree_fails(self, resume):
        application = build_application(
            resume, revised_summary="Master's degree holder in data science."
        )
        checks = run_deterministic_checks(application, resume)
        assert not check_named(checks, "No unsupported degree added")["passed"]

    def test_a_new_date_fails(self, resume):
        application = build_application(
            resume, cover_letter="I have worked in analytics since 2019."
        )
        checks = run_deterministic_checks(application, resume)
        assert not check_named(checks, "No new dates introduced")["passed"]

    def test_an_invented_metric_fails(self, resume):
        application = build_application(
            resume, revised_summary="Improved reporting speed by 45% last term."
        )
        checks = run_deterministic_checks(application, resume)
        assert not check_named(checks, "No invented metrics")["passed"]

    def test_skills_outside_the_master_resume_fail(self, resume):
        application = build_application(
            resume,
            reordered_skills=["SQL", "Excel", "Python", "Tableau", "Power BI"],
        )
        checks = run_deterministic_checks(application, resume)
        assert not check_named(
            checks, "Skills remain grounded in the master resume"
        )["passed"]

    def test_applying_an_unsafe_recommendation_fails(self, resume):
        application = build_application(
            resume, applied_ats_recommendation_ids=["ats_rec_09"]
        )
        checks = run_deterministic_checks(
            application, resume, [SAFE_REC, UNSAFE_REC], ["power bi"]
        )
        assert not check_named(
            checks, "Only safe ATS recommendations were applied"
        )["passed"]

    def test_a_leaked_unsupported_gap_fails(self, resume):
        application = build_application(
            resume,
            revised_summary="Analyst skilled in Power BI reporting.",
        )
        checks = run_deterministic_checks(application, resume, [SAFE_REC], ["power bi"])
        assert not check_named(checks, "Unsupported job gaps were not added")["passed"]

    def test_an_unsupported_skill_in_the_cover_letter_fails(self, resume):
        application = build_application(
            resume,
            cover_letter=(
                "Dear Hiring Team, I build Looker dashboards and maintain ETL "
                "pipelines every week. Thank you for your consideration."
            ),
        )
        checks = run_deterministic_checks(application, resume)
        assert not check_named(
            checks, "Cover letter claims no unsupported skill"
        )["passed"]


class TestClaimReview:
    """Claims are split out and classified, with or without a model."""

    def test_claims_cover_summary_bullets_and_letter(self, resume):
        claims = extract_claims(build_application(resume))
        assert len(claims) >= 4

    def test_the_offline_fallback_flags_an_unsupported_tool(self, resume):
        reviews = review_claims(
            ["Built Power BI dashboards for the executive team."], resume, None
        )
        assert reviews[0].status == "unsupported"
        assert "power bi" in reviews[0].reason.lower()

    def test_the_offline_fallback_supports_a_grounded_claim(self, resume):
        reviews = review_claims(
            ["Created Tableau dashboards for weekly reporting."], resume, None
        )
        assert reviews[0].status == "supported"
        assert "experience_1_bullet_2" in reviews[0].supporting_resume_ids

    def test_llm_reviews_are_used_when_available(self, resume):
        class ScriptedLLM:
            available = True
            model_name = "scripted"

            def generate_structured(self, prompt, schema, *, temperature=0.1):
                return ClaimReviewBatch(
                    reviews=[
                        ClaimReview(
                            claim="A claim.",
                            status="unclear",
                            supporting_resume_ids=[],
                            reason="Partially traceable.",
                        )
                    ]
                )

            def generate_text(self, prompt, *, temperature=0.2):
                return None

        reviews = review_claims(["A claim."], resume, ScriptedLLM())
        assert reviews[0].status == "unclear"


class TestValidationReport:
    """The combined verdict is what the interface and the graph act on."""

    def test_a_truthful_draft_passes(self, resume):
        report = validate_application(
            build_application(resume),
            resume,
            applied_recommendations=[SAFE_REC, UNSAFE_REC],
            unsupported_gaps=["power bi"],
            allowed_organizations=["Lakeside Analytics"],
        )
        assert report.passed
        assert report.valid_source_ids
        assert report.unsupported_claims == []

    def test_a_power_bi_claim_is_blocked(self, resume):
        application = build_application(
            resume,
            revised_summary=(
                "Information systems student who builds Power BI dashboards."
            ),
        )
        report = validate_application(
            application,
            resume,
            applied_recommendations=[SAFE_REC],
            unsupported_gaps=["power bi"],
            allowed_organizations=["Lakeside Analytics"],
        )
        assert not report.passed
        assert report.unsupported_claims
        assert report.failed_checks()

    def test_the_report_lists_every_check(self, resume):
        report = validate_application(build_application(resume), resume)
        assert len(report.deterministic_checks) == 10
