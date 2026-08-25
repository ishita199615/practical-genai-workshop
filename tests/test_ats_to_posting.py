"""Converting a board's structured posting into the agent's JobPosting.

The point of this route is that the employer stated these fields, so the tests
here are mostly about *not* losing or second-guessing what they said.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tools.ats_boards import AtsPosting
from tools.ats_to_posting import to_job_posting, to_job_postings

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
LONG_DESCRIPTION = (
    "Chevron is accepting applications for a data analyst. You will build "
    "reporting in SQL and Excel, partner with operations, and present findings "
    "to stakeholders across the business unit. " * 4
)


def board_posting(**overrides) -> AtsPosting:
    base = dict(
        title="Land Analyst",
        company="Chevron",
        location="Houston, Texas, United States",
        url="https://chevron.wd5.myworkdayjobs.com/jobs/job/Houston/Land-Analyst_R1",
        ats="workday",
        description=LONG_DESCRIPTION,
        posted_at=NOW - timedelta(days=1),
        posted_text="2026-08-24",
    )
    base.update(overrides)
    return AtsPosting(**base)


def convert(**overrides):
    return to_job_posting(
        board_posting(**overrides), freshness_window="last_7_days", retrieved_at=NOW
    )


class TestStatedFieldsSurvive:
    """What the employer published is what the posting carries."""

    def test_title_company_and_location(self):
        job = convert()
        assert job.title == "Land Analyst"
        assert job.company == "Chevron"
        assert job.location == "Houston, Texas, United States"

    def test_the_source_is_labelled_by_its_ats(self):
        """The screen prefixes the category, so the label is the ATS alone."""
        assert convert().source_label == "Workday"
        assert convert(ats="greenhouse").source_label == "Greenhouse"

    def test_the_category_marks_it_as_a_board(self):
        job = convert()
        assert job.query_category == "employer_boards"
        assert job.source_category == "employer_boards"

    def test_a_stated_work_mode_is_kept(self):
        assert convert(work_mode="hybrid").work_mode == "hybrid"

    def test_a_remote_location_implies_remote_when_none_is_stated(self):
        assert convert(location="Remote - Austin").work_mode == "remote"

    def test_an_unstated_work_mode_stays_unknown(self):
        assert convert(location="Houston, TX").work_mode == "unknown"

    def test_missing_title_and_company_get_the_usual_placeholders(self):
        job = convert(title="", company="")
        assert job.title == "Untitled posting"
        assert job.company == "Unknown company"


class TestFreshness:
    """A board that published a date earns a stronger claim than one that did not."""

    def test_a_date_inside_the_window_is_recent(self):
        assert convert().freshness_status == "verified_recent"

    def test_a_bare_date_does_not_claim_a_timestamp(self):
        """"2026-08-24" fixes the day, not the hour, so it says only that."""
        job = convert(posted_text="2026-08-24")
        assert job.freshness_evidence == "date_only"
        assert job.freshness_label() == "Date shown; exact time unavailable"

    def test_a_full_timestamp_earns_the_stronger_claim(self):
        job = convert(posted_text="2026-08-24T09:30:00Z")
        assert job.freshness_evidence == "exact_timestamp"
        assert "Verified" in job.freshness_label()

    def test_a_posting_older_than_the_window_is_stale(self):
        job = convert(posted_at=NOW - timedelta(days=30))
        assert job.freshness_status == "possibly_stale"

    def test_no_date_claims_nothing(self):
        job = convert(posted_at=None, posted_text="")
        assert job.freshness_status == "date_unavailable"
        assert job.freshness_evidence == "unavailable"
        assert job.posting_age_hours is None

    def test_a_workday_age_phrase_is_only_good_to_the_day(self):
        """"Posted Yesterday" is an age, not a timestamp."""
        job = convert(posted_text="Posted Yesterday")
        assert job.freshness_evidence == "date_only"

    def test_age_is_measured_from_the_stated_date(self):
        assert convert().posting_age_hours == pytest.approx(24.0)


class TestDescription:
    def test_the_full_description_is_kept(self):
        assert len(convert().description) > 400

    def test_html_is_reduced_to_text(self):
        job = convert(description="<p>Build " + ("dashboards " * 80) + "</p>")
        assert "<p>" not in job.description

    def test_the_excerpt_does_not_repeat_the_card_header(self):
        job = convert(
            description="Land Analyst Houston, Texas, United States " + LONG_DESCRIPTION
        )
        assert not job.description_excerpt.startswith("Land Analyst Houston")

    def test_a_closed_posting_is_detected(self):
        job = convert(description=LONG_DESCRIPTION + " This job is no longer available.")
        assert job.is_closed


class TestIdentity:
    def test_the_apply_url_is_the_board_url(self):
        job = convert()
        assert job.apply_url == board_posting().url
        assert job.source_domain

    def test_the_same_posting_converts_to_the_same_id(self):
        assert convert().job_id == convert().job_id

    def test_different_postings_get_different_ids(self):
        other = convert(url="https://chevron.wd5.myworkdayjobs.com/jobs/job/X_R2")
        assert other.job_id != convert().job_id


class TestBatch:
    def test_a_whole_set_converts(self):
        jobs = to_job_postings(
            [board_posting(), board_posting(title="IT Business Analyst")],
            freshness_window="last_7_days",
            retrieved_at=NOW,
        )
        assert [job.title for job in jobs] == ["Land Analyst", "IT Business Analyst"]

    def test_an_empty_set_is_fine(self):
        assert to_job_postings([], freshness_window="last_7_days") == []

    def test_the_requested_level_is_recorded_not_copied_onto_the_posting(self):
        """The request is stored beside the detected level, never as it."""
        [job] = to_job_postings(
            [board_posting(title="Land Analyst")],
            freshness_window="last_7_days",
            retrieved_at=NOW,
            requested_experience_level="internship",
        )
        assert job.requested_experience_level == "internship"
        assert job.experience_level != "internship"
