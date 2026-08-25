"""Reading seniority off a retrieved posting, and rejecting the wrong level.

Two rules are load-bearing here. A posting's level comes only from the posting:
asking for senior roles is not evidence that a result is senior, exactly as a
freshness filter is not evidence a posting is recent. And a posting that never
states a level is kept, because there is no evidence on which to drop it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import SAMPLE_DESCRIPTION, make_job, make_raw
from tools.job_filter import filter_and_deduplicate, rejection_reason
from tools.job_normalizer import normalize_job, normalize_jobs

# A body long enough to survive the description-length rule, with no seniority
# wording of its own, so a test can isolate a single piece of evidence.
NEUTRAL_DESCRIPTION = (
    "About the role\n\nYou will support recurring reporting for our commercial "
    "teams. Day to day you will write SQL against the reporting warehouse, use "
    "Python and pandas to clean repeatable data extracts, and publish "
    "dashboards that business partners rely on each week. You will work "
    "alongside analysts and engineers to document data definitions and keep "
    "our metrics consistent across teams. Qualifications include working "
    "knowledge of SQL including joins and aggregation, comfort with "
    "spreadsheets including pivot tables, and exposure to Python for data "
    "analysis. Our office is in Houston and the schedule is hybrid."
)


def make_extracted_raw(**fields: Any):
    """Build a raw result carrying a fixed extraction payload.

    Uses the cached-extraction path so the test exercises normalization without
    a model and without a network call.
    """
    payload: dict[str, Any] = {
        "title": "Data Analyst",
        "company": "Lakeside Analytics",
        "description": NEUTRAL_DESCRIPTION,
        "is_specific_opening": True,
    }
    payload.update(fields)
    return make_raw(metadata={"cached_extraction": payload})


class TestDetectedLevelOnNormalizedPostings:
    """The detected level is read off the page, from the strongest evidence."""

    def test_level_is_detected_from_the_title(self, null_llm):
        raw = make_raw(title="Senior Data Analyst — Lakeside Analytics")
        job, warning = normalize_job(raw, null_llm)
        assert warning is None
        assert job.experience_level == "senior"
        assert "senior" in job.experience_level_evidence.lower()

    def test_internship_titles_are_detected(self, null_llm):
        job, _ = normalize_job(make_raw(), null_llm)
        assert job.experience_level == "internship"

    def test_level_is_detected_from_stated_years(self, null_llm):
        raw = make_extracted_raw(minimum_experience_years=6.0)
        job, warning = normalize_job(raw, null_llm)
        assert warning is None
        assert job.experience_level == "senior"
        assert "6" in job.experience_level_evidence

    def test_level_is_detected_from_the_body(self, null_llm):
        raw = make_extracted_raw(
            description=(
                "This is a senior-level role on our analytics team.\n\n"
                + NEUTRAL_DESCRIPTION
            )
        )
        job, warning = normalize_job(raw, null_llm)
        assert warning is None
        assert job.experience_level == "senior"
        assert "description" in job.experience_level_evidence.lower()

    def test_a_posting_that_states_no_level_stays_unknown(self, null_llm):
        job, warning = normalize_job(make_extracted_raw(), null_llm)
        assert warning is None
        assert job.experience_level == "unknown"
        assert job.experience_level_evidence is None
        assert job.experience_level_label() == "Level not stated"

    def test_the_title_outranks_stated_years(self, null_llm):
        raw = make_extracted_raw(
            title="Data Analyst Intern", minimum_experience_years=6.0
        )
        job, _ = normalize_job(raw, null_llm)
        assert job.experience_level == "internship"


class TestRequestedLevelIsRecordedNotCopied:
    """Storing the request beside the detection is fine; copying it is not."""

    def test_the_request_is_recorded_on_the_posting(self, null_llm):
        job, _ = normalize_job(
            make_raw(), null_llm, requested_experience_level="senior"
        )
        assert job.requested_experience_level == "senior"

    def test_the_request_never_becomes_the_detected_level(self, null_llm):
        job, _ = normalize_job(
            make_extracted_raw(), null_llm, requested_experience_level="senior"
        )
        assert job.requested_experience_level == "senior"
        assert job.experience_level == "unknown"

    def test_the_request_never_overrides_a_detected_level(self, null_llm):
        job, _ = normalize_job(
            make_raw(), null_llm, requested_experience_level="senior"
        )
        assert job.experience_level == "internship"

    def test_omitting_the_request_keeps_todays_behaviour(self, null_llm):
        job, _ = normalize_job(make_raw(), null_llm)
        assert job.requested_experience_level == "unknown"

    def test_normalize_jobs_threads_the_request_through(self, null_llm):
        jobs, _warnings = normalize_jobs(
            [make_raw()], null_llm, requested_experience_level="senior"
        )
        assert [job.requested_experience_level for job in jobs] == ["senior"]
        assert [job.experience_level for job in jobs] == ["internship"]

    def test_normalize_jobs_defaults_to_no_request(self, null_llm):
        jobs, _warnings = normalize_jobs([make_raw()], null_llm)
        assert jobs[0].requested_experience_level == "unknown"


class TestLevelRejection:
    """A posting is dropped only when its own stated level contradicts the ask."""

    def test_a_conflicting_posting_is_rejected(self):
        job = make_job(experience_level="internship")
        reason = rejection_reason(
            job, min_description_chars=400, requested_experience_level="senior"
        )
        assert reason == "Internship posting removed: you searched Senior."

    @pytest.mark.parametrize(
        ("detected", "requested"),
        [
            ("internship", "senior"),
            ("entry", "staff_principal"),
            ("manager", "mid"),
            ("senior", "internship"),
        ],
    )
    def test_the_reason_names_both_levels(self, detected, requested):
        reason = rejection_reason(
            make_job(experience_level=detected),
            min_description_chars=400,
            requested_experience_level=requested,
        )
        assert reason is not None
        assert reason.endswith(".")
        assert "you searched" in reason

    def test_a_matching_posting_is_kept(self):
        job = make_job(title="Senior Data Analyst", experience_level="senior")
        assert (
            rejection_reason(
                job, min_description_chars=400, requested_experience_level="senior"
            )
            is None
        )

    def test_an_unknown_level_posting_is_kept(self):
        job = make_job(experience_level="unknown")
        assert (
            rejection_reason(
                job, min_description_chars=400, requested_experience_level="senior"
            )
            is None
        )

    def test_no_requested_level_rejects_nothing(self):
        job = make_job(experience_level="internship")
        assert rejection_reason(job, min_description_chars=400) is None


class TestLevelFilteringEndToEnd:
    """``filter_and_deduplicate`` applies the rule and reports every removal."""

    def _mixed_jobs(self):
        return [
            make_job(
                job_id="job_senior",
                title="Senior Data Analyst",
                experience_level="senior",
                source_url="https://job-boards.greenhouse.io/lakeside/jobs/1",
            ),
            make_job(
                job_id="job_intern",
                title="Data Analyst Intern",
                experience_level="internship",
                source_url="https://job-boards.greenhouse.io/lakeside/jobs/2",
            ),
            make_job(
                job_id="job_unstated",
                title="Data Analyst",
                experience_level="unknown",
                source_url="https://job-boards.greenhouse.io/lakeside/jobs/3",
            ),
        ]

    def test_only_the_conflicting_posting_is_removed(self):
        outcome = filter_and_deduplicate(
            self._mixed_jobs(), requested_experience_level="senior"
        )
        assert [job.job_id for job in outcome.kept] == ["job_senior", "job_unstated"]
        assert outcome.removed == [
            ("job_intern", "Internship posting removed: you searched Senior.")
        ]

    def test_the_default_request_filters_nothing(self):
        outcome = filter_and_deduplicate(self._mixed_jobs())
        assert len(outcome.kept) == 3
        assert outcome.removed_count == 0

    def test_the_removal_reason_reaches_the_activity_log(self):
        outcome = filter_and_deduplicate(
            self._mixed_jobs(), requested_experience_level="internship"
        )
        assert outcome.reasons() == ["Senior posting removed: you searched Internship."]


class TestRealInternshipTitlesSurviveAnInternshipSearch:
    """The filter must not throw away the postings the user came for.

    Real internship titles carry words the entry markers also claim —
    "Graduate Intern", "Campus Program", "Associate ... Intern". Reading those
    as junior roles meant an Internship search removed genuine internships,
    which is the original complaint inverted rather than fixed.
    """

    @pytest.mark.parametrize(
        "title",
        [
            "Graduate Intern, Data Analytics",
            "Summer 2026 Data Analyst Intern - Campus Program",
            "Associate Analyst Intern",
        ],
    )
    def test_the_posting_is_detected_and_kept(self, title, null_llm):
        raw = make_raw(title=f"{title} — Lakeside Analytics")
        job, warning = normalize_job(
            raw, null_llm, requested_experience_level="internship"
        )
        assert warning is None
        assert job.experience_level == "internship"
        assert (
            rejection_reason(
                job, min_description_chars=1, requested_experience_level="internship"
            )
            is None
        )

    def test_the_whole_batch_survives_the_filter(self, null_llm):
        jobs = [
            make_job(
                job_id=f"job_intern_{index}",
                title=title,
                experience_level="internship",
                source_url=f"https://job-boards.greenhouse.io/lakeside/jobs/{index}",
            )
            for index, title in enumerate(
                ["Graduate Intern", "Data Analyst Intern, Campus Program"], start=1
            )
        ]
        outcome = filter_and_deduplicate(jobs, requested_experience_level="internship")
        assert len(outcome.kept) == 2
        assert outcome.removed_count == 0


ALL_REQUESTS = (
    "internship",
    "entry",
    "mid",
    "senior",
    "staff_principal",
    "manager",
    "unknown",
)


class TestDetectionIsIndependentOfTheRequest:
    """The single property the whole feature rests on.

    Whatever the user asked for, the level on a posting is the level the page
    states. The request is recorded beside it and never becomes it.
    """

    @pytest.mark.parametrize(
        ("title", "detected"),
        [
            ("Data Analyst", "unknown"),
            ("Senior Data Analyst", "senior"),
            ("Data Analyst Intern", "internship"),
            ("Data Analytics Manager", "manager"),
            ("Internal Audit Analyst", "unknown"),
            ("Principal's Office Coordinator", "unknown"),
        ],
    )
    def test_the_detected_level_never_follows_the_request(
        self, null_llm, title: str, detected: str
    ) -> None:
        results = {}
        for requested in ALL_REQUESTS:
            job, _warning = normalize_job(
                make_extracted_raw(title=title),
                null_llm,
                requested_experience_level=requested,
            )
            results[requested] = (job.experience_level, job.requested_experience_level)
        assert results == {
            requested: (detected, requested) for requested in ALL_REQUESTS
        }

    @pytest.mark.parametrize("requested", ALL_REQUESTS)
    def test_a_posting_that_states_no_level_is_always_kept(
        self, null_llm, requested: str
    ) -> None:
        job, _warning = normalize_job(
            make_extracted_raw(title="Data Analyst"),
            null_llm,
            requested_experience_level=requested,
        )
        assert job.experience_level == "unknown"
        outcome = filter_and_deduplicate(
            [job], requested_experience_level=requested
        )
        assert [kept.job_id for kept in outcome.kept] == [job.job_id]
        assert outcome.removed == []
