"""Reading a posting's employment type, and filtering on it."""

from __future__ import annotations

import pytest

from tests.conftest import make_job
from tools.employment_type import detect_employment_type, types_conflict
from tools.job_filter import rejection_reason


def kind(title: str, description: str = "", stated: str | None = None) -> str:
    return detect_employment_type(title, description, stated).employment_type


class TestReadingFromTheTitle:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Data Analyst Intern", "internship"),
            ("Summer 2027 Internship — Analytics", "internship"),
            ("Data Analyst Co-op", "internship"),
            ("Data Analyst (Contract)", "contract"),
            ("Data Analyst — 12 month contract", "contract"),
            ("Freelance Data Analyst", "contract"),
            ("Part-time Data Analyst", "part_time"),
            ("Data Analyst, Full-Time", "full_time"),
            ("Seasonal Data Analyst", "temporary"),
        ],
    )
    def test_the_title_states_the_type(self, title, expected):
        assert kind(title) == expected

    def test_a_plain_title_states_nothing(self):
        assert kind("Data Analyst") == "unknown"


class TestPrecedence:
    def test_an_internship_wins_over_full_time(self):
        """A "Full-time Summer Internship" is an internship."""
        assert kind("Full-time Summer Internship") == "internship"

    def test_a_stated_field_beats_the_title(self):
        assert kind("Data Analyst", stated="Contract") == "contract"

    def test_the_stated_field_is_quoted_as_the_evidence(self):
        detection = detect_employment_type("Data Analyst", stated="Full-time: Remote")
        assert "Full-time: Remote" in detection.evidence

    def test_the_title_beats_the_description(self):
        assert kind("Part-time Analyst", "This is a full-time role.") == "part_time"

    def test_a_meaningless_stated_field_falls_through_to_the_title(self):
        assert kind("Data Analyst Intern", stated="Regular") == "internship"

    def test_only_the_opening_of_a_description_is_read(self):
        """A full-time posting often mentions contractors far down the page."""
        body = "We are hiring a full-time analyst. " + ("filler " * 200) + " contract"
        assert kind("Data Analyst", body) == "full_time"


class TestConflicts:
    def test_a_different_type_conflicts(self):
        assert types_conflict("full_time", "internship")

    def test_the_same_type_does_not(self):
        assert not types_conflict("contract", "contract")

    @pytest.mark.parametrize("requested", ["any", "unknown", "", None])
    def test_not_filtering_never_conflicts(self, requested):
        assert not types_conflict(requested, "internship")

    def test_a_posting_that_states_nothing_is_never_dropped(self):
        """Silence is not evidence the job is the wrong type."""
        assert not types_conflict("full_time", "unknown")


class TestFiltering:
    def test_an_off_type_posting_is_removed_with_a_reason(self):
        job = make_job(title="Data Analyst", employment_type="contract")
        reason = rejection_reason(
            job, min_description_chars=400, requested_employment_type="full_time"
        )
        assert "Contract" in reason and "Full-time" in reason

    def test_a_matching_posting_is_kept(self):
        job = make_job(employment_type="full_time")
        assert (
            rejection_reason(
                job, min_description_chars=400, requested_employment_type="full_time"
            )
            is None
        )

    def test_a_posting_with_no_stated_type_survives_the_filter(self):
        job = make_job(employment_type="unknown")
        assert (
            rejection_reason(
                job, min_description_chars=400, requested_employment_type="full_time"
            )
            is None
        )

    def test_the_default_filters_nothing(self):
        job = make_job(employment_type="contract")
        assert rejection_reason(job, min_description_chars=400) is None
