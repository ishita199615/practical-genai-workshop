"""Duplicate detection by canonical URL and by company + title + location."""

from __future__ import annotations

from tests.conftest import make_job
from tools.job_filter import (
    dedup_key,
    filter_and_deduplicate,
    normalize_company,
    normalize_location,
    normalize_title,
)


class TestNormalizationKeys:
    """Keys ignore punctuation, case, and corporate suffixes."""

    def test_company_suffixes_are_dropped(self):
        assert normalize_company("Lakeside Analytics, Inc.") == normalize_company(
            "Lakeside Analytics"
        )

    def test_titles_normalize_punctuation_and_case(self):
        assert normalize_title("Data Analyst — Intern") == normalize_title(
            "data analyst intern"
        )

    def test_locations_normalize_punctuation(self):
        assert normalize_location("Houston, TX") == normalize_location("houston tx")

    def test_dedup_key_combines_all_three(self):
        key = dedup_key(make_job())
        assert key == "lakeside analytics|data analyst intern|houston tx"


class TestDeduplication:
    """The first occurrence wins; later copies are reported as duplicates."""

    def test_identical_canonical_urls_are_deduplicated(self):
        first = make_job(job_id="job_a")
        second = make_job(
            job_id="job_b",
            company="Different Co",
            source_url=first.source_url + "?utm_source=aggregator",
        )
        outcome = filter_and_deduplicate([first, second])
        assert [job.job_id for job in outcome.kept] == ["job_a"]
        assert "Duplicate" in outcome.reasons()[0]

    def test_same_company_title_location_is_deduplicated(self):
        first = make_job(job_id="job_a")
        second = make_job(
            job_id="job_b",
            company="Lakeside Analytics, Inc.",
            title="Data Analyst  Intern",
            source_url="https://jobs.lever.co/lakeside/other-id",
        )
        outcome = filter_and_deduplicate([first, second])
        assert [job.job_id for job in outcome.kept] == ["job_a"]

    def test_distinct_postings_are_both_kept(self):
        first = make_job(job_id="job_a")
        second = make_job(
            job_id="job_b",
            company="Bayou Insights",
            title="Junior Data Analyst",
            source_url="https://jobs.lever.co/bayou/abc",
        )
        outcome = filter_and_deduplicate([first, second])
        assert {job.job_id for job in outcome.kept} == {"job_a", "job_b"}

    def test_same_title_at_a_different_location_is_kept(self):
        first = make_job(job_id="job_a")
        second = make_job(
            job_id="job_b",
            location="Austin, TX",
            source_url="https://jobs.lever.co/lakeside/austin",
        )
        outcome = filter_and_deduplicate([first, second])
        assert len(outcome.kept) == 2

    def test_deduplication_runs_after_rejection(self):
        closed = make_job(job_id="job_a", is_closed=True)
        valid = make_job(job_id="job_b")
        outcome = filter_and_deduplicate([closed, valid])
        assert [job.job_id for job in outcome.kept] == ["job_b"]
