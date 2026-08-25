"""Rejection rules: closed, snippet-only, generic, stale, and unreadable pages."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from tests.conftest import FIXED_NOW, make_job, make_raw
from tools.firecrawl_search import company_from_url
from tools.job_filter import filter_and_deduplicate, rejection_reason
from tools.job_normalizer import (
    MIN_DESCRIPTION_CHARS,
    clean_description,
    compute_freshness,
    detect_closed,
    make_excerpt,
    normalize_job,
    parse_posted_at,
    split_title_and_company,
)


class TestRejectionRules:
    """A posting is kept only when every rule passes."""

    def test_closed_jobs_are_removed(self):
        job = make_job(is_closed=True)
        assert "closed" in rejection_reason(job, min_description_chars=400).lower()

    def test_missing_title_is_removed(self):
        job = make_job(title="Untitled posting")
        assert "title" in rejection_reason(job, min_description_chars=400).lower()

    def test_missing_company_is_removed(self):
        job = make_job(company="Unknown company")
        assert "company" in rejection_reason(job, min_description_chars=400).lower()

    def test_snippet_only_description_is_removed(self):
        job = make_job(description="Data Analyst Intern in Houston. Apply now.")
        reason = rejection_reason(job, min_description_chars=MIN_DESCRIPTION_CHARS)
        assert "snippet" in reason.lower()

    def test_generic_listing_page_is_removed(self):
        job = make_job(source_url="https://www.linkedin.com/jobs/search/?keywords=x")
        assert "listing" in rejection_reason(job, min_description_chars=400).lower()

    def test_stale_posting_is_removed(self):
        job = make_job(freshness_status="possibly_stale")
        assert "older" in rejection_reason(job, min_description_chars=400).lower()

    def test_a_valid_posting_is_kept(self):
        assert rejection_reason(make_job(), min_description_chars=400) is None

    def test_removals_are_reported_with_reasons(self):
        outcome = filter_and_deduplicate([make_job(is_closed=True)])
        assert outcome.kept == []
        assert outcome.removed_count == 1
        assert outcome.reasons()[0]


class TestPostingTimeParsing:
    """A posting date is read from the source, never inferred."""

    def test_hours_ago_is_an_exact_timestamp(self):
        posted_at, _, evidence = parse_posted_at("Posted 3 hours ago", FIXED_NOW)
        assert evidence == "exact_timestamp"
        assert posted_at == FIXED_NOW - timedelta(hours=3)

    def test_minutes_ago_is_an_exact_timestamp(self):
        _, _, evidence = parse_posted_at("25 minutes ago", FIXED_NOW)
        assert evidence == "exact_timestamp"

    def test_days_ago_is_date_only(self):
        _, _, evidence = parse_posted_at("Posted 2 days ago", FIXED_NOW)
        assert evidence == "date_only"

    def test_iso_datetime_is_an_exact_timestamp(self):
        posted_at, _, evidence = parse_posted_at("2026-08-20T09:30:00Z", FIXED_NOW)
        assert evidence == "exact_timestamp"
        assert posted_at == datetime(2026, 8, 20, 9, 30, tzinfo=timezone.utc)

    def test_bare_date_is_date_only(self):
        posted_at, posting_date, evidence = parse_posted_at("2026-08-19", FIXED_NOW)
        assert evidence == "date_only"
        assert posted_at is None
        assert posting_date == date(2026, 8, 19)

    def test_month_name_date_is_date_only(self):
        _, posting_date, evidence = parse_posted_at("August 18, 2026", FIXED_NOW)
        assert evidence == "date_only"
        assert posting_date == date(2026, 8, 18)

    @pytest.mark.parametrize("value", [None, "", "   ", "Apply now"])
    def test_missing_posting_time_is_unavailable(self, value):
        posted_at, posting_date, evidence = parse_posted_at(value, FIXED_NOW)
        assert (posted_at, posting_date, evidence) == (None, None, "unavailable")


class TestFreshnessLabelling:
    """A search filter is never treated as proof of the posting time."""

    def test_exact_timestamp_inside_the_window_is_verified(self):
        posted_at = FIXED_NOW - timedelta(minutes=30)
        age, evidence, status = compute_freshness(
            posted_at, posted_at.date(), "exact_timestamp", "last_hour", FIXED_NOW
        )
        assert (evidence, status) == ("exact_timestamp", "verified_recent")
        assert age == pytest.approx(0.5)

    def test_exact_timestamp_outside_the_window_is_stale(self):
        posted_at = FIXED_NOW - timedelta(hours=5)
        _, evidence, status = compute_freshness(
            posted_at, posted_at.date(), "exact_timestamp", "last_hour", FIXED_NOW
        )
        assert (evidence, status) == ("exact_timestamp", "possibly_stale")

    def test_date_only_inside_the_range_is_verified_but_date_only(self):
        _, evidence, status = compute_freshness(
            None, FIXED_NOW.date(), "date_only", "last_24_hours", FIXED_NOW
        )
        assert (evidence, status) == ("date_only", "verified_recent")

    def test_missing_evidence_falls_back_to_search_filter_only(self):
        age, evidence, status = compute_freshness(
            None, None, "unavailable", "last_24_hours", FIXED_NOW
        )
        assert (age, evidence, status) == (None, "search_filter_only", "date_unavailable")

    def test_last_hour_is_verified_only_with_an_exact_timestamp(self):
        _, evidence, status = compute_freshness(
            None, FIXED_NOW.date(), "date_only", "last_hour", FIXED_NOW
        )
        assert evidence == "date_only"
        assert status == "verified_recent"
        job = make_job(freshness_window="last_hour", freshness_evidence="date_only")
        assert "exact time unavailable" in job.freshness_label()

    def test_search_filtered_results_say_so(self):
        job = make_job(freshness_evidence="search_filter_only")
        assert "Search-filtered" in job.freshness_label()


class TestDescriptionHandling:
    """The full cleaned description is preserved; the excerpt is derived."""

    def test_markdown_noise_is_stripped(self):
        cleaned = clean_description(
            "![logo](https://x/y.png)\n# Title\n[Apply](https://x/apply)\n<b>Bold</b>"
        )
        assert "https://x/y.png" not in cleaned
        assert "Apply" in cleaned
        assert "<b>" not in cleaned

    def test_long_descriptions_are_clamped(self):
        cleaned = clean_description("word " * 5000, max_chars=500)
        assert len(cleaned) <= 560
        assert "truncated" in cleaned

    def test_excerpt_is_short_and_deterministic(self):
        description = "A " * 400
        first = make_excerpt(description)
        assert first == make_excerpt(description)
        assert len(first) <= 262

    def test_full_description_survives_normalization(self, null_llm):
        job, warning = normalize_job(make_raw(), null_llm)
        assert warning is None
        assert len(job.description) > MIN_DESCRIPTION_CHARS
        assert job.description_excerpt != job.description

    def test_browser_support_boilerplate_is_stripped(self):
        cleaned = clean_description(
            "Sorry, Internet Explorer 11 is no longer supported by SmartRecruiters "
            "Please update to one of the following browsers: - Google Chrome "
            "You can find details about supported web browsers here.\n\n"
            "# Data Analyst Intern\n\nSupport reporting with SQL and Excel."
        )
        assert "Internet Explorer" not in cleaned
        assert "Data Analyst Intern" in cleaned
        assert "SQL and Excel" in cleaned

    def test_cookie_banners_are_stripped(self):
        cleaned = clean_description(
            "We use cookies to improve your experience. Accept All\n\n"
            "# Analyst role\n\nWork with data."
        )
        assert "cookies" not in cleaned.lower()
        assert "Analyst role" in cleaned

    def test_job_content_is_never_edited(self):
        original = "# Data Analyst\n\nYou will write SQL and build dashboards."
        assert "write SQL and build dashboards" in clean_description(original)

    def test_closed_language_is_detected(self):
        assert detect_closed("This role is no longer accepting applications.")
        assert not detect_closed("Applications are reviewed weekly.")


class TestCompanyRecovery:
    """A readable posting is not discarded for want of a model.

    When extraction is unavailable — offline, or under a provider quota limit —
    the employer is recovered from the applicant-tracking URL rather than
    guessed at or left blank.
    """

    @pytest.mark.parametrize(
        ("url", "company"),
        [
            ("https://jobs.lever.co/portcast/f18cc64e", "Portcast"),
            ("https://jobs.ashbyhq.com/joko/6aed4a35", "Joko"),
            ("https://job-boards.greenhouse.io/innodatainc", "Innodatainc"),
            ("https://boards.greenhouse.io/acme/jobs/1", "Acme"),
            (
                "https://jobs.smartrecruiters.com/JobsForHumanity/744000-analyst",
                "Jobs For Humanity",
            ),
            ("https://acme.myworkdayjobs.com/en-US/careers/job/1", "Acme"),
        ],
    )
    def test_company_is_recovered_from_ats_urls(self, url, company):
        assert company_from_url(url) == company

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/jobs/view/123",
            "https://example.com/some/page",
            "https://jobs.lever.co/",
        ],
    )
    def test_unknown_url_shapes_yield_nothing_rather_than_a_guess(self, url):
        assert company_from_url(url) is None

    @pytest.mark.parametrize(
        ("page_title", "expected"),
        [
            ("Data Analyst Intern @ Joko", ("Data Analyst Intern", "Joko")),
            ("Senior Analyst | Acme Corp", ("Senior Analyst", "Acme Corp")),
            ("Analyst — Bayou Insights", ("Analyst", "Bayou Insights")),
            ("Data Analyst", ("Data Analyst", None)),
            ("", ("", None)),
        ],
    )
    def test_titles_split_on_known_separators(self, page_title, expected):
        assert split_title_and_company(page_title) == expected

    def test_a_quota_limited_page_still_becomes_a_posting(self, null_llm):
        raw = make_raw(
            url="https://jobs.lever.co/portcast/f18cc64e",
            final_url="https://jobs.lever.co/portcast/f18cc64e",
            title="Portcast",
        )
        job, warning = normalize_job(raw, null_llm)
        assert warning is None
        assert job.company == "Portcast"

    def test_the_url_company_wins_over_a_title_fragment(self, null_llm):
        raw = make_raw(
            url="https://jobs.ashbyhq.com/joko/abc",
            final_url="https://jobs.ashbyhq.com/joko/abc",
            title="Data Analyst Intern @ Some Aggregator",
        )
        job, _ = normalize_job(raw, null_llm)
        assert job.company == "Joko"


class TestNormalizationSafety:
    """Unsafe or generic pages never become postings."""

    def test_unsafe_url_is_rejected(self, null_llm):
        job, warning = normalize_job(make_raw(final_url="javascript:alert(1)"), null_llm)
        assert job is None
        assert "public web address" in warning

    def test_generic_listing_url_is_rejected(self, null_llm):
        job, warning = normalize_job(
            make_raw(
                url="https://www.linkedin.com/jobs/search/?k=x",
                final_url="https://www.linkedin.com/jobs/search/?k=x",
            ),
            null_llm,
        )
        assert job is None
        assert "listing" in warning
