"""Rejection rules: closed, snippet-only, generic, stale, and unreadable pages."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from tests.conftest import FIXED_NOW, make_job, make_raw
from models.job import normalize_work_mode
from tools.firecrawl_search import company_from_url, location_clause
from tools.job_filter import (
    filter_and_deduplicate,
    location_conflict,
    rejection_reason,
)
from tools.job_normalizer import (
    MIN_DESCRIPTION_CHARS,
    clean_description,
    compute_freshness,
    detect_closed,
    make_excerpt,
    strip_card_header,
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


class TestLocationFiltering:
    """A posting is dropped only when it names a place other than the one asked for."""

    def test_a_posting_in_another_country_is_removed(self):
        job = make_job(location="Riyadh Saudi Arabia")
        assert location_conflict("Houston, TX", job)

    def test_a_posting_in_another_us_city_is_removed(self):
        job = make_job(location="New York, NY", work_mode="onsite")
        assert location_conflict("Houston, TX", job)

    def test_the_requested_city_is_kept(self):
        assert not location_conflict("Houston, TX", make_job(location="Houston, TX"))

    def test_the_state_name_matches_its_abbreviation(self):
        job = make_job(location="Houston, Texas")
        assert not location_conflict("Houston, TX", job)

    def test_another_city_in_the_requested_state_is_removed(self):
        """Austin is in Texas, but it is not a Houston job."""
        job = make_job(location="Austin, TX", work_mode="onsite")
        assert location_conflict("Houston, TX", job)

    def test_a_commuting_suburb_is_kept(self):
        job = make_job(location="Sugar Land, TX", work_mode="onsite")
        assert not location_conflict("Houston, TX", job)

    def test_a_nationwide_posting_is_kept(self):
        job = make_job(location="United States")
        assert not location_conflict("Houston, TX", job)

    def test_a_placeless_remote_posting_is_kept(self):
        job = make_job(location="Remote", work_mode="remote")
        assert not location_conflict("Houston, TX", job)

    def test_a_remote_posting_anchored_abroad_is_removed(self):
        job = make_job(location="Hong Kong / Asia", work_mode="remote")
        assert location_conflict("Houston, TX", job)

    def test_a_posting_without_a_location_is_kept(self):
        assert not location_conflict("Houston, TX", make_job(location=None))

    def test_no_requested_location_filters_nothing(self):
        assert not location_conflict("", make_job(location="Singapore"))

    def test_a_country_search_keeps_every_city_in_it(self):
        """Searching "United States" must not require the words on the posting."""
        for city in ("Austin", "Chicago", "New York, NY", "San Francisco, CA"):
            job = make_job(location=city, work_mode="onsite")
            assert not location_conflict("United States", job), city

    @pytest.mark.parametrize(
        "abroad",
        ["Peru", "Lima, Peru", "Serbia", "Costa Rica", "Cambodia", "Venezuela"],
    )
    def test_a_country_the_parser_did_not_know_is_not_kept_as_unknown(self, abroad):
        """An unlisted country parsed as a city, and unknown places are kept.

        That put a Peru posting on screen under a search for the United States.
        """
        assert location_conflict("United States", make_job(location=abroad))

    def test_one_unreadable_city_does_not_rescue_a_list_of_foreign_ones(self):
        """A role open across Asia is not in the US because one city is unknown."""
        job = make_job(location="Singapore / Bangalore / Chennai / Cebu / Hanoi")
        assert location_conflict("United States", job)

    def test_a_slash_list_is_read_as_every_city_it_names(self):
        """Only the last segment used to be read, losing the rest."""
        job = make_job(location="Singapore / Bangalore / Austin, Texas")
        assert not location_conflict("United States", job)

    def test_a_board_hierarchy_of_non_places_is_still_kept(self):
        """A Lever team and commitment name no place, so they rule nothing out."""
        job = make_job(location="Engineering / Full-time / Remote")
        assert not location_conflict("United States", job)

    def test_a_country_search_drops_other_countries(self):
        for city in ("Bengaluru, India", "London, United Kingdom", "Singapore"):
            job = make_job(location=city, work_mode="onsite")
            assert location_conflict("United States", job), city

    def test_a_continent_is_evidence_of_a_different_country(self):
        job = make_job(location="Asia / South East Asia", work_mode="onsite")
        assert location_conflict("United States", job)

    def test_a_containing_region_still_qualifies(self):
        for label in ("North America", "Global", "Worldwide"):
            job = make_job(location=label, work_mode="onsite")
            assert not location_conflict("United States", job), label

    def test_a_state_search_matches_cities_in_that_state(self):
        for city in ("Houston, TX", "Austin", "Dallas"):
            job = make_job(location=city, work_mode="onsite")
            assert not location_conflict("Texas", job), city

    def test_a_state_search_drops_cities_elsewhere(self):
        for city in ("Chicago", "New York, NY", "San Francisco, CA"):
            job = make_job(location=city, work_mode="onsite")
            assert location_conflict("Texas", job), city

    def test_the_removal_reason_names_both_places(self):
        job = make_job(location="Singapore", work_mode="onsite")
        reason = rejection_reason(
            job, min_description_chars=400, requested_location="Houston, TX"
        )
        assert "Singapore" in reason and "Houston, TX" in reason

    def test_filtering_drops_the_off_location_posting_only(self):
        houston = make_job(location="Houston, TX", source_url="https://a.example/1")
        abroad = make_job(
            location="Riyadh Saudi Arabia", source_url="https://a.example/2"
        )
        outcome = filter_and_deduplicate(
            [houston, abroad], requested_location="Houston, TX"
        )
        assert outcome.kept == [houston]
        assert outcome.removed_count == 1


class TestWorkModeSpelling:
    """The UI label must reach the query and the scorer as a canonical value."""

    def test_the_onsite_label_is_canonicalized(self):
        assert normalize_work_mode("On-site") == "onsite"

    def test_canonical_values_pass_through(self):
        assert normalize_work_mode("remote") == "remote"
        assert normalize_work_mode("hybrid") == "hybrid"

    def test_an_unset_mode_reads_as_any(self):
        assert normalize_work_mode(None) == "any"
        assert normalize_work_mode("") == "any"

    def test_onsite_anchors_the_query_to_the_city(self):
        assert location_clause("Houston, TX", "On-site") == "(Houston)"

    def test_any_still_admits_remote(self):
        assert location_clause("Houston, TX", "Any") == "(Houston OR remote)"


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

    def test_the_card_header_is_not_repeated_in_the_excerpt(self):
        """Title, location, breadcrumb, and apply link already sit on the card."""
        raw = (
            "Entry Level Data Analyst Austin, TX Product / Regular Full-Time / "
            "On-site apply for this job Meds.com is a rapidly growing consumer "
            "technology firm operating a suite of healthcare businesses."
        )
        body = strip_card_header(
            raw, title="Entry Level Data Analyst", location="Austin, TX"
        )
        assert body.startswith("Meds.com is a rapidly growing")

    def test_prose_beginning_with_a_work_mode_word_is_untouched(self):
        """"Remote work is supported" is the posting talking, not page chrome."""
        raw = "Remote work is supported. We are hiring an analyst to own reporting."
        assert strip_card_header(raw, title="Data Analyst", location="Austin, TX") == raw

    def test_a_posting_with_no_header_is_untouched(self):
        raw = "Sysco is the global leader in foodservice distribution and offers a paid internship."
        assert strip_card_header(raw, title="Analytics Intern", location="Houston, TX") == raw

    def test_stripping_never_eats_the_whole_posting(self):
        """A description that is only header keeps the source text rather than emptying."""
        raw = "Data Analyst Houston, TX On-site apply for this job"
        assert strip_card_header(raw, title="Data Analyst", location="Houston, TX") == raw

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
