"""Query construction, freshness mapping, URL safety, and source detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tools.firecrawl_search import (
    build_search_request,
    build_search_query,
    canonicalize_job_url,
    classify_source,
    domain_filter_for,
    freshness_cutoff,
    freshness_to_tbs,
    is_safe_public_job_url,
    looks_like_generic_listing,
    raw_results_to_models,
)

ROLE = "Data Analyst Intern"
LOCATION = "Houston, TX"
NOW = datetime(2026, 8, 20, 15, 0, 0, tzinfo=timezone.utc)


class TestQueryConstruction:
    """Each source category shapes the query in its own documented way."""

    def test_all_public_sources(self):
        assert build_search_query(ROLE, LOCATION, "Any", "all") == (
            '("data analyst intern" OR "entry level data analyst") (Houston OR remote)'
        )

    def test_linkedin_uses_public_job_view_path(self):
        assert build_search_query(ROLE, LOCATION, "Any", "linkedin") == (
            "site:linkedin.com/jobs/view "
            '("data analyst intern" OR "entry level data analyst") (Houston OR remote)'
        )

    def test_indeed_uses_public_viewjob_path(self):
        assert build_search_query(ROLE, LOCATION, "Any", "indeed") == (
            "site:indeed.com/viewjob "
            '("data analyst intern" OR "entry level data analyst") (Houston OR remote)'
        )

    def test_google_jobs_appends_jobs_term(self):
        assert build_search_query(ROLE, LOCATION, "Any", "google_jobs").endswith(
            "(Houston OR remote) jobs"
        )

    def test_company_careers_uses_plain_query(self):
        assert build_search_query(ROLE, LOCATION, "Any", "company_careers") == (
            '("data analyst intern" OR "entry level data analyst") (Houston OR remote)'
        )

    def test_remote_preference_drops_the_city(self):
        query = build_search_query(ROLE, LOCATION, "Remote", "all")
        assert query.endswith("(remote)")
        assert "Houston" not in query


class TestExperienceLevelQueries:
    """The selected seniority shapes the quoted role clause, nothing else."""

    @pytest.mark.parametrize(
        "category", ["all", "linkedin", "indeed", "google_jobs", "company_careers"]
    )
    def test_unknown_level_reproduces_todays_query(self, category):
        assert build_search_query(
            ROLE, LOCATION, "Any", category, "unknown"
        ) == build_search_query(ROLE, LOCATION, "Any", category)

    def test_unknown_level_is_the_default(self):
        assert build_search_query(ROLE, LOCATION, "Any", "all") == (
            '("data analyst intern" OR "entry level data analyst") (Houston OR remote)'
        )

    @pytest.mark.parametrize(
        ("level", "terms"),
        [
            ("internship", ["data analyst intern", "data analyst internship"]),
            (
                "entry",
                [
                    "entry level data analyst",
                    "junior data analyst",
                    "associate data analyst",
                ],
            ),
            ("mid", ["data analyst", "mid level data analyst"]),
            ("senior", ["senior data analyst", "sr data analyst"]),
            (
                "staff_principal",
                [
                    "staff data analyst",
                    "principal data analyst",
                    "lead data analyst",
                ],
            ),
            (
                "manager",
                [
                    "data analyst manager",
                    "director of data analyst",
                    "head of data analyst",
                ],
            ),
        ],
    )
    def test_each_level_produces_its_own_quoted_terms(self, level, terms):
        expected = "(" + " OR ".join(f'"{term}"' for term in terms) + ")"
        query = build_search_query(ROLE, LOCATION, "Any", "all", level)
        assert query == f"{expected} (Houston OR remote)"

    def test_senior_query_matches_the_documented_example(self):
        assert build_search_query(ROLE, LOCATION, "Any", "all", "senior") == (
            '("senior data analyst" OR "sr data analyst") (Houston OR remote)'
        )

    def test_role_already_carrying_a_level_word_is_not_contradicted(self):
        query = build_search_query(
            "Senior Data Analyst", LOCATION, "Any", "all", "internship"
        )
        assert query == (
            '("data analyst intern" OR "data analyst internship") (Houston OR remote)'
        )
        assert "senior" not in query.lower()

    def test_typed_intern_role_upgraded_to_senior_drops_the_intern_word(self):
        query = build_search_query(ROLE, LOCATION, "Any", "all", "senior")
        assert "intern" not in query.lower()

    @pytest.mark.parametrize(
        ("category", "prefix"),
        [
            ("all", ""),
            ("linkedin", "site:linkedin.com/jobs/view "),
            ("indeed", "site:indeed.com/viewjob "),
            ("google_jobs", ""),
            ("company_careers", ""),
        ],
    )
    def test_level_combines_with_each_source_category(self, category, prefix):
        query = build_search_query(ROLE, LOCATION, "Any", category, "senior")
        expected = (
            f'{prefix}("senior data analyst" OR "sr data analyst") (Houston OR remote)'
        )
        if category == "google_jobs":
            expected += " jobs"
        assert query == expected

    def test_level_combines_with_the_remote_work_mode_clause(self):
        query = build_search_query(ROLE, LOCATION, "Remote", "all", "senior")
        assert query == '("senior data analyst" OR "sr data analyst") (remote)'
        assert "Houston" not in query

    def test_level_combines_with_the_hybrid_work_mode_clause(self):
        query = build_search_query(ROLE, LOCATION, "Hybrid", "all", "manager")
        assert query.endswith("(Houston OR hybrid)")

    def test_level_combines_with_an_onsite_other_city(self):
        query = build_search_query(
            "Data Scientist", "Austin, TX", "Onsite", "all", "senior"
        )
        assert query == '("senior data scientist" OR "sr data scientist") (Austin)'

    def test_empty_role_still_produces_a_location_only_query(self):
        assert build_search_query("", LOCATION, "Any", "all", "senior") == (
            "(Houston OR remote)"
        )


class TestExperienceLevelInSearchRequest:
    """The resolved request describes the seniority it was built for."""

    def test_request_defaults_to_unknown_level(self):
        request = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="company_careers",
            freshness_window="last_24_hours",
            now=NOW,
        )
        assert request.experience_level == "unknown"
        assert request.query == (
            '("data analyst intern" OR "entry level data analyst") (Houston OR remote)'
        )

    def test_request_carries_the_selected_level_into_the_query(self):
        request = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="linkedin",
            freshness_window="last_hour",
            experience_level="senior",
            now=NOW,
        )
        assert request.experience_level == "senior"
        assert request.query == (
            "site:linkedin.com/jobs/view "
            '("senior data analyst" OR "sr data analyst") (Houston OR remote)'
        )
        assert request.include_domains == ["linkedin.com"]
        assert request.tbs == "sbd:1,qdr:h"

    def test_level_does_not_disturb_domains_or_freshness(self):
        without = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="company_careers",
            freshness_window="last_3_days",
            now=NOW,
        )
        with_level = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="company_careers",
            freshness_window="last_3_days",
            experience_level="staff_principal",
            now=NOW,
        )
        assert with_level.include_domains == without.include_domains
        assert with_level.tbs == without.tbs
        assert with_level.location == without.location
        assert with_level.query != without.query

    def test_payload_reports_the_level_as_application_metadata(self):
        payload = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="indeed",
            freshness_window="last_24_hours",
            experience_level="entry",
            now=NOW,
        ).as_payload()
        assert payload["experienceLevel"] == "entry"
        assert payload["queryCategory"] == "indeed"
        assert payload["freshnessWindow"] == "last_24_hours"
        # Application metadata only: the level is not a Firecrawl search field.
        assert "experience_level" not in payload

    def test_payload_defaults_to_unknown_level(self):
        payload = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="all",
            freshness_window="last_24_hours",
            now=NOW,
        ).as_payload()
        assert payload["experienceLevel"] == "unknown"


class TestDomainFilters:
    """Domain filters follow the selected category, never the result."""

    def test_linkedin_and_indeed_filters(self):
        assert domain_filter_for("linkedin") == ["linkedin.com"]
        assert domain_filter_for("indeed") == ["indeed.com"]

    def test_company_careers_uses_ats_domains(self):
        domains = domain_filter_for("company_careers")
        assert "boards.greenhouse.io" in domains
        assert "jobs.lever.co" in domains
        assert "jobs.ashbyhq.com" in domains

    def test_broad_categories_have_no_forced_domain(self):
        assert domain_filter_for("all") == []
        assert domain_filter_for("google_jobs") == []


class TestFreshnessMapping:
    """Freshness windows map onto Firecrawl ``tbs`` values."""

    def test_last_hour_uses_qdr_h(self):
        assert freshness_to_tbs("last_hour", NOW) == "sbd:1,qdr:h"

    def test_last_24_hours_uses_qdr_d(self):
        assert freshness_to_tbs("last_24_hours", NOW) == "sbd:1,qdr:d"

    def test_last_7_days_uses_qdr_w(self):
        assert freshness_to_tbs("last_7_days", NOW) == "sbd:1,qdr:w"

    def test_last_3_days_builds_a_runtime_date_range(self):
        tbs = freshness_to_tbs("last_3_days", NOW)
        assert tbs == "sbd:1,cdr:1,cd_min:08/17/2026,cd_max:08/20/2026"

    def test_custom_range_is_not_hard_coded(self):
        later = NOW + timedelta(days=10)
        assert freshness_to_tbs("last_3_days", later) != freshness_to_tbs(
            "last_3_days", NOW
        )

    @pytest.mark.parametrize(
        ("window", "hours"),
        [
            ("last_hour", 1),
            ("last_24_hours", 24),
            ("last_3_days", 72),
            ("last_7_days", 168),
        ],
    )
    def test_local_verification_cutoffs(self, window, hours):
        assert freshness_cutoff(window, NOW) == NOW - timedelta(hours=hours)

    def test_unknown_window_is_rejected(self):
        with pytest.raises(ValueError):
            freshness_to_tbs("last_year", NOW)  # type: ignore[arg-type]


class TestCategoryAndFreshnessCombinations:
    """Category and freshness are independent controls, combined in one request."""

    def test_linkedin_plus_last_hour(self):
        request = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="linkedin",
            freshness_window="last_hour",
            now=NOW,
        )
        assert request.query.startswith("site:linkedin.com/jobs/view")
        assert request.include_domains == ["linkedin.com"]
        assert request.tbs == "sbd:1,qdr:h"

    def test_indeed_plus_last_24_hours(self):
        request = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="indeed",
            freshness_window="last_24_hours",
            now=NOW,
        )
        assert request.query.startswith("site:indeed.com/viewjob")
        assert request.include_domains == ["indeed.com"]
        assert request.tbs == "sbd:1,qdr:d"

    def test_company_careers_plus_last_24_hours(self):
        request = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="company_careers",
            freshness_window="last_24_hours",
            now=NOW,
        )
        assert "site:" not in request.query
        assert "boards.greenhouse.io" in request.include_domains
        assert request.tbs == "sbd:1,qdr:d"

    def test_request_never_exceeds_eight_results(self):
        request = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="all",
            freshness_window="last_24_hours",
            limit=50,
            now=NOW,
        )
        assert request.limit == 8

    def test_payload_carries_application_metadata_separately(self):
        payload = build_search_request(
            role=ROLE,
            location=LOCATION,
            work_mode="Any",
            query_category="linkedin",
            freshness_window="last_hour",
            now=NOW,
        ).as_payload()
        assert payload["queryCategory"] == "linkedin"
        assert payload["freshnessWindow"] == "last_hour"
        assert payload["scrapeOptions"] == {
            "formats": ["markdown"],
            "onlyMainContent": True,
        }


class TestUrlSafety:
    """Only public http/https links are ever rendered or followed."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://boards.greenhouse.io/acme/jobs/1",
            "http://jobs.lever.co/acme/abc",
        ],
    )
    def test_public_urls_are_accepted(self, url):
        assert is_safe_public_job_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(1)",
            "data:text/html;base64,PHNjcmlwdD4=",
            "ftp://example.com/job",
            "https://user:pass@example.com/job",
            "not a url",
            "",
            None,
        ],
    )
    def test_unsafe_urls_are_rejected(self, url):
        assert not is_safe_public_job_url(url)


class TestCanonicalization:
    """Tracking noise is stripped; identity-bearing parameters are kept."""

    def test_tracking_parameters_are_removed(self):
        assert (
            canonicalize_job_url(
                "https://Boards.Greenhouse.io/acme/jobs/123/?utm_source=x&gh_src=y#apply"
            )
            == "https://boards.greenhouse.io/acme/jobs/123"
        )

    def test_identity_parameters_are_preserved(self):
        assert (
            canonicalize_job_url("https://www.indeed.com/viewjob?jk=abc123&from=serp")
            == "https://www.indeed.com/viewjob?jk=abc123&from=serp"
        )

    def test_canonicalization_is_idempotent(self):
        once = canonicalize_job_url("https://jobs.lever.co/acme/abc/?ref=twitter")
        assert canonicalize_job_url(once) == once


class TestSourceClassification:
    """The actual source comes from the final URL, never from the query."""

    @pytest.mark.parametrize(
        ("url", "category", "label"),
        [
            ("https://www.linkedin.com/jobs/view/123", "linkedin", "LinkedIn"),
            ("https://www.indeed.com/viewjob?jk=1", "indeed", "Indeed"),
            (
                "https://boards.greenhouse.io/acme/jobs/1",
                "company_careers",
                "Greenhouse",
            ),
            ("https://jobs.lever.co/acme/1", "company_careers", "Lever"),
            ("https://jobs.ashbyhq.com/acme/1", "company_careers", "Ashby"),
            (
                "https://jobs.smartrecruiters.com/Acme/1",
                "company_careers",
                "SmartRecruiters",
            ),
        ],
    )
    def test_known_domains(self, url, category, label):
        assert classify_source(url) == (category, label)

    def test_unknown_domain_is_other(self):
        category, _ = classify_source("https://careers.exampleco.com/jobs/analyst-1")
        assert category == "other"

    def test_a_linkedin_result_from_an_all_search_is_still_linkedin(self):
        raw, _ = raw_results_to_models(
            [{"url": "https://www.linkedin.com/jobs/view/999", "markdown": "text"}],
            query_category="all",
            freshness_window="last_24_hours",
            retrieved_at=NOW,
        )
        assert raw[0].query_category == "all"
        assert raw[0].detected_source_category == "linkedin"


class TestGenericListingDetection:
    """Search and index pages are not specific openings."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/jobs/search/?keywords=analyst",
            "https://example.com/careers",
            "https://example.com/jobs",
            "https://www.indeed.com/q-data-analyst-jobs.html",
        ],
    )
    def test_listing_pages_are_detected(self, url):
        assert looks_like_generic_listing(url)

    def test_a_specific_opening_is_not_a_listing(self):
        assert not looks_like_generic_listing(
            "https://boards.greenhouse.io/acme/jobs/4410021"
        )


class TestRawResultConversion:
    """Both URLs survive conversion, and unsafe results are dropped."""

    def test_canonical_and_original_urls_are_both_preserved(self):
        models, warnings = raw_results_to_models(
            [
                {
                    "url": "https://jobs.lever.co/acme/abc?utm_source=feed",
                    "markdown": "text",
                }
            ],
            query_category="company_careers",
            freshness_window="last_24_hours",
            retrieved_at=NOW,
        )
        assert warnings == []
        assert models[0].url == "https://jobs.lever.co/acme/abc?utm_source=feed"
        assert models[0].final_url == "https://jobs.lever.co/acme/abc"

    def test_unsafe_results_are_dropped_with_a_warning(self):
        models, warnings = raw_results_to_models(
            [{"url": "javascript:alert(1)", "markdown": "text"}],
            query_category="all",
            freshness_window="last_24_hours",
            retrieved_at=NOW,
        )
        assert models == []
        assert warnings
