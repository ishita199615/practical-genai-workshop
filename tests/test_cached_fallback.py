"""Live-to-cache fallback, and the honesty rules around it."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from agent.nodes import AgentDeps, search_current_jobs
from config import Settings
from models.job import RawJobResult
from services.llm_interface import NullLLMClient
from tests.conftest import FIXED_NOW
from tools.firecrawl_search import (
    FAIL_FAST_TIMEOUT_SECONDS,
    FirecrawlError,
    FirecrawlSearchAdapter,
    build_search_request,
    cached_raw_results,
    filter_to_category,
    load_cache,
    raw_results_to_models,
    search_with_domain_retry,
)

BASE_STATE: dict[str, Any] = {
    "role": "Data Analyst Intern",
    "location": "Houston, TX",
    "work_mode": "Any",
    "query_category": "company_careers",
    "freshness_window": "last_24_hours",
}


class FakeClient:
    """A stand-in Firecrawl client with scripted behaviour."""

    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def search(self, query: str, **kwargs: Any):
        self.calls.append((query, kwargs))
        if self.error:
            raise self.error
        return {"web": self.results}


def make_deps(client: FakeClient | None, demo_mode: str = "auto") -> AgentDeps:
    """Build dependencies wired to a fake search client."""
    settings = Settings(demo_mode=demo_mode, firecrawl_api_key="test-key")
    adapter = FirecrawlSearchAdapter(
        api_key="test-key",
        client_factory=lambda key, url: client,
    )
    return AgentDeps(
        settings=settings,
        search_adapter=adapter,
        llm=NullLLMClient(),
        now=lambda: FIXED_NOW,
    )


LIVE_RESULT = {
    "url": "https://job-boards.greenhouse.io/acme/jobs/1?utm_source=x",
    "title": "Data Analyst Intern",
    "markdown": "Data Analyst Intern at Acme. " + ("Responsibilities include SQL. " * 40),
    "metadata": {"sourceURL": "https://job-boards.greenhouse.io/acme/jobs/1"},
}


class TestAdapter:
    """The adapter sends the resolved request and normalizes the response."""

    def test_the_request_carries_query_domains_and_tbs(self):
        client = FakeClient(results=[LIVE_RESULT])
        adapter = FirecrawlSearchAdapter(
            api_key="k", client_factory=lambda key, url: client
        )
        request = build_search_request(
            role="Data Analyst Intern",
            location="Houston, TX",
            work_mode="Any",
            query_category="linkedin",
            freshness_window="last_hour",
            now=FIXED_NOW,
        )
        adapter.search(request)
        query, kwargs = client.calls[0]
        assert query.startswith("site:linkedin.com/jobs/view")
        assert kwargs["include_domains"] == ["linkedin.com"]
        assert kwargs["tbs"] == "sbd:1,qdr:h"
        assert kwargs["limit"] == 8

    def test_errors_become_firecrawl_errors(self):
        adapter = FirecrawlSearchAdapter(
            api_key="k",
            client_factory=lambda key, url: FakeClient(error=RuntimeError("401")),
        )
        request = build_search_request(
            role="r",
            location="Houston, TX",
            work_mode="Any",
            query_category="all",
            freshness_window="last_24_hours",
            now=FIXED_NOW,
        )
        with pytest.raises(FirecrawlError):
            adapter.search(request)


class TestLiveRetrieval:
    """A successful live call is labelled live."""

    def test_live_results_are_labelled_live(self):
        deps = make_deps(FakeClient(results=[LIVE_RESULT]))
        update = search_current_jobs(dict(BASE_STATE), deps)
        assert update["data_mode"] == "live"
        assert update["raw_jobs"]
        assert update["retrieval_timestamp"] == FIXED_NOW


class LadderClient:
    """Returns results only once a named hint has been dropped.

    Mirrors the measured live behaviour: the time filter is what empties a
    result set, and the domain filter can empty it too.
    """

    def __init__(self, results, empty_while: str = "tbs") -> None:
        self.results = results
        self.empty_while = empty_while
        self.calls: list[dict] = []

    def search(self, query: str, **kwargs: Any):
        self.calls.append(kwargs)
        if self.empty_while == "tbs" and kwargs.get("tbs"):
            return {"web": []}
        if self.empty_while == "domains" and kwargs.get("include_domains"):
            return {"web": []}
        return {"web": self.results}


def make_adapter(client) -> FirecrawlSearchAdapter:
    """Build an adapter wired to a fake client."""
    return FirecrawlSearchAdapter(api_key="k", client_factory=lambda key, url: client)


class TestRetryLadder:
    """Server-side hints are relaxed in order, and each step is reported."""

    def _request(self, category="company_careers"):
        return build_search_request(
            role="Data Analyst Intern",
            location="Houston, TX",
            work_mode="Any",
            query_category=category,
            freshness_window="last_24_hours",
            now=FIXED_NOW,
        )

    def test_a_successful_first_attempt_is_not_retried(self):
        client = FakeClient(results=[LIVE_RESULT])
        outcome = search_with_domain_retry(make_adapter(client), self._request())
        assert len(client.calls) == 1
        assert outcome.notes == []
        assert outcome.results
        assert outcome.time_filter_applied is True

    def test_an_empty_time_filtered_search_drops_only_the_time_filter(self):
        client = LadderClient([LIVE_RESULT], empty_while="tbs")
        outcome = search_with_domain_retry(make_adapter(client), self._request())
        assert len(outcome.results) == 1
        assert len(client.calls) == 2
        assert client.calls[0]["tbs"] == "sbd:1,qdr:d"
        assert "tbs" not in client.calls[1]
        # The domain filter — the part that enforces the user's category — is kept.
        assert client.calls[1]["include_domains"]
        assert outcome.notes and "time filter" in outcome.notes[0].lower()

    def test_dropping_the_time_filter_is_recorded(self):
        client = LadderClient([LIVE_RESULT], empty_while="tbs")
        outcome = search_with_domain_retry(make_adapter(client), self._request())
        assert outcome.time_filter_applied is False

    def test_the_domain_filter_is_dropped_only_as_a_last_resort(self):
        client = LadderClient([LIVE_RESULT], empty_while="domains")
        outcome = search_with_domain_retry(make_adapter(client), self._request())
        assert len(client.calls) == 3
        assert client.calls[0]["include_domains"] and client.calls[0]["tbs"]
        assert client.calls[1]["include_domains"] and "tbs" not in client.calls[1]
        assert "include_domains" not in client.calls[2]
        assert outcome.notes and "domain" in outcome.notes[0].lower()

    def test_the_query_never_changes_across_the_ladder(self):
        client = LadderClient([LIVE_RESULT], empty_while="domains")
        adapter = make_adapter(client)
        request = self._request()
        search_with_domain_retry(adapter, request)
        assert client.calls  # every attempt used the same query string
        assert len({call.get("limit") for call in client.calls}) == 1

    def test_a_broad_category_still_retries_without_the_time_filter(self):
        client = LadderClient([LIVE_RESULT], empty_while="tbs")
        outcome = search_with_domain_retry(make_adapter(client), self._request("all"))
        assert len(client.calls) == 2
        assert outcome.results

    def test_the_time_filtered_attempt_fails_fast(self):
        """A likely-empty first attempt must not eat the demo's time budget."""
        client = LadderClient([LIVE_RESULT], empty_while="tbs")
        request = build_search_request(
            role="Data Analyst Intern",
            location="Houston, TX",
            work_mode="Any",
            query_category="company_careers",
            freshness_window="last_24_hours",
            timeout_seconds=25,
            now=FIXED_NOW,
        )
        search_with_domain_retry(make_adapter(client), request)
        assert client.calls[0]["timeout"] == FAIL_FAST_TIMEOUT_SECONDS * 1000
        # The attempt that actually has to scrape keeps the full budget.
        assert client.calls[1]["timeout"] == 25_000

    def test_a_short_configured_timeout_is_not_extended(self):
        """The cap only shortens; it never grants more time than configured."""
        client = LadderClient([LIVE_RESULT], empty_while="tbs")
        request = build_search_request(
            role="Data Analyst Intern",
            location="Houston, TX",
            work_mode="Any",
            query_category="company_careers",
            freshness_window="last_24_hours",
            timeout_seconds=5,
            now=FIXED_NOW,
        )
        search_with_domain_retry(make_adapter(client), request)
        assert client.calls[0]["timeout"] == 5_000

    def test_exhausting_the_ladder_returns_nothing(self):
        client = FakeClient(results=[])
        outcome = search_with_domain_retry(make_adapter(client), self._request())
        assert outcome.results == []
        assert outcome.notes == []


class TestCategoryEnforcement:
    """A relaxed retry can never put an off-category result on screen."""

    def _models(self, urls: list[str]) -> list[RawJobResult]:
        models, _ = raw_results_to_models(
            [{"url": url, "markdown": "x" * 600} for url in urls],
            query_category="company_careers",
            freshness_window="last_24_hours",
            retrieved_at=FIXED_NOW,
        )
        return models

    def test_off_category_results_are_dropped(self):
        models = self._models(
            [
                "https://jobs.lever.co/acme/1",
                "https://www.linkedin.com/jobs/view/2",
                "https://www.ziprecruiter.com/jobs/3",
            ]
        )
        kept, dropped = filter_to_category(models, "company_careers")
        assert [job.detected_source_label for job in kept] == ["Lever"]
        assert dropped == 2

    def test_broad_categories_pass_everything_through(self):
        models = self._models(
            ["https://jobs.lever.co/acme/1", "https://www.linkedin.com/jobs/view/2"]
        )
        kept, dropped = filter_to_category(models, "all")
        assert len(kept) == 2
        assert dropped == 0

    def test_the_node_enforces_the_category_after_a_retry(self):
        off_category = {
            "url": "https://www.ziprecruiter.com/jobs/entry-level-data-analyst",
            "markdown": "Entry level data analyst. " + ("Responsibilities. " * 40),
        }
        client = LadderClient([off_category], empty_while="none")
        adapter = FirecrawlSearchAdapter(
            api_key="test-key", client_factory=lambda key, url: client
        )
        deps = AgentDeps(
            settings=Settings(demo_mode="auto", firecrawl_api_key="test-key"),
            search_adapter=adapter,
            llm=NullLLMClient(),
            now=lambda: FIXED_NOW,
        )
        update = search_current_jobs(dict(BASE_STATE), deps)
        # Every live result was off-category, so the run falls back to the cache
        # rather than showing a ZipRecruiter page under "Direct Company Careers".
        assert update["data_mode"] == "cached"
        assert any("outside the selected category" in w for w in update["warnings"])


class TestCacheFallback:
    """Failures fall back to the cache, and the cache says it is the cache."""

    def test_an_api_error_falls_back_to_the_cache(self):
        deps = make_deps(FakeClient(error=RuntimeError("boom")))
        update = search_current_jobs(dict(BASE_STATE), deps)
        assert update["data_mode"] == "cached"
        assert update["raw_jobs"]
        assert any("cached" in warning.lower() for warning in update["warnings"])

    def test_an_empty_response_falls_back_to_the_cache(self):
        deps = make_deps(FakeClient(results=[]))
        update = search_current_jobs(dict(BASE_STATE), deps)
        assert update["data_mode"] == "cached"

    def test_cached_mode_never_calls_the_api(self):
        client = FakeClient(results=[LIVE_RESULT])
        deps = make_deps(client, demo_mode="cached")
        update = search_current_jobs(dict(BASE_STATE), deps)
        assert client.calls == []
        assert update["data_mode"] == "cached"

    def test_cached_data_is_never_presented_as_live(self):
        deps = make_deps(FakeClient(error=RuntimeError("boom")))
        update = search_current_jobs(dict(BASE_STATE), deps)
        labels = [item["label"] for item in update["progress_events"]]
        assert any("cached" in label.lower() for label in labels)
        assert not any("Firecrawl searched" in label for label in labels)

    def test_live_only_mode_reports_an_error_instead_of_using_the_cache(self):
        deps = make_deps(FakeClient(results=[]), demo_mode="live")
        update = search_current_jobs(dict(BASE_STATE), deps)
        assert update["data_mode"] == "live"
        assert update["raw_jobs"] == []
        assert update["errors"]


class TestCacheContents:
    """The rehearsal cache satisfies the workshop requirements."""

    def test_the_cache_is_labelled_and_dated(self, settings):
        payload = load_cache(settings.cache_path)
        assert payload["cache_label"]
        assert payload["synthetic"] is True
        assert payload["data_notice"]
        assert payload["originally_retrieved_at"]
        assert payload["modification_log"]

    def test_the_cache_holds_at_least_five_valid_postings(self, settings):
        payload = load_cache(settings.cache_path)
        usable = [
            job
            for job in payload["jobs"]
            if job["metadata"]["cached_extraction"].get("is_specific_opening")
            and not job["metadata"]["cached_extraction"].get("is_closed")
        ]
        assert len(usable) >= 5

    def test_the_cache_spans_two_source_categories(self, settings):
        payload = load_cache(settings.cache_path)
        categories = {job["detected_source_category"] for job in payload["jobs"]}
        assert {"company_careers", "linkedin"} <= categories

    def test_the_cache_includes_a_power_bi_posting(self, settings):
        payload = load_cache(settings.cache_path)
        assert any(
            "Power BI" in job["metadata"]["cached_extraction"].get("required_skills", [])
            for job in payload["jobs"]
        )

    def test_the_cache_includes_a_closed_posting(self, settings):
        payload = load_cache(settings.cache_path)
        assert any(
            job["metadata"]["cached_extraction"].get("is_closed")
            for job in payload["jobs"]
        )

    def test_every_cached_url_is_a_public_web_address(self, settings):
        from tools.firecrawl_search import is_safe_public_job_url

        payload = load_cache(settings.cache_path)
        for job in payload["jobs"]:
            assert is_safe_public_job_url(job["final_url"])

    def test_cached_records_validate_as_raw_results(self, settings):
        payload = load_cache(settings.cache_path)
        for entry in payload["jobs"]:
            record = dict(entry)
            record["query_category"] = "all"
            record["freshness_window"] = "last_24_hours"
            record["retrieved_at"] = datetime.now(timezone.utc)
            assert RawJobResult.model_validate(record)

    def test_a_category_filter_narrows_the_cache(self, settings):
        payload = load_cache(settings.cache_path)
        linkedin = cached_raw_results(
            payload, query_category="linkedin", freshness_window="last_24_hours"
        )
        assert linkedin
        assert all(job["detected_source_category"] == "linkedin" for job in linkedin)

    def test_a_broad_category_returns_everything(self, settings):
        payload = load_cache(settings.cache_path)
        everything = cached_raw_results(
            payload, query_category="all", freshness_window="last_24_hours"
        )
        assert len(everything) == len(payload["jobs"])
