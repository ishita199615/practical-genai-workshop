"""LLM failure handling: quota limits, bad output, and honest fallbacks.

Every LLM step must degrade to a deterministic path rather than to a guess, and
the interface must say when that happened.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.nodes import (
    QUOTA_WARNING,
    AgentDeps,
    draft_application,
    normalize_jobs_node,
)
from config import Settings
from models.job import ExtractedJobFields
from services.gemini_client import GeminiClient, _is_quota_error, build_llm_client
from services.llm_interface import NullLLMClient
from tests.conftest import FIXED_NOW, make_raw
from tools.ats_scorer import assess_ats
from tools.firecrawl_search import FirecrawlSearchAdapter


class BoomClient:
    """A google-genai stand-in whose calls always raise."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs: Any):
        self.calls += 1
        raise self.exc


class TestQuotaDetection:
    """A rate or quota limit is recognised and reported, never hidden."""

    @pytest.mark.parametrize(
        "message",
        [
            "429 RESOURCE_EXHAUSTED. You exceeded your current quota",
            "Rate limit reached for this model",
            "quota exceeded",
        ],
    )
    def test_quota_errors_are_recognised(self, message):
        assert _is_quota_error(Exception(message))

    @pytest.mark.parametrize(
        "message",
        ["404 NOT_FOUND model missing", "500 internal error", "invalid argument"],
    )
    def test_other_errors_are_not_quota_errors(self, message):
        assert not _is_quota_error(Exception(message))

    def test_a_quota_error_sets_the_flag_and_returns_none(self, monkeypatch):
        monkeypatch.setattr("services.gemini_client.QUOTA_BACKOFF_SECONDS", 0.0)
        client = GeminiClient(
            api_key="k", model="m", client=BoomClient(Exception("429 RESOURCE_EXHAUSTED"))
        )
        assert client.generate_structured("p", ExtractedJobFields) is None
        assert client.quota_limited is True

    def test_a_non_quota_error_does_not_set_the_flag(self):
        client = GeminiClient(
            api_key="k", model="m", client=BoomClient(Exception("404 NOT_FOUND"))
        )
        assert client.generate_structured("p", ExtractedJobFields) is None
        assert client.quota_limited is False

    def test_both_attempts_are_made_before_giving_up(self, monkeypatch):
        monkeypatch.setattr("services.gemini_client.QUOTA_BACKOFF_SECONDS", 0.0)
        boom = BoomClient(Exception("429 quota"))
        GeminiClient(api_key="k", model="m", client=boom).generate_structured(
            "p", ExtractedJobFields
        )
        assert boom.calls == 2

    def test_text_generation_also_flags_quota(self):
        client = GeminiClient(
            api_key="k", model="m", client=BoomClient(Exception("429 quota"))
        )
        assert client.generate_text("p") is None
        assert client.quota_limited is True


class TestSecretSafety:
    """Error details are useful without leaking the key."""

    def test_the_api_key_is_redacted_from_error_detail(self):
        secret = "AIza-super-secret-value"
        client = GeminiClient(
            api_key=secret,
            model="m",
            client=BoomClient(Exception(f"401 bad key {secret} rejected")),
        )
        detail = client._safe_detail(Exception(f"401 bad key {secret}"))
        assert secret not in detail
        assert "<redacted>" in detail

    def test_detail_is_truncated(self):
        client = GeminiClient(api_key="k", model="m")
        assert len(client._safe_detail(Exception("x" * 5000))) <= 200


class TestFallbackReporting:
    """A quota-limited run still completes, and says why it degraded."""

    def _deps(self, llm) -> AgentDeps:
        return AgentDeps(
            settings=Settings(demo_mode="cached"),
            search_adapter=FirecrawlSearchAdapter(),
            llm=llm,
            now=lambda: FIXED_NOW,
        )

    def test_normalization_reports_a_quota_limit(self):
        llm = GeminiClient(
            api_key="k", model="m", client=BoomClient(Exception("429 quota"))
        )
        llm.quota_limited = True
        state = {"raw_jobs": [make_raw()], "data_mode": "cached"}
        update = normalize_jobs_node(state, self._deps(llm))
        assert QUOTA_WARNING in update["warnings"]

    def test_normalization_still_produces_postings(self):
        llm = GeminiClient(
            api_key="k", model="m", client=BoomClient(Exception("429 quota"))
        )
        state = {"raw_jobs": [make_raw()], "data_mode": "cached"}
        update = normalize_jobs_node(state, self._deps(llm))
        # The deterministic fallback still reads the page text.
        assert update["normalized_jobs"]

    def test_no_warning_when_the_model_is_healthy(self):
        state = {"raw_jobs": [make_raw()], "data_mode": "cached"}
        update = normalize_jobs_node(state, self._deps(NullLLMClient()))
        assert QUOTA_WARNING not in update["warnings"]

    def test_drafting_reports_a_quota_limit(self, resume, now):
        from tests.conftest import make_job

        llm = GeminiClient(
            api_key="k", model="m", client=BoomClient(Exception("429 quota"))
        )
        llm.quota_limited = True
        job = make_job()
        state = {
            "selected_job": job,
            "resume": resume,
            "ats_assessment": assess_ats(job, resume, now=now),
        }
        update = draft_application(state, self._deps(llm))
        assert QUOTA_WARNING in update["warnings"]
        # The package is still produced, from the deterministic path.
        assert update["tailored_application"] is not None


class TestClientSelection:
    """Configuration decides which client the graph gets."""

    def test_a_key_yields_a_gemini_client(self):
        assert isinstance(build_llm_client("some-key", "gemini-3.6-flash"), GeminiClient)

    def test_no_key_yields_the_null_client(self):
        client = build_llm_client("", "gemini-3.6-flash")
        assert isinstance(client, NullLLMClient)
        assert client.available is False

    def test_the_model_id_comes_from_configuration(self):
        assert build_llm_client("k", "gemini-3.7-flash").model_name == "gemini-3.7-flash"
