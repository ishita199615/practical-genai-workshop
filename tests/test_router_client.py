"""Tests for the model-routing LLM client.

Routing is what keeps the workshop alive when one model's daily quota runs out,
so these tests pin the behaviour that matters: the chain is passed in order, a
failure never raises, quota limits are reported rather than hidden, and the API
key never reaches a log line.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel

from services.router_client import (
    KEYLESS_PROVIDERS,
    RouterClient,
    _strip_code_fence,
    parse_model_chain,
    provider_of,
)


class Shape(BaseModel):
    """Small schema used to exercise structured output."""

    title: str
    count: int = 0


def _reply(content: str, model: str = "served-model"):
    """Build a minimal object shaped like a LiteLLM completion response."""

    class _Message:
        def __init__(self, text: str) -> None:
            self.content = text

    class _Choice:
        def __init__(self, text: str) -> None:
            self.message = _Message(text)

    class _Response:
        def __init__(self, text: str, model_name: str) -> None:
            self.choices = [_Choice(text)]
            self.model = model_name

    return _Response(content, model)


class TestChainParsing:
    def test_splits_and_trims(self) -> None:
        assert parse_model_chain(" gemini/a , gemini/b ") == ["gemini/a", "gemini/b"]

    def test_drops_blank_entries(self) -> None:
        assert parse_model_chain("gemini/a,,  ,gemini/b") == ["gemini/a", "gemini/b"]

    def test_empty_string_is_empty_chain(self) -> None:
        assert parse_model_chain("") == []

    def test_order_is_preserved_because_order_is_the_policy(self) -> None:
        raw = "gemini/first,gemini/second,gemini/third"
        assert parse_model_chain(raw) == ["gemini/first", "gemini/second", "gemini/third"]

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini/gemini-3.5-flash", "gemini"),
            ("ollama/llama3.1", "ollama"),
            ("openrouter/meta-llama/llama-3", "openrouter"),
            ("gemini-3.5-flash", "gemini"),
        ],
    )
    def test_provider_detection(self, model: str, expected: str) -> None:
        assert provider_of(model) == expected


class TestAvailability:
    def test_unavailable_without_models(self) -> None:
        assert RouterClient(api_key="k", models=[]).available is False

    def test_available_with_key(self) -> None:
        assert RouterClient(api_key="k", models=["gemini/x"]).available is True

    def test_local_provider_needs_no_key(self) -> None:
        client = RouterClient(api_key="", models=["ollama/llama3.1"])
        assert client.available is True

    def test_remote_provider_without_key_is_unavailable(self) -> None:
        client = RouterClient(api_key="", models=["gemini/x"])
        assert client.available is False

    def test_keyless_providers_are_local_only(self) -> None:
        assert "ollama" in KEYLESS_PROVIDERS
        assert "gemini" not in KEYLESS_PROVIDERS

    def test_model_name_defaults_to_first_in_chain(self) -> None:
        client = RouterClient(api_key="k", models=["gemini/a", "gemini/b"])
        assert client.model_name == "gemini/a"

    def test_chain_is_a_copy(self) -> None:
        client = RouterClient(api_key="k", models=["gemini/a"])
        client.chain.append("gemini/mutated")
        assert client.chain == ["gemini/a"]


class TestRouting:
    def test_passes_remaining_models_as_fallbacks(self) -> None:
        seen: dict = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return _reply("hi")

        client = RouterClient(
            api_key="k",
            models=["gemini/a", "gemini/b", "gemini/c"],
            completion_fn=fake,
        )
        client.generate_text("prompt")
        assert seen["model"] == "gemini/a"
        assert seen["fallbacks"] == ["gemini/b", "gemini/c"]

    def test_single_model_sends_no_fallbacks(self) -> None:
        seen: dict = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return _reply("hi")

        RouterClient(api_key="k", models=["gemini/a"], completion_fn=fake).generate_text("p")
        assert "fallbacks" not in seen

    def test_records_which_model_actually_served(self) -> None:
        client = RouterClient(
            api_key="k",
            models=["gemini/a", "gemini/b"],
            completion_fn=lambda **_: _reply("ok", model="gemini-b-served"),
        )
        assert client.generate_text("p") == "ok"
        assert client.last_served_by == "gemini-b-served"
        # The UI shows the model that answered, not the one first requested.
        assert client.model_name == "gemini-b-served"

    def test_failure_returns_none_and_does_not_raise(self) -> None:
        def boom(**_):
            raise RuntimeError("everything failed")

        client = RouterClient(api_key="k", models=["gemini/a"], completion_fn=boom)
        assert client.generate_text("p") is None

    def test_quota_error_is_reported_not_hidden(self) -> None:
        def boom(**_):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

        client = RouterClient(api_key="k", models=["gemini/a"], completion_fn=boom)
        assert client.quota_limited is False
        client.generate_text("p")
        assert client.quota_limited is True

    def test_non_quota_error_does_not_set_quota_flag(self) -> None:
        def boom(**_):
            raise RuntimeError("500 internal error")

        client = RouterClient(api_key="k", models=["gemini/a"], completion_fn=boom)
        client.generate_text("p")
        assert client.quota_limited is False

    def test_blank_reply_becomes_none(self) -> None:
        client = RouterClient(
            api_key="k", models=["gemini/a"], completion_fn=lambda **_: _reply("   ")
        )
        assert client.generate_text("p") is None

    def test_unexpected_response_shape_returns_none(self) -> None:
        client = RouterClient(
            api_key="k", models=["gemini/a"], completion_fn=lambda **_: object()
        )
        assert client.generate_text("p") is None

    def test_no_models_returns_none(self) -> None:
        client = RouterClient(api_key="k", models=[], completion_fn=lambda **_: _reply("x"))
        assert client.generate_text("p") is None


class TestStructuredOutput:
    def test_valid_json_is_parsed(self) -> None:
        client = RouterClient(
            api_key="k",
            models=["gemini/a"],
            completion_fn=lambda **_: _reply('{"title": "Analyst", "count": 3}'),
        )
        result = client.generate_structured("p", Shape)
        assert result is not None
        assert (result.title, result.count) == ("Analyst", 3)

    def test_schema_is_forwarded_as_response_format(self) -> None:
        seen: dict = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return _reply('{"title": "x"}')

        RouterClient(api_key="k", models=["gemini/a"], completion_fn=fake).generate_structured(
            "p", Shape
        )
        assert seen["response_format"] is Shape

    def test_code_fenced_json_is_recovered(self) -> None:
        fenced = '```json\n{"title": "Fenced", "count": 1}\n```'
        client = RouterClient(
            api_key="k", models=["gemini/a"], completion_fn=lambda **_: _reply(fenced)
        )
        result = client.generate_structured("p", Shape)
        assert result is not None and result.title == "Fenced"

    def test_invalid_json_retries_then_gives_up(self) -> None:
        calls = {"n": 0}

        def fake(**_):
            calls["n"] += 1
            return _reply("not json at all")

        client = RouterClient(api_key="k", models=["gemini/a"], completion_fn=fake)
        assert client.generate_structured("p", Shape) is None
        # One retry, matching the spec's "retry once, then reject" rule.
        assert calls["n"] == 2

    def test_transport_failure_returns_none_without_retrying(self) -> None:
        calls = {"n": 0}

        def boom(**_):
            calls["n"] += 1
            raise RuntimeError("network down")

        client = RouterClient(api_key="k", models=["gemini/a"], completion_fn=boom)
        assert client.generate_structured("p", Shape) is None
        assert calls["n"] == 1


class TestSecrecy:
    def test_api_key_is_redacted_from_error_detail(self) -> None:
        secret = "super-secret-key-value"
        client = RouterClient(api_key=secret, models=["gemini/a"])
        detail = client._safe_detail(RuntimeError(f"failed using {secret} boom"))
        assert secret not in detail
        assert "<redacted>" in detail

    def test_key_never_appears_in_log_output(self, caplog) -> None:
        secret = "leaky-key-1234567890"

        def boom(**_):
            raise RuntimeError(f"auth failed for {secret}")

        client = RouterClient(api_key=secret, models=["gemini/a"], completion_fn=boom)
        with caplog.at_level(logging.WARNING):
            client.generate_text("p")
        assert secret not in caplog.text


class TestFenceHelper:
    def test_plain_text_unchanged(self) -> None:
        assert _strip_code_fence('{"a": 1}') == '{"a": 1}'

    def test_strips_opening_and_closing_fence(self) -> None:
        assert _strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_bare_fence(self) -> None:
        assert _strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'
