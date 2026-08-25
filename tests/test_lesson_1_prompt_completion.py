"""Step 1 must teach the same lesson whether or not the model answers.

Nothing here touches the network or needs an API key: the LLM is a scripted
stub and the job list comes from the offline cache, exactly as the Learn tab
builds it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from config import Settings, load_settings
from lessons.base import LessonContext, LessonResult, OutputBlock, approx_tokens
from lessons.step_1_prompt_completion import (
    CANNED_CREATIVE,
    CANNED_LABEL,
    CANNED_PREDICTABLE,
    PROMPT,
    STEP,
)
from models.job import JobPosting, RawJobResult
from models.resume import ResumeProfile
from services.llm_interface import NullLLMClient
from tools.firecrawl_search import cached_raw_results, load_cache
from tools.job_normalizer import normalize_jobs


class StubLLM:
    """A scripted stand-in for the project's LLM client.

    ``replies`` maps a temperature to the text that call should return. A
    missing entry (or ``None``) means the provider declined, which is what a
    quota-limited free tier looks like from the caller's side.
    """

    def __init__(
        self,
        *,
        available: bool = True,
        replies: dict[float, str | None] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._available = available
        self._replies = replies if replies is not None else {0.0: "LOW-TEMP REPLY", 1.0: "HIGH-TEMP REPLY"}
        self._raises = raises
        self.calls: list[tuple[str, float]] = []

    @property
    def available(self) -> bool:
        """True when this stub is pretending the provider is reachable."""
        return self._available

    @property
    def model_name(self) -> str:
        """A fake model ID so nothing in the UI needs a real one."""
        return "stub-model-1.0"

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Record the call and return the scripted reply for ``temperature``."""
        self.calls.append((prompt, temperature))
        if self._raises is not None:
            raise self._raises
        return self._replies.get(temperature)


class WrongTypeLLM:
    """A client that answers with something that is not text at all.

    ``generate_text`` is typed ``str | None``, but the client is injected, so a
    future or third-party provider can break that promise. The step must treat
    it as "no usable reply" rather than crashing on ``.strip()``.
    """

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    @property
    def available(self) -> bool:
        """The provider looks perfectly reachable; only its reply is wrong."""
        return True

    @property
    def model_name(self) -> str:
        """A fake model ID so nothing in the UI needs a real one."""
        return "wrong-type-1.0"

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> Any:
        """Return the non-string payload this stub was built with."""
        return self._payload


class HostileLLM:
    """A client whose very ``available`` check explodes."""

    @property
    def available(self) -> bool:
        """Always raises, to prove the step survives a broken client."""
        raise RuntimeError("provider handshake failed")

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Never reached; present only to satisfy the protocol."""
        raise RuntimeError("provider handshake failed")


def cached_job_postings(settings: Settings) -> list[JobPosting]:
    """Build real :class:`JobPosting` objects from the offline cache.

    Mirrors what ``agent/nodes.py`` does in cached mode, so the lesson runs
    against the same data the demo shows.
    """
    payload = load_cache(settings.cache_path)
    entries = cached_raw_results(
        payload, query_category="all", freshness_window="last_24_hours"
    )
    retrieved_at = datetime(2026, 8, 20, 15, 0, tzinfo=timezone.utc)
    raw_jobs: list[RawJobResult] = []
    for entry in entries:
        record: dict[str, Any] = dict(entry)
        record["query_category"] = "all"
        record["freshness_window"] = "last_24_hours"
        record["retrieved_at"] = retrieved_at
        raw_jobs.append(RawJobResult.model_validate(record))
    postings, _warnings = normalize_jobs(
        raw_jobs,
        NullLLMClient(),
        max_description_chars=settings.max_job_description_chars,
        data_mode="cached",
        limit=settings.max_job_results,
    )
    return postings


@pytest.fixture(scope="module")
def lesson_settings() -> Settings:
    """Settings loaded the way the app loads them, forced to cached mode."""
    loaded = load_settings()
    return Settings(
        cache_file=loaded.cache_file,
        max_job_results=loaded.max_job_results,
        max_job_description_chars=loaded.max_job_description_chars,
        demo_mode="cached",
    )


@pytest.fixture(scope="module")
def cached_jobs(lesson_settings: Settings) -> list[JobPosting]:
    """The cached postings, normalized once for the whole module."""
    return cached_job_postings(lesson_settings)


@pytest.fixture
def make_ctx(lesson_settings: Settings, cached_jobs: list[JobPosting], resume: ResumeProfile):
    """Return a factory that builds a LessonContext around a given client."""

    def _build(llm: Any) -> LessonContext:
        return LessonContext(
            settings=lesson_settings, llm=llm, resume=resume, jobs=cached_jobs
        )

    return _build


def blocks_of(result: LessonResult, kind: str) -> list[OutputBlock]:
    """Every block of one kind, in order."""
    return [block for block in result.blocks if block.kind == kind]


def rendered_text(result: LessonResult) -> str:
    """All labels and string bodies flattened, for substring assertions."""
    parts: list[str] = []
    for block in result.blocks:
        parts.append(block.label)
        if isinstance(block.body, str):
            parts.append(block.body)
        elif isinstance(block.body, dict):
            parts.extend(str(value) for value in block.body.values())
    return "\n".join(parts)


class TestStepMetadata:
    """The step satisfies the LessonStep contract the Learn tab relies on."""

    def test_it_is_step_one_and_prefers_the_model(self):
        assert STEP.number == 1
        assert STEP.needs_llm is True
        assert callable(STEP.run)

    def test_every_teaching_field_is_filled_in(self):
        for field_name in ("title", "subtitle", "concept", "why", "deck_reference", "takeaway"):
            assert getattr(STEP, field_name).strip(), f"{field_name} is empty"

    def test_the_takeaway_is_one_sentence(self):
        assert STEP.takeaway.count(".") == 1

    def test_the_on_screen_snippet_stays_readable(self):
        lines = STEP.code.strip().splitlines()
        assert 10 <= len(lines) <= 25
        assert "temperature=0.0" in STEP.code
        assert "temperature=1.0" in STEP.code
        assert "approx_tokens" in STEP.code

    def test_the_snippet_states_the_real_availability_rule(self):
        """The snippet is shown as the code that ran, so it must not simplify.

        Regression: it used to set ``llm_unavailable`` only when *both* calls
        failed, while ``run()`` sets it when *either* does.
        """
        assert "not (low_ok and high_ok)" in STEP.code

    def test_the_snippet_counts_both_replies_like_run_does(self):
        """Regression: the snippet counted one reply, the metric showed two."""
        reply_line = next(
            line for line in STEP.code.splitlines() if line.startswith("reply_tokens")
        )
        assert reply_line.count("approx_tokens") == 2
        assert "left" in reply_line and "right" in reply_line


class TestOfflinePath:
    """With no model reachable the step still teaches the whole idea."""

    @pytest.fixture
    def result(self, make_ctx) -> LessonResult:
        return STEP.execute(make_ctx(NullLLMClient()))

    def test_it_never_raises_and_always_renders(self, result):
        assert len(result.blocks) >= 1

    def test_the_flags_say_the_model_was_not_used(self, result):
        assert result.used_llm is False
        assert result.llm_unavailable is True

    def test_a_note_explains_the_deterministic_path(self, result):
        notes = blocks_of(result, "note")
        assert notes, "the offline path must explain itself"
        assert "offline path" in notes[0].label.lower()

    def test_both_completions_are_labelled_as_canned(self, result):
        compare = blocks_of(result, "compare")[0].body
        assert CANNED_LABEL in compare["left_label"]
        assert CANNED_LABEL in compare["right_label"]

    def test_the_canned_pair_still_shows_a_temperature_difference(self, result):
        compare = blocks_of(result, "compare")[0].body
        assert compare["left"] == CANNED_PREDICTABLE
        assert compare["right"] == CANNED_CREATIVE
        assert compare["left"] != compare["right"]

    def test_the_prompt_and_token_estimates_are_still_shown(self, result):
        code_blocks = blocks_of(result, "code")
        assert code_blocks[0].body == PROMPT
        metrics = blocks_of(result, "metric")
        assert len(metrics) == 2
        assert f"~{approx_tokens(PROMPT)} tokens" in metrics[0].body["value"]

    def test_it_is_not_claimed_to_be_a_model_response(self, result):
        assert not blocks_of(result, "success")

    def test_the_note_does_not_oversell_the_reply_token_count(self, result):
        """The prompt count matches a live run; the reply count cannot.

        Regression: the note claimed every number shown was "exactly what a
        live run would show", but the reply figure is measured from the canned
        text, and a real reply is a different length.
        """
        note = blocks_of(result, "note")[0].body
        assert "canned text" in note
        assert "different length" in note
        assert "token estimates - is calculated locally and is exactly" not in note


class TestLivePath:
    """With a reachable model both replies are real and labelled as such."""

    @pytest.fixture
    def stub(self) -> StubLLM:
        return StubLLM(
            replies={0.0: "A precise, repeatable sentence.", 1.0: "A livelier sentence entirely."}
        )

    @pytest.fixture
    def result(self, make_ctx, stub: StubLLM) -> LessonResult:
        return STEP.execute(make_ctx(stub))

    def test_the_flags_say_the_model_answered(self, result):
        assert result.used_llm is True
        assert result.llm_unavailable is False

    def test_the_same_prompt_is_sent_at_both_temperatures(self, result, stub):
        assert [temperature for _prompt, temperature in stub.calls] == [0.0, 1.0]
        assert {prompt for prompt, _temperature in stub.calls} == {PROMPT}

    def test_both_model_replies_are_displayed(self, result):
        compare = blocks_of(result, "compare")[0].body
        assert compare["left"] == "A precise, repeatable sentence."
        assert compare["right"] == "A livelier sentence entirely."

    def test_nothing_is_marked_canned(self, result):
        assert CANNED_LABEL not in rendered_text(result)

    def test_reply_tokens_are_estimated_from_the_real_replies(self, result):
        expected = approx_tokens("A precise, repeatable sentence.") + approx_tokens(
            "A livelier sentence entirely."
        )
        metrics = blocks_of(result, "metric")
        assert f"~{expected} tokens" in metrics[1].body["value"]


class TestPartialAndBrokenProviders:
    """Anything short of two real replies is reported, never smoothed over."""

    def test_one_missing_reply_is_labelled_canned_on_that_side_only(self, make_ctx):
        stub = StubLLM(replies={0.0: "Only the low-temperature call answered.", 1.0: None})
        result = STEP.execute(make_ctx(stub))
        compare = blocks_of(result, "compare")[0].body
        assert result.used_llm is True
        assert result.llm_unavailable is True
        assert CANNED_LABEL not in compare["left_label"]
        assert CANNED_LABEL in compare["right_label"]
        assert compare["right"] == CANNED_CREATIVE
        assert blocks_of(result, "warning")

    @pytest.mark.parametrize(
        "client",
        [
            None,
            StubLLM(available=False),
            StubLLM(raises=RuntimeError("503 model overloaded")),
            StubLLM(replies={0.0: "   ", 1.0: ""}),
            HostileLLM(),
        ],
        ids=["no-client", "unavailable", "raises-503", "blank-replies", "hostile"],
    )
    def test_every_broken_client_falls_back_cleanly(self, make_ctx, client):
        result = STEP.execute(make_ctx(client))
        assert result.blocks
        assert result.used_llm is False
        assert result.llm_unavailable is True
        assert blocks_of(result, "compare")[0].body["left"] == CANNED_PREDICTABLE

    @pytest.mark.parametrize(
        "payload",
        [{"text": "a dict, not a string"}, ["a", "list"], 12345, object()],
        ids=["dict", "list", "int", "object"],
    )
    def test_a_non_text_reply_is_treated_as_no_reply(self, make_ctx, payload):
        """A reply that is not a string must not crash the step.

        Regression: ``.strip()`` used to run outside the guarded block, so a
        non-string reply reduced the whole step to a generic failure notice and
        left ``llm_unavailable`` False - which reads as "the model was fine".
        """
        result = STEP.execute(make_ctx(WrongTypeLLM(payload)))

        assert [block.kind for block in result.blocks] != ["warning"], (
            "a non-text reply must not collapse the step into a crash notice"
        )
        assert result.used_llm is False
        assert result.llm_unavailable is True
        compare = blocks_of(result, "compare")[0].body
        assert compare["left"] == CANNED_PREDICTABLE
        assert CANNED_LABEL in compare["left_label"]
        assert CANNED_LABEL in compare["right_label"]


class TestBlockShapes:
    """The app renders blocks blindly, so their bodies must be well formed."""

    @pytest.fixture(params=["offline", "live"])
    def result(self, request, make_ctx) -> LessonResult:
        client = NullLLMClient() if request.param == "offline" else StubLLM()
        return STEP.execute(make_ctx(client))

    def test_the_step_is_timed(self, result):
        assert result.elapsed_seconds >= 0.0

    def test_metric_bodies_carry_a_value_and_help(self, result):
        for block in blocks_of(result, "metric"):
            assert set(block.body) == {"value", "help"}
            assert block.body["value"] and block.body["help"]

    def test_compare_bodies_carry_both_labelled_sides(self, result):
        for block in blocks_of(result, "compare"):
            assert set(block.body) == {"left_label", "left", "right_label", "right"}
            assert all(block.body.values())
            assert "0.0" in block.body["left_label"]
            assert "1.0" in block.body["right_label"]

    def test_the_four_chars_per_token_rule_is_explained(self, result):
        text = rendered_text(result).lower()
        assert "four characters" in text or "/ 4" in text

    def test_no_api_key_material_is_rendered(self, result, lesson_settings):
        text = rendered_text(result)
        assert "api_key" not in text.lower()
        if lesson_settings.gemini_api_key:
            assert lesson_settings.gemini_api_key not in text


class TestContextRealism:
    """The lesson runs against the same cached data the demo uses."""

    def test_the_offline_cache_yields_usable_postings(self, cached_jobs):
        assert cached_jobs, "the cached demonstration data should normalize offline"
        assert all(job.description for job in cached_jobs)
