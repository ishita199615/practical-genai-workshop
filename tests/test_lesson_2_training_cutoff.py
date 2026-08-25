"""Step 2 must teach the training-cutoff idea with or without a live model.

Nothing here touches the network or needs an API key: the jobs come from the
repository's cached demonstration data, and every model is a stub.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import pytest

from config import Settings
from lessons.base import LessonContext, LessonResult
from lessons.step_2_training_cutoff import (
    CANNED_HALLUCINATION,
    MEMORY_PROMPT,
    STEP,
    describe_reply,
    find_urls,
    looks_like_refusal,
    retrieved_jobs_table,
    run,
)
from models.job import RawJobResult
from services.llm_interface import NullLLMClient
from tools.firecrawl_search import cached_raw_results, load_cache
from tools.job_normalizer import normalize_jobs

SETTINGS = Settings(demo_mode="cached")


class StubLLM:
    """A model stand-in whose availability and reply are both scripted."""

    def __init__(self, text: str | None = None, *, available: bool = True) -> None:
        self.available = available
        self.model_name = "stub"
        self.text = text
        self.prompts: list[str] = []

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        self.prompts.append(prompt)
        return self.text


class BoomLLM:
    """A model stand-in that claims to be available and then fails."""

    available = True
    model_name = "boom"

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        raise RuntimeError("503 model overloaded")


REFUSAL_REPLY = (
    "I don't have access to real-time job listings, so I can't tell you what "
    "was posted in Houston in the last 24 hours. Try Indeed or LinkedIn."
)

LINKS_REPLY = (
    "Here are three roles:\n"
    "1. Data Analyst Intern - Acme Corp https://careers.acme.example/jobs/1234\n"
    "2. Junior Data Analyst - Beta LLC https://jobs.beta.example/openings/77.\n"
    "3. BI Intern - Gamma Inc https://gamma.example/careers/bi-intern"
)

HEDGED_REPLY = (
    "I don't have real-time access to job boards, but here are some examples:\n"
    "https://careers.acme.example/jobs/1234\n"
    "https://jobs.beta.example/openings/77"
)

ADVICE_REPLY = (
    "Search for entry-level analytics roles on the major job boards and filter "
    "by the last day. Set up alerts for the Houston metro area."
)


@lru_cache(maxsize=1)
def _cached_jobs() -> tuple:
    """Normalize the offline cache exactly the way the agent does.

    Mirrors ``agent.nodes._load_cached_jobs``: read the cache, stamp the
    selected category, window, and original retrieval time onto each record,
    then normalize with an offline LLM so the deterministic path is exercised.
    """
    payload = load_cache(SETTINGS.cache_path)
    entries = cached_raw_results(
        payload, query_category="all", freshness_window="last_24_hours"
    )
    retrieved_at = datetime.fromisoformat(
        str(payload["originally_retrieved_at"]).replace("Z", "+00:00")
    )
    raw: list[RawJobResult] = []
    for entry in entries:
        record = dict(entry)
        record["query_category"] = "all"
        record["freshness_window"] = "last_24_hours"
        record["retrieved_at"] = retrieved_at
        raw.append(RawJobResult.model_validate(record))
    jobs, _ = normalize_jobs(raw, NullLLMClient(), data_mode="cached", limit=8)
    return tuple(jobs)


def make_ctx(llm: Any, resume, *, jobs: list | None = None) -> LessonContext:
    """Build a lesson context from cached data and a stub model."""
    return LessonContext(
        settings=SETTINGS,
        llm=llm,
        resume=resume,
        jobs=list(_cached_jobs()) if jobs is None else jobs,
    )


def kinds(result: LessonResult) -> list[str]:
    """The block kinds a run produced, in order."""
    return [block.kind for block in result.blocks]


def body_text(result: LessonResult) -> str:
    """All block bodies flattened to one lowercase string for searching."""
    return "\n".join(str(block.label) + "\n" + str(block.body) for block in result.blocks).lower()


class TestCachedJobsAreUsable:
    """The offline fixture really does yield checkable postings."""

    def test_the_cache_produces_jobs(self):
        assert len(_cached_jobs()) >= 3

    def test_every_job_has_a_url_and_a_retrieval_time(self):
        for job in _cached_jobs():
            assert job.source_url.startswith("http")
            assert job.retrieved_at is not None


class TestOfflinePath:
    """No model reachable: the step must still be complete and honest."""

    @pytest.fixture
    def result(self, resume) -> LessonResult:
        return STEP.execute(make_ctx(NullLLMClient(), resume))

    def test_it_reports_the_model_was_unavailable(self, result):
        assert result.llm_unavailable is True
        assert result.used_llm is False

    def test_it_still_produces_blocks(self, result):
        assert len(result.blocks) >= 1

    def test_the_canned_example_is_labelled_as_canned(self, result):
        canned = [b for b in result.blocks if CANNED_HALLUCINATION in str(b.body)]
        assert canned, "the canned illustration must be shown"
        assert "canned illustration" in canned[0].label.lower()

    def test_it_explains_that_the_deterministic_path_ran(self, result):
        notes = [b for b in result.blocks if b.kind == "note"]
        assert notes
        assert "not reachable" in notes[0].label.lower()

    def test_it_still_shows_the_real_retrieved_jobs(self, result):
        tables = [b for b in result.blocks if b.kind == "table"]
        assert tables
        rows = tables[0].body
        assert rows and set(rows[0]) == {
            "Title",
            "Company",
            "Source URL",
            "Retrieved at",
        }

    def test_the_canned_links_cannot_reach_a_real_employer(self, result):
        for url in find_urls(CANNED_HALLUCINATION):
            assert ".example/" in url

    def test_a_failing_model_falls_back_to_the_offline_path(self, resume):
        result = STEP.execute(make_ctx(BoomLLM(), resume))
        assert result.llm_unavailable is True
        assert result.used_llm is False
        assert any(b.kind == "table" for b in result.blocks)

    def test_an_available_model_returning_nothing_counts_as_unavailable(self, resume):
        result = STEP.execute(make_ctx(StubLLM(None), resume))
        assert result.llm_unavailable is True
        assert result.used_llm is False


class TestLivePath:
    """A reachable model: show its real answer and describe it accurately."""

    def test_the_unanswerable_question_is_what_gets_asked(self, resume):
        llm = StubLLM(REFUSAL_REPLY)
        run(make_ctx(llm, resume))
        assert llm.prompts == [MEMORY_PROMPT]

    def test_flags_are_set_for_a_live_call(self, resume):
        result = STEP.execute(make_ctx(StubLLM(REFUSAL_REPLY), resume))
        assert result.used_llm is True
        assert result.llm_unavailable is False

    def test_the_actual_reply_is_shown(self, resume):
        result = STEP.execute(make_ctx(StubLLM(LINKS_REPLY), resume))
        assert any(LINKS_REPLY.strip() in str(b.body) for b in result.blocks)

    def test_no_canned_example_appears_when_the_model_answered(self, resume):
        result = STEP.execute(make_ctx(StubLLM(LINKS_REPLY), resume))
        assert CANNED_HALLUCINATION not in body_text(result)

    def test_a_refusal_is_not_called_a_hallucination(self, resume):
        result = STEP.execute(make_ctx(StubLLM(REFUSAL_REPLY), resume))
        assert "success" in kinds(result)
        assert "refused" in body_text(result)

    def test_invented_links_are_warned_about_and_listed(self, resume):
        result = STEP.execute(make_ctx(StubLLM(LINKS_REPLY), resume))
        assert "warning" in kinds(result)
        listed = [b for b in result.blocks if b.kind == "json"]
        assert listed and len(listed[0].body["unverified_links"]) == 3

    def test_the_retrieved_jobs_table_is_always_shown(self, resume):
        for reply in (REFUSAL_REPLY, LINKS_REPLY, HEDGED_REPLY, ADVICE_REPLY):
            result = STEP.execute(make_ctx(StubLLM(reply), resume))
            assert any(b.kind == "table" for b in result.blocks), reply

    def test_a_very_long_reply_is_truncated_for_the_screen(self, resume):
        result = STEP.execute(make_ctx(StubLLM("word " * 2000), resume))
        code_blocks = [b for b in result.blocks if b.kind == "code"]
        assert any("truncated" in str(b.body) for b in code_blocks)


class TestReplyAssessment:
    """The step describes what happened, not what it hoped would happen."""

    def test_a_clean_refusal(self):
        assert describe_reply(REFUSAL_REPLY).kind == "refused"

    def test_links_with_no_disclaimer(self):
        assessment = describe_reply(LINKS_REPLY)
        assert assessment.kind == "listed_links"
        assert assessment.block_kind == "warning"

    def test_a_disclaimer_followed_by_links_anyway(self):
        assessment = describe_reply(HEDGED_REPLY)
        assert assessment.kind == "hedged_but_listed"
        assert len(assessment.urls) == 2

    def test_generic_advice_with_no_links(self):
        assert describe_reply(ADVICE_REPLY).kind == "no_links"

    def test_urls_are_deduplicated_and_stripped_of_punctuation(self):
        text = "see https://a.example/x. and https://a.example/x again"
        assert find_urls(text) == ["https://a.example/x"]

    def test_refusal_markers_are_case_insensitive(self):
        assert looks_like_refusal("I CANNOT browse the web")
        assert not looks_like_refusal("Here are three great roles for you")

    def test_empty_text_is_handled(self):
        assert find_urls("") == []
        assert looks_like_refusal("") is False


class TestTheStepNeverPutsWordsInTheModelsMouth:
    """The headline is a claim about what the model said. It must be true.

    Every reply below is a *confident* answer. Calling any of them a refusal
    would print "the model said it cannot know" over text that said the
    opposite - the exact mistake this step exists to warn a room about.
    """

    CONFIDENT_REPLIES = [
        "Here are three up-to-date postings from the last 24 hours:\n"
        "https://acme.example/1\nhttps://beta.example/2",
        "Browse these three openings posted today:\nhttps://acme.example/1",
        "Pulled from live data feeds this morning:\nhttps://acme.example/1",
        "These come straight off the real-time job boards:\nhttps://acme.example/1",
    ]

    ADVICE_REPLIES = [
        "You can browse Indeed and LinkedIn, and filter by the last 24 hours.",
        "Check that your saved searches are up to date on each job board.",
        "Don't forget to verify each posting before applying.",
    ]

    @pytest.mark.parametrize("reply", CONFIDENT_REPLIES)
    def test_a_confident_answer_is_never_called_a_refusal(self, reply):
        assessment = describe_reply(reply)
        assert assessment.kind == "listed_links"
        assert assessment.block_kind == "warning"
        assert "cannot know" not in assessment.headline

    @pytest.mark.parametrize("reply", ADVICE_REPLIES)
    def test_generic_advice_is_not_dressed_up_as_a_refusal(self, reply, resume):
        assert describe_reply(reply).kind == "no_links"
        result = STEP.execute(make_ctx(StubLLM(reply), resume))
        assert "success" not in kinds(result)
        assert "the model refused" not in body_text(result)

    @pytest.mark.parametrize(
        "reply",
        [
            "I'm unable to browse the internet, so I can't list today's postings.",
            "My knowledge cutoff means I have not seen anything posted this year.",
            "My knowledge ends before today, so any listing would be invented.",
            "Job postings are not in my training data at that granularity.",
        ],
    )
    def test_a_real_refusal_is_still_recognised(self, reply):
        assert describe_reply(reply).kind == "refused"

    def test_markdown_decoration_is_stripped_from_links(self):
        text = "See **https://acme.example/jobs/1** and `https://b.example/2`."
        assert find_urls(text) == [
            "https://acme.example/jobs/1",
            "https://b.example/2",
        ]

    def test_the_canned_example_cannot_be_confused_with_a_retrieved_job(self):
        """The contrast only reads if the two lists share nothing."""
        canned = CANNED_HALLUCINATION.lower()
        for job in _cached_jobs():
            assert job.company.lower() not in canned
            assert job.source_url.lower() not in canned


class TestHostileClients:
    """A provider can misbehave in ways no stub author anticipated."""

    class AvailabilityRaises:
        model_name = "x"

        @property
        def available(self) -> bool:
            raise RuntimeError("client not constructed yet")

        def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str:
            return "hi"

    class Blank:
        available = True
        model_name = "x"

        def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str:
            return "   \n\t  \n "

    class NotText:
        available = True
        model_name = "x"

        def generate_text(self, prompt: str, *, temperature: float = 0.2) -> Any:
            return {"text": "nope"}

    class NoMethod:
        available = True
        model_name = "x"

    class WrongSignature:
        available = True
        model_name = "x"

        def generate_text(self, prompt: str) -> str:
            return "positional only"

    ALL = [AvailabilityRaises, Blank, NotText, NoMethod, WrongSignature]

    @pytest.mark.parametrize("client", ALL)
    def test_the_step_still_teaches(self, client, resume):
        result = STEP.execute(make_ctx(client(), resume))
        assert any(b.kind == "table" for b in result.blocks)

    @pytest.mark.parametrize("client", ALL)
    def test_the_two_flags_never_contradict_each_other(self, client, resume):
        result = STEP.execute(make_ctx(client(), resume))
        assert result.used_llm is not result.llm_unavailable

    @pytest.mark.parametrize("client", ALL)
    def test_no_block_is_rendered_empty(self, client, resume):
        """An empty box under "what the model actually replied" teaches nothing."""
        result = STEP.execute(make_ctx(client(), resume))
        for block in result.blocks:
            if isinstance(block.body, str):
                assert block.body.strip(), block.label

    @pytest.mark.parametrize("client", ALL)
    def test_canned_and_live_output_never_appear_together(self, client, resume):
        result = STEP.execute(make_ctx(client(), resume))
        labels = [b.label for b in result.blocks]
        canned = any("CANNED" in label for label in labels)
        live = any("actually replied" in label for label in labels)
        assert not (canned and live)

    def test_a_blank_reply_counts_as_no_reply(self, resume):
        result = STEP.execute(make_ctx(self.Blank(), resume))
        assert result.used_llm is False
        assert result.llm_unavailable is True


class TestNeverRaises:
    """A student clicking Run must always see something."""

    def test_no_jobs_loaded(self, resume):
        result = STEP.execute(make_ctx(NullLLMClient(), resume, jobs=[]))
        assert result.blocks
        assert any(b.kind == "warning" for b in result.blocks)

    def test_a_malformed_job_object_does_not_crash_the_table(self):
        rows = retrieved_jobs_table([object()])
        assert rows == [
            {
                "Title": "unknown",
                "Company": "unknown",
                "Source URL": "not available",
                "Retrieved at": "not recorded",
            }
        ]

    def test_a_naive_datetime_is_formatted(self):
        class Job:
            title = "T"
            company = "C"
            source_url = "https://x.example/1"
            retrieved_at = datetime(2026, 8, 20, 14, 5, tzinfo=timezone.utc)

        assert retrieved_jobs_table([Job()])[0]["Retrieved at"].startswith("2026-08-20")

    def test_a_context_with_a_none_model(self, resume):
        result = STEP.execute(make_ctx(None, resume))
        assert result.llm_unavailable is True
        assert result.blocks

    def test_the_table_shows_at_most_three_rows(self, resume):
        result = STEP.execute(make_ctx(NullLLMClient(), resume))
        table = next(b for b in result.blocks if b.kind == "table")
        assert 1 <= len(table.body) <= 3


class TestStepMetadata:
    """The Learn tab reads these fields directly."""

    def test_it_is_step_two(self):
        assert STEP.number == 2
        assert STEP.deck_reference

    def test_it_declares_that_it_prefers_the_model(self):
        assert STEP.needs_llm is True

    def test_the_teaching_snippet_stays_readable_on_a_projector(self):
        lines = STEP.code.strip().splitlines()
        assert 10 <= len(lines) <= 25
        assert max(len(line) for line in lines) <= 88

    def test_the_takeaway_is_one_sentence(self):
        assert STEP.takeaway.count(".") <= 1

    def test_the_prose_fields_are_filled_in(self):
        for field in (STEP.title, STEP.subtitle, STEP.concept, STEP.why):
            assert field.strip()

    def test_the_run_callable_is_the_module_function(self):
        assert STEP.run is run
