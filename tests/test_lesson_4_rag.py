"""Tests for lesson step 4 (RAG).

Everything here runs with no network and no API key. Jobs come from the
clearly-labelled cached demonstration data through the same normalization and
filtering the app itself uses, so the lesson is exercised against real
:class:`JobPosting` objects rather than a hand-written stub.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from config import Settings, load_settings
from lessons.base import LessonContext, LessonResult
from lessons.context import load_cached_postings
from lessons.step_4_rag import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    QUESTION,
    STEP,
    TOP_K,
    build_augmented_prompt,
    build_chunks,
    extractive_answer,
    retrieve,
    split_into_chunks,
    unique_postings,
)
from models.job import JobPosting, RawJobResult
from models.resume import ResumeProfile
from services.llm_interface import NullLLMClient
from tools.firecrawl_search import cached_raw_results, load_cache
from tools.job_filter import filter_and_deduplicate
from tools.job_normalizer import normalize_jobs

CANNED_ANSWER = (
    "Lakeside Analytics and Bayou Insights both name Power BI [chunk_003]."
)


# --------------------------------------------------------------------------
# Stub clients
# --------------------------------------------------------------------------


class OfflineLLM:
    """A client that cannot reach a model, like a quota-limited free tier."""

    available = False

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        raise AssertionError("An unavailable client must never be called.")


class ScriptedLLM:
    """A client that returns canned text and records the prompt it received."""

    available = True

    def __init__(self, reply: str = CANNED_ANSWER) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        self.prompts.append(prompt)
        return self.reply


class ExplodingLLM:
    """A client that claims to be available and then fails mid-call."""

    available = True

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        raise RuntimeError("503 model overloaded")


class BlankReplyLLM:
    """A client that answers with whitespace — a real Gemini safety-block shape."""

    available = True

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        return "   \n\t  "


class NonStringLLM:
    """A client that breaks its contract and returns a non-string payload."""

    available = True

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> Any:
        return {"candidates": []}


class UnreadableAvailabilityLLM:
    """A client whose availability probe itself raises."""

    @property
    def available(self) -> bool:
        raise RuntimeError("credential lookup failed")

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        return "should never be reached"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _cached_job_postings(settings: Settings) -> list[JobPosting]:
    """Build validated postings from the offline cache, as the app does."""
    payload = load_cache(settings.cache_path)
    entries = cached_raw_results(
        payload, query_category="all", freshness_window="last_24_hours"
    )
    retrieved_at = datetime.fromisoformat(
        str(payload["originally_retrieved_at"]).replace("Z", "+00:00")
    )
    raw: list[RawJobResult] = []
    for entry in entries:
        record: dict[str, Any] = dict(entry)
        record["query_category"] = "all"
        record["freshness_window"] = "last_24_hours"
        record["retrieved_at"] = retrieved_at
        raw.append(RawJobResult.model_validate(record))
    postings, _warnings = normalize_jobs(
        raw, NullLLMClient(), data_mode="cached", limit=settings.max_job_results
    )
    return filter_and_deduplicate(postings).kept


@pytest.fixture(scope="module")
def cached_jobs() -> list[JobPosting]:
    """The offline corpus the lesson retrieves from."""
    jobs = _cached_job_postings(load_settings())
    assert jobs, "The cached demonstration data must yield at least one posting."
    return jobs


@pytest.fixture
def app_jobs(settings: Settings) -> list[JobPosting]:
    """The corpus the Learn tab really passes in.

    Distinct from :func:`cached_jobs`: the tab builds its context with
    :func:`lessons.context.load_cached_postings`, which does *not* run
    ``filter_and_deduplicate``. The cache deliberately holds the same Greenhouse
    posting under two tracking URLs, so this list contains a duplicate and the
    step must cope with exactly that.
    """
    jobs = load_cached_postings(settings)
    assert jobs, "The Learn tab must have postings to teach with."
    return jobs


def make_ctx(
    jobs: list[JobPosting], resume: ResumeProfile, llm: Any, settings: Settings
) -> LessonContext:
    """Assemble a lesson context around a given stub client."""
    return LessonContext(settings=settings, llm=llm, resume=resume, jobs=jobs)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_split_into_chunks_respects_size_and_overlap() -> None:
    """Windows are the requested size and consecutive windows overlap."""
    text = " ".join(f"word{index:04d}" for index in range(600))
    chunks = split_into_chunks(text, size=500, overlap=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)
    # The tail of one chunk reappears at the head of the next.
    assert chunks[0][-40:] in chunks[1]


def test_split_into_chunks_handles_empty_and_short_text() -> None:
    """No text yields no chunks; short text yields exactly one."""
    assert split_into_chunks("") == []
    assert split_into_chunks("   \n  ") == []
    assert split_into_chunks("Short posting body.") == ["Short posting body."]


def test_build_chunks_tags_every_piece_with_its_source(
    cached_jobs: list[JobPosting],
) -> None:
    """Each chunk carries a unique ID and the job it came from."""
    chunks = build_chunks(cached_jobs)

    assert len(chunks) > len(cached_jobs)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    job_ids = {job.job_id for job in cached_jobs}
    assert {chunk.job_id for chunk in chunks} <= job_ids
    assert all(chunk.text for chunk in chunks)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def test_retrieve_returns_top_k_scored_chunks(cached_jobs: list[JobPosting]) -> None:
    """Retrieval returns TOP_K chunks in descending similarity order."""
    chunks = build_chunks(cached_jobs)
    ranked, vocabulary, shape, warning = retrieve(QUESTION, chunks, top_k=TOP_K)

    assert warning is None
    assert len(ranked) == TOP_K
    assert vocabulary > 0
    assert shape == (len(chunks), vocabulary)
    scores = [score for _chunk, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.0


def test_retrieve_finds_power_bi_chunks(cached_jobs: list[JobPosting]) -> None:
    """The demo question surfaces chunks that actually mention Power BI."""
    chunks = build_chunks(cached_jobs)
    ranked, _vocab, _shape, _warning = retrieve(QUESTION, chunks, top_k=TOP_K)

    assert any("power bi" in chunk.text.lower() for chunk, _score in ranked)


def test_retrieve_on_empty_corpus_is_safe() -> None:
    """An empty corpus returns empty results rather than raising."""
    assert retrieve(QUESTION, [], top_k=TOP_K) == ([], 0, (0, 0), None)


# --------------------------------------------------------------------------
# Prompt assembly and the offline answer
# --------------------------------------------------------------------------


def test_augmented_prompt_contains_question_and_only_top_chunks(
    cached_jobs: list[JobPosting],
) -> None:
    """The prompt carries the question and the retrieved chunks, nothing more."""
    chunks = build_chunks(cached_jobs)
    ranked, _vocab, _shape, _warning = retrieve(QUESTION, chunks, top_k=TOP_K)
    prompt = build_augmented_prompt(QUESTION, ranked)

    assert QUESTION in prompt
    for chunk, _score in ranked:
        assert chunk.chunk_id in prompt
        assert chunk.text in prompt
    # Chunks that were not retrieved must not leak into the prompt.
    retrieved_ids = {chunk.chunk_id for chunk, _score in ranked}
    others = [chunk for chunk in chunks if chunk.chunk_id not in retrieved_ids]
    assert others, "The corpus should be larger than the retrieved set."
    assert all(chunk.text not in prompt for chunk in others)
    # And the prompt must be far smaller than the whole corpus.
    corpus = "\n\n".join(job.description for job in cached_jobs)
    assert len(prompt) < len(corpus)


def test_extractive_answer_cites_its_chunks(cached_jobs: list[JobPosting]) -> None:
    """The offline answer names a job and cites the chunk it read."""
    chunks = build_chunks(cached_jobs)
    ranked, _vocab, _shape, _warning = retrieve(QUESTION, chunks, top_k=TOP_K)
    answer = extractive_answer(ranked, cached_jobs)

    assert "Power BI" in answer
    assert any(chunk.chunk_id in answer for chunk, _score in ranked)


def test_extractive_answer_admits_when_context_is_silent(
    cached_jobs: list[JobPosting],
) -> None:
    """With irrelevant chunks the answer says so instead of guessing."""
    chunks = build_chunks(cached_jobs)
    silent = [
        (chunk, 0.0)
        for chunk in chunks
        if "power bi" not in chunk.text.lower()
    ][:TOP_K]
    assert silent, "The corpus should contain chunks without Power BI."

    answer = extractive_answer(silent, cached_jobs)
    assert "does not say" in answer.lower()


# --------------------------------------------------------------------------
# run() — offline path
# --------------------------------------------------------------------------


def test_run_offline_produces_full_lesson(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """With no model, every stage still runs and the flags are honest."""
    ctx = make_ctx(cached_jobs, resume, OfflineLLM(), settings)
    result = STEP.execute(ctx)

    assert isinstance(result, LessonResult)
    assert len(result.blocks) >= 1
    assert result.used_llm is False
    assert result.llm_unavailable is True

    labels = " | ".join(block.label for block in result.blocks)
    assert "1 · CHUNK" in labels
    assert "2 · EMBED" in labels
    assert "3 · RETRIEVE" in labels
    assert "4 · GENERATE" in labels
    assert not any(block.kind == "warning" for block in result.blocks)


def test_run_offline_shows_prompt_and_labelled_fallback(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """The assembled prompt is shown, and the offline answer is labelled."""
    ctx = make_ctx(cached_jobs, resume, OfflineLLM(), settings)
    result = STEP.execute(ctx)

    prompt_blocks = [
        block for block in result.blocks if block.kind == "code" and QUESTION in block.body
    ]
    assert prompt_blocks, "The augmented prompt must be visible on screen."

    texts = [f"{block.label}\n{block.body}".lower() for block in result.blocks]
    assert any("offline extractive path" in block.label.lower() for block in result.blocks)
    assert any("deterministic path ran" in text for text in texts)
    assert any("power bi" in text for text in texts)
    # Nothing may be presented as a model response when no model was reached.
    assert not any(block.kind == "success" for block in result.blocks)


def test_run_offline_with_null_client(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """The project's own NullLLMClient takes the same deterministic path."""
    ctx = make_ctx(cached_jobs, resume, NullLLMClient(), settings)
    result = STEP.execute(ctx)

    assert result.used_llm is False
    assert result.llm_unavailable is True
    assert len(result.blocks) >= 1


def test_run_with_missing_client(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """A ``None`` client is treated as unavailable, not as an error."""
    ctx = make_ctx(cached_jobs, resume, None, settings)
    result = STEP.execute(ctx)

    assert result.used_llm is False
    assert result.llm_unavailable is True


def test_run_survives_a_failing_client(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """A client that 503s mid-call falls back instead of crashing the lab."""
    ctx = make_ctx(cached_jobs, resume, ExplodingLLM(), settings)
    result = STEP.execute(ctx)

    assert result.used_llm is False
    assert result.llm_unavailable is True
    assert len(result.blocks) >= 1


def test_run_is_deterministic_offline(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """Same inputs, same screen — the numbers never wobble between rehearsals."""
    ctx = make_ctx(cached_jobs, resume, OfflineLLM(), settings)
    first = STEP.execute(ctx)
    second = STEP.execute(ctx)

    assert [(b.kind, b.label, b.body) for b in first.blocks] == [
        (b.kind, b.label, b.body) for b in second.blocks
    ]


# --------------------------------------------------------------------------
# run() — model-available path
# --------------------------------------------------------------------------


def test_run_uses_the_model_when_available(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """With a working client the grounded answer is shown and flagged."""
    client = ScriptedLLM()
    ctx = make_ctx(cached_jobs, resume, client, settings)
    result = STEP.execute(ctx)

    assert result.used_llm is True
    assert result.llm_unavailable is False
    assert len(client.prompts) == 1
    assert QUESTION in client.prompts[0]
    assert any(CANNED_ANSWER in str(block.body) for block in result.blocks)
    assert any(block.kind == "success" for block in result.blocks)


def test_model_prompt_carries_only_retrieved_context(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """The model is sent the retrieved chunks, not the whole corpus."""
    client = ScriptedLLM()
    ctx = make_ctx(cached_jobs, resume, client, settings)
    STEP.execute(ctx)

    prompt = client.prompts[0]
    corpus = "\n\n".join(job.description for job in cached_jobs)
    assert len(prompt) < len(corpus)
    assert all(job.description not in prompt for job in cached_jobs)


def test_empty_reply_is_treated_as_unavailable(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """An empty model reply falls back rather than showing a blank answer."""
    ctx = make_ctx(cached_jobs, resume, ScriptedLLM(reply=""), settings)
    result = STEP.execute(ctx)

    assert result.used_llm is False
    assert result.llm_unavailable is True


# --------------------------------------------------------------------------
# Degenerate input and step metadata
# --------------------------------------------------------------------------


def test_run_with_no_jobs_still_returns_a_block(
    resume: ResumeProfile, settings: Settings
) -> None:
    """An empty corpus produces an explanation, never an exception."""
    ctx = make_ctx([], resume, OfflineLLM(), settings)
    result = STEP.execute(ctx)

    assert len(result.blocks) >= 1
    assert result.blocks[0].kind == "warning"
    assert result.used_llm is False


def test_token_comparison_metric_is_present(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """The cost argument is made concrete with a token metric."""
    ctx = make_ctx(cached_jobs, resume, OfflineLLM(), settings)
    result = STEP.execute(ctx)

    metrics = [block for block in result.blocks if block.kind == "metric"]
    assert len(metrics) >= 3  # chunk count, embed shape, token comparison
    token_metric = metrics[-1]
    assert "tokens" in str(token_metric.body["value"])
    assert "smaller" in str(token_metric.body["value"])


def test_step_metadata_is_workshop_ready() -> None:
    """The step declares itself correctly for the Learn tab."""
    assert STEP.number == 4
    assert STEP.title
    assert STEP.subtitle
    assert STEP.deck_reference
    assert callable(STEP.run)
    assert STEP.takeaway.count(".") <= 1
    # The on-screen snippet must stay readable for beginners.
    assert 10 <= len(STEP.code.strip().splitlines()) <= 25
    # The step prefers the model but never requires it.
    assert STEP.needs_llm is False
    assert CHUNK_CHARS > CHUNK_OVERLAP > 0


# --------------------------------------------------------------------------
# Regressions found while adversarially verifying this step
# --------------------------------------------------------------------------


def test_blank_model_reply_never_renders_as_a_model_answer(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """A whitespace-only reply takes the offline path, not a blank green box.

    Regression: a reply of ``"   "`` is truthy, so it used to set
    ``used_llm=True`` and render an empty "Grounded answer from the model"
    success block — the screen claiming the model spoke when it said nothing,
    and the extractive answer suppressed.
    """
    ctx = make_ctx(cached_jobs, resume, BlankReplyLLM(), settings)
    result = STEP.execute(ctx)

    assert result.used_llm is False
    assert result.llm_unavailable is True
    assert not any(block.kind == "success" for block in result.blocks)
    assert any("offline extractive path" in block.label.lower() for block in result.blocks)
    # No block may be rendered with an empty body.
    assert all(
        str(block.body).strip()
        for block in result.blocks
        if isinstance(block.body, str)
    )


def test_non_string_reply_falls_back_instead_of_losing_the_lesson(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """A contract-breaking reply must not collapse all four stages."""
    ctx = make_ctx(cached_jobs, resume, NonStringLLM(), settings)
    result = STEP.execute(ctx)

    assert result.used_llm is False
    assert result.llm_unavailable is True
    labels = " | ".join(block.label for block in result.blocks)
    for stage in ("1 · CHUNK", "2 · EMBED", "3 · RETRIEVE", "4 · GENERATE"):
        assert stage in labels
    assert not any(block.kind == "warning" for block in result.blocks)


def test_unreadable_availability_is_reported_as_unavailable(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """A client whose ``available`` probe raises is unavailable, not 'pure Python'.

    The Learn tab captions a step with ``llm_unavailable`` to explain which path
    ran, so this flag must be set rather than left False.
    """
    ctx = make_ctx(cached_jobs, resume, UnreadableAvailabilityLLM(), settings)
    result = STEP.execute(ctx)

    assert result.used_llm is False
    assert result.llm_unavailable is True
    assert len(result.blocks) >= 1
    assert not any(block.kind == "warning" for block in result.blocks)


def test_unique_postings_drops_the_repeated_cache_entry(
    app_jobs: list[JobPosting],
) -> None:
    """The real Learn-tab corpus contains a duplicate; the index must not."""
    assert len(app_jobs) > len(unique_postings(app_jobs)), (
        "The cached data is expected to contain a repeated posting."
    )
    unique = unique_postings(app_jobs)
    assert len({job.job_id for job in unique}) == len(unique)


def test_run_on_the_real_learn_tab_corpus_indexes_each_job_once(
    app_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """Against the corpus the app really supplies, no job is chunked twice.

    Regression: the duplicated cache entry used to fill two of the three
    retrieval slots with the same posting's text, so the augmented prompt
    carried the same requirement list twice.
    """
    ctx = make_ctx(app_jobs, resume, OfflineLLM(), settings)
    result = STEP.execute(ctx)

    chunk_metric = next(block for block in result.blocks if "CHUNK" in block.label)
    expected = len(unique_postings(app_jobs))
    assert f"from {expected} job descriptions" in chunk_metric.body["value"]
    assert "repeated posting" in chunk_metric.body["help"]

    # Every retrieved chunk is a distinct piece of text.
    chunks = build_chunks(unique_postings(app_jobs))
    ranked, _vocab, _shape, _warning = retrieve(QUESTION, chunks, top_k=TOP_K)
    texts = [chunk.text for chunk, _score in ranked]
    assert len(set(texts)) == len(texts)


def test_cost_metric_never_claims_a_saving_it_did_not_make(
    cached_jobs: list[JobPosting], resume: ResumeProfile, settings: Settings
) -> None:
    """With a corpus smaller than the prompt, the metric says so plainly.

    Regression: it used to print "0% smaller" over a prompt more than twice the
    size of the corpus, under help text promising "You pay less".
    """
    tiny = cached_jobs[0].model_copy(
        update={"description": cached_jobs[0].description[:300]}
    )
    ctx = make_ctx([tiny], resume, OfflineLLM(), settings)
    result = STEP.execute(ctx)

    cost = [block for block in result.blocks if block.kind == "metric"][-1]
    assert "no saving yet" in cost.body["value"]
    assert "smaller" not in cost.body["value"]
    assert "You pay less" not in cost.body["help"]


def test_overlap_cannot_explode_the_chunk_count() -> None:
    """An overlap at or above the window size must not step one char at a time.

    An instructor editing ``CHUNK_OVERLAP`` live to show what overlap does
    should not hang the projector.
    """
    text = "x" * 2000
    assert len(split_into_chunks(text, size=100, overlap=100)) < 60
    assert len(split_into_chunks(text, size=100, overlap=10_000)) < 60
    # The documented default geometry is untouched by the clamp.
    assert split_into_chunks(text, size=500, overlap=80)[0][-40:] in (
        split_into_chunks(text, size=500, overlap=80)[1]
    )
