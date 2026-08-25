"""Step 3 (retrieval) must teach correctly with no network and no API key.

The lesson is deterministic by design, so every assertion here holds whether the
language model is reachable or not. Jobs come from the clearly-labelled cached
demonstration data through the same normalizer and filter the app uses.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from config import Settings
from lessons.base import LessonContext, LessonResult, OutputBlock, approx_tokens
from lessons.step_3_retrieval import (
    RAW_EXCERPT_CHARS,
    STEP,
    excerpt_parts,
    format_utc,
    freshness_note,
    mode_summary,
    pick_excerpt_job,
    posted_display,
    raw_excerpt,
    retrieval_rows,
    run,
)
from models.job import JobPosting, RawJobResult
from services.llm_interface import NullLLMClient
from tests.conftest import make_job
from tools.firecrawl_search import cached_raw_results, load_cache
from tools.job_filter import filter_and_deduplicate
from tools.job_normalizer import normalize_jobs


class ScriptedLLM:
    """An LLM client that is reachable and always answers with canned text."""

    def __init__(self, text: str = "canned model text") -> None:
        self.text = text
        self.calls = 0

    @property
    def available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "scripted-test-model"

    def generate_structured(self, prompt: str, schema: Any, *, temperature: float = 0.1):
        self.calls += 1
        return None

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        self.calls += 1
        return self.text


def cached_jobs(settings: Settings) -> list[JobPosting]:
    """Build real postings from the offline cache, exactly as the app does.

    Mirrors ``agent.nodes``: cached entries become :class:`RawJobResult`
    records, the normalizer reads them with an offline LLM client, and the
    filter removes closed, snippet-only, and duplicate pages.
    """
    payload = load_cache(settings.cache_path)
    entries = cached_raw_results(
        payload, query_category="company_careers", freshness_window="last_24_hours"
    )
    retrieved_at = datetime.fromisoformat(payload["originally_retrieved_at"])
    raw_results: list[RawJobResult] = []
    for entry in entries:
        record = dict(entry)
        record["query_category"] = "company_careers"
        record["freshness_window"] = "last_24_hours"
        record["retrieved_at"] = retrieved_at
        raw_results.append(RawJobResult.model_validate(record))

    postings, _ = normalize_jobs(
        raw_results, NullLLMClient(), data_mode="cached", limit=8
    )
    return filter_and_deduplicate(postings).kept


@pytest.fixture
def jobs(settings: Settings) -> list[JobPosting]:
    """Postings from the cached demonstration data."""
    return cached_jobs(settings)


@pytest.fixture
def offline_ctx(settings, resume, jobs) -> LessonContext:
    """A context whose model cannot be reached at all."""
    return LessonContext(settings=settings, llm=NullLLMClient(), resume=resume, jobs=jobs)


@pytest.fixture
def online_ctx(settings, resume, jobs) -> LessonContext:
    """A context whose model is reachable and returns canned text."""
    return LessonContext(settings=settings, llm=ScriptedLLM(), resume=resume, jobs=jobs)


def kinds(result: LessonResult) -> list[str]:
    """Return the block kinds a result produced."""
    return [block.kind for block in result.blocks]


def block_of(result: LessonResult, kind: str) -> OutputBlock:
    """Return the first block of a given kind."""
    return next(block for block in result.blocks if block.kind == kind)


class TestCachedFixture:
    """The offline fixture really does hold retrieved page text."""

    def test_the_cache_yields_several_usable_postings(self, jobs):
        assert len(jobs) >= 3

    def test_every_posting_carries_real_page_text(self, jobs):
        # Full descriptions, not search snippets.
        assert all(len(job.description) > 500 for job in jobs)


class TestRunsWithoutAModel:
    """No network, no API key, still a complete lesson."""

    def test_it_never_raises_and_returns_blocks(self, offline_ctx):
        result = run(offline_ctx)
        assert len(result.blocks) >= 1

    def test_the_step_is_marked_deterministic(self, offline_ctx):
        result = run(offline_ctx)
        # The step never wants a model, so an unreachable one is not a failure.
        assert result.used_llm is False
        assert result.llm_unavailable is False

    def test_the_llm_is_never_called(self, settings, resume, jobs):
        scripted = ScriptedLLM()
        ctx = LessonContext(settings=settings, llm=scripted, resume=resume, jobs=jobs)
        run(ctx)
        assert scripted.calls == 0

    def test_it_declares_that_it_needs_no_model(self):
        assert STEP.needs_llm is False


class TestSameResultWithAModelAvailable:
    """A reachable model changes nothing: this step is deterministic."""

    def test_flags_are_identical_either_way(self, online_ctx):
        result = run(online_ctx)
        assert result.used_llm is False
        assert result.llm_unavailable is False

    def test_output_is_identical_either_way(self, offline_ctx, online_ctx):
        offline = [(b.kind, b.label, b.body) for b in run(offline_ctx).blocks]
        online = [(b.kind, b.label, b.body) for b in run(online_ctx).blocks]
        assert offline == online

    def test_repeated_runs_agree(self, offline_ctx):
        first = [(b.kind, b.body) for b in run(offline_ctx).blocks]
        second = [(b.kind, b.body) for b in run(offline_ctx).blocks]
        assert first == second


class TestProvenanceTable:
    """Every row shows where the text came from."""

    def test_one_row_per_retrieved_page(self, jobs):
        assert len(retrieval_rows(jobs)) == len(jobs)

    def test_rows_share_their_keys(self, jobs):
        rows = retrieval_rows(jobs)
        assert all(row.keys() == rows[0].keys() for row in rows)

    def test_a_row_shows_source_link_time_and_size(self, jobs):
        row = retrieval_rows(jobs)[0]
        job = jobs[0]
        assert row["Source"] == job.source_label
        assert row["Source URL"] == job.source_url
        assert row["Page text (chars)"] == len(job.description)
        assert "UTC" in str(row["Retrieved (UTC)"])

    def test_an_unsafe_link_is_withheld_not_rendered(self):
        job = make_job(source_url="javascript:alert(1)")
        row = retrieval_rows([job])[0]
        assert row["Link checked"] == "rejected"
        assert "javascript:" not in str(row["Source URL"])

    def test_the_table_block_is_present(self, offline_ctx):
        assert "table" in kinds(run(offline_ctx))

    def test_the_row_shows_what_the_page_said_about_its_own_posting_time(self):
        # The point of the step: our retrieval clock and the source's posting
        # claim are two different facts and both are on screen.
        row = retrieval_rows([make_job()])[0]
        assert row["Posted (per the page)"] == "2026-08-20 15:00 UTC"
        assert row["Retrieved (UTC)"] == "2026-08-20 15:00 UTC"
        assert row["Posted (per the page)"] != row["Posting evidence"]

    def test_a_date_only_page_says_date_only(self):
        job = make_job(posted_at=None, freshness_evidence="date_only")
        assert retrieval_rows([job])[0]["Posted (per the page)"] == (
            "2026-08-20 (date only)"
        )

    def test_a_page_with_no_date_says_so_rather_than_guessing(self):
        job = make_job(
            posted_at=None, posting_date=None, freshness_evidence="unavailable"
        )
        assert retrieval_rows([job])[0]["Posted (per the page)"] == "not shown on page"
        assert posted_display(job) == "not shown on page"


class TestRawExcerpt:
    """Students see genuine scraped text, not a summary of it."""

    def test_the_excerpt_is_a_prefix_of_the_real_page_text(self, jobs):
        job = pick_excerpt_job(jobs)
        excerpt = raw_excerpt(job)
        assert job.description.startswith(excerpt[:RAW_EXCERPT_CHARS].rstrip())

    def test_long_text_is_truncated_with_an_honest_marker(self, jobs):
        job = pick_excerpt_job(jobs)
        assert "more characters]" in raw_excerpt(job)

    def test_short_text_is_shown_whole(self):
        job = make_job(description="Short posting text.")
        assert raw_excerpt(job) == "Short posting text."

    def test_the_chosen_page_has_a_verifiable_link(self, jobs):
        assert pick_excerpt_job(jobs).source_url.startswith("https://")

    def test_no_job_yields_no_choice(self):
        assert pick_excerpt_job([]) is None

    def test_a_code_block_carries_the_excerpt(self, offline_ctx):
        result = run(offline_ctx)
        assert "code" in kinds(result)
        assert len(block_of(result, "code").body) > 100


class TestTokenCostMetric:
    """The metric sets up why chunking is needed next."""

    def test_the_metric_block_exists(self, offline_ctx):
        assert "metric" in kinds(run(offline_ctx))

    def test_it_reports_pages_characters_and_tokens(self, offline_ctx, jobs):
        body = block_of(run(offline_ctx), "metric").body
        total_chars = len("\n\n".join(job.description for job in jobs))
        assert f"{len(jobs)} pages" in body["value"]
        assert f"{total_chars:,} chars" in body["value"]
        assert f"{approx_tokens('x' * total_chars):,} tokens" in body["value"].replace(
            "~", ""
        )

    def test_the_help_text_points_at_chunking(self, offline_ctx):
        assert "chunk" in block_of(run(offline_ctx), "metric").body["help"].lower()


class TestFreshnessHonesty:
    """A search filter is never presented as proof of a posting time."""

    def test_the_note_separates_filter_from_evidence(self, jobs):
        note = freshness_note(jobs)
        assert "filter" in note.lower()
        assert "never proves" in note.lower() or "not a receipt" in note.lower()

    def test_the_note_names_the_requested_window(self, jobs):
        assert "Last 24 hours" in freshness_note(jobs)

    def test_a_note_block_is_emitted(self, offline_ctx):
        assert "note" in kinds(run(offline_ctx))

    def test_undated_pages_are_not_called_verified(self):
        job = make_job(
            posted_at=None,
            posting_date=None,
            posting_age_hours=None,
            freshness_evidence="search_filter_only",
            freshness_status="date_unavailable",
        )
        assert "verified" not in job.freshness_label().lower()
        assert job.freshness_label() in freshness_note([job])

    def test_the_note_quotes_only_labels_the_table_really_shows(self):
        # The paragraph and the rows are generated from the same call, so the
        # prose can never describe a label the audience is not looking at.
        jobs = [
            make_job(job_id=f"j{i}", freshness_evidence=evidence)
            for i, evidence in enumerate(
                ("exact_timestamp", "date_only", "search_filter_only", "unavailable")
            )
        ]
        note = freshness_note(jobs)
        for row in retrieval_rows(jobs):
            assert str(row["Posting evidence"]) in note

    def test_each_evidence_kind_is_counted_separately(self):
        # "search-filtered" and "no evidence at all" are different claims and
        # must not be collapsed into one bucket.
        jobs = [
            make_job(job_id="a", freshness_evidence="search_filter_only"),
            make_job(job_id="b", freshness_evidence="unavailable"),
        ]
        note = freshness_note(jobs)
        assert "Search-filtered; source timestamp unavailable" in note
        assert "Posting time unavailable" in note
        assert "2 with" not in note  # two separate buckets of one, not one of two

    def test_the_strongest_evidence_is_described_first(self):
        jobs = [
            make_job(job_id="a", freshness_evidence="unavailable"),
            make_job(job_id="b", freshness_evidence="exact_timestamp"),
        ]
        note = freshness_note(jobs)
        assert note.index("exact timestamp on the page") < note.index(
            "no posting evidence at all"
        )

    def test_a_mixed_window_search_still_reads_as_english(self):
        note = freshness_note(
            [
                make_job(job_id="a", freshness_window="last_hour"),
                make_job(job_id="b", freshness_window="last_7_days"),
            ]
        )
        assert "asked for a time window" in note
        assert "the selected window" not in note


class TestHonestModeLabelling:
    """Cached text is never presented as if it were fetched just now."""

    def test_cached_runs_say_so(self, offline_ctx):
        body = block_of(run(offline_ctx), "markdown").body
        assert "cached" in body.lower()

    def test_live_runs_say_live(self, settings, resume):
        ctx = LessonContext(
            settings=settings,
            llm=NullLLMClient(),
            resume=resume,
            jobs=[make_job(data_mode="live")],
        )
        assert "live" in block_of(run(ctx), "markdown").body.lower()


class TestFailsSoftly:
    """A student clicking Run always sees something useful."""

    def test_no_jobs_gives_a_warning_not_a_crash(self, settings, resume):
        ctx = LessonContext(
            settings=settings, llm=NullLLMClient(), resume=resume, jobs=[]
        )
        result = run(ctx)
        assert kinds(result) == ["warning"]
        assert result.used_llm is False

    def test_a_page_with_no_text_is_reported(self, settings, resume):
        ctx = LessonContext(
            settings=settings,
            llm=NullLLMClient(),
            resume=resume,
            jobs=[make_job(description="", description_excerpt="")],
        )
        result = run(ctx)
        assert "warning" in kinds(result)
        assert len(result.blocks) >= 2

    def test_execute_times_the_step_and_survives_a_bad_context(self, settings, resume):
        broken = LessonContext(
            settings=settings, llm=None, resume=resume, jobs=None  # type: ignore[arg-type]
        )
        result = STEP.execute(broken)
        assert result.blocks
        assert result.elapsed_seconds >= 0.0


class TestStepMetadata:
    """The contract the Learn tab renders."""

    def test_it_is_step_three(self):
        assert STEP.number == 3

    def test_the_teaching_snippet_is_short_and_readable(self):
        lines = STEP.code.strip().splitlines()
        assert 10 <= len(lines) <= 25

    def test_the_snippet_shows_the_real_helpers_the_step_uses(self):
        assert "approx_tokens" in STEP.code
        assert "is_safe_public_job_url" in STEP.code

    def test_every_teaching_field_is_filled_in(self):
        for field in (STEP.title, STEP.subtitle, STEP.concept, STEP.why,
                      STEP.deck_reference, STEP.takeaway):
            assert field.strip()

    def test_the_deck_reference_points_at_slide_eight(self):
        assert "8" in STEP.deck_reference


class TestTheProseNeverOutrunsTheEvidence:
    """Claims about the captured text must survive the awkward cases."""

    def test_no_evidence_claim_is_made_over_a_page_with_no_text(
        self, settings, resume
    ):
        # Captioning an empty page "raw page text as captured" would be exactly
        # the dishonesty this step argues against.
        ctx = LessonContext(
            settings=settings,
            llm=NullLLMClient(),
            resume=resume,
            jobs=[make_job(description="", description_excerpt="")],
        )
        result = run(ctx)
        assert "code" not in kinds(result)
        assert not any("That text is the evidence" == b.label for b in result.blocks)
        assert "warning" in kinds(result)

    def test_an_unverified_link_never_prints_an_empty_domain(
        self, settings, resume
    ):
        ctx = LessonContext(
            settings=settings,
            llm=NullLLMClient(),
            resume=resume,
            jobs=[make_job(source_url="javascript:alert(1)")],
        )
        body = next(
            b.body for b in run(ctx).blocks if b.label == "That text is the evidence"
        )
        assert "at ``" not in body
        assert "failed validation" in body
        assert "withheld" in body

    def test_a_verifiable_link_names_its_domain(self, offline_ctx):
        body = next(
            b.body
            for b in run(offline_ctx).blocks
            if b.label == "That text is the evidence"
        )
        assert "job-boards.greenhouse.io" in body

    def test_the_excerpt_is_preferred_from_a_page_that_has_text(self):
        empty = make_job(job_id="empty", description="")
        real = make_job(job_id="real")
        assert pick_excerpt_job([empty, real]) is real

    def test_the_caption_arithmetic_adds_up(self, jobs):
        for job in jobs:
            _, shown, remaining = excerpt_parts(job)
            assert shown + remaining == len(job.description)

    def test_the_caption_reports_the_characters_actually_shown(self, jobs):
        job = pick_excerpt_job(jobs)
        shown_text, shown, _ = excerpt_parts(job)
        assert len(shown_text) == shown
        assert job.description.startswith(shown_text)


class TestCachedLabellingCannotContradictTheTable:
    """One stated retrieval time is only claimed when there is only one."""

    def test_differing_cached_times_are_not_flattened_into_one_claim(self):
        early = make_job(
            job_id="early",
            data_mode="cached",
            retrieved_at=datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc),
        )
        late = make_job(
            job_id="late",
            data_mode="cached",
            retrieved_at=datetime(2026, 8, 19, 23, 30, tzinfo=timezone.utc),
        )
        summary = mode_summary([early, late])
        assert "originally retrieved at" not in summary
        assert "each row carries" in summary

    def test_one_shared_cached_time_is_still_stated_plainly(self, jobs):
        summary = mode_summary(jobs)
        assert "originally retrieved at" in summary
        assert format_utc(jobs[0].retrieved_at) in summary


class TestItReadsAsEnglishOnStage:
    """A room of a hundred people should not see "These 1 pages"."""

    def test_a_single_live_page_is_singular(self):
        summary = mode_summary([make_job(data_mode="live")])
        assert "This 1 page was" in summary
        assert "pages" not in summary

    def test_a_single_cached_page_is_singular(self):
        summary = mode_summary([make_job(data_mode="cached")])
        assert "This 1 page is" in summary
        assert "demonstration result**" in summary

    def test_two_pages_are_plural(self):
        summary = mode_summary(
            [make_job(job_id="a", data_mode="live"), make_job(job_id="b")]
        )
        assert "These 2 pages were" in summary

    def test_a_single_page_metric_is_singular(self, settings, resume):
        ctx = LessonContext(
            settings=settings, llm=NullLLMClient(), resume=resume, jobs=[make_job()]
        )
        assert "1 page |" in block_of(run(ctx), "metric").body["value"]

    def test_the_warning_counts_pages_not_problems(self, settings, resume):
        # One page with both problems is one unusable page, not two.
        ctx = LessonContext(
            settings=settings,
            llm=NullLLMClient(),
            resume=resume,
            jobs=[
                make_job(job_id="bad", description="", source_url="javascript:x"),
                make_job(job_id="good"),
            ],
        )
        warning = block_of(run(ctx), "warning").body
        assert "1 of 2 retrieved pages" in warning
        assert "a single page can have both problems" in warning

    def test_a_lone_problem_drops_the_both_problems_caveat(
        self, settings, resume
    ):
        ctx = LessonContext(
            settings=settings,
            llm=NullLLMClient(),
            resume=resume,
            jobs=[make_job(source_url="javascript:x")],
        )
        warning = block_of(run(ctx), "warning").body
        assert "both problems" not in warning
        assert "1 of 1 retrieved page" in warning


class TestTokenCostIsMeasuredNotAsserted:
    """Every number on screen comes from the text in front of it."""

    def test_only_pages_with_text_are_counted(self, settings, resume):
        real = make_job(job_id="real")
        ctx = LessonContext(
            settings=settings,
            llm=NullLLMClient(),
            resume=resume,
            jobs=[make_job(job_id="empty", description=""), real],
        )
        value = block_of(run(ctx), "metric").body["value"]
        assert "1 page |" in value
        assert f"{len(real.description):,} chars" in value

    def test_no_metric_is_shown_when_nothing_was_readable(self, settings, resume):
        ctx = LessonContext(
            settings=settings,
            llm=NullLLMClient(),
            resume=resume,
            jobs=[make_job(description="")],
        )
        result = run(ctx)
        assert "metric" not in kinds(result)
        # The student still gets the honest story.
        assert kinds(result) == ["markdown", "table", "note", "warning"]

    def test_the_character_count_matches_the_table(self, offline_ctx, jobs):
        value = block_of(run(offline_ctx), "metric").body["value"]
        rows_total = sum(int(row["Page text (chars)"]) for row in retrieval_rows(jobs))
        # The joined corpus adds one blank line between pages.
        separators = 2 * (len(jobs) - 1)
        assert f"{rows_total + separators:,} chars" in value


class TestTheSnippetIsRealCode:
    """The code students are shown must actually execute."""

    def test_the_snippet_runs_against_real_postings(self):
        namespace: dict[str, Any] = {
            "jobs": [
                make_job(job_id="a"),
                make_job(job_id="b", source_url="javascript:alert(1)"),
            ]
        }
        exec(compile(STEP.code, "<STEP.code>", "exec"), namespace)
        assert namespace["all_text"]

    def test_the_snippet_matches_what_the_step_does(self):
        # The step keeps unverifiable pages visible and marked; the snippet
        # must not quietly skip them, or the code and the screen disagree.
        assert "continue" not in STEP.code
        assert "rejected" in STEP.code


class TestFormatting:
    """Timestamps are shown in UTC so the room sees one clock."""

    def test_naive_timestamps_are_treated_as_utc(self):
        assert format_utc(datetime(2026, 8, 20, 14, 5)) == "2026-08-20 14:05 UTC"

    def test_aware_timestamps_are_converted(self):
        stamp = datetime(2026, 8, 20, 14, 5, tzinfo=timezone.utc)
        assert format_utc(stamp) == "2026-08-20 14:05 UTC"

    def test_a_missing_timestamp_is_labelled(self):
        assert format_utc(None) == "not recorded"
