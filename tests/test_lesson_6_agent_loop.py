"""Step 6 (the agent loop) must teach correctly with no network and no API key.

The loop itself is pure Python over this project's own tools, so every
assertion here holds offline. The language model is only ever offered an
optional narration, and the step has to report honestly which path it took.
"""

from __future__ import annotations

from typing import Any

import pytest

from config import Settings
from lessons.base import LessonContext, LessonResult
from lessons.step_6_agent_loop import (
    MAX_TURNS,
    STEP,
    decide_next_action,
    run_agent_loop,
)
from models.job import JobPosting, RawJobResult
from models.resume import ResumeProfile
from services.llm_interface import NullLLMClient
from tests.conftest import FIXED_NOW, make_job
from tools.firecrawl_search import cached_raw_results, load_cache
from tools.job_normalizer import normalize_jobs


# --------------------------------------------------------------------------
# Offline helpers
# --------------------------------------------------------------------------


class StubLLM:
    """A scripted LLM client matching the slice lessons are allowed to use."""

    def __init__(self, *, available: bool, text: str | None = None) -> None:
        self._available = available
        self._text = text
        self.calls = 0

    @property
    def available(self) -> bool:
        """True when this stub should pretend a model is reachable."""
        return self._available

    @property
    def model_name(self) -> str:
        """A readable placeholder name."""
        return "stub-model"

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Return the canned text, recording that a call was made."""
        self.calls += 1
        return self._text


class ExplodingLLM(StubLLM):
    """Claims to be available, then fails - the exact live-demo failure mode."""

    def __init__(self) -> None:
        super().__init__(available=True)

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Raise the way a 503 or quota error does."""
        self.calls += 1
        raise RuntimeError("503 model overloaded")


def cached_jobs(settings: Settings) -> list[JobPosting]:
    """Build normalized postings from the offline cache, as the app does.

    Mirrors ``agent.nodes._load_cached_jobs`` plus ``normalize_jobs`` so the
    lesson is exercised against the real demonstration data, with no network
    call and no API key.
    """
    payload = load_cache(settings.cache_path)
    entries = cached_raw_results(
        payload, query_category="all", freshness_window="last_24_hours"
    )
    raw_results: list[RawJobResult] = []
    for entry in entries:
        record = dict(entry)
        record["query_category"] = "all"
        record["freshness_window"] = "last_24_hours"
        record["retrieved_at"] = payload.get("originally_retrieved_at") or FIXED_NOW
        raw_results.append(RawJobResult.model_validate(record))
    jobs, _warnings = normalize_jobs(raw_results, NullLLMClient(), data_mode="cached")
    return jobs


def make_context(
    settings: Settings,
    resume: ResumeProfile,
    jobs: list[JobPosting],
    llm: Any,
) -> LessonContext:
    """Assemble a :class:`LessonContext` with an injected LLM stub."""
    return LessonContext(settings=settings, llm=llm, resume=resume, jobs=jobs)


@pytest.fixture
def jobs(settings: Settings) -> list[JobPosting]:
    """Normalized postings straight from the cached demonstration data."""
    return cached_jobs(settings)


# --------------------------------------------------------------------------
# The step contract
# --------------------------------------------------------------------------


class TestStepMetadata:
    """The module exports a well-formed step the Learn tab can render."""

    def test_exports_step_six(self):
        assert STEP.number == 6
        assert STEP.title
        assert STEP.subtitle
        assert STEP.takeaway
        assert "18" in STEP.deck_reference

    def test_step_does_not_require_the_model(self):
        assert STEP.needs_llm is False

    def test_teaching_snippet_is_short_and_real(self):
        lines = STEP.code.strip().splitlines()
        assert 10 <= len(lines) <= 30
        assert "decide_next_action" in STEP.code
        assert "break" in STEP.code


class TestOfflineRun:
    """With no model, the step still produces the complete lesson."""

    def test_runs_and_reports_the_model_was_unavailable(self, settings, resume, jobs):
        ctx = make_context(settings, resume, jobs, StubLLM(available=False))
        result = STEP.execute(ctx)

        assert isinstance(result, LessonResult)
        assert len(result.blocks) >= 1
        assert result.used_llm is False
        assert result.llm_unavailable is True

    def test_no_warning_block_is_emitted(self, settings, resume, jobs):
        ctx = make_context(settings, resume, jobs, StubLLM(available=False))
        result = STEP.execute(ctx)
        assert [block for block in result.blocks if block.kind == "warning"] == []

    def test_emits_the_loop_table_with_the_required_columns(
        self, settings, resume, jobs
    ):
        ctx = make_context(settings, resume, jobs, StubLLM(available=False))
        result = STEP.execute(ctx)

        tables = [block for block in result.blocks if block.kind == "table"]
        assert len(tables) == 1
        rows = tables[0].body
        assert 3 <= len(rows) <= MAX_TURNS
        for row in rows:
            assert set(row) == {"Step", "Reason", "Action", "Observation"}

    def test_table_names_the_real_project_functions(self, settings, resume, jobs):
        ctx = make_context(settings, resume, jobs, StubLLM(available=False))
        result = STEP.execute(ctx)
        actions = " ".join(
            row["Action"]
            for block in result.blocks
            if block.kind == "table"
            for row in block.body
        )
        assert "job_scorer.score_job" in actions
        assert "ats_scorer.assess_ats" in actions

    def test_maps_the_loop_to_harness_and_graph(self, settings, resume, jobs):
        ctx = make_context(settings, resume, jobs, StubLLM(available=False))
        result = STEP.execute(ctx)
        prose = " ".join(
            str(block.body) for block in result.blocks if block.kind == "markdown"
        )
        assert "Harness" in prose
        assert "Graph" in prose
        assert "agent/graph.py" in prose

    def test_explains_the_stop_condition(self, settings, resume, jobs):
        ctx = make_context(settings, resume, jobs, StubLLM(available=False))
        result = STEP.execute(ctx)
        notes = " ".join(
            str(block.body) for block in result.blocks if block.kind == "note"
        )
        assert "stop condition" in notes

    def test_null_client_and_missing_client_behave_the_same(
        self, settings, resume, jobs
    ):
        for client in (NullLLMClient(), None):
            result = STEP.execute(make_context(settings, resume, jobs, client))
            assert result.used_llm is False
            assert result.llm_unavailable is True
            assert len(result.blocks) >= 1

    def test_a_failing_model_does_not_break_the_lesson(self, settings, resume, jobs):
        llm = ExplodingLLM()
        result = STEP.execute(make_context(settings, resume, jobs, llm))
        assert llm.calls == 1
        assert result.used_llm is False
        assert result.llm_unavailable is True
        assert any(block.kind == "table" for block in result.blocks)


class TestModelAvailableRun:
    """When the model answers, the narration is added and reported as used."""

    def test_narration_is_shown_and_flags_are_set(self, settings, resume, jobs):
        llm = StubLLM(available=True, text="The agent ranked jobs and scored a resume.")
        result = STEP.execute(make_context(settings, resume, jobs, llm))

        assert llm.calls == 1
        assert result.used_llm is True
        assert result.llm_unavailable is False
        prose = " ".join(
            str(block.body) for block in result.blocks if block.kind == "markdown"
        )
        assert "The agent ranked jobs and scored a resume." in prose

    def test_the_deterministic_table_is_still_produced(self, settings, resume, jobs):
        llm = StubLLM(available=True, text="A one-line narration.")
        result = STEP.execute(make_context(settings, resume, jobs, llm))
        assert any(block.kind == "table" for block in result.blocks)

    def test_blank_model_output_counts_as_unavailable(self, settings, resume, jobs):
        llm = StubLLM(available=True, text="   ")
        result = STEP.execute(make_context(settings, resume, jobs, llm))
        assert result.used_llm is False
        assert result.llm_unavailable is True


class TestLoopMechanics:
    """The reasoner, the trace, and the bound behave as taught."""

    def test_reasoner_advances_one_fact_at_a_time(self):
        assert decide_next_action({}).action_key == "load_jobs"
        assert decide_next_action({"jobs": []}).action_key == "score_jobs"
        assert decide_next_action({"jobs": [], "matches": []}).action_key == "score_ats"
        assert (
            decide_next_action({"jobs": [], "matches": [], "ats": None}).action_key
            == "stop"
        )

    def test_loop_stops_on_the_goal_not_on_the_bound(self, settings, resume, jobs):
        ctx = make_context(settings, resume, jobs, NullLLMClient())
        trace = run_agent_loop(ctx)
        assert len(trace) == 4
        assert len(trace) < MAX_TURNS
        assert [turn.step for turn in trace] == [1, 2, 3, 4]

    def test_observations_report_both_deterministic_scores(
        self, settings, resume, jobs
    ):
        trace = run_agent_loop(make_context(settings, resume, jobs, NullLLMClient()))
        observations = " ".join(turn.observation for turn in trace)
        assert "Demo Job Match Score" in observations
        assert "Demo ATS Readiness Score" in observations

    def test_loop_is_reproducible(self, settings, resume, jobs):
        ctx = make_context(settings, resume, jobs, NullLLMClient())
        first = [turn.as_row() for turn in run_agent_loop(ctx)]
        second = [turn.as_row() for turn in run_agent_loop(ctx)]
        assert first == second

    def test_empty_job_list_still_completes(self, settings, resume):
        ctx = make_context(settings, resume, [], NullLLMClient())
        trace = run_agent_loop(ctx)
        assert len(trace) == 4
        assert trace[-1].action == "stop - no tool call"

        result = STEP.execute(ctx)
        assert len(result.blocks) >= 1
        assert [block for block in result.blocks if block.kind == "warning"] == []

    def test_a_broken_tool_is_recorded_and_the_loop_still_ends(
        self, settings, resume, monkeypatch
    ):
        def boom(*args: Any, **kwargs: Any):
            raise RuntimeError("scikit-learn is unhappy")

        monkeypatch.setattr("lessons.step_6_agent_loop.score_job", boom)
        ctx = make_context(settings, resume, [make_job()], NullLLMClient())
        trace = run_agent_loop(ctx)

        assert len(trace) <= MAX_TURNS
        assert "failed" in trace[1].observation
        assert trace[-1].action == "stop - no tool call"

    def test_the_loop_can_never_exceed_its_bound(self, settings, resume):
        """Even a reasoner that never reaches its goal is capped."""
        ctx = make_context(settings, resume, [], NullLLMClient())
        monkeyed = run_agent_loop(ctx, max_turns=2)
        assert len(monkeyed) == 2


class HostileLLM:
    """A client whose ``available`` check itself blows up.

    Not hypothetical politeness: ``available`` is a property on every real
    client in ``services/``, so a misconfigured provider can raise before a
    single token is requested. The lesson must survive it.
    """

    @property
    def available(self) -> bool:
        """Fail the way a broken provider config does."""
        raise RuntimeError("provider misconfigured")

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Never reached."""
        raise AssertionError("generate_text must not be called")


class TestHonestStopReporting:
    """The loop must never announce a goal it did not reach."""

    def test_real_run_reports_the_goal_and_says_so_once(self, settings, resume, jobs):
        trace = run_agent_loop(make_context(settings, resume, jobs, NullLLMClient()))
        assert trace[-1].goal_reached is True
        assert "Goal met" in trace[-1].reason
        assert "Goal met" in trace[-1].observation

    def test_metric_help_claims_the_goal_only_on_a_real_run(
        self, settings, resume, jobs
    ):
        result = STEP.execute(make_context(settings, resume, jobs, NullLLMClient()))
        (metric,) = [block for block in result.blocks if block.kind == "metric"]
        assert "goal was met" in metric.body["help"]

    def test_no_jobs_means_no_goal_claim_anywhere_on_screen(self, settings, resume):
        ctx = make_context(settings, resume, [], NullLLMClient())
        trace = run_agent_loop(ctx)

        assert trace[-1].goal_reached is False
        assert "Goal met" not in trace[-1].reason
        assert "Goal met" not in trace[-1].observation

        result = STEP.execute(ctx)
        (metric,) = [block for block in result.blocks if block.kind == "metric"]
        assert "goal was met" not in metric.body["help"]
        rendered = " ".join(str(block.body) for block in result.blocks)
        assert "Goal met" not in rendered

    def test_a_broken_tool_never_becomes_a_reported_success(
        self, settings, resume, monkeypatch
    ):
        def boom(*args: Any, **kwargs: Any):
            raise RuntimeError("scikit-learn is unhappy")

        monkeypatch.setattr("lessons.step_6_agent_loop.score_job", boom)
        ctx = make_context(settings, resume, [make_job()], NullLLMClient())
        trace = run_agent_loop(ctx)

        assert "failed" in trace[1].observation
        assert trace[-1].goal_reached is False
        assert "Goal met" not in trace[-1].observation

    def test_goal_is_not_met_when_the_ats_step_produced_nothing(self, settings, resume, monkeypatch):
        def boom(*args: Any, **kwargs: Any):
            raise RuntimeError("ats scorer is unhappy")

        monkeypatch.setattr("lessons.step_6_agent_loop.assess_ats", boom)
        ctx = make_context(settings, resume, [make_job()], NullLLMClient())
        trace = run_agent_loop(ctx)

        assert "Demo Job Match Score" in trace[1].observation
        assert trace[-1].goal_reached is False


class TestHostileClients:
    """A broken model client may cost the narration, never the lesson."""

    def test_a_raising_available_check_does_not_delete_the_lesson(
        self, settings, resume, jobs
    ):
        result = STEP.execute(make_context(settings, resume, jobs, HostileLLM()))

        assert [block for block in result.blocks if block.kind == "warning"] == []
        assert any(block.kind == "table" for block in result.blocks)
        assert result.used_llm is False
        assert result.llm_unavailable is True

    def test_an_empty_trace_is_never_handed_to_the_model(
        self, settings, resume, jobs, monkeypatch
    ):
        """Nothing happened, so there is nothing for the model to narrate."""
        llm = StubLLM(available=True, text="The agent worked hard.")
        monkeypatch.setattr(
            "lessons.step_6_agent_loop.run_agent_loop",
            lambda *args, **kwargs: [],
        )
        result = STEP.execute(make_context(settings, resume, jobs, llm))

        assert llm.calls == 0
        assert result.used_llm is False
        assert "The agent worked hard." not in " ".join(
            str(block.body) for block in result.blocks
        )
