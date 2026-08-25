"""Tests for lesson data loading and the step registry.

The lab's promise is that it opens and teaches with no API key and no network.
These tests pin that promise: the sample data loads offline, every step is
registered in order, and a broken cache degrades instead of crashing.
"""

from __future__ import annotations

import pytest

from config import load_settings
from lessons import ALL_STEPS, step_by_number
from lessons.base import LessonStep
from lessons.context import (
    build_lesson_context,
    load_cached_postings,
    load_sample_resume,
)


@pytest.fixture(scope="module")
def settings():
    """Real project settings; no network access is performed."""
    return load_settings()


class TestSampleResume:
    def test_loads_the_fictional_candidate(self, settings) -> None:
        resume = load_sample_resume(settings)
        assert resume.candidate_id == "demo_candidate_001"
        assert resume.name == "Alex Morgan"

    def test_has_tableau_but_not_power_bi(self, settings) -> None:
        # The whole guardrail lesson depends on this gap being real.
        skills = {skill.lower() for skill in load_sample_resume(settings).skills}
        assert "tableau" in skills
        assert "power bi" not in skills

    def test_every_bullet_has_a_stable_id(self, settings) -> None:
        index = load_sample_resume(settings).bullet_index()
        assert "experience_1_bullet_1" in index
        assert all(bullet_id and text for bullet_id, text in index.items())


class TestCachedPostings:
    def test_loads_postings_without_network_or_key(self, settings) -> None:
        jobs = load_cached_postings(settings)
        assert len(jobs) >= 3

    def test_postings_carry_full_descriptions(self, settings) -> None:
        # A search snippet is not enough to teach retrieval or RAG with.
        for job in load_cached_postings(settings):
            assert len(job.description) > 200

    def test_at_least_one_posting_requires_power_bi(self, settings) -> None:
        jobs = load_cached_postings(settings)
        mentions = [
            job
            for job in jobs
            if any(
                "power bi" in skill.lower()
                for skill in job.required_skills + job.preferred_skills
            )
        ]
        assert mentions, "the RAG and guardrail lessons need a Power BI posting"

    def test_every_posting_has_a_public_source_url(self, settings) -> None:
        for job in load_cached_postings(settings):
            assert job.source_url.startswith(("http://", "https://"))

    def test_loading_is_deterministic(self, settings) -> None:
        first = [job.job_id for job in load_cached_postings(settings)]
        second = [job.job_id for job in load_cached_postings(settings)]
        assert first == second

    def test_missing_cache_returns_empty_rather_than_raising(
        self, settings, tmp_path
    ) -> None:
        from dataclasses import replace

        broken = replace(settings, cache_file=str(tmp_path / "does_not_exist.json"))
        assert load_cached_postings(broken) == []


class TestLessonContext:
    def test_builds_without_an_llm(self, settings) -> None:
        ctx = build_lesson_context(settings, llm=None)
        assert ctx.resume is not None
        assert ctx.jobs

    def test_llm_text_reports_unavailable_when_no_client(self, settings) -> None:
        ctx = build_lesson_context(settings, llm=None)
        text, available = ctx.llm_text("anything")
        assert text is None and available is False

    def test_llm_text_survives_a_raising_client(self, settings) -> None:
        class Exploding:
            available = True

            def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str:
                raise RuntimeError("provider down")

        ctx = build_lesson_context(settings, llm=Exploding())
        text, available = ctx.llm_text("anything")
        assert text is None and available is False

    def test_llm_text_treats_empty_reply_as_unavailable(self, settings) -> None:
        class Blank:
            available = True

            def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str:
                return ""

        ctx = build_lesson_context(settings, llm=Blank())
        assert ctx.llm_text("anything") == (None, False)

    def test_llm_text_returns_a_real_reply(self, settings) -> None:
        class Canned:
            available = True

            def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str:
                return "hello"

        ctx = build_lesson_context(settings, llm=Canned())
        assert ctx.llm_text("anything") == ("hello", True)


class TestStepRegistry:
    def test_seven_steps_registered(self) -> None:
        assert len(ALL_STEPS) == 7

    def test_numbered_one_through_seven_in_order(self) -> None:
        assert [step.number for step in ALL_STEPS] == [1, 2, 3, 4, 5, 6, 7]

    def test_every_step_is_a_lesson_step(self) -> None:
        assert all(isinstance(step, LessonStep) for step in ALL_STEPS)

    def test_lookup_by_number(self) -> None:
        assert step_by_number(4) is not None
        assert step_by_number(4).number == 4
        assert step_by_number(99) is None

    @pytest.mark.parametrize("step", ALL_STEPS, ids=lambda s: f"step{s.number}")
    def test_teaching_fields_are_filled_in(self, step: LessonStep) -> None:
        assert step.title.strip()
        assert step.subtitle.strip()
        assert len(step.concept.strip()) > 40
        assert len(step.why.strip()) > 40
        assert step.deck_reference.strip()
        assert step.takeaway.strip()

    @pytest.mark.parametrize("step", ALL_STEPS, ids=lambda s: f"step{s.number}")
    def test_code_snippet_is_short_enough_to_read_on_a_slide(
        self, step: LessonStep
    ) -> None:
        lines = step.code.strip().splitlines()
        assert 5 <= len(lines) <= 30, f"step {step.number} has {len(lines)} code lines"

    @pytest.mark.parametrize("step", ALL_STEPS, ids=lambda s: f"step{s.number}")
    def test_runs_offline_without_raising(self, settings, step: LessonStep) -> None:
        ctx = build_lesson_context(settings, llm=None)
        result = step.execute(ctx)
        assert result.blocks, f"step {step.number} produced no output offline"
        assert result.used_llm is False
        assert result.elapsed_seconds >= 0

    @pytest.mark.parametrize("step", ALL_STEPS, ids=lambda s: f"step{s.number}")
    def test_runs_offline_when_the_provider_raises(
        self, settings, step: LessonStep
    ) -> None:
        class Exploding:
            available = True

            def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str:
                raise RuntimeError("429 quota exhausted")

            def generate_structured(self, *args, **kwargs):
                raise RuntimeError("429 quota exhausted")

        ctx = build_lesson_context(settings, llm=Exploding())
        result = step.execute(ctx)
        assert result.blocks
        assert result.used_llm is False
