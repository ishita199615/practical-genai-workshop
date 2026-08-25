"""Step 7 must teach the guardrail with no network and no API key.

Every test here runs the real validator against the real cached demonstration
jobs. Two stub clients stand in for Gemini: one that is unavailable (the free
tier's usual state during a live workshop) and one that returns canned text.
The step has to behave correctly, and report honestly, under both.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from config import Settings
from lessons.base import LessonContext, LessonResult, LessonStep
from lessons.context import build_lesson_context
from lessons.step_7_guardrails import (
    STEP,
    UNSUPPORTED_REQUIREMENT,
    build_dishonest_draft,
    build_honest_draft,
    pick_job,
    power_bi_verdict,
    quotable,
    unproven_skills,
)
from models.job import JobPosting, RawJobResult
from models.resume import ResumeProfile
from services.llm_interface import NullLLMClient
from tests.conftest import make_job
from tools.ats_scorer import snapshot_from_resume
from tools.claim_validator import run_deterministic_checks
from tools.firecrawl_search import cached_raw_results, load_cache
from tools.job_filter import filter_and_deduplicate
from tools.job_normalizer import normalize_jobs

CANNED_TEXT = "A validator is a rule, not a request. Rules keep working offline."


class StubLLM:
    """A scripted client with the same surface the lesson context expects."""

    def __init__(self, *, available: bool, text: str | None = CANNED_TEXT) -> None:
        self._available = available
        self._text = text
        self.prompts: list[str] = []

    @property
    def available(self) -> bool:
        """Whether this stub claims it can reach a model."""
        return self._available

    @property
    def model_name(self) -> str:
        """A readable placeholder name."""
        return "stub"

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Record the prompt and return the canned reply."""
        self.prompts.append(prompt)
        return self._text

    def generate_structured(
        self, prompt: str, schema: type[BaseModel], *, temperature: float = 0.1
    ) -> BaseModel | None:
        """Decline structured output so deterministic paths stay in charge."""
        self.prompts.append(prompt)
        return None


class ExplodingLLM:
    """A client that fails the way an overloaded provider does."""

    @property
    def available(self) -> bool:
        """Claims availability, then fails on use."""
        return True

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Raise the way a 503 surfaces through an SDK."""
        raise RuntimeError("503 model overloaded")


@pytest.fixture(scope="module")
def cached_jobs() -> list[JobPosting]:
    """Build real ``JobPosting`` objects from the offline cache.

    Mirrors ``agent/nodes.py``: load the cache, normalize with an unavailable
    LLM so the deterministic extraction path runs, then filter and deduplicate.
    """
    settings = Settings(demo_mode="cached")
    payload = load_cache(settings.cache_path)
    entries = cached_raw_results(
        payload, query_category="all", freshness_window="last_7_days"
    )
    raws: list[RawJobResult] = []
    for entry in entries:
        record = dict(entry)
        record.setdefault("query_category", "all")
        record.setdefault("freshness_window", "last_7_days")
        record.setdefault("retrieved_at", payload.get("originally_retrieved_at"))
        raws.append(RawJobResult.model_validate(record))
    postings, _ = normalize_jobs(raws, NullLLMClient(), data_mode="cached")
    return filter_and_deduplicate(postings).kept


def make_context(
    resume: ResumeProfile, jobs: list[JobPosting], llm: Any
) -> LessonContext:
    """Assemble a lesson context without touching the network."""
    return LessonContext(
        settings=Settings(demo_mode="cached"), llm=llm, resume=resume, jobs=list(jobs)
    )


class TestCacheFixture:
    """The offline cache must actually supply the step with something to teach."""

    def test_cache_yields_usable_postings(self, cached_jobs: list[JobPosting]) -> None:
        assert cached_jobs, "the cached demonstration data produced no valid jobs"

    def test_a_posting_asks_for_power_bi(self, cached_jobs: list[JobPosting]) -> None:
        chosen = pick_job(cached_jobs)
        assert chosen is not None
        haystack = f"{chosen.title} {chosen.description} {chosen.required_skills}".lower()
        assert UNSUPPORTED_REQUIREMENT.lower() in haystack


class TestOfflineRun:
    """The unavailable-model case is the one the workshop actually hits."""

    @pytest.fixture
    def result(self, resume: ResumeProfile, cached_jobs: list[JobPosting]) -> LessonResult:
        return STEP.execute(make_context(resume, cached_jobs, NullLLMClient()))

    def test_reports_the_model_was_not_used(self, result: LessonResult) -> None:
        assert result.used_llm is False
        assert result.llm_unavailable is True

    def test_still_produces_a_full_result(self, result: LessonResult) -> None:
        assert len(result.blocks) >= 8

    def test_shows_the_guardrail_blocking_the_lie(self, result: LessonResult) -> None:
        warnings = [block for block in result.blocks if block.kind == "warning"]
        assert warnings, "the dishonest draft must produce a warning block"
        assert UNSUPPORTED_REQUIREMENT in warnings[0].body

    def test_shows_the_honest_draft_passing(self, result: LessonResult) -> None:
        successes = [block for block in result.blocks if block.kind == "success"]
        assert successes, "the honest draft must produce a success block"
        assert "10 of 10" in successes[0].label

    def test_explains_that_python_did_the_work(self, result: LessonResult) -> None:
        notes = [block for block in result.blocks if block.kind == "note"]
        assert notes, "an unavailable model must be stated, not hidden"
        assert "Python" in notes[0].body

    def test_renders_the_required_learning_gap_statement(self, result: LessonResult) -> None:
        code_blocks = [
            block.body for block in result.blocks if block.kind == "code"
        ]
        verdict = next(body for body in code_blocks if "Job requirement:" in body)
        assert "Job requirement: Power BI" in verdict
        assert "Evidence in master resume: None" in verdict
        assert "Decision: Power BI was not added to the resume" in verdict
        assert "Recommendation: Treat it as a learning gap" in verdict

    def test_covers_the_human_approval_gate(self, result: LessonResult) -> None:
        prose = "\n".join(
            str(block.body) for block in result.blocks if block.kind == "markdown"
        )
        for phrase in ("Approve", "Request Changes", "Reject"):
            assert phrase in prose
        assert "No application is ever submitted." in prose


class TestAvailableModelRun:
    """A reachable model adds commentary and nothing else."""

    @pytest.fixture
    def stub(self) -> StubLLM:
        return StubLLM(available=True)

    @pytest.fixture
    def result(
        self, resume: ResumeProfile, cached_jobs: list[JobPosting], stub: StubLLM
    ) -> LessonResult:
        return STEP.execute(make_context(resume, cached_jobs, stub))

    def test_reports_the_model_was_used(self, result: LessonResult) -> None:
        assert result.used_llm is True
        assert result.llm_unavailable is False

    def test_shows_the_model_reply(self, result: LessonResult) -> None:
        bodies = [str(block.body) for block in result.blocks]
        assert any(CANNED_TEXT in body for body in bodies)

    def test_the_deterministic_verdict_is_unchanged(
        self, resume: ResumeProfile, cached_jobs: list[JobPosting], stub: StubLLM
    ) -> None:
        def tables(run: LessonResult) -> list[Any]:
            return [block.body for block in run.blocks if block.kind == "table"]

        with_model = STEP.execute(make_context(resume, cached_jobs, stub))
        without_model = STEP.execute(make_context(resume, cached_jobs, NullLLMClient()))
        assert tables(with_model) == tables(without_model)


class TestNeverRaises:
    """A student clicking Run must always see something."""

    @pytest.mark.parametrize(
        "llm",
        [NullLLMClient(), StubLLM(available=True), StubLLM(available=True, text=None), ExplodingLLM(), None],
        ids=["null", "canned", "empty-reply", "exploding", "missing"],
    )
    def test_survives_any_client(self, resume: ResumeProfile, llm: Any) -> None:
        result = STEP.execute(make_context(resume, [], llm))
        assert len(result.blocks) >= 1

    def test_survives_no_jobs(self, resume: ResumeProfile) -> None:
        result = STEP.execute(make_context(resume, [], NullLLMClient()))
        assert result.used_llm is False
        assert any(block.kind == "success" for block in result.blocks)

    def test_survives_a_job_with_hostile_text(self, resume: ResumeProfile) -> None:
        """A live posting whose title carries digits and an unproven tool.

        The honest draft must refuse to quote it rather than import the claim.
        """
        hostile = make_job(
            title="Power BI Analyst 2029",
            company="Vendor 9 Labs",
            required_skills=["Power BI"],
        )
        result = STEP.execute(make_context(resume, [hostile], NullLLMClient()))
        successes = [block for block in result.blocks if block.kind == "success"]
        assert successes and "10 of 10" in successes[0].label


class TestDrafts:
    """The two drafts must differ in exactly the way the lesson claims."""

    def test_dishonest_draft_is_blocked_on_several_counts(
        self, resume: ResumeProfile
    ) -> None:
        draft = build_dishonest_draft("job_1", resume)
        checks = run_deterministic_checks(
            draft, resume, unsupported_gaps=[UNSUPPORTED_REQUIREMENT]
        )
        failed = {check["name"] for check in checks if not check["passed"]}
        assert "Skills remain grounded in the master resume" in failed
        assert "Unsupported job gaps were not added" in failed
        assert "Cover letter claims no unsupported skill" in failed
        assert "No invented metrics" in failed

    def test_dishonest_draft_still_cites_a_real_bullet(
        self, resume: ResumeProfile
    ) -> None:
        """Only the lie is on trial, so the traceable parts must stay valid."""
        draft = build_dishonest_draft("job_1", resume)
        checks = run_deterministic_checks(
            draft, resume, unsupported_gaps=[UNSUPPORTED_REQUIREMENT]
        )
        by_name = {check["name"]: check["passed"] for check in checks}
        assert by_name["Source IDs verified"] is True
        assert by_name["Original bullet text matches the master resume"] is True

    def test_honest_draft_clears_every_check(
        self, resume: ResumeProfile, cached_jobs: list[JobPosting]
    ) -> None:
        snapshot = snapshot_from_resume(resume)
        for job in cached_jobs:
            draft = build_honest_draft(job, resume, snapshot)
            checks = run_deterministic_checks(
                draft,
                resume,
                unsupported_gaps=[UNSUPPORTED_REQUIREMENT],
                allowed_organizations=[job.company],
            )
            failed = [check["name"] for check in checks if not check["passed"]]
            assert not failed, f"{job.title}: {failed}"

    def test_honest_draft_never_names_power_bi(
        self, resume: ResumeProfile, cached_jobs: list[JobPosting]
    ) -> None:
        snapshot = snapshot_from_resume(resume)
        draft = build_honest_draft(pick_job(cached_jobs), resume, snapshot)
        text = " ".join(
            [
                draft.revised_summary,
                draft.cover_letter,
                *draft.reordered_skills,
                *[bullet.revised_text for bullet in draft.revised_bullets],
            ]
        ).lower()
        assert UNSUPPORTED_REQUIREMENT.lower() not in text

    def test_honest_draft_only_reorders_skills(self, resume: ResumeProfile) -> None:
        draft = build_honest_draft(make_job(), resume, snapshot_from_resume(resume))
        assert sorted(draft.reordered_skills) == sorted(resume.skills)

    def test_honest_draft_revises_exactly_two_bullets(
        self, resume: ResumeProfile
    ) -> None:
        draft = build_honest_draft(make_job(), resume, snapshot_from_resume(resume))
        index = resume.bullet_index()
        assert len(draft.revised_bullets) == 2
        for bullet in draft.revised_bullets:
            assert index[bullet.source_bullet_id] == bullet.original_text


class TestHelpers:
    """The small guards the honest draft depends on."""

    def test_power_bi_has_no_resume_evidence(self, resume: ResumeProfile) -> None:
        verdict = power_bi_verdict(resume, snapshot_from_resume(resume))
        assert "Evidence in master resume: None" in verdict

    def test_unproven_skills_flags_power_bi_but_not_tableau(
        self, resume: ResumeProfile
    ) -> None:
        snapshot = snapshot_from_resume(resume)
        found = unproven_skills("We use Power BI and Tableau daily.", resume, snapshot)
        assert "power bi" in [name.lower() for name in found]
        assert "tableau" not in [name.lower() for name in found]

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Data Analyst Intern", True),
            ("Power BI Analyst", False),
            ("Analyst 2029", False),
            ("", False),
            (None, False),
        ],
    )
    def test_quotable_rejects_risky_text(
        self, resume: ResumeProfile, text: str | None, expected: bool
    ) -> None:
        assert quotable(text, resume, snapshot_from_resume(resume)) is expected

    def test_pick_job_returns_none_without_jobs(self) -> None:
        assert pick_job([]) is None


class TestStepMetadata:
    """The Learn tab reads these fields directly."""

    def test_is_a_lesson_step_numbered_seven(self) -> None:
        assert isinstance(STEP, LessonStep)
        assert STEP.number == 7

    def test_declares_it_does_not_need_a_model(self) -> None:
        assert STEP.needs_llm is False

    def test_every_teaching_field_is_filled(self) -> None:
        for field_name in ("title", "subtitle", "concept", "why", "deck_reference", "takeaway"):
            assert getattr(STEP, field_name).strip()

    def test_the_snippet_stays_readable_on_a_projector(self) -> None:
        lines = STEP.code.strip().splitlines()
        assert 10 <= len(lines) <= 25
        assert "run_deterministic_checks" in STEP.code


class TestModelReplyHonesty:
    """A reply of nothing must never be presented as the model's answer."""

    @pytest.mark.parametrize(
        "reply", ["", "   ", "\n\n", "  \n \t "], ids=["empty", "spaces", "newlines", "mixed"]
    )
    def test_a_blank_reply_is_not_counted_as_a_model_answer(
        self, resume: ResumeProfile, reply: str
    ) -> None:
        """A whitespace-only reply is not a reply.

        Reporting ``used_llm`` here would make the page caption claim the model
        was called and answered, under a heading with nothing beneath it.
        """
        result = STEP.execute(make_context(resume, [], StubLLM(available=True, text=reply)))
        assert result.used_llm is False
        assert result.llm_unavailable is True
        assert any(block.kind == "note" for block in result.blocks)

    def test_no_block_is_ever_rendered_blank(self, resume: ResumeProfile) -> None:
        """Every prose block must carry text, whatever the model returned."""
        for llm in (NullLLMClient(), StubLLM(available=True), StubLLM(available=True, text="  ")):
            result = STEP.execute(make_context(resume, [], llm))
            for block in result.blocks:
                if block.kind in {"markdown", "code", "note", "warning", "success"}:
                    assert str(block.body).strip(), f"{block.label} rendered empty"

    def test_a_real_reply_is_still_shown_and_credited(self, resume: ResumeProfile) -> None:
        result = STEP.execute(make_context(resume, [], StubLLM(available=True)))
        assert result.used_llm is True
        assert any(CANNED_TEXT in str(block.body) for block in result.blocks)


class TestUntrustedJobText:
    """Retrieved job text must never break the honest draft on stage."""

    @pytest.mark.parametrize(
        "title",
        [
            "Data Analyst at Acme",
            "Data Analyst Intern at Globex Corporation",
            "Analyst for Northwind",
            "Intern - Bachelors preferred",
            "Certified Data Analyst",
            "Power BI Analyst 2029",
        ],
    )
    def test_a_hostile_title_never_costs_draft_b_a_check(
        self, resume: ResumeProfile, title: str
    ) -> None:
        """Quoting a scraped title must not import an employer, degree or metric.

        Each of these would otherwise trip a different deterministic check and
        turn the lesson's punchline into a failure in front of the room.
        """
        job = make_job(title=title, company="Vendor 9 Labs", required_skills=["Power BI"])
        result = STEP.execute(make_context(resume, [job], NullLLMClient()))
        successes = [block for block in result.blocks if block.kind == "success"]
        assert successes and "10 of 10" in successes[0].label

    def test_quotable_rejects_employer_and_credential_wording(
        self, resume: ResumeProfile
    ) -> None:
        snapshot = snapshot_from_resume(resume)
        for risky in ("Data Analyst at Acme", "Analyst for Northwind", "Certified Analyst"):
            assert quotable(risky, resume, snapshot) is False

    def test_the_honest_draft_is_itself_validated_before_it_is_shown(
        self, resume: ResumeProfile
    ) -> None:
        """The fallback is structural, not a word list: the draft is re-checked."""
        job = make_job(title="Data Analyst at Acme", company="Acme")
        draft = build_honest_draft(job, resume, snapshot_from_resume(resume))
        checks = run_deterministic_checks(
            draft, resume, unsupported_gaps=[UNSUPPORTED_REQUIREMENT]
        )
        assert all(check["passed"] for check in checks)
        assert "Acme" not in draft.revised_summary + draft.cover_letter


class TestMatchesWhatTheProjectorShows:
    """The snippet on screen and the live output must agree."""

    def test_draft_a_swaps_the_tool_the_resume_actually_proves(
        self, resume: ResumeProfile
    ) -> None:
        """The lie should replace Tableau, exactly as the code panel shows."""
        draft = build_dishonest_draft("job_1", resume)
        bullet = draft.revised_bullets[0]
        assert bullet.source_bullet_id in STEP.code
        assert bullet.original_text in STEP.code
        assert "tableau" in bullet.original_text.lower()

    def test_the_snippets_claimed_output_is_what_the_validator_says(
        self, resume: ResumeProfile
    ) -> None:
        """Comments claiming printed output must match the real detail strings."""
        draft = build_dishonest_draft("job_1", resume)
        details = {
            check["detail"]
            for check in run_deterministic_checks(
                draft, resume, unsupported_gaps=[UNSUPPORTED_REQUIREMENT]
            )
            if not check["passed"]
        }
        assert "Unsupported number(s): 40." in details
        assert "Unsupported number(s): 40." in STEP.code


class TestTheContextTheAppActuallyBuilds:
    """The Learn tab passes an unfiltered list; the tests must use the same one."""

    def test_honest_draft_clears_every_check_for_every_posting_the_app_loads(
        self, resume: ResumeProfile
    ) -> None:
        """Guards against a cache refresh degrading the demo silently."""
        context = build_lesson_context(Settings(demo_mode="cached"), llm=None)
        assert context.jobs, "the lesson context loaded no postings"
        snapshot = snapshot_from_resume(context.resume)
        for job in context.jobs:
            draft = build_honest_draft(job, context.resume, snapshot)
            checks = run_deterministic_checks(
                draft,
                context.resume,
                unsupported_gaps=[UNSUPPORTED_REQUIREMENT],
                allowed_organizations=[job.company],
            )
            failed = [check["name"] for check in checks if not check["passed"]]
            assert not failed, f"{job.title} @ {job.company}: {failed}"

    def test_the_step_runs_clean_on_the_real_context(self, resume: ResumeProfile) -> None:
        context = build_lesson_context(Settings(demo_mode="cached"), llm=None)
        result = STEP.execute(context)
        assert result.used_llm is False and result.llm_unavailable is True
        successes = [block for block in result.blocks if block.kind == "success"]
        assert successes and "10 of 10" in successes[0].label
