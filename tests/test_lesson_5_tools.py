"""Step 5 must teach the same lesson with or without a language model.

Nothing here touches the network or needs an API key: jobs come from the
clearly-labelled cached demonstration data, and the model is a stub.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from config import Settings
from lessons.base import LessonContext, LessonResult, OutputBlock
from lessons.step_5_tools import (
    COMPONENT_WEIGHTS,
    STEP,
    build_model_score_prompt,
    component_rows,
    describe_model_answers,
    fingerprint,
    parse_model_score,
    reference_time,
)
from models.job import JobPosting, RawJobResult
from models.resume import ResumeProfile
from services.llm_interface import NullLLMClient
from tests.conftest import make_job
from tools.firecrawl_search import cached_raw_results, load_cache
from tools.job_normalizer import normalize_jobs
from tools.job_scorer import score_job


class ScriptedLLM:
    """A reachable model whose replies are fixed in advance."""

    def __init__(self, replies: list[str | None], model_name: str = "stub-model") -> None:
        self.replies = list(replies)
        self.calls: list[str] = []
        self._model_name = model_name

    @property
    def available(self) -> bool:
        """Always True; this stub stands in for a working provider."""
        return True

    @property
    def model_name(self) -> str:
        """The name the comparison block prints."""
        return self._model_name

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Return the next scripted reply, repeating the last one if exhausted."""
        self.calls.append(prompt)
        if not self.replies:
            return None
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[index]


class ExplodingLLM:
    """A model that claims availability but raises on every call.

    Mirrors a live 503 or quota error: :meth:`LessonContext.llm_text` must
    absorb it and report the model as unreachable.
    """

    @property
    def available(self) -> bool:
        """Advertised as reachable, which is what makes this case interesting."""
        return True

    @property
    def model_name(self) -> str:
        """A readable placeholder name."""
        return "exploding-model"

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Always fail, the way an overloaded free-tier endpoint does."""
        raise RuntimeError("503 Service Unavailable")


def cached_jobs(settings: Settings) -> list[JobPosting]:
    """Build real :class:`JobPosting` objects from the offline cache.

    Mirrors ``agent.nodes._load_cached_jobs``: cached records carry a previously
    extracted payload, so normalization needs no API call.
    """
    payload = load_cache(settings.cache_path)
    entries = cached_raw_results(
        payload, query_category="company_careers", freshness_window="last_24_hours"
    )
    retrieved_at = datetime.fromisoformat(
        str(payload["originally_retrieved_at"]).replace("Z", "+00:00")
    )
    raw_results: list[RawJobResult] = []
    for entry in entries:
        record = dict(entry)
        record["query_category"] = "company_careers"
        record["freshness_window"] = "last_24_hours"
        record["retrieved_at"] = retrieved_at
        raw_results.append(RawJobResult.model_validate(record))
    postings, _ = normalize_jobs(raw_results, NullLLMClient(), data_mode="cached")
    return postings


@pytest.fixture
def jobs(settings: Settings) -> list[JobPosting]:
    """Postings normalized from the cached demonstration data."""
    postings = cached_jobs(settings)
    assert postings, "the cached demonstration data should yield postings"
    return postings


def make_ctx(
    settings: Settings,
    resume: ResumeProfile,
    jobs: list[JobPosting],
    llm: object,
) -> LessonContext:
    """Assemble a :class:`LessonContext` for one scenario."""
    return LessonContext(settings=settings, llm=llm, resume=resume, jobs=jobs)


def kinds(result: LessonResult) -> list[str]:
    """Return the block kinds a result produced, in order."""
    return [block.kind for block in result.blocks]


class TestStepMetadata:
    """The module honours the LessonStep contract the app renders against."""

    def test_the_module_exports_step_5(self):
        assert STEP.number == 5
        assert STEP.title
        assert STEP.subtitle
        assert STEP.deck_reference

    def test_the_prose_fields_are_plain_english_and_short(self):
        assert 1 <= STEP.concept.count(".") <= 3
        assert 1 <= STEP.why.count(".") <= 3
        assert STEP.takeaway.count(".") == 1

    def test_the_teaching_snippet_is_short_and_real(self):
        lines = STEP.code.strip().splitlines()
        assert 10 <= len(lines) <= 25
        assert "from tools.job_scorer import score_job" in STEP.code

    def test_the_step_does_not_require_a_model(self):
        assert STEP.needs_llm is False

    def test_the_weighting_matches_claude_md_section_15(self):
        weights = [round(weight * 100) for _, _, weight in COMPONENT_WEIGHTS]
        assert weights == [45, 20, 15, 10, 10]
        assert sum(weights) == 100


class TestOfflinePath:
    """With no model reachable, the deterministic lesson still lands."""

    def test_it_never_raises_and_returns_blocks(self, settings, resume, jobs, null_llm):
        result = STEP.execute(make_ctx(settings, resume, jobs, null_llm))
        assert len(result.blocks) >= 1
        assert all(isinstance(block, OutputBlock) for block in result.blocks)

    def test_the_flags_report_the_offline_path_honestly(
        self, settings, resume, jobs, null_llm
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, null_llm))
        assert result.used_llm is False
        assert result.llm_unavailable is True

    def test_it_explains_that_the_model_half_was_skipped(
        self, settings, resume, jobs, null_llm
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, null_llm))
        notes = [block for block in result.blocks if block.kind == "note"]
        assert notes, "the offline path must say the model half did not run"
        assert "could not be reached" in notes[-1].body

    def test_no_comparison_block_is_faked_without_a_model(
        self, settings, resume, jobs, null_llm
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, null_llm))
        assert "compare" not in kinds(result)

    def test_determinism_is_proved_with_a_success_block(
        self, settings, resume, jobs, null_llm
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, null_llm))
        successes = [block for block in result.blocks if block.kind == "success"]
        assert len(successes) == 1
        assert "identical" in successes[0].label.lower()
        assert "same signature" in successes[0].body.lower()

    def test_the_component_breakdown_is_shown(self, settings, resume, jobs, null_llm):
        result = STEP.execute(make_ctx(settings, resume, jobs, null_llm))
        tables = [block for block in result.blocks if block.kind == "table"]
        assert len(tables) == 2
        components = [row["Component"] for row in tables[0].body]
        assert "Required-skill coverage" in components
        assert "Demo Job Match Score" in components
        assert len(tables[1].body) == 3

    def test_a_model_that_raises_is_treated_as_unavailable(
        self, settings, resume, jobs
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, ExplodingLLM()))
        assert result.used_llm is False
        assert result.llm_unavailable is True
        assert "compare" not in kinds(result)

    def test_a_missing_llm_object_is_survivable(self, settings, resume, jobs):
        result = STEP.execute(make_ctx(settings, resume, jobs, None))
        assert result.llm_unavailable is True
        assert len(result.blocks) >= 1


class TestModelPath:
    """With a reachable model, the comparison half runs and stays factual."""

    def test_two_different_answers_are_compared(self, settings, resume, jobs):
        llm = ScriptedLLM(["72", "58"])
        result = STEP.execute(make_ctx(settings, resume, jobs, llm))
        assert result.used_llm is True
        assert result.llm_unavailable is False
        compares = [block for block in result.blocks if block.kind == "compare"]
        assert len(compares) == 1
        body = compares[0].body
        assert set(body) == {"left_label", "left", "right_label", "right"}
        assert "Ask 1: 72" in body["right"]
        assert "Ask 2: 58" in body["right"]
        assert "stub-model" in body["right_label"]
        assert len(llm.calls) == 2

    def test_the_model_is_asked_the_same_question_both_times(
        self, settings, resume, jobs
    ):
        llm = ScriptedLLM(["72", "58"])
        STEP.execute(make_ctx(settings, resume, jobs, llm))
        assert llm.calls[0] == llm.calls[1]

    def test_a_repeated_model_answer_is_reported_fairly(self, settings, resume, jobs):
        llm = ScriptedLLM(["77", "77"])
        result = STEP.execute(make_ctx(settings, resume, jobs, llm))
        assert result.used_llm is True
        notes = [block for block in result.blocks if block.kind == "note"]
        assert any("returned 77 twice" in block.body for block in notes)
        warnings = [block for block in result.blocks if block.kind == "warning"]
        assert not warnings, "a repeated answer must not be overstated as a failure"

    def test_the_python_score_is_unchanged_by_the_model(self, settings, resume, jobs):
        offline = STEP.execute(make_ctx(settings, resume, jobs, NullLLMClient()))
        online = STEP.execute(make_ctx(settings, resume, jobs, ScriptedLLM(["3", "99"])))
        offline_metric = [b for b in offline.blocks if b.kind == "metric"][0]
        online_metric = [b for b in online.blocks if b.kind == "metric"][0]
        assert offline_metric.body["value"] == online_metric.body["value"]

    def test_a_second_call_that_fails_is_reported_not_hidden(
        self, settings, resume, jobs
    ):
        llm = ScriptedLLM(["81", None])
        result = STEP.execute(make_ctx(settings, resume, jobs, llm))
        assert result.used_llm is True
        assert result.llm_unavailable is False
        compares = [block for block in result.blocks if block.kind == "compare"]
        assert "did not answer a second time" in compares[0].body["right"]

    def test_unparseable_model_output_is_shown_verbatim(self, settings, resume, jobs):
        llm = ScriptedLLM(["it depends on the team", "hard to say"])
        result = STEP.execute(make_ctx(settings, resume, jobs, llm))
        compares = [block for block in result.blocks if block.kind == "compare"]
        assert "no number found" in compares[0].body["right"]


class TestNoJobs:
    """An empty job list is a message, never a crash."""

    def test_it_warns_instead_of_raising(self, settings, resume, null_llm):
        result = STEP.execute(make_ctx(settings, resume, [], null_llm))
        assert len(result.blocks) == 1
        assert result.blocks[0].kind == "warning"


class TestHelpers:
    """The small pure functions behave the way the step depends on."""

    def test_scoring_the_same_pair_repeatedly_is_identical(self, resume, jobs):
        job = jobs[0]
        signatures = {
            fingerprint(
                score_job(
                    job,
                    resume,
                    location="Houston, TX",
                    work_mode="Any",
                    now=reference_time(job),
                )
            )
            for _ in range(5)
        }
        assert len(signatures) == 1

    def test_reference_time_uses_the_posting_retrieval_time(self):
        job = make_job()
        assert reference_time(job) == job.retrieved_at

    def test_component_rows_carry_the_weighted_points(self, resume, jobs):
        job = jobs[0]
        match = score_job(
            job, resume, location="Houston, TX", work_mode="Any", now=reference_time(job)
        )
        rows = component_rows(match)
        assert len(rows) == len(COMPONENT_WEIGHTS) + 1
        skill_row = rows[0]
        assert skill_row["Weight"] == "45 pts"
        assert skill_row["Points earned"] == round(match.skill_score * 0.45, 1)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("72", 72),
            ("  85  ", 85),
            ("I would say 64 out of 100.", 64),
            ("0", 0),
            ("100", 100),
            ("no idea", None),
            (None, None),
            ("", None),
        ],
    )
    def test_model_scores_are_parsed_or_rejected(self, text, expected):
        assert parse_model_score(text) == expected

    def test_the_model_prompt_carries_both_inputs(self, resume, jobs):
        prompt = build_model_score_prompt(resume, jobs[0])
        assert "CANDIDATE RESUME" in prompt
        assert "JOB POSTING" in prompt
        assert jobs[0].title in prompt
        assert resume.name in prompt

    @pytest.mark.parametrize(
        ("answers", "kind"),
        [([72, 58], "warning"), ([77, 77], "note"), ([81], "note"), ([], "note")],
    )
    def test_the_verdict_matches_what_the_model_actually_did(self, answers, kind):
        assert describe_model_answers(answers)[0] == kind

    def test_the_verdict_never_calls_a_repeat_a_failure(self):
        _, sentence = describe_model_answers([77, 77])
        assert "not guaranteed" in sentence


class BadlyBehavedLLM:
    """A client that violates the ``str | None`` contract of ``generate_text``.

    Stands in for a swapped-in provider that hands back a response object or a
    raw number. The optional half may suffer; the lesson must not.
    """

    available = True
    model_name = "badly-behaved"

    def generate_text(self, prompt: str, *, temperature: float = 0.2):
        """Return an int, which is not what the protocol promises."""
        return 55


class AvailabilityRaisesLLM:
    """A client whose ``available`` check itself blows up."""

    model_name = "availability-raises"

    @property
    def available(self) -> bool:
        """Fail before any call can be attempted."""
        raise RuntimeError("credential check exploded")

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str:
        """Never reached."""
        return "50"


class TestTheOptionalHalfCannotTakeDownTheLesson:
    """A misbehaving provider may cost the comparison, never the teaching."""

    def test_a_non_string_reply_does_not_erase_the_deterministic_blocks(
        self, settings, resume, jobs
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, BadlyBehavedLLM()))
        assert "success" in kinds(result), "the determinism proof must survive"
        assert "metric" in kinds(result)
        assert len(result.blocks) >= 6

    def test_an_exploding_availability_check_leaves_the_lesson_standing(
        self, settings, resume, jobs
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, AvailabilityRaisesLLM()))
        assert "success" in kinds(result)
        assert "compare" not in kinds(result), "no comparison may be invented"
        assert result.llm_unavailable is True
        notes = [block for block in result.blocks if block.kind == "note"]
        assert any("could not run" in block.label for block in notes)

    def test_the_deterministic_score_is_the_same_however_the_model_misbehaves(
        self, settings, resume, jobs, null_llm
    ):
        baseline = STEP.execute(make_ctx(settings, resume, jobs, null_llm))
        broken = STEP.execute(make_ctx(settings, resume, jobs, BadlyBehavedLLM()))
        value = [b for b in baseline.blocks if b.kind == "metric"][0].body["value"]
        assert [b for b in broken.blocks if b.kind == "metric"][0].body["value"] == value


class TestTheComparisonReportsWhatActuallyHappened:
    """Nothing on the screen may be a stand-in for an unobserved value."""

    def test_the_run_lines_come_from_the_actual_runs_not_a_repeated_constant(
        self, settings, resume, jobs, null_llm, monkeypatch
    ):
        """Force the scorer to drift and check the step reports the drift.

        The scorer is deterministic, so this is the only way to prove the
        comparison prints observed totals rather than one value copied three
        times - which would fake evidence for the exact claim under test.
        """
        totals = iter([81, 62, 47])

        def drifting_score_job(job, resume_arg, **kwargs):
            base = score_job(job, resume_arg, **kwargs)
            return base.model_copy(update={"total_score": next(totals)})

        monkeypatch.setattr("lessons.step_5_tools.score_job", drifting_score_job)
        result = STEP.execute(make_ctx(settings, resume, jobs, ScriptedLLM(["70", "70"])))

        compare = [b for b in result.blocks if b.kind == "compare"][0]
        assert "Run 1: 81" in compare.body["left"]
        assert "Run 2: 62" in compare.body["left"]
        assert "Run 3: 47" in compare.body["left"]
        warnings = [b for b in result.blocks if b.kind == "warning"]
        assert any("did not match" in block.label for block in warnings)

    def test_an_empty_reply_is_labelled_rather_than_left_dangling(
        self, settings, resume, jobs
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, ScriptedLLM(["   ", "  "])))
        compare = [b for b in result.blocks if b.kind == "compare"][0]
        assert "the reply was empty" in compare.body["right"]
        assert "no number found: \n" not in compare.body["right"]

    def test_two_unreadable_replies_are_not_called_one_answer(
        self, settings, resume, jobs
    ):
        result = STEP.execute(
            make_ctx(settings, resume, jobs, ScriptedLLM(["maybe", "unclear"]))
        )
        notes = [b for b in result.blocks if b.kind == "note"]
        assert any("answered twice" in block.body for block in notes)
        assert not any("Only one usable" in block.body for block in notes)

    def test_the_metric_shows_the_unrounded_weighted_total(
        self, settings, resume, jobs, null_llm
    ):
        result = STEP.execute(make_ctx(settings, resume, jobs, null_llm))
        metric = [b for b in result.blocks if b.kind == "metric"][0]
        assert "add up to" in metric.body["help"]
        assert "rounded to" in metric.body["help"]


class TestTheVerdictDistinguishesSilenceFromNonsense:
    """"No answer" and "two unusable answers" are different facts."""

    def test_zero_answers_from_two_replies_says_so(self):
        kind, sentence = describe_model_answers([], 2)
        assert kind == "note"
        assert "answered twice" in sentence

    def test_zero_answers_from_no_replies_says_something_different(self):
        _, silent = describe_model_answers([], 0)
        _, nonsense = describe_model_answers([], 2)
        assert silent != nonsense
        assert "No readable number" in silent

    def test_one_usable_answer_is_still_reported_as_one(self):
        _, sentence = describe_model_answers([81], 1)
        assert "Only one usable" in sentence


class TestTheParserOnlyReportsNumbersTheModelMeant:
    """A misread digit printed as "the model's answer" would be a lie."""

    @pytest.mark.parametrize(
        "text",
        ["As of 2026, hard to say", "about 0.85", "1000", "score: 250", "v1.2.3"],
    )
    def test_numbers_that_are_not_a_0_to_100_verdict_are_rejected(self, text):
        assert parse_model_score(text) is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("85/100", 85), ("I'd say 64.", 64), ("Score: 7", 7)],
    )
    def test_real_verdicts_are_still_read(self, text, expected):
        assert parse_model_score(text) == expected
