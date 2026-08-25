"""Step 6 - the agent loop: Reason -> Act -> Observe.

Deck slide 18 draws three boxes: Loop, Harness, Graph. This step makes the
first box concrete by running an honest, visible agent loop over the tools this
project already uses. A rule-based reasoner picks one action per turn, the
action calls a real function from ``tools/``, and the observation goes back into
working memory so the next turn can reason about it.

Nothing here needs a language model. The model is offered exactly one optional
job - narrating the finished trace in a sentence - and when it is unavailable
the step says so instead of inventing a narration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from lessons.base import LessonContext, LessonResult, LessonStep
from models.ats import BAND_LABELS
from tools.ats_scorer import assess_ats
from tools.job_scorer import score_job

# The loop is bounded. Even if every tool call fails, it cannot run forever.
MAX_TURNS = 6


@dataclass(frozen=True)
class Plan:
    """What the reasoner decided to do on one turn.

    ``memory_key`` is the fact this action is supposed to produce. Recording it
    even when the action fails is what stops the loop retrying the same broken
    step until it hits ``MAX_TURNS``.
    """

    reason: str
    action_key: str
    action_label: str
    memory_key: str | None = None


@dataclass(frozen=True)
class LoopTurn:
    """One completed pass of Reason -> Act -> Observe.

    ``goal_reached`` is the state of the goal *after* this turn, recorded so
    the screen can report how the loop ended instead of assuming it succeeded.
    """

    step: int
    reason: str
    action: str
    observation: str
    goal_reached: bool = False

    def as_row(self) -> dict[str, Any]:
        """Render this turn as a flat row for a ``table`` output block."""
        return {
            "Step": self.step,
            "Reason": self.reason,
            "Action": self.action,
            "Observation": self.observation,
        }


# --------------------------------------------------------------------------
# Reason
# --------------------------------------------------------------------------


def goal_is_met(memory: dict[str, Any]) -> bool:
    """True only when both deterministic results really are in memory.

    Deliberately different from the ``in memory`` checks below. Routing asks
    "have I already tried this?" so a failed step is not retried forever;
    reporting asks "did I actually get it?" so the loop never announces a goal
    it did not reach.
    """
    return bool(memory.get("matches")) and memory.get("ats") is not None


def decide_next_action(memory: dict[str, Any]) -> Plan:
    """Choose the next action by looking at what is already known.

    This is the whole "reasoning" step: a short chain of rules over working
    memory. A real agent asks a model this question instead, but the shape of
    the answer - one action, plus a reason it can state out loud - is identical.
    """
    if "jobs" not in memory:
        return Plan(
            reason="I have no candidate jobs yet, so I cannot rank anything.",
            action_key="load_jobs",
            action_label="read ctx.jobs (retrieved by tools.firecrawl_search)",
            memory_key="jobs",
        )
    if "matches" not in memory:
        return Plan(
            reason="I have jobs but no ranking, so I cannot say which one fits best.",
            action_key="score_jobs",
            action_label="tools.job_scorer.score_job (once per job)",
            memory_key="matches",
        )
    if "ats" not in memory:
        return Plan(
            reason="I have a best job but I have not checked the resume against it.",
            action_key="score_ats",
            action_label="tools.ats_scorer.assess_ats",
            memory_key="ats",
        )
    if goal_is_met(memory):
        reason = "Goal met: the jobs are ranked and the resume is scored."
    else:
        reason = (
            "Stopping short of the goal: every action has been tried once and "
            "no remaining one could add anything."
        )
    return Plan(
        reason=reason,
        action_key="stop",
        action_label="stop - no tool call",
        memory_key=None,
    )


# --------------------------------------------------------------------------
# Act
# --------------------------------------------------------------------------


def act_load_jobs(ctx: LessonContext, memory: dict[str, Any]) -> str:
    """Put the retrieved postings into working memory."""
    jobs = list(ctx.jobs or [])
    memory["jobs"] = jobs
    if not jobs:
        return "0 postings available, so later turns will have nothing to rank."
    sources = sorted({job.source_label for job in jobs})
    return f"{len(jobs)} postings in memory, from {', '.join(sources)}."


def act_score_jobs(ctx: LessonContext, memory: dict[str, Any]) -> str:
    """Score every posting in memory and remember the best one."""
    jobs = memory.get("jobs") or []
    scored = [
        (
            score_job(
                job,
                ctx.resume,
                location=ctx.resume.location,
                work_mode="any",
            ),
            job,
        )
        for job in jobs
    ]
    # Ties break on job_id so two runs over the same data agree exactly.
    scored.sort(key=lambda pair: (-pair[0].total_score, pair[0].job_id))
    memory["matches"] = scored
    if not scored:
        return "Nothing to score."
    best_match, best_job = scored[0]
    memory["best_job"] = best_job
    memory["best_match"] = best_match
    return (
        f"Scored {len(scored)} postings. Best: {best_job.title} at "
        f"{best_job.company}, Demo Job Match Score {best_match.total_score}/100."
    )


def act_score_ats(ctx: LessonContext, memory: dict[str, Any]) -> str:
    """Run the ATS readiness rubric against the best-ranked job."""
    best_job = memory.get("best_job")
    if best_job is None:
        memory["ats"] = None
        return "No best job, so there is nothing to check the resume against."
    assessment = assess_ats(
        best_job,
        ctx.resume,
        threshold=ctx.settings.ats_recommendation_threshold,
    )
    memory["ats"] = assessment
    band = BAND_LABELS.get(assessment.band, assessment.band)
    return (
        f"Demo ATS Readiness Score {assessment.total_score}/100 - {band}; "
        f"{len(assessment.recommendations)} prioritized change(s) proposed."
    )


def act_stop(ctx: LessonContext, memory: dict[str, Any]) -> str:
    """End the loop and report honestly how it went."""
    calls = int(memory.get("tool_calls", 0))
    if goal_is_met(memory):
        return (
            f"Goal met after {calls} tool call(s). The loop exits here rather "
            "than calling another tool."
        )
    return (
        f"Stopping after {calls} tool call(s) without reaching the goal: a step "
        "above produced nothing, and repeating it would not change that."
    )


ACTIONS: dict[str, Callable[[LessonContext, dict[str, Any]], str]] = {
    "load_jobs": act_load_jobs,
    "score_jobs": act_score_jobs,
    "score_ats": act_score_ats,
    "stop": act_stop,
}


# --------------------------------------------------------------------------
# Observe (and repeat)
# --------------------------------------------------------------------------


def run_agent_loop(
    ctx: LessonContext, *, max_turns: int = MAX_TURNS
) -> list[LoopTurn]:
    """Run the bounded Reason -> Act -> Observe cycle and return its trace.

    Never raises: a failing tool becomes an observation, which is exactly how a
    real agent should treat one.
    """
    memory: dict[str, Any] = {"tool_calls": 0}
    trace: list[LoopTurn] = []

    for step in range(1, max_turns + 1):
        plan = decide_next_action(memory)
        action = ACTIONS[plan.action_key]
        try:
            observation = action(ctx, memory)
        except Exception as exc:  # noqa: BLE001 - a bad tool must not kill the loop
            observation = (
                f"The tool call failed ({type(exc).__name__}). The loop records "
                "the failure and carries on."
            )
            if plan.memory_key:
                memory.setdefault(plan.memory_key, None)
        if plan.action_key != "stop":
            memory["tool_calls"] = int(memory.get("tool_calls", 0)) + 1
        trace.append(
            LoopTurn(
                step=step,
                reason=plan.reason,
                action=plan.action_label,
                observation=observation,
                goal_reached=goal_is_met(memory),
            )
        )
        if plan.action_key == "stop":
            break

    return trace


# --------------------------------------------------------------------------
# Teaching copy
# --------------------------------------------------------------------------

THREE_LEVELS = """
**Loop** - the rows above. Reason about what is missing, act with one tool,
observe what came back, repeat. That is the entire idea, and it fits on a
screen.

**Harness** - everything wrapped around the loop so it is safe to run in front
of people: this Streamlit page, the Pydantic models that check every tool
result, `tools/claim_validator.py`, and the human approval gate. The loop
decides what to do next; the harness decides what the loop is allowed to do.

**Graph** - `agent/graph.py` is this same loop written down in advance. Rather
than a rule choosing an action each turn, LangGraph wires named nodes together:
`search_current_jobs` -> `normalize_jobs` -> `filter_and_deduplicate_jobs` ->
`score_jobs` -> ... -> `human_approval`. Fixed nodes and conditional edges
replace the free-running cycle, which is what makes an eight-minute live demo
predictable. The graph version is what powers the main tab of this app.
""".strip()

STOP_CONDITION_NOTE = """
The loop stopped for a reason it can state out loud - read the last row of the
table. `decide_next_action` returns `stop` as soon as no remaining action would
add anything new to working memory, and it says plainly whether it got there or
gave up. The loop is also bounded by `for step in range(1, MAX_TURNS + 1)`, so
it cannot spin forever even if every tool call fails.

An agent with no stop condition is the classic agent bug: it repeats the same
call, burns tokens and money, and never reports back.
""".strip()

NO_NARRATION_NOTE = """
The language model was unavailable or returned nothing usable, so the optional
narration is missing - and nothing else is. Every row in the table was produced
by Python calling this project's own tools, so the lesson is complete without
the model. No sentence on this screen was invented to paper over the gap.
""".strip()

NO_TRACE_NOTE = """
The loop produced no turns this run, so there was nothing for a model to
narrate and the step did not call one. An agent with nothing to report should
say so, not ask a model to fill the silence - a fluent sentence about work that
never happened is worse than a blank space.
""".strip()

TEACHING_CODE = '''\
def decide_next_action(memory: dict[str, Any]) -> Plan:
    """REASON: look at what is known, choose exactly one next action."""
    if "jobs" not in memory:
        return Plan("No candidate jobs yet.", "load_jobs", "ctx.jobs")
    if "matches" not in memory:
        return Plan("Jobs but no ranking.", "score_jobs", "job_scorer.score_job")
    if "ats" not in memory:
        return Plan("No resume fit yet.", "score_ats", "ats_scorer.assess_ats")
    # Say which one it is. An agent that always reports success is not honest.
    reason = "Goal met." if goal_is_met(memory) else "Nothing left to try."
    return Plan(reason, "stop", "stop")


memory: dict[str, Any] = {}
trace: list[LoopTurn] = []

for step in range(1, MAX_TURNS + 1):          # bounded, never "while True"
    plan = decide_next_action(memory)          # REASON
    observation = ACTIONS[plan.action_key](ctx, memory)   # ACT: a real tool
    trace.append(LoopTurn(step, plan.reason, plan.action_label, observation))
    if plan.action_key == "stop":              # OBSERVE, then the stop condition
        break
'''


def run(ctx: LessonContext) -> LessonResult:
    """Run the visible agent loop and explain how it scales up to a graph."""
    result = LessonResult()

    try:
        trace = run_agent_loop(ctx)
    except Exception as exc:  # noqa: BLE001 - belt and braces; the lab stays alive
        result.add(
            "warning",
            "The loop could not run",
            f"{type(exc).__name__}: {exc}",
        )
        trace = []

    if trace:
        result.add(
            "table",
            "One agent loop: Reason -> Act -> Observe",
            [turn.as_row() for turn in trace],
        )
        # Derived from the trace, never asserted: the screen must not claim the
        # goal was met on a run where it was not.
        if len(trace) >= MAX_TURNS:
            stop_help = (
                "The loop stopped because it ran out of turns. The bound ended "
                "it, not the goal."
            )
        elif trace[-1].goal_reached:
            stop_help = (
                "The loop stopped because the goal was met, not because it ran "
                "out of turns."
            )
        else:
            stop_help = (
                "The loop stopped because no remaining action could add "
                "anything. It did not reach its goal, and it did not run out "
                "of turns."
            )
        result.add(
            "metric",
            "Turns before the stop condition fired",
            {"value": f"{len(trace)} of {MAX_TURNS} allowed", "help": stop_help},
        )

    result.add("markdown", "The same idea at three sizes", THREE_LEVELS)
    result.add("note", "A loop needs a stop condition", STOP_CONDITION_NOTE)

    if not trace:
        # No trace means no work to describe. Asking anyway would invite the
        # model to invent one, which is precisely what this lesson warns about.
        result.add("note", "No model narration this run", NO_TRACE_NOTE)
        return result

    trace_text = "\n".join(
        f"{turn.step}. {turn.reason} -> {turn.action} -> {turn.observation}"
        for turn in trace
    )
    prompt = (
        "Below is a trace of an agent loop. In one sentence of plain English "
        "for a non-programmer, say what the agent did. Use only the trace; add "
        "nothing that is not in it.\n\n"
        f"{trace_text}"
    )
    try:
        text, available = ctx.llm_text(prompt, temperature=0.2)
    except Exception:  # noqa: BLE001 - a broken client must not delete the lesson
        text, available = None, False
    narration = (text or "").strip()
    if available and narration:
        result.used_llm = True
        result.add(
            "markdown",
            "Optional: the model narrates the trace it was shown",
            narration,
        )
    else:
        result.llm_unavailable = True
        result.add("note", "No model narration this run", NO_NARRATION_NOTE)

    return result


STEP = LessonStep(
    number=6,
    title="The agent loop: Reason -> Act -> Observe",
    subtitle="One bounded cycle over this project's real tools",
    concept=(
        "An agent does not produce one giant answer. It takes one turn at a "
        "time: work out what it still needs (reason), call a single tool to "
        "get it (act), then read what came back (observe). It repeats that "
        "cycle until the goal is met, and then it stops."
    ),
    why=(
        "Every agent framework you will hear about is this loop with extra "
        "machinery around it, so seeing the turns written out in plain Python "
        "makes the rest of the vocabulary easy to place. It also shows where "
        "agents go wrong: a loop is only as good as the rule that picks the "
        "next action, and only as safe as the condition that makes it stop."
    ),
    deck_reference="Slide 18 - Loop / Harness / Graph",
    code=TEACHING_CODE,
    run=run,
    takeaway=(
        "An agent is a loop that decides, acts, looks at what happened, and "
        "knows when to stop."
    ),
    needs_llm=False,
)
