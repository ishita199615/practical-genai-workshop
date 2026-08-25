"""Step 5 - Tools: deterministic Python, not model guesswork.

Deck slide 18 (the harness and its tools); SPECIFICATION.md section 15.

The lesson in one line: a language model is a poor calculator. Ask it to rate
the same resume against the same job twice and it can hand back two different
numbers, with no breakdown anyone can audit. So the agent does not ask. It
calls a plain Python function - :func:`tools.job_scorer.score_job` - which
returns the same number every time and shows exactly how it got there. The
model's job is to *explain* the number, never to produce it.

The step is offline-safe by design. The determinism demonstration is pure
Python, so it always runs. The model comparison is the optional half: when the
provider is rate limited or unconfigured, the step says so plainly instead of
inventing a "model answer".
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from lessons.base import BlockKind, LessonContext, LessonResult, LessonStep
from models.job import JobPosting
from models.match import MatchResult
from models.resume import ResumeProfile
from tools.job_scorer import score_job

# The 45/20/15/10/10 weighting from SPECIFICATION.md section 15, as
# (display name, MatchResult field, weight).
COMPONENT_WEIGHTS: tuple[tuple[str, str, float], ...] = (
    ("Required-skill coverage", "skill_score", 0.45),
    ("Resume/job text similarity", "similarity_score", 0.20),
    ("Role-title alignment", "role_score", 0.15),
    ("Experience alignment", "experience_score", 0.10),
    ("Location and work-mode fit", "preference_score", 0.10),
)

DEMO_LOCATION = "Houston, TX"
DEMO_WORK_MODE = "Any"

# How many times the same scoring call is repeated to prove it is stable.
REPEAT_RUNS = 3

# Keep the model prompt small so the optional half stays inside the demo's
# time budget on a workshop laptop.
RESUME_PROMPT_CHARS = 1500
JOB_PROMPT_CHARS = 2000

# A standalone 1-3 digit whole number. The lookarounds matter: without them
# "2026" reads as 202 then 6, and "0.85" reads as 0 - printing a number on the
# screen that the model never offered as its answer. A trailing sentence period
# is still fine, so "I'd say 64." reads as 64.
_NUMBER_RE = re.compile(r"(?<![\d.])\d{1,3}(?!\d|\.\d)")

TEACHING_CODE = '''from tools.job_scorer import score_job

# One function call. No model involved, no network, no randomness.
match = score_job(job, resume, location="Houston, TX", work_mode="Any")

match.skill_score        # 0-100, worth 45 points
match.similarity_score   # 0-100, worth 20 points
match.role_score         # 0-100, worth 15 points
match.experience_score   # 0-100, worth 10 points
match.preference_score   # 0-100, worth 10 points

total = round(
    0.45 * match.skill_score
    + 0.20 * match.similarity_score
    + 0.15 * match.role_score
    + 0.10 * match.experience_score
    + 0.10 * match.preference_score
)

# Score the same pair three times and keep only the distinct answers.
totals = {score_job(job, resume, location="Houston, TX",
                    work_mode="Any").total_score for _ in range(3)}
assert len(totals) == 1  # same input, same number, every time
'''


def reference_time(job: JobPosting) -> datetime:
    """Return the fixed clock used for scoring.

    Experience alignment compares resume dates against "now", so pinning the
    clock to the posting's retrieval time keeps repeated runs comparable. The
    clock is part of the input, not a hidden variable.
    """
    retrieved = getattr(job, "retrieved_at", None)
    if isinstance(retrieved, datetime):
        return retrieved
    return datetime.now(timezone.utc)


def component_rows(match: MatchResult) -> list[dict[str, object]]:
    """Return one table row per scoring component, with the points it earned."""
    rows: list[dict[str, object]] = []
    for label, field_name, weight in COMPONENT_WEIGHTS:
        raw = int(getattr(match, field_name, 0))
        rows.append(
            {
                "Component": label,
                "Weight": f"{round(weight * 100)} pts",
                "Score (0-100)": raw,
                "Points earned": round(raw * weight, 1),
            }
        )
    rows.append(
        {
            "Component": "Demo Job Match Score",
            "Weight": "100 pts",
            "Score (0-100)": match.total_score,
            "Points earned": float(match.total_score),
        }
    )
    return rows


def fingerprint(match: MatchResult) -> str:
    """Return a compact signature of every number the scorer produced.

    Comparing signatures rather than totals alone proves the whole breakdown
    repeated, not just the headline figure.
    """
    parts = [str(match.total_score)]
    parts.extend(
        str(getattr(match, field_name, 0)) for _, field_name, _ in COMPONENT_WEIGHTS
    )
    return "|".join(parts)


def build_model_score_prompt(resume: ResumeProfile, job: JobPosting) -> str:
    """Build the prompt that asks the model for the same 0-100 judgement."""
    resume_text = resume.as_plain_text()[:RESUME_PROMPT_CHARS]
    description = (job.description or "")[:JOB_PROMPT_CHARS]
    return (
        "Rate how well this candidate matches this job posting.\n"
        "Answer with a single whole number from 0 to 100 and nothing else.\n\n"
        "CANDIDATE RESUME\n"
        f"{resume_text}\n\n"
        "JOB POSTING\n"
        f"{job.title} at {job.company}\n"
        f"{description}\n"
    )


def parse_model_score(text: str | None) -> int | None:
    """Pull the first plausible 0-100 score out of a model reply."""
    for token in _NUMBER_RE.findall(text or ""):
        value = int(token)
        if 0 <= value <= 100:
            return value
    return None


def describe_model_answers(
    answers: list[int], replies: int = 0
) -> tuple[BlockKind, str]:
    """Return an honest (block kind, sentence) for the model's replies.

    ``answers`` holds the replies a number could be read from; ``replies`` is
    how many replies arrived at all. The two differ when the model answered but
    said something like "it depends", and the sentence must not round that off
    into "only one answer came back".

    Deliberately fair: if the model happened to repeat itself, the step says so
    rather than pretending the point was proved.
    """
    replies = max(replies, len(answers))
    if not answers:
        if replies >= 2:
            return (
                "note",
                "The model answered twice, but neither reply contained a "
                "number that could be read, so there was nothing to compare. "
                "Both replies are printed above exactly as they arrived. The "
                "Python score is unaffected.",
            )
        return (
            "note",
            "No readable number came back from the model, so there was nothing "
            "to compare. The Python score above is unaffected.",
        )
    if len(answers) < 2:
        return (
            "note",
            "Only one usable model answer came back, so the two replies could "
            "not be compared this time. The Python score above is unaffected.",
        )
    if answers[0] == answers[1]:
        return (
            "note",
            f"This time the model returned {answers[0]} twice. That is a fair "
            "result and worth saying out loud - it is not guaranteed. Nothing "
            "in the model promises the same answer on the next run, and it "
            "still gave no breakdown anyone could check or argue with.",
        )
    return (
        "warning",
        f"Same prompt, same settings, two different answers: {answers[0]} and "
        f"{answers[1]} - a gap of {abs(answers[0] - answers[1])} points. "
        "Neither came with a breakdown that could be audited.",
    )


def _add_model_comparison(
    ctx: LessonContext,
    result: LessonResult,
    *,
    resume: ResumeProfile,
    job: JobPosting,
    python_totals: list[int],
    python_fingerprint: str,
) -> None:
    """Ask the model for the same score twice and compare, when it is reachable.

    Sets ``used_llm`` and ``llm_unavailable`` on ``result`` to match what
    actually happened. Never raises.
    """
    prompt = build_model_score_prompt(resume, job)
    replies: list[str] = []
    answers: list[int] = []
    reached = False

    for _ in range(2):
        text, available = ctx.llm_text(prompt, temperature=0.2)
        if not available:
            break
        reached = True
        cleaned = str(text or "").strip()
        replies.append(cleaned)
        parsed = parse_model_score(cleaned)
        if parsed is not None:
            answers.append(parsed)

    if not reached:
        result.llm_unavailable = True
        result.used_llm = False
        result.add(
            "note",
            "The model half was skipped",
            "The language model could not be reached, so the optional "
            "comparison did not run. Everything above is the deterministic "
            "Python path, which is the part the agent actually depends on - "
            "and that is precisely why the score lives in code rather than in "
            "a prompt.",
        )
        return

    result.used_llm = True

    right_lines: list[str] = []
    for index, reply in enumerate(replies, start=1):
        parsed = parse_model_score(reply)
        if parsed is not None:
            shown = str(parsed)
        elif reply:
            shown = f"no number found: {reply[:60]}"
        else:
            shown = "no number found: the reply was empty"
        right_lines.append(f"Ask {index}: {shown}")
    if len(replies) < 2:
        right_lines.append("Ask 2: the model did not answer a second time.")
    right_lines.append("No component breakdown was returned.")

    # Print what each run actually returned. Repeating one number REPEAT_RUNS
    # times would be inventing evidence for the very claim under test.
    left_lines = [
        f"Run {index}: {total}" for index, total in enumerate(python_totals, start=1)
    ]
    left_lines.append(f"Fingerprint: {python_fingerprint}")
    left_lines.append("Full breakdown returned every time.")

    model_name = str(getattr(ctx.llm, "model_name", "") or "the model")
    result.add(
        "compare",
        "Python (repeatable) vs the model (varies)",
        {
            "left_label": "Python - tools/job_scorer.py",
            "left": "\n".join(left_lines),
            "right_label": f"Model - {model_name}",
            "right": "\n".join(right_lines),
        },
    )

    kind, sentence = describe_model_answers(answers, len(replies))
    result.add(kind, "What the two model answers show", sentence)


def run(ctx: LessonContext) -> LessonResult:
    """Score one real posting in Python three times, then compare with the model.

    Produces the weighted component breakdown, proof that repeated runs are
    identical, and - only when the provider is reachable - the model's own two
    attempts at the same judgement.
    """
    result = LessonResult()

    jobs = list(ctx.jobs or [])
    if not jobs:
        result.add(
            "warning",
            "No job posting was available",
            "This step scores a retrieved posting and none were loaded. Run "
            "the search on the Demo tab, or check that the cached "
            "demonstration data is present.",
        )
        return result

    job = jobs[0]
    resume = ctx.resume
    now = reference_time(job)

    try:
        matches = [
            score_job(
                job,
                resume,
                location=DEMO_LOCATION,
                work_mode=DEMO_WORK_MODE,
                now=now,
            )
            for _ in range(REPEAT_RUNS)
        ]
    except Exception as exc:  # noqa: BLE001 - a lesson must never crash the app
        result.add(
            "warning",
            "The scorer could not run",
            f"{type(exc).__name__}: {exc}",
        )
        return result

    first = matches[0]
    fingerprints = [fingerprint(match) for match in matches]

    result.add(
        "markdown",
        "The input",
        f"**Resume:** {resume.name} (fictional)  \n"
        f"**Job:** {job.title} at {job.company}  \n"
        f"**Preferences:** {DEMO_LOCATION}, work mode {DEMO_WORK_MODE}\n\n"
        "Both go into one Python function. No language model is called to "
        "produce any number on this screen.",
    )

    result.add(
        "table",
        "Demo Job Match Score - component breakdown (SPECIFICATION.md section 15)",
        component_rows(first),
    )

    weighted = sum(
        getattr(first, field_name, 0) * weight
        for _, field_name, weight in COMPONENT_WEIGHTS
    )
    result.add(
        "metric",
        "Demo Job Match Score",
        {
            "value": f"{first.total_score} / 100",
            "help": (
                f"The weighted points add up to {weighted:g}, rounded to "
                f"{first.total_score}. Calculated in tools/job_scorer.py - the "
                "model cannot change it."
            ),
        },
    )

    result.add(
        "json",
        "The evidence behind the skill component",
        {
            "matched_skills": first.matched_skills,
            "missing_skills": first.missing_skills,
            "concerns": first.concerns,
        },
    )

    result.add(
        "table",
        f"The same call, run {REPEAT_RUNS} times",
        [
            {
                "Run": index,
                "Total": match.total_score,
                "Fingerprint (total|skill|similarity|role|experience|preference)": value,
            }
            for index, (match, value) in enumerate(zip(matches, fingerprints), start=1)
        ],
    )

    if len(set(fingerprints)) == 1:
        result.add(
            "success",
            "Byte-identical every run",
            f"All {REPEAT_RUNS} runs returned the same signature "
            f"`{fingerprints[0]}` - not merely the same total, but the same "
            "five components. Same input, same number, every time. That is "
            "what makes the score auditable: anyone can re-run it and "
            "challenge a single component.",
        )
    else:
        result.add(
            "warning",
            "The runs did not match",
            "The repeated runs returned different signatures: "
            + ", ".join(sorted(set(fingerprints)))
            + ". That is a bug in the demo, not a teaching point - the scorer "
            "is supposed to be deterministic.",
        )

    try:
        _add_model_comparison(
            ctx,
            result,
            resume=resume,
            job=job,
            python_totals=[match.total_score for match in matches],
            python_fingerprint=fingerprints[0],
        )
    except Exception as exc:  # noqa: BLE001 - the optional half must never
        # take the lesson down with it. The deterministic blocks above are the
        # teaching point; a misbehaving provider may cost us the comparison,
        # never the lesson.
        if not result.used_llm:
            result.llm_unavailable = True
        result.add(
            "note",
            "The model half could not run",
            f"The optional comparison failed ({type(exc).__name__}), so it was "
            "left out rather than filled in with a made-up answer. Everything "
            "above is the deterministic Python path, which is the part the "
            "agent actually depends on.",
        )

    return result


STEP = LessonStep(
    number=5,
    title="Tools: deterministic Python, not model guesswork",
    subtitle="The agent calculates the score in code, then lets the model explain it",
    concept=(
        "An agent is a language model plus tools. A tool is ordinary code the "
        "agent calls whenever it needs an exact answer. Here the tool is a "
        "Python function that scores a resume against a job on a published "
        "100-point scale: 45 for skills, 20 for text similarity, 15 for the "
        "job title, 10 for experience, and 10 for location and work mode."
    ),
    why=(
        "Language models are fluent but inconsistent at arithmetic and ranking. "
        "Ask one to rate the same resume twice and you can get two different "
        "numbers, with no breakdown you can check. Putting the number in code "
        "makes it repeatable, explainable, and arguable, and leaves the model "
        "to do what it is genuinely good at: putting the result into words."
    ),
    deck_reference="Slide 18 - the harness and its tools; SPECIFICATION.md section 15",
    code=TEACHING_CODE,
    run=run,
    takeaway=(
        "If the answer has to be right every time, write it in code and let the "
        "model explain it."
    ),
    needs_llm=False,
)
