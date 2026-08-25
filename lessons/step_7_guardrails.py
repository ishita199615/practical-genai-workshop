"""Step 7 - Guardrails: the agent refuses to lie, and a human approves.

The previous steps let the agent *write* something. This one shows what stops
it from writing something untrue.

Two drafts of the same application are built here. The first one lies: it
claims Power BI, a tool the fictional master resume cannot evidence. The second
one is truthful. Both are handed to the *same* validator the live demo uses -
:func:`tools.claim_validator.run_deterministic_checks` - and the validator, not
a prompt and not good intentions, is what catches the lie.

Nothing in this step needs a language model. That is the point: validation and
human permission are ordinary Python and an ordinary button, and they keep
working when the model is rate limited, offline, or simply wrong.
"""

from __future__ import annotations

from typing import Any

from lessons.base import LessonContext, LessonResult, LessonStep
from models.application import RevisedBullet, TailoredApplication
from models.job import JobPosting
from models.resume import ResumeProfile
from tools.ats_scorer import ResumeSnapshot, find_evidence, snapshot_from_resume
from tools.claim_validator import (
    extract_claims,
    review_claims,
    run_deterministic_checks,
)
from tools.job_scorer import canonical_skill, canonical_skill_set, skills_from_text

#: The requirement the demo deliberately cannot support. The fictional resume
#: lists Tableau, never Power BI, so every honest draft must leave it out.
UNSUPPORTED_REQUIREMENT = "Power BI"

_FALLBACK_ROLE = "Data Analyst Intern"

#: Words that turn a job title into a claim about who employed the candidate,
#: e.g. "Data Analyst at Globex". Quoting one would invent an employer.
_EMPLOYER_CONNECTORS = ("at", "with", "for", "@")

#: Credential words the fictional resume cannot back beyond its own degree.
_CREDENTIAL_WORDS = (
    "bachelor",
    "master",
    "mba",
    "phd",
    "ph.d",
    "doctorate",
    "associate",
    "certified",
    "certification",
)

#: The bullet the dishonest draft overwrites. Swapping a tool the resume proves
#: (Tableau) for one it does not is the clearest version of the mistake.
_LIE_ANCHOR = "tableau"

CODE_SNIPPET = '''\
from tools.claim_validator import run_deterministic_checks

# A draft that lies: the master resume lists Tableau, never Power BI.
dishonest = TailoredApplication(
    job_id=job.job_id,
    revised_summary="Information systems student and Power BI developer.",
    revised_bullets=[RevisedBullet(
        source_bullet_id="experience_1_bullet_2",
        original_text="Created Tableau dashboards for weekly reporting.",
        revised_text="Built Power BI dashboards that cut reporting time by 40%.",
    )],
    reordered_skills=["Power BI", "Tableau", "SQL"],
    cover_letter="I have delivered Power BI reporting for large teams.",
)

# The same validator the live demo runs. Plain Python, no model involved.
checks = run_deterministic_checks(dishonest, resume, unsupported_gaps=["Power BI"])
for check in checks:
    if not check["passed"]:
        print("BLOCKED:", check["name"], "-", check["detail"])

# BLOCKED: No invented metrics - Unsupported number(s): 40.
# BLOCKED: Skills remain grounded in the master resume - ... Power BI.
# BLOCKED: Unsupported job gaps were not added - ... Power BI.
# BLOCKED: Cover letter claims no unsupported skill - ... power bi.
'''


def unproven_skills(text: str, resume: ResumeProfile, snapshot: ResumeSnapshot) -> list[str]:
    """Return skills named in ``text`` that the master resume cannot back up.

    Mirrors the rule the validator applies: a skill counts as proven when it is
    listed on the resume outright, or when a bullet evidences the concept (a
    Tableau dashboard proves "data visualization"). Anything else is a claim the
    agent is not allowed to make.
    """
    proven = canonical_skill_set(list(resume.skills))
    unproven: list[str] = []
    for name in skills_from_text(text or ""):
        canonical = canonical_skill(name)
        if canonical in proven or find_evidence(canonical, snapshot):
            continue
        unproven.append(name)
    return unproven


def quotable(text: str | None, resume: ResumeProfile, snapshot: ResumeSnapshot) -> bool:
    """True when retrieved text is safe to repeat inside an honest draft.

    Live job text is untrusted input, and every rule here mirrors a check the
    validator will run afterwards. A title carrying a number would import a
    metric the resume never proved; a title naming an unproven tool would
    smuggle the very claim this step refuses to make; a title like "Data Analyst
    at Globex" would name an employer the candidate never worked for; and a
    title mentioning a degree would claim a credential the resume lacks.
    """
    value = (text or "").strip()
    if not value or any(character.isdigit() for character in value):
        return False
    lowered = f" {value.lower()} "
    if any(f" {word} " in lowered for word in _EMPLOYER_CONNECTORS):
        return False
    if any(word in lowered for word in _CREDENTIAL_WORDS):
        return False
    return not unproven_skills(value, resume, snapshot)


def pick_job(jobs: list[JobPosting]) -> JobPosting | None:
    """Choose the posting to demonstrate against, preferring a Power BI ad."""
    if not jobs:
        return None
    needle = UNSUPPORTED_REQUIREMENT.lower()
    for job in jobs:
        haystack = " ".join([job.title, job.description, *job.required_skills]).lower()
        if needle in haystack:
            return job
    return jobs[0]


def build_dishonest_draft(job_id: str, resume: ResumeProfile) -> TailoredApplication:
    """Build a draft that overstates the candidate in four separate ways.

    Deliberately realistic: this is what an unguarded "make my resume match the
    job" prompt produces when nothing checks its output.
    """
    bullet_id, original = _first_bullet(resume)
    return TailoredApplication(
        job_id=job_id,
        revised_summary=(
            f"Information systems student and {UNSUPPORTED_REQUIREMENT} developer "
            "with two years of experience building executive dashboards."
        ),
        revised_bullets=[
            RevisedBullet(
                source_bullet_id=bullet_id,
                original_text=original,
                revised_text=(
                    f"Built {UNSUPPORTED_REQUIREMENT} dashboards that cut weekly "
                    "reporting time by 40%."
                ),
            )
        ],
        reordered_skills=[UNSUPPORTED_REQUIREMENT, *resume.skills],
        keywords_used=[UNSUPPORTED_REQUIREMENT],
        cover_letter=(
            f"I am a certified {UNSUPPORTED_REQUIREMENT} professional and have "
            f"delivered {UNSUPPORTED_REQUIREMENT} reporting for large teams."
        ),
        unsupported_ats_gaps_not_applied=[UNSUPPORTED_REQUIREMENT],
    )


def build_honest_draft(
    job: JobPosting | None,
    resume: ResumeProfile,
    snapshot: ResumeSnapshot,
) -> TailoredApplication:
    """Build a draft assembled only from facts already in the master resume.

    Every sentence is composed from resume text, resume skills, and - when it
    passes :func:`quotable` - the role name from the posting. Nothing else is
    allowed in, which is why it clears all ten checks.

    The draft is then run past the validator before it is returned. If quoting
    the posting introduced anything the resume cannot prove, the draft is rebuilt
    using only the resume's own target role. A word list can be outrun by a job
    title nobody anticipated; asking the validator cannot.
    """
    drafted = _compose_honest_draft(
        job, resume, _safe_role_label(job, resume, snapshot)
    )
    if _clears_every_check(drafted, resume):
        return drafted
    return _compose_honest_draft(job, resume, _resume_role_label(resume))


def _compose_honest_draft(
    job: JobPosting | None, resume: ResumeProfile, role_label: str
) -> TailoredApplication:
    """Assemble the honest draft around one already-vetted role label."""
    ordered_skills = _reorder_skills(job, resume)
    summary = (
        f"{role_label} candidate. {resume.professional_summary.rstrip('.')}, "
        f"working with {', '.join(ordered_skills[:4])}."
    )
    return TailoredApplication(
        job_id=job.job_id if job else "demo_job",
        revised_summary=summary,
        revised_bullets=_honest_bullets(resume),
        reordered_skills=ordered_skills,
        keywords_used=[skill for skill in ordered_skills[:3]],
        cover_letter=_honest_cover_letter(role_label, resume),
        missing_requirements=[UNSUPPORTED_REQUIREMENT],
        unsupported_ats_gaps_not_applied=[UNSUPPORTED_REQUIREMENT],
    )


def _clears_every_check(
    application: TailoredApplication, resume: ResumeProfile
) -> bool:
    """True when a draft passes every check without needing any allowance.

    Deliberately stricter than the run: no company is whitelisted here, so a
    draft that clears this also clears the checks shown on screen.
    """
    try:
        checks = run_deterministic_checks(
            application, resume, unsupported_gaps=[UNSUPPORTED_REQUIREMENT]
        )
    except Exception:  # noqa: BLE001 - a lesson must never crash the app
        return False
    return bool(checks) and all(check.get("passed") for check in checks)


def checks_table(checks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Turn the validator's raw check records into rows the UI can draw."""
    return [
        {
            "Check": str(check.get("name", "")),
            "Result": "PASS" if check.get("passed") else "BLOCKED",
            "What the validator saw": str(check.get("detail", "")),
        }
        for check in checks
    ]


def power_bi_verdict(resume: ResumeProfile, snapshot: ResumeSnapshot) -> str:
    """Render the exact learning-gap statement SPECIFICATION.md section 19 requires."""
    evidence = find_evidence(canonical_skill(UNSUPPORTED_REQUIREMENT), snapshot)
    found = ", ".join(evidence) if evidence else "None"
    return (
        f"Job requirement: {UNSUPPORTED_REQUIREMENT}\n"
        f"Evidence in master resume: {found}\n"
        f"Decision: {UNSUPPORTED_REQUIREMENT} was not added to the resume\n"
        f"Recommendation: Treat it as a learning gap"
    )


def _first_bullet(resume: ResumeProfile) -> tuple[str, str]:
    """Return a real bullet ID and its exact text, so only the lie is on trial.

    Prefers the bullet naming the tool the lie will replace, so the swap on
    screen reads as the substitution it is rather than a random rewrite.
    """
    index = list(resume.bullet_index().items())
    if not index:
        return ("experience_1_bullet_1", "")
    for bullet_id, text in index:
        if _LIE_ANCHOR in text.lower():
            return bullet_id, text
    return index[0]


def _honest_bullets(resume: ResumeProfile) -> list[RevisedBullet]:
    """Rewrite exactly two real bullets without adding a single new fact."""
    clauses = (
        " to support recurring analysis and reporting.",
        " and data visualization.",
    )
    bullets: list[RevisedBullet] = []
    for position, (bullet_id, text) in enumerate(list(resume.bullet_index().items())[:2]):
        bullets.append(
            RevisedBullet(
                source_bullet_id=bullet_id,
                original_text=text,
                revised_text=f"{text.rstrip('.')}{clauses[position]}",
            )
        )
    return bullets


def _reorder_skills(job: JobPosting | None, resume: ResumeProfile) -> list[str]:
    """Reorder - never extend - the resume's own skills to lead with matches."""
    required = canonical_skill_set(list(job.required_skills)) if job else set()
    return sorted(resume.skills, key=lambda skill: canonical_skill(skill) not in required)


def _safe_role_label(
    job: JobPosting | None, resume: ResumeProfile, snapshot: ResumeSnapshot
) -> str:
    """Use the posting's own role name when quoting it introduces no new claim."""
    if job and quotable(job.title, resume, snapshot):
        return job.title.strip()
    return _resume_role_label(resume)


def _resume_role_label(resume: ResumeProfile) -> str:
    """A role name taken from the resume itself, so it can claim nothing new."""
    for role in resume.target_roles:
        if role and role.strip():
            return role.strip()
    return _FALLBACK_ROLE


def _honest_cover_letter(role_label: str, resume: ResumeProfile) -> str:
    """Compose a short letter whose every sentence traces to the resume."""
    return (
        f"Dear Hiring Team,\n\n"
        f"I am applying for the {role_label} role. As an information systems "
        "student, I analyze survey data with Python and Excel, and I build "
        "Tableau dashboards that keep weekly reporting on schedule. A campus "
        "project gave me practice using SQL and Pandas to clean and summarize "
        "event attendance data, which is close to the recurring analysis your "
        "team describes. I am comfortable moving between querying, cleaning, "
        "and presenting results, and I like work where the reporting has to be "
        "right every week. I would welcome the chance to discuss how this "
        f"experience fits your needs.\n\nThank you for your time,\n{resume.name}"
    )


def _draft_preview(application: TailoredApplication) -> dict[str, Any]:
    """Show the three fields the validator reads, not the whole object."""
    return {
        "revised_summary": application.revised_summary,
        "revised_bullets": [
            {
                "source_bullet_id": bullet.source_bullet_id,
                "revised_text": bullet.revised_text,
            }
            for bullet in application.revised_bullets
        ],
        "reordered_skills": application.reordered_skills,
        "cover_letter": application.cover_letter,
    }


APPROVAL_NOTE = """\
### The second guardrail: a human holds the pen

Validation catches untruths. It does not decide whether this application should
exist. That decision is not the agent's to make, so the graph **stops** and
waits:

| Choice | What the agent does next |
| --- | --- |
| **Approve** | Writes `output/application_package_<timestamp>.md` and `.json`. Nothing else. |
| **Request Changes** | Takes the reviewer's note, redrafts, re-validates, and pauses again. |
| **Reject** | Stops. No package is written, and the review screen stays on screen. |

Export is the **only** side effect in this demo, and it happens only after
Approve. There is no submit button, no email, no login, no form fill. The agent
cannot apply for a job on anyone's behalf - not because it was asked nicely,
but because the code to do it was never written.

**No application is ever submitted.**
"""


def run(ctx: LessonContext) -> LessonResult:
    """Show the validator blocking a dishonest draft and clearing an honest one."""
    result = LessonResult()
    resume = ctx.resume
    snapshot = snapshot_from_resume(resume)
    job = pick_job(list(ctx.jobs or []))
    company = job.company if job and quotable(job.company, resume, snapshot) else None
    job_id = job.job_id if job else "demo_job"

    result.add(
        "markdown",
        "The experiment",
        "Two drafts of the same application"
        + (f" for **{job.title}**" + (f" at **{company}**" if company else "") if job else "")
        + ". One lies about Power BI; one does not. Both go through the *same* "
        "Python validator the live demo runs. Notice that the model is not "
        "asked to police itself.",
    )

    # --- Draft A: the lie -------------------------------------------------
    dishonest = build_dishonest_draft(job_id, resume)
    result.add("json", "Draft A - written to impress", _draft_preview(dishonest))

    checks_a = _safe_checks(dishonest, resume, company)
    failed = [check for check in checks_a if not check.get("passed")]
    result.add("table", f"Draft A - all {len(checks_a)} deterministic checks", checks_table(checks_a))
    result.add(
        "warning",
        f"Guardrail fired: {len(failed)} of {len(checks_a)} checks blocked this draft",
        "\n\n".join(f"**{check['name']}** - {check['detail']}" for check in failed)
        or "No check failed, which would mean the guardrail needs repair.",
    )

    reviews = _safe_reviews(dishonest, resume)
    if reviews:
        result.add(
            "table",
            "Second layer - claim-by-claim review (deterministic reviewer)",
            [
                {
                    "Claim": review.claim[:110],
                    "Status": review.status,
                    "Why": review.reason,
                }
                for review in reviews
            ],
        )

    # --- Draft B: the truth -----------------------------------------------
    honest = build_honest_draft(job, resume, snapshot)
    result.add("json", "Draft B - written to be true", _draft_preview(honest))

    checks_b = _safe_checks(honest, resume, company)
    passed_b = [check for check in checks_b if check.get("passed")]
    result.add("table", f"Draft B - the same {len(checks_b)} checks", checks_table(checks_b))
    result.add(
        "success" if len(passed_b) == len(checks_b) else "warning",
        f"Draft B passed {len(passed_b)} of {len(checks_b)} checks",
        "Same validator, same resume, different draft. The guardrail is not "
        "hostile to the agent - it is indifferent to it. It only asks whether "
        "each claim traces back to the master resume."
        if len(passed_b) == len(checks_b)
        else "\n\n".join(
            f"**{check['name']}** - {check['detail']}"
            for check in checks_b
            if not check.get("passed")
        ),
    )

    result.add(
        "compare",
        "The one sentence that decides it",
        {
            "left_label": "Draft A (blocked)",
            "left": dishonest.revised_summary,
            "right_label": "Draft B (cleared)",
            "right": honest.revised_summary,
        },
    )

    result.add(
        "code",
        "What the demo shows on screen when a requirement cannot be supported",
        power_bi_verdict(resume, snapshot),
    )

    result.add("markdown", "Then the agent stops and asks", APPROVAL_NOTE)

    _add_model_gloss(ctx, result, len(failed))
    return result


def _safe_checks(
    application: TailoredApplication, resume: ResumeProfile, company: str | None
) -> list[dict[str, Any]]:
    """Run the real validator, degrading to a visible record on failure."""
    try:
        return run_deterministic_checks(
            application,
            resume,
            unsupported_gaps=[UNSUPPORTED_REQUIREMENT],
            allowed_organizations=[company] if company else [],
        )
    except Exception as exc:  # noqa: BLE001 - a lesson must never crash the app
        return [
            {
                "name": "Validator could not run",
                "passed": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        ]


def _safe_reviews(application: TailoredApplication, resume: ResumeProfile) -> list[Any]:
    """Run the claim reviewer's offline path so the result is the same in every room."""
    try:
        return review_claims(extract_claims(application), resume, llm=None)
    except Exception:  # noqa: BLE001 - the deterministic checks are the point
        return []


def _add_model_gloss(ctx: LessonContext, result: LessonResult, failed_count: int) -> None:
    """Optionally let the model restate the lesson, and say plainly when it did not."""
    prompt = (
        "In two sentences, for an audience of university staff new to AI agents, "
        f"explain why a Python validator that blocked {failed_count} claim(s) in a "
        "resume draft is a better safeguard than instructing the language model "
        "to be honest. Plain English, no jargon, no lists."
    )
    text, available = ctx.llm_text(prompt, temperature=0.3)
    # A reply of nothing but whitespace is not a reply. Rendering it would put an
    # empty section under a heading promising the model's words, and would let the
    # page caption claim a call succeeded when it returned nothing.
    gloss = (text or "").strip()
    if available and gloss:
        result.used_llm = True
        result.add("markdown", "The model's own take on the guardrail", gloss)
        return

    result.used_llm = False
    result.llm_unavailable = True
    result.add(
        "note",
        "No model was called - and nothing above is missing",
        "The optional closing commentary is the only part of this step that "
        "wanted a language model, and it was unavailable. Every draft, every "
        "check, and every verdict above was produced by Python. That is exactly "
        "the property a guardrail needs: it does not go offline when the model "
        "does.",
    )


STEP = LessonStep(
    number=7,
    title="Guardrails: the agent refuses to lie, and a human approves",
    subtitle="Validation and permission are parts of the agent, not extras",
    concept=(
        "The agent writes a draft, and then a separate piece of plain Python "
        "checks every claim in it against the master resume. Anything the "
        "resume cannot prove is blocked before a person ever sees it. After the "
        "draft passes, the agent stops and waits for a human to approve, "
        "request changes, or reject."
    ),
    why=(
        "A language model asked to make a resume match a job will happily add a "
        "skill the candidate does not have, and it will sound confident doing "
        "it. Telling the model to be honest is a request; a validator is a "
        "rule. Pairing the rule with a human approval gate is what makes an "
        "agent safe enough to put in front of students."
    ),
    deck_reference="Slide 14 - UHS guardrails (SPECIFICATION.md sections 19 and 20)",
    code=CODE_SNIPPET,
    run=run,
    takeaway=(
        "An agent you can trust is one that can be told no - by a validator it "
        "cannot argue with, and by a human it cannot skip."
    ),
    needs_llm=False,
)
