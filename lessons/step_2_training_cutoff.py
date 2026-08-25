"""Step 2 - why the model cannot know today's jobs.

The deck's training-cutoff slide says a language model's knowledge is frozen at
the moment training stopped. This step makes that concrete: it asks the model a
question no amount of memory can answer honestly - "which jobs were posted in
Houston in the last 24 hours?" - and then puts the answer next to the postings
this app actually retrieved, each one carrying a source URL and a retrieval
timestamp.

The step is deliberately honest about the reply it gets. A well-behaved model
refuses the question, and the step says so rather than pretending it caught a
hallucination. When the model cannot be reached at all, the step labels a canned
illustration as canned and still shows the real retrieved-jobs table, because a
lab full of students cannot wait on a rate limit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from lessons.base import BlockKind, LessonContext, LessonResult, LessonStep

# The question the model is asked. It is unanswerable from memory by design:
# the answer changes every hour and lives on pages the model never saw.
MEMORY_PROMPT = (
    "List three Data Analyst Intern jobs posted in Houston, TX within the last "
    "24 hours. For each one give the job title, the company, and a working URL "
    "to the posting."
)

# Phrases that only ever appear when a model names its own cutoff.
CUTOFF_MARKERS: tuple[str, ...] = (
    "knowledge cutoff",
    "knowledge cut-off",
    "training cutoff",
    "training data",
    "my knowledge is",
    "my knowledge ends",
    "my knowledge only",
    "beyond my knowledge",
    "outside my knowledge",
    "as of my last",
    "as of my knowledge",
    "would be fabricated",
    "would be made up",
    "i made them up",
    "i would be inventing",
    "i would have to invent",
)

# A refusal is the model saying it *lacks* a capability, so both halves have to
# be present: an inability phrase and, close behind it, the thing it cannot do.
# Matching either half alone is what makes this detector lie. "browse" appears
# in "I can't browse the web" (a refusal) and in "Browse these three openings"
# (a confident invention); "up to date" appears in "I don't have up-to-date
# listings" and in "here are three up-to-date postings". Requiring the pair
# keeps the step from announcing a refusal that never happened.
_REFUSAL_PATTERN = re.compile(
    r"(?:i (?:do not|don't) have|i cannot|i can't|i lack|"
    r"(?:i (?:am|'m) )?(?:unable|not able) to|"
    r"no (?:access|ability|way) to|without access to)"
    r".{0,60}?"
    r"(?:access|brows|internet|the web|search the web|real[- ]time|"
    r"live (?:data|job|listing|posting)|up[- ]to[- ]date|"
    r"current (?:job|listing|posting|opening)|verify)",
    re.IGNORECASE | re.DOTALL,
)

_URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]]+")

# Markdown decoration a model wraps around a link: **url**, `url`, _url_.
_URL_TRAILING_NOISE = ".,;:!?*_~`"

# A written-by-us illustration of the answer this question usually attracts.
# The hostnames use the IANA-reserved ``.example`` top-level domain so nothing
# here can resolve to, or be mistaken for, a real employer's careers page.
CANNED_HALLUCINATION = """Here are three Data Analyst Intern roles posted in Houston in the last 24 hours:

1. Data Analyst Intern - Summit Data Partners
   https://careers.summitdatapartners.example/jobs/data-analyst-intern-8827134
2. Junior Data Analyst - Magnolia Health Systems
   https://jobs.magnoliahealth.example/openings/junior-data-analyst
3. Business Intelligence Intern - Pinewood Freight Group
   https://pinewoodfreight.example/careers/bi-intern-summer-2026

All three are accepting applications now."""

ReplyKind = Literal["refused", "hedged_but_listed", "listed_links", "no_links"]

MAX_REPLY_CHARS = 1600


@dataclass(frozen=True)
class ReplyAssessment:
    """What the model's reply actually did, stated without exaggeration."""

    kind: ReplyKind
    urls: list[str]
    headline: str
    explanation: str
    block_kind: BlockKind


def find_urls(text: str) -> list[str]:
    """Return the distinct URLs in ``text``, in the order they appear.

    Trailing sentence punctuation and Markdown decoration are stripped, so
    ``...intern.`` and ``**https://...**`` both yield a clean link.
    """
    seen: list[str] = []
    for match in _URL_PATTERN.findall(text or ""):
        url = match.rstrip(_URL_TRAILING_NOISE)
        if url and url not in seen:
            seen.append(url)
    return seen


def looks_like_refusal(text: str) -> bool:
    """True when the reply explicitly says the model cannot know this.

    Deliberately strict. A false positive here would put the words "the model
    refused" on the screen when the model actually answered with confidence,
    which is the exact mistake this step exists to warn against.
    """
    lowered = (text or "").lower()
    if any(marker in lowered for marker in CUTOFF_MARKERS):
        return True
    return _REFUSAL_PATTERN.search(lowered) is not None


def describe_reply(text: str) -> ReplyAssessment:
    """Classify what the model did, so the step reports facts, not a script.

    Four outcomes are possible, and each gets its own honest wording: a clean
    refusal, a disclaimer followed by links anyway, links with no disclaimer,
    and an answer with no links at all.
    """
    urls = find_urls(text)
    hedged = looks_like_refusal(text)
    count = len(urls)

    if urls and hedged:
        return ReplyAssessment(
            kind="hedged_but_listed",
            urls=urls,
            headline=(
                f"The model said it cannot know - and then listed {count} link(s) anyway."
            ),
            explanation=(
                "This is the most dangerous shape of answer, because the warning "
                "makes the list look supervised. No page was fetched. Every link "
                "was assembled from patterns in the training data, so it may lead "
                "nowhere, to a filled role, or to a posting that never existed. "
                "Nothing in the reply carries a retrieval time you could check."
            ),
            block_kind="warning",
        )

    if urls:
        return ReplyAssessment(
            kind="listed_links",
            urls=urls,
            headline=f"The model answered confidently with {count} link(s).",
            explanation=(
                "The model has no connection to the web and no clock. These links "
                "were produced from memory of what job URLs look like, not from a "
                "page anyone loaded, so they are unverifiable and may be invented. "
                "A truthful answer here is a refusal plus an offer to search."
            ),
            block_kind="warning",
        )

    if hedged:
        return ReplyAssessment(
            kind="refused",
            urls=urls,
            headline="The model refused, which is the correct answer.",
            explanation=(
                "Good behaviour: it recognised that today's postings are outside "
                "its training data and declined to invent any. Note that a correct "
                "refusal is still not a job list - the student ends up with "
                "nothing. Closing that gap is what retrieval is for."
            ),
            block_kind="success",
        )

    return ReplyAssessment(
        kind="no_links",
        urls=urls,
        headline="The model answered with no links and no explicit refusal.",
        explanation=(
            "Usually this is generic advice - where to look, what to search for - "
            "rather than the three current postings that were requested. It may "
            "also be a refusal worded in a way this step does not recognise; read "
            "the reply above and judge it yourself. Either way the student is left "
            "with no title, no employer, and nothing dated they could open."
        ),
        block_kind="warning",
    )


def _format_timestamp(value: Any) -> str:
    """Render a retrieval timestamp for the screen, tolerating odd input."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M %Z").strip()
    return str(value) if value else "not recorded"


def retrieved_jobs_table(jobs: list[Any], limit: int = 3) -> list[dict[str, str]]:
    """Build the contrast table: facts a model could not have held in memory.

    Each row is one posting this app retrieved, with the direct public URL and
    the moment it was fetched.
    """
    rows: list[dict[str, str]] = []
    for job in jobs[:limit]:
        rows.append(
            {
                "Title": str(getattr(job, "title", "") or "unknown"),
                "Company": str(getattr(job, "company", "") or "unknown"),
                "Source URL": str(getattr(job, "source_url", "") or "not available"),
                "Retrieved at": _format_timestamp(getattr(job, "retrieved_at", None)),
            }
        )
    return rows


def _add_prompt(result: LessonResult) -> None:
    """Show the question before showing any answer to it."""
    result.add(
        "markdown",
        "The question",
        "We ask the model something it cannot answer from memory. The answer "
        "changes every hour, and it lives on pages the model was never shown.",
    )
    result.add("code", "Prompt sent to the model", MEMORY_PROMPT)


def _add_retrieval_contrast(result: LessonResult, ctx: LessonContext) -> None:
    """Put the app's own retrieved postings next to the model's answer."""
    rows = retrieved_jobs_table(list(ctx.jobs or []))
    if not rows:
        result.add(
            "warning",
            "No retrieved postings are loaded",
            "The contrast table needs the retrieved job data. Run the agent on "
            "the demo tab, or check data/cached_jobs.json.",
        )
        return

    result.add(
        "table",
        "What this app retrieved instead - every row is checkable",
        rows,
    )
    result.add(
        "markdown",
        "Why this table is different",
        "Each row carries a direct public URL you can open on the projector and "
        "the exact time it was fetched. None of that came from the model's "
        "memory; a tool went and got it. That is the whole idea behind the next "
        "step.",
    )


def _model_reply(ctx: LessonContext) -> str | None:
    """Ask the model once, tolerating a client that misbehaves in any way.

    Returns clean reply text, or ``None`` when nothing usable came back: an
    unreachable provider, a client that raised, an empty or whitespace-only
    answer, or a value that is not text at all. A blank answer counts as
    unreachable on purpose - printing an empty box under the heading "what the
    model actually replied" would teach nothing and imply a reply that was not
    given.
    """
    try:
        reply, available = ctx.llm_text(MEMORY_PROMPT, temperature=0.2)
    except Exception:  # noqa: BLE001 - a lesson must never crash the lab
        return None
    if not available or reply is None:
        return None
    text = reply if isinstance(reply, str) else str(reply)
    return text.strip() or None


def run(ctx: LessonContext) -> LessonResult:
    """Ask the model an unanswerable question, then show verifiable retrieval.

    Never raises: on any failure the step still returns the deterministic
    retrieved-jobs contrast.
    """
    result = LessonResult()
    _add_prompt(result)

    reply = _model_reply(ctx)

    if reply:
        result.used_llm = True
        shown = reply
        if len(shown) > MAX_REPLY_CHARS:
            shown = shown[:MAX_REPLY_CHARS].rstrip() + "\n... (reply truncated for the screen)"
        result.add("code", "What the model actually replied", shown)

        assessment = describe_reply(reply)
        result.add(
            assessment.block_kind,
            assessment.headline,
            assessment.explanation,
        )
        if assessment.urls:
            result.add(
                "json",
                "Links the model produced - none of these were fetched",
                {"unverified_links": assessment.urls},
            )
    else:
        result.used_llm = False
        result.llm_unavailable = True
        result.add(
            "note",
            "The model was not reachable, so the deterministic path ran",
            "No API key, or the provider declined the request. Nothing below is "
            "a live model response. The step still teaches: the next block is a "
            "canned illustration written for this workshop, and the table under "
            "it is real retrieved data from this app.",
        )
        result.add(
            "code",
            "CANNED ILLUSTRATION - written by us, not by a model",
            CANNED_HALLUCINATION,
        )
        result.add(
            "warning",
            "Why an answer shaped like this cannot be trusted",
            "It reads like a search result, but no page was ever loaded. There "
            "is no retrieval time, no source, and nothing to click through to. "
            "The links above deliberately use the reserved .example domain so "
            "this teaching example cannot point at a real employer - a real "
            "hallucination would look just as plausible and would not be marked.",
        )

    _add_retrieval_contrast(result, ctx)
    return result


TEACHING_CODE = '''# The model's knowledge stops at its training cutoff. It has no clock
# and no connection to the web, so this question is unanswerable from memory.
MEMORY_PROMPT = (
    "List three Data Analyst Intern jobs posted in Houston, TX within the "
    "last 24 hours, with a working URL for each."
)

reply = _model_reply(ctx)            # None when the provider cannot be reached

if reply:
    urls = find_urls(reply)          # a plain regex over the reply text
    if urls and not looks_like_refusal(reply):
        print("Unverified links invented from memory:", len(urls))
    # A truthful model refuses here - and then the student still has no jobs.

# The agent's own answer is checkable, because a tool went and fetched it:
for job in ctx.jobs[:3]:
    print(job.title, job.company, job.source_url, job.retrieved_at)
'''

STEP = LessonStep(
    number=2,
    title="Why the model cannot know today's jobs",
    subtitle="A frozen brain with no clock and no browser",
    concept=(
        "A language model learns from text collected up to a fixed date, and "
        "then stops learning. It has no clock, no browser, and no memory of "
        "anything that happened after that date. Ask it about a job posted this "
        "morning and the only thing it can do is produce text that looks like "
        "an answer."
    ),
    why=(
        "This is the single most expensive misunderstanding in the room: people "
        "treat a confident answer as a checked answer. A made-up job link costs "
        "a student an afternoon, and the same failure in a business process "
        "costs a great deal more. Seeing the model produce, or correctly refuse "
        "to produce, an answer it cannot know is what makes the next step - "
        "going and fetching the real page - feel necessary rather than fancy."
    ),
    deck_reference="Slide 4 - Training cutoff",
    code=TEACHING_CODE,
    run=run,
    takeaway=(
        "The model can only tell you what it read before its cutoff; anything "
        "about today has to be fetched, not remembered."
    ),
    needs_llm=True,
)
