"""Step 3 - Retrieval: fetching real, current text.

Deck slide 8 (retrieval-augmented generation). A model only knows what was in
its training data. An agent knows what it just fetched. This step opens the
retrieval stage and shows the evidence: which public page each piece of text
came from, when it was captured, how much of it there is, and what it would
cost to paste all of it into a prompt.

The step is deliberately **fully deterministic**. It never calls the language
model, so it teaches identically on a conference network, on a quota-limited
free tier, and with no API key at all.

Every sentence it prints is generated from the postings in front of it. Where
prose and table could drift apart - a claimed retrieval time, a freshness
label, the domain a quotation came from - the prose is built from the same
values the table shows, so the two cannot contradict each other on stage.
"""

from __future__ import annotations

from datetime import datetime, timezone

from lessons.base import LessonContext, LessonResult, LessonStep, approx_tokens
from models.job import FRESHNESS_LABELS, JobPosting
from tools.firecrawl_search import is_safe_public_job_url, registrable_domain

# How much genuine page text students see on screen. Long enough to look like a
# real job page, short enough to read from the back of the room.
RAW_EXCERPT_CHARS = 600

# The freshness evidence a page can carry, strongest first. The order is the
# teaching order: hard proof first, "we do not know" last.
EVIDENCE_ORDER: tuple[str, ...] = (
    "exact_timestamp",
    "date_only",
    "search_filter_only",
    "unavailable",
)

# Plain-English names for each kind of evidence.
EVIDENCE_NAMES: dict[str, str] = {
    "exact_timestamp": "an exact timestamp on the page",
    "date_only": "a date but no clock time",
    "search_filter_only": "no posting time of its own",
    "unavailable": "no posting evidence at all",
}


def plural(count: int, word: str, many: str | None = None) -> str:
    """Return ``word`` or its plural, so the screen never reads "1 pages"."""
    return word if count == 1 else (many or word + "s")


def format_utc(value: datetime | None) -> str:
    """Format a timestamp for the screen, always in UTC.

    Naive timestamps are treated as UTC rather than as local time, so the
    retrieval clock shown to the room never shifts with the laptop's timezone.
    """
    if not isinstance(value, datetime):
        return "not recorded"
    stamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def page_text_of(job: JobPosting) -> str:
    """Return the cleaned page text captured for one posting."""
    return job.description or ""


def posted_display(job: JobPosting) -> str:
    """Show what the page itself said about when it was posted.

    Never inferred. A page with no date says so, which is the point of the whole
    step: the retrieval time is ours, the posting time is the source's.
    """
    if job.posted_at is not None:
        return format_utc(job.posted_at)
    if job.posting_date is not None:
        return f"{job.posting_date.isoformat()} (date only)"
    return "not shown on page"


def usable_pages(jobs: list[JobPosting]) -> list[JobPosting]:
    """Return the pages that actually carried text we can quote and count."""
    return [job for job in jobs if page_text_of(job)]


def retrieval_rows(jobs: list[JobPosting]) -> list[dict[str, object]]:
    """Build one table row per retrieved page.

    Each row is provenance, not summary: the source, the link that was checked,
    what the page said about its own posting time, when we captured it, and how
    many characters of real text we kept.
    """
    rows: list[dict[str, object]] = []
    for index, job in enumerate(jobs, start=1):
        link_ok = is_safe_public_job_url(job.source_url)
        rows.append(
            {
                "#": index,
                "Source": job.source_label,
                "Job": f"{job.title} - {job.company}",
                "Page text (chars)": len(page_text_of(job)),
                "Posted (per the page)": posted_display(job),
                "Retrieved (UTC)": format_utc(job.retrieved_at),
                "Posting evidence": job.freshness_label(),
                "Link checked": "public URL" if link_ok else "rejected",
                "Source URL": job.source_url if link_ok else "withheld (unsafe link)",
            }
        )
    return rows


def pick_excerpt_job(jobs: list[JobPosting]) -> JobPosting | None:
    """Choose one posting whose captured text is worth showing.

    Deterministic on purpose: the first page that has a verified public link and
    real text, so the same job appears at every rehearsal. Falls back to any page
    that at least has text, and only then to the first page at all.
    """
    for job in jobs:
        if page_text_of(job) and is_safe_public_job_url(job.source_url):
            return job
    for job in jobs:
        if page_text_of(job):
            return job
    return jobs[0] if jobs else None


def excerpt_parts(
    job: JobPosting, limit: int = RAW_EXCERPT_CHARS
) -> tuple[str, int, int]:
    """Split a page's text into ``(shown_text, shown_chars, remaining_chars)``.

    ``shown_chars + remaining_chars`` always equals the full length, so the
    caption on screen adds up to the number in the table.
    """
    text = page_text_of(job)
    if len(text) <= limit:
        return text, len(text), 0
    head = text[:limit].rstrip()
    return head, len(head), len(text) - len(head)


def raw_excerpt(job: JobPosting, limit: int = RAW_EXCERPT_CHARS) -> str:
    """Return the opening characters of a page exactly as they were captured.

    Line breaks are preserved and nothing is paraphrased. Students should see
    scraped text, not a tidy summary of it.
    """
    if not page_text_of(job):
        return "(no page text was captured for this posting)"
    shown, _, remaining = excerpt_parts(job, limit)
    if not remaining:
        return shown
    return f"{shown}\n\n[... {remaining:,} more characters]"


def freshness_note(jobs: list[JobPosting]) -> str:
    """Explain the difference between a search filter and proof of freshness.

    The per-page breakdown quotes :meth:`JobPosting.freshness_label` - the very
    strings printed in the table - so this paragraph can never claim a label the
    rows do not show.
    """
    windows = {job.freshness_window for job in jobs}
    if len(windows) == 1:
        window_text = FRESHNESS_LABELS.get(next(iter(windows)), "a time window")
        opening = (
            f"**The search asked for {window_text}. That is a filter, not a receipt.**"
        )
    else:
        opening = (
            "**The search asked for a time window. That is a filter, not a receipt.**"
        )

    counts: dict[tuple[str, str], int] = {}
    for job in jobs:
        key = (job.freshness_evidence, job.freshness_label())
        counts[key] = counts.get(key, 0) + 1

    def rank(item: tuple[tuple[str, str], int]) -> tuple[int, str]:
        """Order buckets by evidence strength, then by label for stability."""
        evidence, label = item[0]
        position = (
            EVIDENCE_ORDER.index(evidence)
            if evidence in EVIDENCE_ORDER
            else len(EVIDENCE_ORDER)
        )
        return position, label

    breakdown = "; ".join(
        f"{count} with {EVIDENCE_NAMES.get(evidence, 'no posting evidence at all')}, "
        f'shown as "{label}"'
        for (evidence, label), count in sorted(counts.items(), key=rank)
    )

    return (
        f"{opening}\n\n"
        f"A time filter narrows what the search engine returns. It never proves "
        f"when a job was posted. So each page is labelled by the evidence the "
        f"page itself carried: {breakdown}.\n\n"
        f"Only a page that exposes an exact timestamp is called *verified*. A "
        f"page that shows a date but no clock time says exactly that. A page "
        f"with no posting time of its own says so too, instead of borrowing "
        f"confidence from the filter. And if a narrow window returns nothing, "
        f"the app asks before widening it - it never widens the search quietly "
        f"and calls the result fresh."
    )


def mode_summary(jobs: list[JobPosting]) -> str:
    """State plainly whether this text was fetched live or replayed from cache."""
    count = len(jobs)
    lead = "This" if count == 1 else "These"
    pages = f"{count} {plural(count, 'page')}"
    modes = {job.data_mode for job in jobs}

    if modes == {"cached"}:
        stamps = {format_utc(job.retrieved_at) for job in jobs}
        when = (
            f", originally retrieved at {next(iter(stamps))}"
            if len(stamps) == 1
            else "; each row carries the time it was originally retrieved"
        )
        noun = (
            "a clearly labelled cached demonstration result"
            if count == 1
            else "clearly labelled cached demonstration results"
        )
        verb = "is" if count == 1 else "are"
        return (
            f"{lead} {pages} {verb} **{noun}**{when}. The timestamps below are "
            f"from that run, not from right now."
        )
    if modes == {"live"}:
        verb = "was" if count == 1 else "were"
        return (
            f"{lead} {pages} {verb} fetched **live** from public job sites "
            f"during this run."
        )
    verb = "is" if count == 1 else "are"
    return (
        f"{lead} {pages} {verb} a mix of live and cached results; each row "
        f"carries its own retrieval time."
    )


def run(ctx: LessonContext) -> LessonResult:
    """Show where the agent's job text came from, and what it costs to use.

    Deterministic end to end: no model call, so ``used_llm`` is False and
    ``llm_unavailable`` stays False - the step never wanted a model in the first
    place.
    """
    result = LessonResult(used_llm=False, llm_unavailable=False)
    try:
        jobs = list(ctx.jobs or [])
    except Exception:  # noqa: BLE001 - a lesson must never crash the lab
        jobs = []

    if not jobs:
        result.add(
            "warning",
            "No retrieved pages are loaded",
            "This step reads the pages the agent already fetched. Run the agent "
            "on the main tab, or check that the cached demonstration data is "
            "present, then run this step again.",
        )
        return result

    try:
        result.add("markdown", "Where this text came from", mode_summary(jobs))
        result.add("table", "Pages the agent actually fetched", retrieval_rows(jobs))

        sample = pick_excerpt_job(jobs)
        # Quote only a page that actually gave us text. A caption promising "raw
        # page text" over an empty page would be exactly the dishonesty this
        # step exists to argue against.
        if sample is not None and page_text_of(sample):
            _, shown, remaining = excerpt_parts(sample)
            total = shown + remaining
            result.add(
                "code",
                (
                    f"Raw page text as captured - {sample.company} via "
                    f"{sample.source_label} ({total:,} chars total, first "
                    f"{shown:,} shown)"
                ),
                raw_excerpt(sample),
                language="text",
            )
            host = (
                registrable_domain(sample.source_url)
                if is_safe_public_job_url(sample.source_url)
                else ""
            )
            origin = (
                f"It came from the public page at `{host}`."
                if host
                else (
                    "It came from a page whose link failed validation, which is "
                    "why the link is withheld in the table above."
                )
            )
            result.add(
                "markdown",
                "That text is the evidence",
                f"Nothing above was written by a model. {origin} It has been "
                f"cleaned of menus and cookie banners and otherwise left alone. "
                f"Every score later in the demo is computed against this text.",
            )

        readable = usable_pages(jobs)
        corpus = "\n\n".join(page_text_of(job) for job in readable)
        total_chars = len(corpus)
        naive_tokens = approx_tokens(corpus) if corpus else 0
        if readable:
            page_word = plural(len(readable), "page")
            result.add(
                "metric",
                "Cost of pasting every page into one prompt",
                {
                    "value": (
                        f"{len(readable)} {page_word} | {total_chars:,} chars | "
                        f"~{naive_tokens:,} tokens"
                    ),
                    "help": (
                        f"{len(readable)} public {page_word} of real text, "
                        f"{total_chars:,} characters in all. Pasting the lot into "
                        f"a single prompt costs roughly {naive_tokens:,} tokens "
                        f"(about 4 characters per token) before the model even "
                        f"reads the question - and most of it is irrelevant to "
                        f"any one job. That is why Step 4 splits the text into "
                        f"chunks and sends only the parts that matter."
                    ),
                },
            )

        result.add(
            "note",
            "Freshness: a filter is not proof",
            freshness_note(jobs),
        )

        no_text = [job for job in jobs if not page_text_of(job)]
        bad_link = [job for job in jobs if not is_safe_public_job_url(job.source_url)]
        affected = {id(job) for job in no_text} | {id(job) for job in bad_link}
        if affected:
            problems: list[str] = []
            if no_text:
                problems.append(f"{len(no_text)} returned no readable text")
            if bad_link:
                problems.append(
                    f"{len(bad_link)} had a link that failed public-URL validation"
                )
            caveat = (
                " (a single page can have both problems)" if len(problems) > 1 else ""
            )
            result.add(
                "warning",
                "Some pages could not be used as evidence",
                f"{len(affected)} of {len(jobs)} retrieved "
                f"{plural(len(jobs), 'page')} could not be used: "
                + " and ".join(problems)
                + caveat
                + ". They stay visible and marked, never quietly quoted or "
                "invented around.",
            )
    except Exception as exc:  # noqa: BLE001 - keep the lab alive
        result.add(
            "warning",
            "This step could not finish",
            f"{type(exc).__name__}: {exc}",
        )
    return result


CODE_SNIPPET = '''\
from lessons.base import approx_tokens
from tools.firecrawl_search import is_safe_public_job_url

# `jobs` are pages the agent fetched. The model was never asked what jobs exist.
corpus = []
for job in jobs:
    # Every link is checked. A page we cannot verify is marked, not hidden.
    link = job.source_url if is_safe_public_job_url(job.source_url) else "rejected"
    page_text = job.description       # real text captured from the public page
    if page_text:
        corpus.append(page_text)
    print(job.source_label, len(page_text), "chars", job.retrieved_at, link)

    # A time filter narrows the search. It is not proof of when a job posted,
    # so each page is labelled by the evidence the page itself carried.
    print("  freshness:", job.freshness_label())

all_text = "\\n\\n".join(corpus)
print("Pasting every page into one prompt:", approx_tokens(all_text), "tokens")
'''


STEP = LessonStep(
    number=3,
    title="Retrieval: fetching real, current text",
    subtitle="The agent looks things up instead of remembering them",
    concept=(
        "A language model can only repeat what it saw during training, and its "
        "training stopped months or years ago. Retrieval fixes that: the agent "
        "fetches real public web pages right now and hands that text to the "
        "model. The pages below are actual job postings, not recollections."
    ),
    why=(
        "Jobs posted this week did not exist when the model was trained, so asking "
        "the model to list them produces confident fiction. Fetching the pages "
        "gives every later step something checkable - a source, a link, and a "
        "capture time an instructor can open in a browser. It also shows the "
        "problem Step 4 solves: all this text is far too much to paste into one "
        "prompt."
    ),
    deck_reference="Slide 8 - Retrieval-Augmented Generation (RAG)",
    code=CODE_SNIPPET,
    run=run,
    takeaway=(
        "An agent is current because it fetches the page, not because it "
        "remembers the answer."
    ),
    needs_llm=False,
)
