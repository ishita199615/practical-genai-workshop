"""Step 1 - Prompt to Completion (deck slides 3 and 4).

The first thing a room of non-programmers needs to see is that a prompt is
ordinary text, that the reply is a *prediction* of the text that should come
next, and that both halves are metered in tokens.

The step sends one short fixed prompt twice - once at ``temperature=0.0`` and
once at ``temperature=1.0`` - so the effect of the temperature dial is visible
on screen rather than merely described. When the model cannot be reached (the
free Gemini tier is quota limited and refuses requests regularly), the step
says so plainly and shows two clearly-labelled canned completions instead. It
never presents canned text as a model response.
"""

from __future__ import annotations

from lessons.base import LessonContext, LessonResult, LessonStep, approx_tokens

#: The one prompt this step sends. Short on purpose: students can count the
#: characters themselves and check the token estimate.
PROMPT = "In one sentence, describe what a Data Analyst Intern does day to day."

#: Attached to every canned completion so nothing on screen can be mistaken for
#: something the model actually said.
CANNED_LABEL = "canned illustration - the model was not reached"

#: A low-temperature answer looks like this: plain, safe, predictable wording.
CANNED_PREDICTABLE = (
    "A Data Analyst Intern gathers data, cleans it, analyses it with tools such "
    "as SQL and Excel, and summarises the findings in reports and dashboards "
    "for the team."
)

#: A high-temperature answer to the same prompt: same facts, freer wording.
CANNED_CREATIVE = (
    "Most days open with a spreadsheet nobody wants to touch: the intern "
    "untangles it, writes a few SQL queries to test the story it tells, and "
    "finishes by turning the result into a dashboard the team actually reads."
)

TEACHING_CODE = '''PROMPT = "In one sentence, describe what a Data Analyst Intern does day to day."

# A prompt is just text. The model answers with the text it predicts should
# come next. It is not looking anything up.
predictable, low_ok = ctx.llm_text(PROMPT, temperature=0.0)
creative, high_ok = ctx.llm_text(PROMPT, temperature=1.0)

# temperature 0.0 -> always take the most likely next word (repeatable)
# temperature 1.0 -> sample among the likely ones (wording varies)

# Say what actually happened. A side that did not answer is swapped for a
# hand-written illustration and labelled as one - never passed off as a reply.
result.used_llm = low_ok or high_ok
result.llm_unavailable = not (low_ok and high_ok)
left = predictable if low_ok else CANNED_PREDICTABLE
right = creative if high_ok else CANNED_CREATIVE

# A token is roughly 4 characters of English. You pay for the prompt you
# send AND for every reply you get back - here, both of them.
prompt_tokens = approx_tokens(PROMPT)                    # len(text) / 4
reply_tokens = approx_tokens(left) + approx_tokens(right)
'''


def ask_once(ctx: LessonContext, temperature: float) -> tuple[str | None, bool]:
    """Send :data:`PROMPT` once and report whether a real reply came back.

    Returns ``(text, available)``. ``available`` is False whenever the model
    could not be reached or returned nothing usable, so the caller can fall
    back to a labelled illustration instead of inventing a completion.
    """
    try:
        text, available = ctx.llm_text(PROMPT, temperature=temperature)
        if not available or not isinstance(text, str) or not text.strip():
            return None, False
        return text.strip(), True
    except Exception:  # noqa: BLE001 - a lesson must never crash the app
        return None, False


def model_label(ctx: LessonContext) -> str:
    """Return a readable name for the configured model, never a key."""
    try:
        return str(getattr(ctx.llm, "model_name", "") or "unavailable")
    except Exception:  # noqa: BLE001 - the client is injected, so stay defensive
        return "unavailable"


def _side_label(temperature: float, headline: str, is_real: bool) -> str:
    """Label one half of the comparison, marking canned text as canned."""
    suffix = "" if is_real else f" ({CANNED_LABEL})"
    return f"temperature = {temperature:.1f} - {headline}{suffix}"


def run(ctx: LessonContext) -> LessonResult:
    """Send one prompt at two temperatures and report the size of each half.

    Flag semantics, so the screen never overstates what happened:

    * both calls answered - ``used_llm=True``, ``llm_unavailable=False``
    * one call answered   - ``used_llm=True``, ``llm_unavailable=True``, and the
      missing side is shown as a labelled canned illustration
    * neither answered    - ``used_llm=False``, ``llm_unavailable=True``, and
      both sides are labelled canned illustrations
    """
    result = LessonResult()

    predictable, low_ok = ask_once(ctx, 0.0)
    creative, high_ok = ask_once(ctx, 1.0)

    result.used_llm = low_ok or high_ok
    result.llm_unavailable = not (low_ok and high_ok)

    left = predictable if low_ok else CANNED_PREDICTABLE
    right = creative if high_ok else CANNED_CREATIVE

    result.add(
        "code",
        "The exact prompt sent to the model",
        PROMPT,
        language="text",
    )

    if low_ok and high_ok:
        result.add(
            "success",
            "Both replies came from the model",
            f"Model: `{model_label(ctx)}`. The same prompt was sent twice. "
            "The only thing that changed between the two calls was the "
            "temperature setting.",
        )
    elif result.used_llm:
        missing = "temperature 1.0" if low_ok else "temperature 0.0"
        result.add(
            "warning",
            "Only one reply came back from the model",
            f"The {missing} call did not return text, so that side below is a "
            f"{CANNED_LABEL}. The other side is a real model reply. Nothing "
            "here is a guess presented as a completion.",
        )
    else:
        result.add(
            "note",
            "The model was not reached, so the offline path ran",
            "No reply came back - the API key is missing, the free tier is "
            "rate limited, or the machine is offline. Both completions below "
            f'are marked "{CANNED_LABEL}": they were written by hand for this '
            "slide so the temperature difference is still visible. The prompt "
            "and its token count are exactly what a live run would show. The "
            "reply token count below is measured from the canned text on "
            "screen - a live reply would be a different length, counted the "
            "same way.",
        )

    result.add(
        "compare",
        "One prompt, two temperatures",
        {
            "left_label": _side_label(0.0, "the safe, most-likely wording", low_ok),
            "left": left,
            "right_label": _side_label(1.0, "free to pick less-likely words", high_ok),
            "right": right,
        },
    )

    result.add(
        "markdown",
        "What temperature changed",
        "Temperature is a randomness dial, not a quality dial.\n\n"
        "- **0.0** - the model takes the single most likely next word every "
        "time, so the same prompt gives near-identical answers. Use it for "
        "extraction and anything you need to be repeatable.\n"
        "- **1.0** - the model samples among the plausible next words, so the "
        "wording moves around. Use it for brainstorming and variety.\n\n"
        "Neither setting makes the answer more *true*. That is why this demo "
        "keeps every number in Python and lets the model handle only wording.",
    )

    prompt_tokens = approx_tokens(PROMPT)
    left_tokens = approx_tokens(left)
    right_tokens = approx_tokens(right)
    reply_tokens = left_tokens + right_tokens

    result.add(
        "metric",
        "Prompt size (estimated)",
        {
            "value": f"~{prompt_tokens} tokens",
            "help": (
                f"{len(PROMPT)} characters / 4 = about {prompt_tokens} tokens. "
                "The prompt is billed every single time it is sent."
            ),
        },
    )

    reply_source = "both replies" if (low_ok and high_ok) else "both completions shown"
    result.add(
        "metric",
        "Reply size, both completions added up (estimated)",
        {
            "value": f"~{reply_tokens} tokens",
            "help": (
                f"About {left_tokens} tokens at temperature 0.0 and "
                f"{right_tokens} at temperature 1.0, across {reply_source}. "
                "Replies are billed too, and you cannot know their length in "
                "advance."
            ),
        },
    )

    result.add(
        "markdown",
        "Why the token count matters",
        "A token is roughly **four characters** of English - about "
        "three-quarters of a word. `approx_tokens()` in `lessons/base.py` is "
        "literally `len(text) / 4`, which is close enough to reason about "
        "cost.\n\n"
        f"The whole exchange above is about **{prompt_tokens + reply_tokens} "
        "tokens**. That is tiny. But the full agent later in this demo sends "
        "an entire job description plus the resume for every posting it looks "
        "at, and each of those runs into the thousands. Providers bill per "
        "token in both directions, so a longer prompt is a permanently higher "
        "bill - which is a good reason to send the model only the text it "
        "actually needs.",
    )

    return result


STEP = LessonStep(
    number=1,
    title="Prompt to Completion",
    subtitle="Text goes in, predicted text comes out - and both halves cost tokens",
    concept=(
        "A prompt is just text you send to the model, and the reply is the text "
        "the model predicts should come next. Nothing is looked up and nothing "
        "is remembered. A setting called temperature decides how adventurous "
        "that prediction is allowed to be, which is why the same prompt can "
        "come back worded two different ways."
    ),
    why=(
        "Everything else in this workshop sits on top of this one exchange. "
        "Once you can see that the model is predicting rather than retrieving, "
        "its confident mistakes stop being a surprise and the tools, scoring, "
        "and validation in the rest of the demo start to look necessary. And "
        "because you are billed for the text you send plus the text you get "
        "back, prompt length is a running cost rather than a free choice."
    ),
    deck_reference="Deck slides 3 and 4",
    code=TEACHING_CODE,
    run=run,
    takeaway=(
        "A prompt is input, the reply is a prediction, and you pay by the token "
        "for both halves."
    ),
    needs_llm=True,
)
