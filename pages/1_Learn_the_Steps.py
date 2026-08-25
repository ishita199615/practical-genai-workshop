"""The Learn tab: seven runnable steps that build up to the agent.

Each step states one idea, shows the real code behind it, runs that code, and
prints what came back. Every step works with no API key and no network, so the
lab never depends on a quota surviving the workshop.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config import load_settings
from lessons import ALL_STEPS
from lessons.base import LessonResult, LessonStep, OutputBlock
from lessons.context import build_lesson_context
from services.gemini_client import build_llm_client

settings = load_settings()


@st.cache_resource(show_spinner=False)
def get_llm() -> Any:
    """Build the routing LLM client once per session."""
    return build_llm_client(
        settings.gemini_api_key, settings.gemini_model, settings.model_chain
    )


@st.cache_resource(show_spinner="Loading the fictional resume and cached postings…")
def get_context() -> Any:
    """Build the lesson data once; it is identical on every machine."""
    return build_lesson_context(settings, llm=get_llm())


def render_block(block: OutputBlock) -> None:
    """Draw one output block according to its kind."""
    kind = block.kind
    if kind == "markdown":
        if block.label:
            st.markdown(f"**{block.label}**")
        st.markdown(block.body)
    elif kind == "code":
        if block.label:
            st.markdown(f"**{block.label}**")
        st.code(str(block.body), language=block.language or "text")
    elif kind == "json":
        if block.label:
            st.markdown(f"**{block.label}**")
        st.json(block.body, expanded=False)
    elif kind == "table":
        if block.label:
            st.markdown(f"**{block.label}**")
        rows = block.body if isinstance(block.body, list) else []
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No rows to show.")
    elif kind == "metric":
        body = block.body if isinstance(block.body, dict) else {"value": block.body}
        st.metric(block.label, body.get("value", "—"), help=body.get("help"))
    elif kind == "note":
        st.info(block.body, icon="ℹ️")
    elif kind == "warning":
        st.warning(block.body, icon="⚠️")
    elif kind == "success":
        st.success(block.body, icon="✅")
    elif kind == "compare":
        body = block.body if isinstance(block.body, dict) else {}
        if block.label:
            st.markdown(f"**{block.label}**")
        left, right = st.columns(2)
        with left:
            st.caption(str(body.get("left_label", "Left")))
            st.code(str(body.get("left", "")), language="text")
        with right:
            st.caption(str(body.get("right_label", "Right")))
            st.code(str(body.get("right", "")), language="text")
    else:  # pragma: no cover - defensive
        st.write(block.body)


def render_result(step: LessonStep, result: LessonResult) -> None:
    """Draw a step's live output plus an honest note about which path ran."""
    if result.used_llm:
        served = getattr(get_llm(), "last_served_by", None)
        st.caption(
            f"⚡ Ran in {result.elapsed_seconds:.2f}s · the language model was called"
            + (f" · served by `{served}`" if served else "")
        )
    else:
        detail = (
            "the model was not reached, so the deterministic path ran"
            if result.llm_unavailable
            else "no model needed — this step is pure Python"
        )
        st.caption(f"⚡ Ran in {result.elapsed_seconds:.2f}s · {detail}")

    for block in result.blocks:
        render_block(block)


def run_step(step: LessonStep) -> None:
    """Execute one step and stash its result in session state."""
    with st.spinner(f"Running step {step.number}…"):
        st.session_state[f"lesson_result_{step.number}"] = step.execute(get_context())


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("📚 Learn the steps")
st.caption(
    "Seven small ideas. Each one runs real code and solves the problem the "
    "step before it exposed."
)

st.info(
    "Every step works with no API key and no internet. When the language model "
    "is unreachable, a step says so and runs its deterministic path instead — "
    "it never pretends a call happened.",
    icon="ℹ️",
)

with st.expander("How the seven steps fit together", expanded=False):
    st.markdown(
        """
| # | Step | The problem it solves |
|---|------|----------------------|
| 1 | Prompt → Completion | What a model actually does, and why output varies |
| 2 | The training cutoff | The model cannot know today's jobs |
| 3 | Retrieval | So fetch real, current pages instead of trusting memory |
| 4 | **RAG** | Pages are too big — chunk, embed, retrieve only what matters |
| 5 | Tools | Models are inconsistent at numbers — let Python compute |
| 6 | The agent loop | Wrap it all in Reason → Act → Observe |
| 7 | Guardrails | Refuse to lie, and make a human approve |

The **Full Demo** page in the sidebar is these seven ideas assembled into one
working agent.
        """
    )

# --------------------------------------------------------------------------
# Sidebar — what is actually configured
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Setup")
    llm = get_llm()
    if getattr(llm, "available", False):
        st.success("Language model: configured", icon="✅")
        chain = getattr(llm, "chain", None)
        if chain:
            st.caption("Model routing chain (tried in order):")
            for index, model in enumerate(chain, 1):
                st.caption(f"{index}. `{model}`")
            st.caption(
                "Gemini meters free quota per model, so a chain keeps the lab "
                "running when one model is exhausted."
            )
        served = getattr(llm, "last_served_by", None)
        if served:
            st.caption(f"Last served by: `{served}`")
    else:
        st.warning("Language model: not configured", icon="⚠️")
        st.caption("Every step still runs. LLM steps use their offline path.")

    context = get_context()
    st.metric("Sample postings loaded", len(context.jobs))
    st.caption(f"Fictional candidate: {context.resume.name}")

    if st.button("Run every step", use_container_width=True):
        for step in ALL_STEPS:
            run_step(step)

    if st.button("Clear all results", use_container_width=True):
        for step in ALL_STEPS:
            st.session_state.pop(f"lesson_result_{step.number}", None)

# --------------------------------------------------------------------------
# The steps
# --------------------------------------------------------------------------

for step in ALL_STEPS:
    result_key = f"lesson_result_{step.number}"
    has_result = result_key in st.session_state

    with st.expander(
        f"Step {step.number} · {step.title}", expanded=(step.number == 1 or has_result)
    ):
        st.markdown(f"*{step.subtitle}*")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**What this is**")
            st.markdown(step.concept)
            st.markdown("**Why it matters**")
            st.markdown(step.why)
            st.caption(f"📊 Workshop deck: {step.deck_reference}")
        with right:
            st.markdown("**The code that runs**")
            st.code(step.code, language="python")

        if st.button(
            f"▶ Run step {step.number}",
            key=f"run_{step.number}",
            type="primary" if step.number == 1 else "secondary",
        ):
            run_step(step)
            has_result = True

        if has_result:
            st.divider()
            render_result(step, st.session_state[result_key])
            st.success(f"**Takeaway** — {step.takeaway}", icon="🎯")
