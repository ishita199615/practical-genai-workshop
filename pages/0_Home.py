"""Cougar Career Agent — workshop home page.

Rendered through ``app.py``, which owns navigation and page config.

Two ways in:

* **Learn the steps** builds the agent one idea at a time, each step runnable on
  its own with no API key required.
* **Full Demo** is the complete Job Hunter agent from the workshop script.

Both live in the sidebar. This page just orients the room and reports whether
the machine is configured for a live run.
"""

from __future__ import annotations

import streamlit as st

from config import load_settings
from lessons import ALL_STEPS

settings = load_settings()

st.title("🎓 Cougar Career Agent")
st.caption("From prompts to agents — a hands-on workshop build.")

st.warning(
    "Demo uses a fictional resume and public job data. It does not submit "
    "applications. ATS readiness is estimated using a transparent workshop "
    "rubric, not an employer's proprietary ATS.",
    icon="⚠️",
)

left, right = st.columns(2)

with left:
    st.subheader("📚 Learn the steps")
    st.markdown(
        """
Seven small steps, each one runnable. Every step shows the real code, runs it,
and prints the result:

1. **Prompt → Completion** — what a model actually does
2. **The training cutoff** — why it cannot know today's jobs
3. **Retrieval** — fetching real, current pages
4. **RAG** — chunk, embed, retrieve, generate
5. **Tools** — let Python do the math
6. **The agent loop** — Reason → Act → Observe
7. **Guardrails** — refuse to lie, then ask a human

Every step works offline. No API key needed.
        """
    )
    st.page_link(
        "pages/1_Learn_the_Steps.py", label="Open the lab", icon="📚"
    )

with right:
    st.subheader("🎯 Full Demo")
    st.markdown(
        """
The complete Job Hunter agent, exactly as the workshop script runs it:

- Searches **current** public job pages
- Ranks them with a deterministic **Job Match Score**
- Scores the resume for **ATS readiness** against one posting
- Drafts a **truthful** resume patch and cover letter
- **Blocks** any claim the resume does not support
- **Pauses for human approval** before exporting

Nothing is ever submitted.
        """
    )
    st.page_link("pages/2_Full_Demo.py", label="Open the demo", icon="🎯")

st.divider()

st.subheader("Is this machine ready?")

status = st.columns(3)

with status[0]:
    if settings.has_firecrawl:
        st.success("**Firecrawl** configured", icon="✅")
        st.caption("Live public job retrieval is available.")
    else:
        st.info("**Firecrawl** not configured", icon="ℹ️")
        st.caption("The demo will use clearly labelled cached results.")

with status[1]:
    if settings.has_gemini:
        st.success("**Language model** configured", icon="✅")
        chain = settings.model_chain
        st.caption(
            f"Routing across {len(chain)} model(s); a quota limit falls through "
            "to the next."
            if len(chain) > 1
            else "Single model configured."
        )
    else:
        st.info("**Language model** not configured", icon="ℹ️")
        st.caption("Steps and the demo fall back to deterministic paths.")

with status[2]:
    st.success(f"**{len(ALL_STEPS)} teaching steps** ready", icon="✅")
    st.caption("The lab runs with no key and no network.")

for warning in settings.startup_warnings:
    st.caption(f"ℹ️ {warning}")

st.divider()
st.caption(
    "University of Houston System · Practical Generative AI. "
    "Fictional data only. No application is ever submitted."
)
