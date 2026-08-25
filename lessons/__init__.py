"""The Learn tab: seven small steps that build up to the agent.

The order is the argument. Each step exposes a problem that the next step
solves:

1. A prompt produces a completion - and temperature makes it vary.
2. So the model cannot know today's jobs, because its knowledge is frozen.
3. So we retrieve real, current pages instead of trusting memory.
4. So we chunk, embed, and retrieve only what matters, then ground the answer.
5. So we hand the arithmetic to Python, which gives the same answer every time.
6. So we wrap it all in a loop that reasons, acts, and observes.
7. So we add a validator that refuses to lie, and a human who approves.

The full Cougar Career Agent in the Demo tab is these seven ideas assembled.
"""

from __future__ import annotations

from lessons.base import (
    LessonContext,
    LessonResult,
    LessonStep,
    OutputBlock,
    approx_tokens,
)
from lessons.step_1_prompt_completion import STEP as STEP_1
from lessons.step_2_training_cutoff import STEP as STEP_2
from lessons.step_3_retrieval import STEP as STEP_3
from lessons.step_4_rag import STEP as STEP_4
from lessons.step_5_tools import STEP as STEP_5
from lessons.step_6_agent_loop import STEP as STEP_6
from lessons.step_7_guardrails import STEP as STEP_7

ALL_STEPS: tuple[LessonStep, ...] = (
    STEP_1,
    STEP_2,
    STEP_3,
    STEP_4,
    STEP_5,
    STEP_6,
    STEP_7,
)


def step_by_number(number: int) -> LessonStep | None:
    """Return the step with the given number, or ``None`` if there is none."""
    for step in ALL_STEPS:
        if step.number == number:
            return step
    return None


__all__ = [
    "ALL_STEPS",
    "LessonContext",
    "LessonResult",
    "LessonStep",
    "OutputBlock",
    "approx_tokens",
    "step_by_number",
]
