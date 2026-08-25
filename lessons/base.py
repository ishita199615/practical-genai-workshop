"""The shared contract every teaching step implements.

The Learn tab is a sequence of small, self-contained steps. Each one states a
single idea in plain English, shows the real code that implements it, runs that
code live, and prints what came back.

Two rules keep the lab usable in a room full of students:

1. **Every step must run offline.** A step may *prefer* the language model, but
   it must still produce a correct, teachable result when the API is rate
   limited, overloaded, or unconfigured. Steps report which path they took via
   :attr:`LessonResult.used_llm` so the screen never implies a call happened
   that did not.
2. **Pure logic here, rendering in the app.** A step returns data structures;
   Streamlit decides how to draw them. That keeps steps unit-testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

from config import Settings
from models.job import JobPosting
from models.resume import ResumeProfile

BlockKind = Literal[
    "markdown",
    "code",
    "json",
    "table",
    "metric",
    "note",
    "warning",
    "success",
    "compare",
]


@dataclass(frozen=True)
class OutputBlock:
    """One renderable piece of a step's live output.

    ``kind`` tells the UI how to draw ``body``:

    * ``markdown`` - prose; ``body`` is a string
    * ``code``     - a code or raw-text block; ``body`` is a string
    * ``json``     - a structured object; ``body`` is JSON-serializable
    * ``table``    - ``body`` is a list of flat dicts sharing keys
    * ``metric``   - ``body`` is ``{"value": ..., "help": ...}``
    * ``note`` / ``warning`` / ``success`` - a callout; ``body`` is a string
    * ``compare``  - ``body`` is ``{"left_label", "left", "right_label", "right"}``
    """

    kind: BlockKind
    label: str
    body: Any
    language: str = "text"


@dataclass
class LessonResult:
    """What one step produced when it ran."""

    blocks: list[OutputBlock] = field(default_factory=list)
    used_llm: bool = False
    llm_unavailable: bool = False
    elapsed_seconds: float = 0.0

    def add(
        self, kind: BlockKind, label: str, body: Any, language: str = "text"
    ) -> "LessonResult":
        """Append a block and return self so calls can be chained."""
        self.blocks.append(OutputBlock(kind=kind, label=label, body=body, language=language))
        return self


class LessonLLM(Protocol):
    """The slice of the language-model client a lesson may use."""

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        ...


@dataclass(frozen=True)
class LessonContext:
    """Everything a step needs, injected so steps stay testable.

    ``jobs`` is always pre-loaded from the cached demonstration data, so no step
    depends on a live network call to have something to teach with.
    """

    settings: Settings
    llm: Any
    resume: ResumeProfile
    jobs: list[JobPosting]

    def llm_text(
        self, prompt: str, *, temperature: float = 0.2
    ) -> tuple[str | None, bool]:
        """Ask the model for text.

        Returns ``(text, available)``. ``available`` is False when the model
        could not be reached at all, which the caller must surface rather than
        hide.
        """
        client = self.llm
        if client is None or not getattr(client, "available", False):
            return None, False
        try:
            text = client.generate_text(prompt, temperature=temperature)
        except Exception:  # noqa: BLE001 - a lesson must never crash the app
            return None, False
        return (text, True) if text else (None, False)


@dataclass(frozen=True)
class LessonStep:
    """One numbered teaching step."""

    number: int
    title: str
    subtitle: str
    concept: str
    why: str
    deck_reference: str
    code: str
    run: Callable[[LessonContext], LessonResult]
    takeaway: str
    needs_llm: bool = False

    def execute(self, ctx: LessonContext) -> LessonResult:
        """Run the step, timing it and guaranteeing it never raises."""
        started = time.perf_counter()
        try:
            result = self.run(ctx)
        except Exception as exc:  # noqa: BLE001 - keep the lab alive
            result = LessonResult()
            result.add(
                "warning",
                "This step could not complete",
                f"{type(exc).__name__}: {exc}",
            )
        result.elapsed_seconds = time.perf_counter() - started
        return result


def approx_tokens(text: str) -> int:
    """Estimate token count the way the deck explains it (~4 chars per token).

    Deliberately the simple rule from slide 4 rather than a real tokenizer: the
    point is the order of magnitude and the cost intuition, not exactness.
    """
    return max(1, round(len(text) / 4))
