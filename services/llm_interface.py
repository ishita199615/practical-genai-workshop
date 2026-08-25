"""The provider-neutral LLM boundary used by every graph node.

Nodes never import a vendor SDK. They call :class:`LLMClient`, so a local
provider (for example Ollama) can be added later without changing the graph,
and tests can inject a scripted client.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMClient(Protocol):
    """Structured and free-text generation with validated output."""

    @property
    def available(self) -> bool:
        """True when the client can actually reach a model."""
        ...

    @property
    def model_name(self) -> str:
        """The configured model identifier."""
        ...

    def generate_structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        temperature: float = 0.1,
    ) -> ModelT | None:
        """Return a validated model instance, or ``None`` when unavailable."""
        ...

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Return plain text, or ``None`` when unavailable."""
        ...


class NullLLMClient:
    """Offline stand-in used when no API key is configured.

    Returning ``None`` is deliberate: callers fall back to deterministic
    behaviour rather than silently inventing content.
    """

    def __init__(self, reason: str = "No LLM provider configured.") -> None:
        self.reason = reason

    @property
    def available(self) -> bool:
        """Always False; the null client cannot reach a model."""
        return False

    @property
    def model_name(self) -> str:
        """A readable placeholder name for the UI."""
        return "unavailable"

    def generate_structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        temperature: float = 0.1,
    ) -> ModelT | None:
        """Return ``None`` so the caller uses its deterministic fallback."""
        return None

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Return ``None`` so the caller uses its deterministic fallback."""
        return None
