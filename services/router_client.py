"""A model-routing LLM client built on LiteLLM.

Why this exists
---------------
Gemini's free tier meters quota *per model*: ``gemini-3.6-flash`` allows only 20
requests per day, and one full demo run makes about 13. A workshop that depends
on a single model therefore dies part-way through the second run, which is
exactly what happened in rehearsal.

Routing fixes it. This client takes an ordered chain of models and asks LiteLLM
to walk it: the first model that answers wins, and a ``429`` or ``503`` moves
quietly to the next. Because the chain crosses model families - and can cross
providers entirely, including a local Ollama model that has no quota at all -
the demo keeps working when any single endpoint is exhausted.

The client satisfies :class:`services.llm_interface.LLMClient`, so nodes and
lessons need no changes: they still call ``generate_text`` and
``generate_structured`` and still get ``None`` when every model fails, which
triggers the deterministic fallbacks.
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

# Providers whose credentials LiteLLM reads from a conventional variable name.
PROVIDER_ENV_VARS: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Providers that run locally and therefore need no key.
KEYLESS_PROVIDERS: frozenset[str] = frozenset({"ollama", "ollama_chat"})


def _is_quota_error(exc: Exception) -> bool:
    """True when an exception looks like a rate or quota limit."""
    text = str(exc).lower()
    return any(
        term in text
        for term in ("rate limit", "ratelimit", "429", "quota", "resource_exhausted")
    )


def parse_model_chain(raw: str) -> list[str]:
    """Parse a comma-separated model chain into an ordered list.

    Blank entries are dropped and order is preserved, because order *is* the
    routing policy: earlier models are preferred.
    """
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def provider_of(model: str) -> str:
    """Return the LiteLLM provider prefix of a model string.

    ``"gemini/gemini-3.5-flash"`` -> ``"gemini"``. A model with no prefix is
    assumed to be Gemini, matching this project's default.
    """
    return model.split("/", 1)[0].strip().lower() if "/" in model else "gemini"


class RouterClient:
    """Route generation across an ordered chain of models.

    The API key is held here only. It is copied into the process environment
    because that is how LiteLLM reads provider credentials, and it is redacted
    from every log line this class emits.
    """

    def __init__(
        self,
        api_key: str = "",
        models: list[str] | None = None,
        *,
        completion_fn: Any | None = None,
        extra_keys: dict[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._models = list(models or [])
        self._completion_fn = completion_fn
        self._init_error: str | None = None
        self.quota_limited = False
        # The model that actually served the most recent call. Shown in the UI
        # so the room can see routing happen rather than take it on faith.
        self.last_served_by: str | None = None
        self._export_keys(extra_keys or {})

    def _export_keys(self, extra: dict[str, str]) -> None:
        """Publish credentials in the form LiteLLM expects, without logging them."""
        if self._api_key:
            for model in self._models:
                env_var = PROVIDER_ENV_VARS.get(provider_of(model))
                if env_var and not os.environ.get(env_var):
                    os.environ[env_var] = self._api_key
        for name, value in extra.items():
            if value:
                os.environ[name] = value

    @property
    def available(self) -> bool:
        """True when at least one model could plausibly be reached.

        A local provider needs no key, so a chain containing only Ollama is
        available even with no API key configured.
        """
        if not self._models:
            return False
        if self._completion_fn is not None:
            return True
        if self._api_key:
            return True
        return any(provider_of(model) in KEYLESS_PROVIDERS for model in self._models)

    @property
    def model_name(self) -> str:
        """The primary model, or the one that last served a request."""
        return self.last_served_by or (self._models[0] if self._models else "unavailable")

    @property
    def chain(self) -> list[str]:
        """The configured routing chain, in preference order."""
        return list(self._models)

    def _completion(self) -> Any | None:
        """Return the LiteLLM completion callable, or None if unusable."""
        if self._completion_fn is not None:
            return self._completion_fn
        try:
            import litellm  # noqa: PLC0415 - optional dependency

            litellm.suppress_debug_info = True
            os.environ.setdefault("LITELLM_LOG", "ERROR")
            self._completion_fn = litellm.completion
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            self._init_error = type(exc).__name__
            logger.warning("LiteLLM unavailable: %s", self._init_error)
            return None
        return self._completion_fn

    def _call(
        self,
        prompt: str,
        temperature: float,
        response_format: type[BaseModel] | None,
    ) -> str | None:
        """Send one routed request and return the raw reply text."""
        completion = self._completion()
        if completion is None or not self._models:
            return None

        kwargs: dict[str, Any] = {
            "model": self._models[0],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "num_retries": 0,
        }
        if len(self._models) > 1:
            kwargs["fallbacks"] = self._models[1:]
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            if _is_quota_error(exc):
                self.quota_limited = True
            logger.warning(
                "Routed request failed across %d model(s): %s - %s",
                len(self._models),
                type(exc).__name__,
                self._safe_detail(exc),
            )
            return None

        self.last_served_by = getattr(response, "model", None) or self._models[0]
        try:
            content = response.choices[0].message.content
        except Exception:  # noqa: BLE001 - unexpected response shape
            return None
        return content.strip() if isinstance(content, str) and content.strip() else None

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Generate short free text, or ``None`` when every model fails."""
        return self._call(prompt, temperature, None)

    def generate_structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        temperature: float = 0.1,
    ) -> ModelT | None:
        """Return a validated model instance, or ``None`` when unavailable.

        Retries once on malformed structured output, matching the behaviour the
        project spec requires of the Gemini client.
        """
        for attempt in (1, 2):
            raw = self._call(prompt, temperature, schema)
            if raw is None:
                return None
            try:
                return schema.model_validate_json(raw)
            except ValidationError:
                try:
                    return schema.model_validate_json(_strip_code_fence(raw))
                except ValidationError:
                    logger.warning(
                        "Structured output failed validation (attempt %d).", attempt
                    )
        return None

    def _safe_detail(self, exc: Exception) -> str:
        """Return a short error detail with the API key redacted."""
        detail = " ".join(str(exc).split())[:200]
        if self._api_key:
            detail = detail.replace(self._api_key, "<redacted>")
        return detail


def _strip_code_fence(text: str) -> str:
    """Remove a Markdown code fence some models wrap JSON in."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()
