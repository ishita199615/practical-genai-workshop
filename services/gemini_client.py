"""Gemini implementation of :class:`LLMClient` using ``google-genai``.

Responsibilities are deliberately narrow. Gemini extracts, explains,
recommends wording, drafts, and reviews claims. It never calculates a score and
never decides whether a job is fresh.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

# Free Gemini tiers cap requests per minute. One short pause lets a burst of
# concurrent extractions recover instead of all failing at once.
QUOTA_BACKOFF_SECONDS = 4.0

_QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "rate limit")


def _is_quota_error(exc: Exception) -> bool:
    """True when an exception looks like a rate or quota limit."""
    lowered = str(exc).lower()
    return any(marker in lowered for marker in _QUOTA_MARKERS)


class GeminiClient:
    """Structured-output client with one automatic retry.

    The API key is held here only; it is never logged, never returned, and
    never placed in application state.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._client = client
        self._init_error: str | None = None
        # Set when the provider reports a quota limit, so the interface can say
        # why a step fell back instead of degrading silently.
        self.quota_limited = False

    @property
    def available(self) -> bool:
        """True when an API key (or injected client) is present."""
        return bool(self._api_key) or self._client is not None

    @property
    def model_name(self) -> str:
        """The configured Gemini model ID, read from ``GEMINI_MODEL``."""
        return self._model

    def _get_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not self._api_key:
            return None
        try:
            from google import genai  # noqa: PLC0415 - optional dependency

            self._client = genai.Client(api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            self._init_error = type(exc).__name__
            logger.warning("Gemini client init failed: %s", self._init_error)
            return None
        return self._client

    def generate_structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        temperature: float = 0.1,
    ) -> ModelT | None:
        """Generate JSON constrained to ``schema`` and validate it.

        Retries once on invalid structured output and gives up afterwards so a
        malformed record is rejected rather than guessed at.
        """
        client = self._get_client()
        if client is None:
            return None

        config = self._build_config(schema, temperature)
        for attempt in (1, 2):
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001 - adapter boundary
                logger.warning(
                    "Gemini request failed (attempt %s): %s — %s",
                    attempt,
                    type(exc).__name__,
                    self._safe_detail(exc),
                )
                if attempt == 1 and _is_quota_error(exc):
                    self.quota_limited = True
                    time.sleep(QUOTA_BACKOFF_SECONDS)
                continue

            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                return parsed
            text = getattr(response, "text", None)
            if text:
                try:
                    return schema.model_validate_json(text)
                except ValidationError:
                    logger.warning(
                        "Gemini returned invalid structured output (attempt %s).",
                        attempt,
                    )
        return None

    def generate_text(self, prompt: str, *, temperature: float = 0.2) -> str | None:
        """Generate short free text, or ``None`` when the call fails."""
        client = self._get_client()
        if client is None:
            return None
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._build_config(None, temperature),
            )
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            logger.warning(
                "Gemini text request failed: %s — %s",
                type(exc).__name__,
                self._safe_detail(exc),
            )
            if _is_quota_error(exc):
                self.quota_limited = True
            return None
        text = getattr(response, "text", None)
        return text.strip() if text else None

    def _safe_detail(self, exc: Exception) -> str:
        """Return a short error detail with the API key redacted.

        Enough to diagnose a quota or model problem; never enough to leak a
        secret or echo prompt content into a log file.
        """
        detail = " ".join(str(exc).split())[:200]
        if self._api_key:
            detail = detail.replace(self._api_key, "<redacted>")
        return detail

    def _build_config(self, schema: type[BaseModel] | None, temperature: float) -> Any:
        """Build a generation config, tolerating SDK differences."""
        try:
            from google.genai import types  # noqa: PLC0415 - optional dependency
        except Exception:  # noqa: BLE001
            return None
        kwargs: dict[str, Any] = {"temperature": temperature}
        if schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema
        try:
            return types.GenerateContentConfig(**kwargs)
        except Exception:  # noqa: BLE001 - unknown field on this SDK version
            return types.GenerateContentConfig(temperature=temperature)


def build_llm_client(
    api_key: str, model: str, models: list[str] | None = None
) -> Any:
    """Return the best available LLM client for the configuration.

    Prefers the routing client when a model chain is configured and LiteLLM is
    importable, because routing survives a per-model quota limit. Falls back to
    the single-model Gemini client, then to the null client, so the app degrades
    instead of failing.
    """
    if models:
        try:
            from services.router_client import RouterClient  # noqa: PLC0415

            import litellm  # noqa: F401, PLC0415 - probe that routing can work

            router = RouterClient(api_key=api_key, models=models)
            if router.available:
                return router
        except Exception as exc:  # noqa: BLE001 - fall back to the direct client
            logger.warning("Model routing unavailable (%s); using the direct client.", type(exc).__name__)
    if api_key:
        return GeminiClient(api_key=api_key, model=model)
    from services.llm_interface import NullLLMClient  # noqa: PLC0415

    return NullLLMClient("GEMINI_API_KEY is not set.")
