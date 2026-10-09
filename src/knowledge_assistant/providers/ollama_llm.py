"""Ollama-backed LLM provider (default, fully local)."""

from __future__ import annotations

from typing import Any

import ollama
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from knowledge_assistant.core.errors import ProviderError
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.providers.base import ChatMessage

logger = get_logger(__name__)


class OllamaLLMProvider:
    """Text generation via a local or remote Ollama server."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        timeout: float = 120.0,
        temperature: float = 0.1,
        num_ctx: int = 8192,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._client = ollama.Client(host=base_url, timeout=timeout)
        self._resolved_model: str | None = None

    @property
    def model_name(self) -> str:
        return self._model

    def _available_models(self) -> list[str]:
        try:
            response = self._client.list()
        except Exception:  # noqa: BLE001
            return []
        names: list[str] = []
        for item in getattr(response, "models", None) or []:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name:
                names.append(str(name))
        return names

    def _effective_model(self) -> str:
        """Resolve to the configured model, or fall back to any pulled model."""
        if self._resolved_model is not None:
            return self._resolved_model
        available = self._available_models()
        if self._model in available or not available:
            self._resolved_model = self._model
        else:
            self._resolved_model = available[0]
            logger.warning(
                "ollama_model_fallback", configured=self._model, using=self._resolved_model
            )
        return self._resolved_model

    @retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        options: dict[str, Any] = {
            "temperature": self._temperature if temperature is None else temperature,
            "num_ctx": self._num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        model = self._effective_model()
        try:
            response = self._client.chat(
                model=model,
                messages=[m.model_dump() for m in messages],
                options=options,
            )
        except Exception as exc:
            logger.warning("ollama_generate_failed", error=str(exc), model=model)
            raise ProviderError(f"Ollama generation failed: {exc}") from exc

        content = response.get("message", {}).get("content")
        if not content:
            raise ProviderError("Ollama returned an empty completion.")
        return str(content)

    def health_check(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ollama_health_check_failed", error=str(exc))
            return False
