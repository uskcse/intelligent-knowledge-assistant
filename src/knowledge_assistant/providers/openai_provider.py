"""OpenAI-backed providers (optional alternative to the local stack).

Imports of the ``openai`` SDK are lazy so the default local deployment does not
require the package. Install with the ``openai`` extra to enable.
"""

from __future__ import annotations

from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from knowledge_assistant.core.errors import ConfigurationError, ProviderError
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.providers.base import ChatMessage

logger = get_logger(__name__)

# Embedding dimensions for common OpenAI models (avoids a probe call).
_OPENAI_EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _make_client(api_key: str | None, base_url: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ConfigurationError(
            "OpenAI provider selected but the 'openai' package is not installed. "
            "Install with: pip install 'knowledge-assistant[openai]'"
        ) from exc
    if not api_key:
        raise ConfigurationError("KA_OPENAI_API_KEY must be set to use the OpenAI provider.")
    return OpenAI(api_key=api_key, base_url=base_url or None)


class OpenAILLMProvider:
    """Chat completions via the OpenAI API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = _make_client(api_key, base_url)

    @property
    def model_name(self) -> str:
        return self._model

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
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[m.model_dump() for m in messages],
                temperature=self._temperature if temperature is None else temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI generation failed: {exc}") from exc
        content = response.choices[0].message.content
        if not content:
            raise ProviderError("OpenAI returned an empty completion.")
        return str(content)

    def health_check(self) -> bool:
        try:
            self._client.models.retrieve(self._model)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("openai_health_check_failed", error=str(exc))
            return False


class OpenAIEmbeddingProvider:
    """Embeddings via the OpenAI API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._client = _make_client(api_key, base_url)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return _OPENAI_EMBED_DIMS.get(self._model, 1536)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:
            raise ProviderError(f"OpenAI embedding failed: {exc}") from exc
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
