"""Local embedding provider backed by sentence-transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_assistant.core.device import resolve_device
from knowledge_assistant.core.errors import ProviderError
from knowledge_assistant.core.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

# bge-* retrieval models expect this instruction prefixed to queries only.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbeddingProvider:
    """CPU/GPU embeddings using a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = resolve_device(device)
        self._batch_size = batch_size
        self._normalize = normalize
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        # Only bge models benefit from the query instruction.
        self._query_instruction = _BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("loading_embedding_model", model=self._model_name, device=self._device)
                self._model = SentenceTransformer(self._model_name, device=self._device)
                # Method was renamed across sentence-transformers versions.
                get_dim = getattr(
                    self._model, "get_embedding_dimension", None
                ) or self._model.get_sentence_embedding_dimension
                self._dimension = get_dim()
            except Exception as exc:
                raise ProviderError(f"Failed to load embedding model: {exc}") from exc
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
        assert self._dimension is not None
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = model.encode(
            f"{self._query_instruction}{text}",
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.tolist()
