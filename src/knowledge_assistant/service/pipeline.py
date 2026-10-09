"""The high-level query service: a single entrypoint for answering questions.

It invokes the compiled LangGraph (which performs routing internally) and maps
the resulting state into a transport-agnostic :class:`QueryResult`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from knowledge_assistant.agent.graph import AgentDeps, build_agent_graph
from knowledge_assistant.agent.state import Step
from knowledge_assistant.agent.tools import DocumentSearchTool
from knowledge_assistant.core.config import Settings, get_settings
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import Citation
from knowledge_assistant.providers.base import LLMProvider
from knowledge_assistant.providers.factory import build_embedding_provider, build_llm_provider
from knowledge_assistant.rag.generator import RagGenerator
from knowledge_assistant.retrieval.retriever import HybridRetriever, build_reranker
from knowledge_assistant.vectorstore.chroma_store import ChromaVectorStore

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = get_logger(__name__)


class QueryResult(BaseModel):
    """Transport-agnostic result of answering a question."""

    answer: str
    abstained: bool
    grounded: bool
    workflow: str
    citations: list[Citation] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    sub_results: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_count: int = 0
    latency_ms: float = 0.0


class QueryService:
    """Facade over the agent graph plus store/provider health."""

    def __init__(
        self,
        settings: Settings,
        store: ChromaVectorStore,
        retriever: HybridRetriever,
        llm: LLMProvider,
        graph: CompiledStateGraph,
    ) -> None:
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.llm = llm
        self.graph = graph

    def warmup(self) -> None:
        """Build the BM25 index so the first request is not penalised."""
        try:
            self.retriever.refresh()
        except Exception as exc:  # noqa: BLE001
            logger.warning("warmup_failed", error=str(exc))

    def answer(
        self,
        question: str,
        mode: str = "auto",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> QueryResult:
        start = time.perf_counter()
        initial = {
            "question": question,
            "mode": mode,
            "top_k": top_k or self.settings.retrieval_top_k,
            "filters": filters,
        }
        final = self.graph.invoke(initial)
        latency_ms = (time.perf_counter() - start) * 1000.0

        result = QueryResult(
            answer=final.get("answer", ""),
            abstained=bool(final.get("abstained", False)),
            grounded=bool(final.get("grounded", False)),
            workflow=final.get("route", "rag"),
            citations=final.get("citations", []),
            steps=final.get("steps", []),
            sub_questions=final.get("sub_questions", []),
            sub_results=final.get("sub_results", []),
            retrieved_count=len(final.get("retrieved", [])),
            latency_ms=round(latency_ms, 1),
        )
        logger.info(
            "query_answered",
            workflow=result.workflow,
            abstained=result.abstained,
            grounded=result.grounded,
            citations=len(result.citations),
            latency_ms=result.latency_ms,
        )
        return result

    def health(self) -> dict[str, bool]:
        return {
            "vector_store": self.store.health_check(),
            "llm": self.llm.health_check(),
        }

    def list_documents(self) -> list[dict[str, Any]]:
        return self.store.list_documents()


def build_query_service(settings: Settings | None = None) -> QueryService:
    """Construct a fully wired :class:`QueryService` from settings."""
    settings = settings or get_settings()
    llm = build_llm_provider(settings)
    embeddings = build_embedding_provider(settings)
    store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
    )
    reranker = build_reranker(settings)
    retriever = HybridRetriever(settings, store, embeddings, reranker)
    generator = RagGenerator(llm)
    search_tool = DocumentSearchTool(retriever)
    deps = AgentDeps(
        settings=settings,
        retriever=retriever,
        generator=generator,
        llm=llm,
        search_tool=search_tool,
    )
    graph = build_agent_graph(deps)
    return QueryService(settings, store, retriever, llm, graph)
