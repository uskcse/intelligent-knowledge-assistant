"""Assemble the LangGraph state machine for the agentic workflow."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from knowledge_assistant.agent.nodes import (
    agent_retrieve_node,
    decompose_node,
    router_node,
    select_route,
    simple_rag_node,
    synthesize_node,
)
from knowledge_assistant.agent.state import AgentState
from knowledge_assistant.agent.tools import DocumentSearchTool
from knowledge_assistant.core.config import Settings
from knowledge_assistant.providers.base import LLMProvider
from knowledge_assistant.rag.generator import RagGenerator
from knowledge_assistant.retrieval.retriever import HybridRetriever

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


@dataclass
class AgentDeps:
    """Dependencies injected into graph nodes."""

    settings: Settings
    retriever: HybridRetriever
    generator: RagGenerator
    llm: LLMProvider
    search_tool: DocumentSearchTool


def build_agent_graph(deps: AgentDeps) -> CompiledStateGraph:
    """Build and compile the routing + RAG + agentic graph."""
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("router", partial(router_node, deps=deps))
    builder.add_node("simple_rag", partial(simple_rag_node, deps=deps))
    builder.add_node("decompose", partial(decompose_node, deps=deps))
    builder.add_node("agent_retrieve", partial(agent_retrieve_node, deps=deps))
    builder.add_node("synthesize", partial(synthesize_node, deps=deps))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        select_route,
        {"rag": "simple_rag", "agentic": "decompose"},
    )
    builder.add_edge("simple_rag", END)
    builder.add_edge("decompose", "agent_retrieve")
    builder.add_edge("agent_retrieve", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()
