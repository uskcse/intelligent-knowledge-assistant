"""Agentic workflow built on LangGraph.

A router node decides between a single-step RAG path and a multi-step agentic
path (decompose -> multi-retrieve -> synthesize). Both paths converge on a
grounded, cited answer with full step tracking for transparency.
"""

from knowledge_assistant.agent.graph import AgentDeps, build_agent_graph
from knowledge_assistant.agent.state import AgentState, Step

__all__ = ["AgentDeps", "AgentState", "Step", "build_agent_graph"]
