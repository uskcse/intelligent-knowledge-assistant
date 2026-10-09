"""Chainlit demo UI for the Knowledge Assistant.

A thin client over the REST API: it sends questions to ``/ask`` and renders the
answer, the workflow used (RAG vs agentic), and expandable source citations.

Run with:  chainlit run ui/chainlit_app.py
"""

from __future__ import annotations

import os

import chainlit as cl
import httpx
from chainlit.input_widget import Select, Slider

API_URL = os.getenv("KA_API_URL", "http://localhost:8080")
REQUEST_TIMEOUT = float(os.getenv("KA_UI_TIMEOUT", "180"))

WELCOME = (
    "### Intelligent Knowledge Assistant\n"
    "Ask questions about **RAG architecture**, **vector databases**, or "
    "**agentic AI frameworks**.\n\n"
    "- Simple questions use a single-step **RAG** flow.\n"
    "- Comparative / multi-step questions use the **agentic** flow.\n"
    "- Questions outside the corpus are answered with an honest *I don't know*.\n"
)


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""
    lines = ["\n\n---\n**Sources**"]
    for i, src in enumerate(sources, start=1):
        doc = src.get("document", "?")
        page = src.get("page", 0)
        score = src.get("score", 0.0)
        snippet = (src.get("snippet", "") or "").strip()
        lines.append(f"\n**[{i}]** `{doc}` — p.{page} (score {score:.3f})")
        if snippet:
            lines.append(f"\n> {snippet}")
    return "\n".join(lines)


def _format_meta(data: dict) -> str:
    workflow = data.get("workflow", "rag")
    badge = "🧭 Agentic (multi-step)" if workflow == "agentic" else "⚡ RAG (single-step)"
    latency = data.get("latency_ms", 0.0)
    grounded = "grounded" if data.get("grounded") else "not grounded"
    meta = f"\n\n`{badge}` · `{grounded}` · `{latency:.0f} ms`"
    subs = data.get("sub_questions") or []
    if subs:
        meta += "\n\n*Sub-questions:*\n" + "\n".join(f"- {s}" for s in subs)
    return meta


@cl.on_chat_start
async def start() -> None:
    await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Workflow",
                values=["auto", "rag", "agentic"],
                initial_index=0,
            ),
            Slider(id="top_k", label="Top-K chunks", initial=5, min=1, max=15, step=1),
        ]
    ).send()
    cl.user_session.set("mode", "auto")
    cl.user_session.set("top_k", 5)
    await cl.Message(content=WELCOME).send()


@cl.on_settings_update
async def update_settings(settings: dict) -> None:
    cl.user_session.set("mode", settings.get("mode", "auto"))
    cl.user_session.set("top_k", int(settings.get("top_k", 5)))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    mode = cl.user_session.get("mode") or "auto"
    top_k = cl.user_session.get("top_k") or 5
    thinking = cl.Message(content="")
    await thinking.send()

    payload = {"question": message.content, "mode": mode, "top_k": int(top_k)}
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(f"{API_URL}/ask", json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        thinking.content = f"API error ({exc.response.status_code}). Please try again."
        await thinking.update()
        return
    except httpx.HTTPError as exc:
        thinking.content = f"Could not reach the API at {API_URL}: {exc}"
        await thinking.update()
        return

    body = data.get("answer", "")
    body += _format_meta(data)
    body += _format_sources(data.get("sources", []))
    thinking.content = body
    await thinking.update()
