"""Optional RAGAS-based evaluation adapter.

RAGAS and its LLM/embedding backends are imported lazily so the core evaluation
harness has no hard dependency on them. Enable with the ``ragas`` extra:

    pip install 'knowledge-assistant[ragas]'

RAGAS is wired to the same local models (Ollama + sentence-transformers) so no
external API calls are made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from knowledge_assistant.core.config import get_settings
from knowledge_assistant.core.errors import ConfigurationError
from knowledge_assistant.service.pipeline import QueryService


def evaluate_with_ragas(service: QueryService, eval_set_path: Path) -> dict[str, Any]:
    """Compute RAGAS faithfulness / answer-relevancy / context-precision.

    Only answerable questions (``expect_abstain: false``) are scored.
    """
    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ConfigurationError(
            "RAGAS requires the 'ragas' extra: pip install 'knowledge-assistant[ragas]'"
        ) from exc

    settings = get_settings()
    items = yaml.safe_load(eval_set_path.read_text("utf-8"))

    questions, answers, contexts, references = [], [], [], []
    for item in items:
        if item.get("expect_abstain", False):
            continue
        question = item["question"]
        chunks = service.retriever.retrieve(question, top_k=settings.retrieval_top_k)
        result = service.answer(question, mode="auto")
        questions.append(question)
        answers.append(result.answer)
        contexts.append([c.chunk.text for c in chunks])
        references.append(" ".join(item.get("key_facts", [])) or result.answer)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "reference": references,
        }
    )

    judge_llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    judge_emb = OllamaEmbeddings(model=settings.ollama_model, base_url=settings.ollama_base_url)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_emb,
    )
    return dict(result)
