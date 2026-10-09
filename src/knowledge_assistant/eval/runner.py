"""Evaluation runner: executes the eval set and aggregates metrics."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.eval.metrics import (
    abstention_correct,
    build_faithfulness_messages,
    build_relevance_messages,
    key_fact_coverage,
    parse_yes_no,
    retrieval_metrics,
    routing_correct,
)
from knowledge_assistant.rag.prompts import format_context
from knowledge_assistant.service.pipeline import QueryService

logger = get_logger(__name__)


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.mean(clean), 3) if clean else None


class EvalReport(BaseModel):
    """Aggregated evaluation results."""

    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    judge_enabled: bool = False
    rows: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    per_category: dict[str, Any] = Field(default_factory=dict)

    def save(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "eval_report.json"
        md_path = out_dir / "eval_report.md"
        json_path.write_text(self.model_dump_json(indent=2), "utf-8")
        md_path.write_text(self.to_markdown(), "utf-8")
        return json_path, md_path

    def to_markdown(self) -> str:
        lines = ["# Evaluation Report", "", f"_Generated: {self.generated_at}_", ""]
        lines.append("## Summary")
        for key, value in self.summary.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
        lines.append("## Per-category")
        lines.append("")
        header = "| Category | N | Routing acc | Abstain acc | hit@k | MRR | Faithful | Relevant |"
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|---|")
        for cat, m in self.per_category.items():
            lines.append(
                f"| {cat} | {m.get('n', 0)} | {m.get('routing_accuracy', '-')} | "
                f"{m.get('abstention_accuracy', '-')} | {m.get('hit_at_k', '-')} | "
                f"{m.get('mrr', '-')} | {m.get('faithfulness', '-')} | {m.get('relevance', '-')} |"
            )
        lines.append("")
        lines.append("## Per-question")
        lines.append("")
        q_header = "| id | category | workflow | abstained | routing | abstain | hit@k | coverage |"
        lines.append(q_header)
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in self.rows:
            lines.append(
                f"| {r['id']} | {r['category']} | {r['workflow']} | {r['abstained']} | "
                f"{r.get('routing_correct', '-')} | {r.get('abstention_correct', '-')} | "
                f"{r.get('hit_at_k', '-')} | {r.get('key_fact_coverage', '-')} |"
            )
        return "\n".join(lines)


class EvalRunner:
    """Run the evaluation set against a live :class:`QueryService`."""

    def __init__(self, service: QueryService, use_judge: bool = True) -> None:
        self.service = service
        self.use_judge = use_judge

    def run(self, eval_set_path: Path) -> EvalReport:
        questions = yaml.safe_load(eval_set_path.read_text("utf-8"))
        top_k = self.service.settings.retrieval_top_k
        rows: list[dict[str, Any]] = []

        for item in questions:
            row = self._evaluate_one(item, top_k)
            rows.append(row)
            logger.info("eval_item_done", id=row["id"], workflow=row["workflow"])

        report = EvalReport(judge_enabled=self.use_judge, rows=rows)
        report.summary = self._summarise(rows)
        report.per_category = self._by_category(rows)
        return report

    def _evaluate_one(self, item: dict[str, Any], top_k: int) -> dict[str, Any]:
        question = item["question"]
        expected_docs = item.get("expected_documents", []) or []
        chunks = self.service.retriever.retrieve(question, top_k=top_k)
        retrieved_docs = [c.metadata.document for c in chunks]

        result = self.service.answer(question, mode="auto", top_k=top_k)

        row: dict[str, Any] = {
            "id": item.get("id"),
            "category": item.get("category"),
            "question": question,
            "workflow": result.workflow,
            "abstained": result.abstained,
            "grounded": result.grounded,
            "latency_ms": result.latency_ms,
            "routing_correct": routing_correct(item.get("expected_workflow"), result.workflow),
            "abstention_correct": abstention_correct(
                bool(item.get("expect_abstain", False)), result.abstained
            ),
            "key_fact_coverage": key_fact_coverage(result.answer, item.get("key_facts", [])),
            **retrieval_metrics(expected_docs, retrieved_docs, top_k),
        }

        if self.use_judge and not result.abstained and not item.get("expect_abstain", False):
            context = format_context(chunks)
            llm = self.service.llm
            row["faithfulness"] = parse_yes_no(
                llm.generate(build_faithfulness_messages(question, result.answer, context))
            )
            row["relevance"] = parse_yes_no(
                llm.generate(build_relevance_messages(question, result.answer))
            )
        return row

    @staticmethod
    def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "routing_accuracy": _mean([r.get("routing_correct") for r in rows]),
            "abstention_accuracy": _mean([r.get("abstention_correct") for r in rows]),
            "hit_at_k": _mean([r.get("hit_at_k") for r in rows]),
            "recall_at_k": _mean([r.get("recall_at_k") for r in rows]),
            "mrr": _mean([r.get("mrr") for r in rows]),
            "key_fact_coverage": _mean([r.get("key_fact_coverage") for r in rows]),
            "faithfulness": _mean([r.get("faithfulness") for r in rows]),
            "relevance": _mean([r.get("relevance") for r in rows]),
            "mean_latency_ms": _mean([r.get("latency_ms") for r in rows]),
        }

    @staticmethod
    def _by_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
        categories: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            categories.setdefault(row["category"], []).append(row)
        result: dict[str, Any] = {}
        for cat, items in categories.items():
            result[cat] = {
                "n": len(items),
                "routing_accuracy": _mean([r.get("routing_correct") for r in items]),
                "abstention_accuracy": _mean([r.get("abstention_correct") for r in items]),
                "hit_at_k": _mean([r.get("hit_at_k") for r in items]),
                "mrr": _mean([r.get("mrr") for r in items]),
                "faithfulness": _mean([r.get("faithfulness") for r in items]),
                "relevance": _mean([r.get("relevance") for r in items]),
            }
        return result
