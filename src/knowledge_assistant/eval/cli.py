"""CLI entrypoint for evaluation (``ka-eval``)."""

from __future__ import annotations

from pathlib import Path

import typer

from knowledge_assistant.core.config import get_settings
from knowledge_assistant.core.logging import configure_logging, get_logger
from knowledge_assistant.eval.runner import EvalRunner
from knowledge_assistant.service.pipeline import build_query_service

app = typer.Typer(help="Evaluate retrieval, answer quality, and agent behaviour.")
logger = get_logger(__name__)


@app.callback()
def main() -> None:
    """Evaluation commands (keeps 'run' as an explicit subcommand)."""


@app.command()
def run(
    eval_set: Path = typer.Option(
        Path("configs/eval_set.yaml"), "--eval-set", help="Path to the eval YAML."
    ),
    out_dir: Path = typer.Option(Path("eval_results"), "--out", help="Output directory."),
    judge: bool = typer.Option(
        True, "--judge/--no-judge", help="Use the LLM-as-judge for answer quality."
    ),
    ragas: bool = typer.Option(False, "--ragas", help="Also run the optional RAGAS metrics."),
) -> None:
    """Run the evaluation harness and write a JSON + Markdown report."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    service = build_query_service(settings)
    service.warmup()

    runner = EvalRunner(service, use_judge=judge)
    report = runner.run(eval_set)
    json_path, md_path = report.save(out_dir)

    typer.echo("\n=== Evaluation summary ===")
    for key, value in report.summary.items():
        typer.echo(f"  {key}: {value}")
    typer.echo(f"\nReports written to:\n  {json_path}\n  {md_path}")

    if ragas:
        from knowledge_assistant.eval.ragas_adapter import evaluate_with_ragas

        try:
            scores = evaluate_with_ragas(service, eval_set)
            typer.echo(f"\nRAGAS: {scores}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"\nRAGAS skipped: {exc}")


if __name__ == "__main__":
    app()
