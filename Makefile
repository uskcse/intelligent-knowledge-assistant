# Knowledge Assistant - developer convenience targets
.DEFAULT_GOAL := help
SHELL := /bin/bash

PY ?= python
UV ?= uv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Create venv and install project with dev+ui extras
	$(UV) venv
	$(UV) pip install -e ".[dev,ui,ragas]"

.PHONY: lint
lint: ## Run ruff + mypy
	$(UV) run ruff check src tests scripts
	$(UV) run mypy src

.PHONY: fmt
fmt: ## Auto-format / fix lint
	$(UV) run ruff check --fix src tests scripts
	$(UV) run ruff format src tests scripts

.PHONY: test
test: ## Run unit + e2e tests (no live services needed)
	$(UV) run pytest

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(UV) run pytest --cov --cov-report=term-missing

.PHONY: ingest
ingest: ## Ingest the corpus into Chroma (requires chroma + embeddings)
	$(UV) run ka-ingest run

.PHONY: ingest-rebuild
ingest-rebuild: ## Rebuild the knowledge base from scratch
	$(UV) run ka-ingest run --rebuild

.PHONY: evaluate
evaluate: ## Run the evaluation harness
	$(UV) run ka-eval run

.PHONY: api
api: ## Run the API locally (requires chroma + ollama)
	$(UV) run uvicorn knowledge_assistant.api.app:app --host 0.0.0.0 --port 8080 --reload

.PHONY: ui
ui: ## Run the Chainlit UI locally
	$(UV) run chainlit run ui/chainlit_app.py --host 0.0.0.0 --port 8501

.PHONY: up
up: ## Start the full stack (auto-detects GPU/CPU)
	./scripts/run.sh

.PHONY: up-cpu
up-cpu: ## Start the full stack, CPU-only (always portable)
	docker compose up --build

.PHONY: down
down: ## Stop the stack and remove volumes
	docker compose down -v

.PHONY: compose-ingest
compose-ingest: ## Run ingestion as a one-shot compose job
	docker compose run --rm ingest
