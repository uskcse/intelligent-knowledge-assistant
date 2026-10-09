# Single application image used by the api, ui, and ingest services.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    KA_CONFIG_FILE=configs/default.yaml

WORKDIR /app

# Build tools for packages that compile native extensions, plus curl + CA tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optionally trust a corporate root CA for TLS-intercepting proxies (e.g. Zscaler)
# so pip/uv/HuggingFace downloads verify during the build. certs/ holds only a
# .gitkeep on open networks; drop corp-ca.crt there behind a proxy.
COPY certs/ /usr/local/share/ca-certificates/corp/
RUN update-ca-certificates || true
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    UV_SYSTEM_CERTS=true \
    UV_HTTP_TIMEOUT=900 \
    UV_CONCURRENT_DOWNLOADS=4

RUN pip install uv

# Install dependencies against a stub package so this heavy layer (torch, etc.)
# stays cached independently of application code changes.
COPY pyproject.toml README.md ./
RUN mkdir -p src/knowledge_assistant \
    && printf '__version__ = "0.1.0"\n' > src/knowledge_assistant/__init__.py
# Install CPU-only torch first so the (CPU) image doesn't pull ~3 GB of unused
# CUDA/triton wheels — much smaller and far more reliable behind a slow proxy.
# Only the CPU index is used here (uv gives extra indexes priority, which would
# otherwise resolve the CUDA build from PyPI).
RUN uv pip install --system \
      --index-url https://download.pytorch.org/whl/cpu \
      torch
RUN uv pip install --system -e ".[ui]"

# Bake the local embedding + reranker models (cached unless the script changes).
COPY scripts/download_models.py ./scripts/download_models.py
RUN python scripts/download_models.py

# Models are baked above; force offline at runtime so no HF network/SSL calls occur.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Copy real application code last so edits don't bust the cached layers above.
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src
COPY ui ./ui

# Create a non-root user and the data directory.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "knowledge_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
