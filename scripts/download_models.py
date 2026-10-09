"""Pre-download local models into the image/cache for offline operation.

Run during the Docker build so the embedding and reranker models are baked in and
no download is required at runtime. Reads model names from the environment (with
defaults) so it has no dependency on the application package or config file, which
keeps the Docker layer order clean (dependency install + model bake stay cached
independently of application code changes).
"""

from __future__ import annotations

import os


def main() -> None:
    embedding_model = os.getenv("KA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model = os.getenv("KA_RERANKER_MODEL", "BAAI/bge-reranker-base")
    reranker_enabled = os.getenv("KA_RERANKER_ENABLED", "true").lower() in {"1", "true", "yes"}

    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"Downloading embedding model: {embedding_model}")
    SentenceTransformer(embedding_model, device="cpu")

    if reranker_enabled:
        print(f"Downloading reranker model: {reranker_model}")
        CrossEncoder(reranker_model, device="cpu")

    print("Model download complete.")


if __name__ == "__main__":
    main()
