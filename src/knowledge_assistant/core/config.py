"""Typed, layered application configuration.

Precedence (low -> high): field defaults < ``configs/default.yaml`` < ``.env``
< process environment. Secrets (API keys) are only ever read from the
environment, never from the YAML file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

_DEFAULT_CONFIG_FILE = os.getenv("KA_CONFIG_FILE", "configs/default.yaml")


class Settings(BaseSettings):
    """Central application settings."""

    model_config = SettingsConfigDict(
        env_prefix="KA_",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file=_DEFAULT_CONFIG_FILE,
        extra="ignore",
        case_sensitive=False,
    )

    # Runtime
    env: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = False

    # LLM
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: float = 120.0
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_provider: Literal["sentence_transformers", "openai"] = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "auto"
    openai_embedding_model: str = "text-embedding-3-small"

    # Reranker
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 5

    # Vector store
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "knowledge_base"

    # Paths
    corpus_dir: str = "data/corpus"
    data_dir: str = "data"

    # Chunking
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 90
    min_chunk_tokens: int = 50

    # Retrieval
    retrieval_top_k: int = 5
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    hybrid_enabled: bool = True
    bm25_enabled: bool = True

    # Grounding / abstention
    grounding_enabled: bool = True
    grounding_min_score: float = 0.15

    # Agent
    agent_enabled: bool = True
    agent_max_steps: int = 8
    router_mode: Literal["auto", "rag", "agentic"] = "auto"
    query_rewrite_enabled: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_base_url: str = "http://localhost:8080"
    cors_origins: str = "*"

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse the comma-separated CORS origins into a list."""
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def corpus_path(self) -> Path:
        return Path(self.corpus_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def manifest_path(self) -> Path:
        return self.data_path / "manifest.json"

    @property
    def chroma_url(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if Path(_DEFAULT_CONFIG_FILE).is_file():
            sources.append(YamlConfigSettingsSource(settings_cls))
        sources.append(file_secret_settings)
        return tuple(sources)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide settings instance."""
    return Settings()
