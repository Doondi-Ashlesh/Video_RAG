"""Centralised settings loaded from .env via pydantic-settings.

Single source of truth for model names, paths, and pipeline parameters. Every
other module imports `settings` rather than reading os.environ directly.
"""
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # NVIDIA NIM (free hosted endpoint, OpenAI-compatible). Used for QAM
    # decomposition and grounded answer generation.
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL",
    )
    qam_model: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1",
        alias="QAM_MODEL",
    )
    answer_model: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1",
        alias="ANSWER_MODEL",
    )

    # Embedding + reranker (run locally via sentence-transformers — no API cost)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL",
    )

    # ChromaDB
    chroma_persist_dir: Path = Field(default=Path("./chroma_db"), alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="lecture_chunks", alias="CHROMA_COLLECTION")

    # Retrieval / chunking parameters
    top_k_retrieval: int = Field(default=30, alias="TOP_K_RETRIEVAL")
    chunk_window_seconds: int = Field(default=90, alias="CHUNK_WINDOW_SECONDS")
    chunk_overlap_seconds: int = Field(default=20, alias="CHUNK_OVERLAP_SECONDS")
    min_chunk_words: int = Field(default=20, alias="MIN_CHUNK_WORDS")

    # Confidence threshold (cross-encoder logit). ms-marco-MiniLM scores are
    # roughly in [-10, +10]; below -2 the chunk is usually a tangential mention.
    rerank_confidence_threshold: float = Field(
        default=-2.0, alias="RERANK_CONFIDENCE_THRESHOLD"
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


settings = Settings()
