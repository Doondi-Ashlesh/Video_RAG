"""Skill: vector_search.

Embed the QAM semantic_element, run a metadata-filtered cosine search against
the vector store, and return the top-k candidate chunks. If the metadata
filter is too restrictive (zero hits), retry without it and log a warning.
"""
from __future__ import annotations

from typing import Any

from loguru import logger
from sentence_transformers import SentenceTransformer

from src.db import VectorStore
from src.ingestion.embedder import get_embedding_model


def vector_search(
    semantic_element: str,
    where_filter: dict[str, Any] | None,
    store: VectorStore,
    model: SentenceTransformer | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return up to top_k candidate chunks ranked by cosine similarity."""
    model = model or get_embedding_model()
    query_embedding = model.encode(
        [semantic_element], normalize_embeddings=True
    ).tolist()

    if where_filter:
        logger.info("Applying metadata filter: {}", where_filter)
    else:
        logger.info("No metadata filter — searching full corpus")

    results = store.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
    )
    chunks = _parse_results(results)

    if not chunks and where_filter is not None:
        logger.warning(
            "No results with metadata filter — retrying without filter"
        )
        results = store.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=None,
        )
        chunks = _parse_results(results)

    logger.info("Vector search returned {} candidates", len(chunks))
    return chunks


def _parse_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the ChromaDB query response into a list of chunk dicts."""
    if not results.get("documents") or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks: list[dict[str, Any]] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine distance is in [0, 2]; similarity = 1 - distance.
        chunks.append(
            {
                "text": doc,
                "metadata": meta,
                "cosine_score": float(1 - dist),
            }
        )
    return chunks
