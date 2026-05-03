"""ChromaDB wrapper.

Wrapping ChromaDB behind a thin interface so the production swap to Qdrant is
a single-file change. The rest of the pipeline only ever sees `VectorStore`.

For MVP we use a local PersistentClient. Cosine distance (not the default L2)
because our embeddings are L2-normalised by sentence-transformers and cosine
gives more stable scoring across queries.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from src.config import settings


class VectorStore:
    """Thin facade around a single ChromaDB collection."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "ChromaDB collection ready: {} ({} chunks)",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------ writes
    def add_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            logger.warning("add_chunks called with empty list — nothing to do")
            return
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_video(self, video_id: str) -> int:
        """Remove every chunk belonging to a video. Returns count deleted."""
        existing = self._collection.get(where={"video_id": video_id})
        n = len(existing.get("ids", []))
        if n:
            self._collection.delete(where={"video_id": video_id})
            logger.info("Deleted {} existing chunks for video_id={}", n, video_id)
        return n

    def video_exists(self, video_id: str) -> bool:
        existing = self._collection.get(where={"video_id": video_id}, limit=1)
        return bool(existing.get("ids"))

    # ------------------------------------------------------------------ reads
    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self._collection.count()


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Process-wide cached store. The CLI scripts and tests share this handle."""
    return VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )
