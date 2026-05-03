"""Skill: embed_and_store.

Embeds chunk text with sentence-transformers and writes the result into
ChromaDB along with the lecturer-corpus metadata used by QAM filtering at
query time.

The embedding model is loaded once via lru_cache so re-ingesting multiple
videos in one process doesn't pay the load cost repeatedly.

Per-chunk transcript segments are JSON-serialised into the `segments`
metadata field so the query-time sentence_anchor skill can refine the URL
timestamp without an extra DB round-trip or a transcript re-fetch.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.db import VectorStore
from src.models import TranscriptChunk, VideoMeta


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    name = model_name or settings.embedding_model
    logger.info("Loading embedding model: {}", name)
    return SentenceTransformer(name)


def _find_chapter(start_time: int, chapters: list[dict]) -> dict | None:
    """Return the chapter whose [start, end) range contains start_time."""
    for ch in chapters:
        if ch["start"] <= start_time < ch["end"]:
            return ch
    return None


def _build_metadata(chunk: TranscriptChunk, meta: VideoMeta) -> dict[str, Any]:
    """Construct the metadata payload stored alongside the embedding.

    ChromaDB metadata values must be primitive types (str / int / float / bool).
    None values are dropped — ChromaDB rejects nulls in `where` clauses.
    """
    payload: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "video_id": chunk.video_id,
        "start_time": chunk.start_time,
        "end_time": chunk.end_time,
        "title": meta.title,
        "instructor": meta.instructor,
        "upload_date": meta.upload_date,
        "upload_year": meta.upload_year,
        # JSON-encoded list of [text, start] pairs — read by sentence_anchor
        # at query time to refine the URL timestamp.
        "segments": json.dumps(chunk.segments, separators=(",", ":")),
    }
    if meta.course_id:
        payload["course_id"] = meta.course_id
    if meta.lecture_number is not None:
        payload["lecture_number"] = int(meta.lecture_number)
    if meta.topic:
        payload["topic"] = meta.topic

    # Attach the YouTube chapter this chunk falls within (if any). The chapter
    # boundaries become the URL anchor when the QAM topic matches the chapter
    # title — much more accurate than chunk-level timestamps for broad
    # explanatory queries.
    chapter = _find_chapter(chunk.start_time, meta.chapters)
    if chapter:
        payload["chapter_title"] = chapter["title"]
        payload["chapter_start"] = int(chapter["start"])
        payload["chapter_end"] = int(chapter["end"])
    return payload


def embed_and_store(
    chunks: list[TranscriptChunk],
    video_meta: VideoMeta,
    store: VectorStore,
    model: SentenceTransformer | None = None,
    batch_size: int = 32,
) -> int:
    """Embed each chunk and write it to the vector store.

    Returns the number of chunks stored.
    """
    if not chunks:
        logger.warning("embed_and_store called with no chunks")
        return 0

    model = model or get_embedding_model()
    texts = [c.text for c in chunks]

    logger.info("Embedding {} chunks (batch_size={})", len(chunks), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    store.add_chunks(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[_build_metadata(c, video_meta) for c in chunks],
    )
    logger.success(
        "Stored {} chunks for {} ({})",
        len(chunks),
        video_meta.title,
        video_meta.video_id,
    )
    return len(chunks)
