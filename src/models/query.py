"""Query-side data models: QAM decomposition output, re-ranked chunks, and the
final structured response returned by the Query & Retrieval Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QAMResult:
    """Output of the QAM (Query Attribute Modeling) decomposition step.

    metadata_filters is keyed by the same fields the ingester writes into chunk
    metadata (course_id, instructor, lecture_number, topic, upload_year).
    Empty dict means no hard filtering — search the full corpus.

    semantic_element is always present. It is the *rephrased* retrieval intent
    used to embed the query for cosine search. The original raw query is kept
    around for cross-encoder re-ranking and answer generation.
    """

    metadata_filters: dict[str, Any]
    semantic_element: str
    original_query: str


@dataclass
class RankedChunk:
    """A chunk that has survived vector search and cross-encoder re-ranking."""

    chunk_id: str
    video_id: str
    text: str
    start_time: int
    end_time: int
    metadata: dict[str, Any]
    cosine_score: float
    cross_encoder_score: float


@dataclass
class ClipReference:
    """A secondary clip surfaced alongside the primary answer URL.

    Broad queries (e.g. "explain linear regression") often span multiple
    chunks; we synthesise the answer across the top-N chunks and let the
    student jump to any of the moments that contributed.
    """

    snippet_url: str
    video_title: str
    instructor: str
    start_time: int
    end_time: int
    cross_encoder_score: float


@dataclass
class VideoRAGResponse:
    """Final structured response returned to the caller.

    chunk_text and qam_result are kept for transparency / debugging — the demo
    surfaces them so a reviewer can see exactly which transcript segment
    grounded the answer and how the query was decomposed.

    additional_clips holds the secondary chunks (top-2/top-3 from rerank)
    that contributed context to the answer; the UI can render them as
    secondary jumps so the student isn't locked into the primary URL.
    """

    explanation: str
    snippet_url: str
    video_title: str
    instructor: str
    start_time: int
    end_time: int
    chunk_text: str
    qam_result: QAMResult
    cross_encoder_score: float
    low_confidence: bool = False
    additional_clips: list[ClipReference] = field(default_factory=list)
    # Set when chapter-aware override fired — UI shows this so the user can
    # see the URL was anchored to a YouTube-labelled chapter, not a chunk.
    chapter_title: str = ""
