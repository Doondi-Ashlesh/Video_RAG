"""Ingestion-side data models: raw transcript segments, video metadata, and
the final timestamped chunks that get embedded into the vector store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawTranscriptSegment:
    """One segment as returned by youtube-transcript-api."""

    text: str
    start: float       # seconds since video start
    duration: float    # seconds


@dataclass
class VideoMeta:
    """Metadata about an ingested lecture video.

    The `course_id`, `instructor`, `lecture_number`, and `topic` fields drive
    QAM hard-filtering at query time. They replace the original spec's
    `channel_name` / `topic_category` because in a single-lecturer corpus those
    coarse fields carry almost no signal.
    """

    video_id: str
    title: str
    instructor: str
    upload_date: str            # YYYY-MM-DD
    duration_seconds: int

    # Lecturer-corpus filter fields
    course_id: Optional[str] = None        # e.g. "CS231"
    lecture_number: Optional[int] = None   # e.g. 5
    topic: Optional[str] = None            # e.g. "backpropagation"

    # YouTube chapter markers (the lecturer-defined topical segmentation).
    # Each entry: {"start": int_seconds, "end": int_seconds, "title": str}.
    # When present, query-time chapter matching can override the chunk-level
    # URL anchor with the chapter's official start/end — which is dramatically
    # better for broad queries (e.g. "explain linear regression" lands at the
    # start of the "Linear Regression Algorithm" chapter, not mid-derivation).
    chapters: list[dict] = field(default_factory=list)

    @property
    def upload_year(self) -> int:
        return int(self.upload_date[:4])


@dataclass
class TranscriptChunk:
    """A timestamped chunk produced by the chunker.

    `start_time` is the chunk-level anchor used as the URL fallback. It is set
    once here and must never be modified by downstream code.

    `segments` is the ordered list of (text, start) pairs from the raw
    transcript that were merged into this chunk. The query-time
    `sentence_anchor` skill uses this list to refine the URL timestamp from
    the chunk boundary (potentially up to 45s before the answer) to the
    actual moment where the answer is spoken.
    """

    chunk_id: str
    video_id: str
    text: str
    start_time: int   # integer seconds — chunk window start (URL fallback)
    end_time: int
    segments: list[tuple[str, float]] = field(default_factory=list)
