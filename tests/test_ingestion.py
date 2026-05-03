"""Smoke tests for the ingestion side: video-id parsing and chunker behaviour."""
from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_transcript
from src.ingestion.transcript import extract_video_id
from src.models import RawTranscriptSegment


# ----------------------------------------------------------- extract_video_id
@pytest.mark.parametrize(
    "url,expected",
    [
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id_valid(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


def test_extract_video_id_invalid() -> None:
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/foo")


# ---------------------------------------------------------------- chunker
def _seg(text: str, start: float, duration: float = 2.0) -> RawTranscriptSegment:
    return RawTranscriptSegment(text=text, start=start, duration=duration)


def test_chunker_produces_overlapping_windows() -> None:
    # Synthesise 120 seconds of content with one segment per 2s.
    sentence = "This is a synthetic transcript segment used for chunker testing."
    segments = [_seg(sentence, start=i * 2.0) for i in range(60)]

    chunks = chunk_transcript(
        segments,
        video_id="VID12345abc",
        window_seconds=45,
        overlap_seconds=10,
        min_words=20,
    )

    assert len(chunks) >= 2
    # start_time must be int and non-negative
    assert all(isinstance(c.start_time, int) and c.start_time >= 0 for c in chunks)
    # chunk_ids must be unique and follow the {video_id}_chunk_{i} convention
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(c.chunk_id.startswith("VID12345abc_chunk_") for c in chunks)
    # Adjacent chunks should advance by (window - overlap) seconds
    if len(chunks) >= 2:
        assert chunks[1].start_time - chunks[0].start_time == 35


def test_chunker_preserves_segments_per_chunk() -> None:
    """The new sentence_anchor skill needs the per-segment list on each chunk."""
    sentence = "This is a synthetic transcript segment used for chunker testing."
    segments = [_seg(sentence, start=i * 2.0) for i in range(60)]

    chunks = chunk_transcript(
        segments, video_id="VIDsegments", window_seconds=45, overlap_seconds=10
    )

    assert chunks
    first = chunks[0]
    # Segments are list[tuple[str, float]] with start times within the chunk window
    assert isinstance(first.segments, list)
    assert all(
        isinstance(s, tuple)
        and len(s) == 2
        and isinstance(s[0], str)
        and isinstance(s[1], float)
        for s in first.segments
    )
    # Segment starts must be inside the chunk's time window
    assert all(
        first.start_time <= start <= first.end_time + 5  # +5 grace for the last seg
        for _text, start in first.segments
    )
    # And at least a few segments per chunk (a 45s window with 2s segments has ~22)
    assert len(first.segments) >= 5


def test_chunker_drops_short_segments() -> None:
    short_segments = [_seg("[Music]", start=0.0, duration=5.0)]
    chunks = chunk_transcript(short_segments, video_id="VID00000000", min_words=20)
    assert chunks == []


def test_chunker_strips_filler_tags() -> None:
    sentence = "We will now derive the gradient of the loss with respect to the weights."
    segments = [_seg(f"[Music] {sentence}", start=i * 2.0) for i in range(30)]
    chunks = chunk_transcript(segments, video_id="VID11111111", min_words=10)
    assert chunks
    assert "[Music]" not in chunks[0].text
