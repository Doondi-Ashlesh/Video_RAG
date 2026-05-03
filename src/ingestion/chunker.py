"""Skill: chunk_with_timestamps.

Merge raw transcript segments into overlapping fixed-duration chunks. Each
chunk carries an integer `start_time` (in seconds) that becomes the YouTube
URL anchor downstream — never modify it after this stage.
"""
from __future__ import annotations

import re

from loguru import logger

from src.models import RawTranscriptSegment, TranscriptChunk

# Strip [Music], [Applause], [Laughter] etc.
_FILLER_RE = re.compile(r"\[.*?\]")
_WHITESPACE_RE = re.compile(r"\s+")


def chunk_transcript(
    segments: list[RawTranscriptSegment],
    video_id: str,
    window_seconds: int = 45,
    overlap_seconds: int = 10,
    min_words: int = 20,
) -> list[TranscriptChunk]:
    """Window the transcript into overlapping chunks.

    Args:
        segments: ordered raw segments from youtube-transcript-api.
        video_id: used to namespace chunk_ids.
        window_seconds: chunk duration.
        overlap_seconds: overlap between adjacent chunks (must be < window).
        min_words: chunks shorter than this are dropped (silence / music).
    """
    if not segments:
        return []
    if overlap_seconds >= window_seconds:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be < window_seconds ({window_seconds})"
        )

    total_duration = segments[-1].start + segments[-1].duration
    step = window_seconds - overlap_seconds

    chunks: list[TranscriptChunk] = []
    chunk_index = 0
    t = 0.0

    while t < total_duration:
        window_end = t + window_seconds
        window_segs = [s for s in segments if t <= s.start < window_end]

        if not window_segs:
            t += step
            continue

        raw_text = " ".join(s.text for s in window_segs)
        clean_text = _FILLER_RE.sub("", raw_text).strip()
        clean_text = _WHITESPACE_RE.sub(" ", clean_text)

        if len(clean_text.split()) >= min_words:
            # Preserve the per-segment (cleaned_text, start) list so the
            # query-time sentence_anchor skill can refine URL timestamps
            # to the actual moment of the answer rather than the chunk boundary.
            seg_pairs: list[tuple[str, float]] = []
            for s in window_segs:
                seg_text = _WHITESPACE_RE.sub(" ", _FILLER_RE.sub("", s.text)).strip()
                if seg_text:
                    seg_pairs.append((seg_text, float(s.start)))

            chunks.append(
                TranscriptChunk(
                    chunk_id=f"{video_id}_chunk_{chunk_index}",
                    video_id=video_id,
                    text=clean_text,
                    start_time=int(t),
                    end_time=int(min(window_end, total_duration)),
                    segments=seg_pairs,
                )
            )
            chunk_index += 1

        t += step

    logger.info(
        "Built {} chunks from {} segments (video_id={})",
        len(chunks),
        len(segments),
        video_id,
    )
    return chunks
