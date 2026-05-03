"""Skill: sentence_anchor.

After the chunk-level cross-encoder rerank picks the winning chunk, this skill
refines the URL timestamp from the chunk boundary (potentially up to 45s before
the answer) to the actual moment where the answer is spoken.

How it works:
1. The winning chunk carries a JSON-encoded list of (segment_text, start)
   pairs from the raw transcript — written at ingestion time.
2. We group consecutive raw segments into sub-chunk windows (~15s each) so
   the cross-encoder has enough context per scoring.
3. We cross-encode the original user query against each sub-chunk window.
4. We return the best window's start time as the refined URL timestamp.

Falls back to the chunk-level start_time if:
- segments metadata is missing (older indices)
- segments list is empty / can't be parsed
- only a single sub-chunk window can be formed
- the cross-encoder is unavailable (delegated to get_reranker fallback)
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

from src.models import RankedChunk
from src.retrieval.reranker import get_reranker


def _make_subchunks(
    segments: list[tuple[str, float]],
    window_seconds: float = 15.0,
    min_words: int = 6,
) -> list[tuple[str, float]]:
    """Group consecutive raw segments into ~window_seconds sub-chunks.

    Each sub-chunk has enough context for the cross-encoder to score against
    the query, while still being fine-grained enough that the start time of
    the winner usefully narrows down where in the chunk the answer sits.
    """
    if not segments:
        return []

    subchunks: list[tuple[str, float]] = []
    current_text: list[str] = []
    current_start: float = float(segments[0][1])

    for text, start in segments:
        start = float(start)
        if start - current_start >= window_seconds and current_text:
            joined = " ".join(current_text).strip()
            if len(joined.split()) >= min_words:
                subchunks.append((joined, current_start))
            current_text = [text]
            current_start = start
        else:
            current_text.append(text)

    if current_text:
        joined = " ".join(current_text).strip()
        if len(joined.split()) >= min_words:
            subchunks.append((joined, current_start))

    return subchunks


def _parse_segments(raw: Any) -> list[tuple[str, float]]:
    """Coerce metadata's segments field into a list[tuple[str, float]]."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    else:
        data = raw

    out: list[tuple[str, float]] = []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), float(item[1])))
    return out


def anchor_to_segment(
    query: str,
    winner: RankedChunk,
    window_seconds: float = 15.0,
) -> int:
    """Return the refined URL start_time for the winning chunk.

    Falls back to winner.start_time on any failure path so the caller can
    treat this as a strict-improvement step that never breaks the URL.
    """
    segments = _parse_segments(winner.metadata.get("segments"))
    if not segments:
        logger.debug(
            "sentence_anchor: no segments in metadata; using chunk start_time={}s",
            winner.start_time,
        )
        return winner.start_time

    subchunks = _make_subchunks(segments, window_seconds=window_seconds)
    if len(subchunks) <= 1:
        logger.debug(
            "sentence_anchor: only {} sub-chunk(s); using chunk start_time={}s",
            len(subchunks),
            winner.start_time,
        )
        return winner.start_time

    try:
        model = get_reranker()
    except Exception as e:
        logger.warning(
            "sentence_anchor: cross-encoder unavailable ({}); using chunk start_time", e
        )
        return winner.start_time

    pairs = [(query, text) for text, _ in subchunks]
    scores = model.predict(pairs).tolist()

    # Log the full breakdown so it's visible during demo --debug runs.
    for (text, start), score in zip(subchunks, scores):
        preview = text[:60].replace("\n", " ")
        logger.debug(
            "sentence_anchor candidate: start={}s score={:.3f}  {!r}",
            int(start),
            score,
            preview,
        )

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    refined = int(subchunks[best_idx][1])
    logger.info(
        "sentence_anchor refined start_time: chunk={}s -> segment={}s (chosen sub-chunk score={:.3f})",
        winner.start_time,
        refined,
        scores[best_idx],
    )
    return refined
