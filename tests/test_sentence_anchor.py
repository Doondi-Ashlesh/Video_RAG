"""Tests for the sentence_anchor skill.

Mocks the cross-encoder so the suite stays offline. Verifies:
  - sub-chunk windowing groups raw segments into ~window-second windows
  - the anchor returns the start of the highest-scoring sub-chunk
  - all defined fallback paths (missing metadata, single sub-chunk, model fail)
    return the chunk's own start_time
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from src.models import RankedChunk
from src.retrieval import sentence_anchor


class _DummyCE:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        import numpy as np

        return np.array(self._scores[: len(pairs)], dtype=float)


def _winner(segments: list[tuple[str, float]] | None, start_time: int = 0) -> RankedChunk:
    md: dict[str, Any] = {
        "chunk_id": "VID_chunk_0",
        "video_id": "VID",
        "start_time": start_time,
        "end_time": start_time + 45,
    }
    if segments is not None:
        md["segments"] = json.dumps(segments)
    return RankedChunk(
        chunk_id="VID_chunk_0",
        video_id="VID",
        text="dummy",
        start_time=start_time,
        end_time=start_time + 45,
        metadata=md,
        cosine_score=0.5,
        cross_encoder_score=3.0,
    )


# ----------------------------------------------------- sub-chunk windowing
def test_subchunks_group_into_window_seconds() -> None:
    # 30 segments at 2s spacing across 60s should produce ~4 sub-chunks at 15s windows
    segs = [(f"word{i} word{i+1} word{i+2} word{i+3} word{i+4} word{i+5}", float(i * 2)) for i in range(30)]
    out = sentence_anchor._make_subchunks(segs, window_seconds=15.0)
    assert len(out) >= 3
    # Each sub-chunk should have at least min_words content
    assert all(len(text.split()) >= 6 for text, _ in out)
    # Sub-chunk starts must be monotonically non-decreasing
    starts = [s for _, s in out]
    assert starts == sorted(starts)


def test_subchunks_empty_input_returns_empty_list() -> None:
    assert sentence_anchor._make_subchunks([]) == []


# --------------------------------------------------------- anchor behaviour
def test_anchor_picks_highest_scoring_subchunk(monkeypatch: pytest.MonkeyPatch) -> None:
    # Three sub-chunks worth of content (each ~15s wide).
    segs: list[tuple[str, float]] = []
    for i in range(8):
        segs.append((f"early word a b c d e f g h ({i})", float(i)))     # 0-7s    -> sub-chunk 1
    for i in range(8):
        segs.append((f"middle relevant deep explanation example ({i})", float(15 + i)))  # 15-22s -> sub-chunk 2
    for i in range(8):
        segs.append((f"late tangent unrelated content ({i})", float(30 + i)))  # 30-37s -> sub-chunk 3

    winner = _winner(segs, start_time=0)

    # Mock the cross-encoder so the *middle* window scores highest
    monkeypatch.setattr(sentence_anchor, "get_reranker", lambda: _DummyCE([0.1, 5.0, -1.0]))

    refined = sentence_anchor.anchor_to_segment("explain the deep concept", winner, window_seconds=15.0)

    # Refined start should land in the middle window (around t=15s), not at chunk boundary t=0
    assert 14 <= refined <= 22


def test_anchor_falls_back_to_chunk_start_when_no_segments_metadata() -> None:
    winner = _winner(segments=None, start_time=120)
    assert sentence_anchor.anchor_to_segment("anything", winner) == 120


def test_anchor_falls_back_to_chunk_start_with_only_one_subchunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two segments, both within a 15s window -> only one sub-chunk -> fallback
    segs = [("hello world this is a short transcript", 100.0)]
    winner = _winner(segs, start_time=100)
    refined = sentence_anchor.anchor_to_segment("anything", winner, window_seconds=15.0)
    assert refined == 100


def test_anchor_falls_back_when_cross_encoder_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segs = []
    for i in range(20):
        segs.append((f"word filler one two three four ({i})", float(i * 2)))
    winner = _winner(segs, start_time=0)

    def _boom() -> Any:
        raise RuntimeError("model files missing")

    monkeypatch.setattr(sentence_anchor, "get_reranker", _boom)
    refined = sentence_anchor.anchor_to_segment("anything", winner, window_seconds=15.0)
    assert refined == winner.start_time


# ----------------------------------------------- segments JSON parse path
def test_parse_segments_handles_dict_string_and_list_inputs() -> None:
    # JSON-string form (real ChromaDB metadata path)
    raw = json.dumps([("hello", 1.5), ("world", 3.2)])
    parsed = sentence_anchor._parse_segments(raw)
    assert parsed == [("hello", 1.5), ("world", 3.2)]

    # Pass-through list form
    assert sentence_anchor._parse_segments([("a", 0.0), ("b", 1.0)]) == [("a", 0.0), ("b", 1.0)]

    # Bad inputs -> empty list, never raise
    assert sentence_anchor._parse_segments(None) == []
    assert sentence_anchor._parse_segments("not json") == []
    assert sentence_anchor._parse_segments("") == []
