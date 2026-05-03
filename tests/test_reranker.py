"""Tests for the cross-encoder rerank skill.

We don't load the real cross-encoder in unit tests (slow + downloads weights).
We monkey-patch get_reranker to inject a dummy model with deterministic scores.
"""
from __future__ import annotations

from typing import Any

import pytest

from src.retrieval import reranker


class _DummyCE:
    """Stand-in cross-encoder that returns a fixed list of scores."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        # Return scores aligned to candidate order. Mimics .tolist() shape via list.
        import numpy as np

        return np.array(self._scores[: len(pairs)], dtype=float)


def _candidate(chunk_id: str, text: str, cosine: float, start: int) -> dict[str, Any]:
    return {
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "video_id": "VID",
            "start_time": start,
            "end_time": start + 45,
        },
        "cosine_score": cosine,
    }


def test_rerank_returns_highest_cross_encoder_score(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [
        _candidate("VID_chunk_0", "passing mention of RAG", cosine=0.85, start=0),
        _candidate("VID_chunk_1", "deep explanation of RAG step by step", cosine=0.82, start=120),
        _candidate("VID_chunk_2", "unrelated content about pandas", cosine=0.78, start=240),
    ]
    # Cross-encoder prefers chunk_1 even though chunk_0 has the higher cosine score.
    monkeypatch.setattr(reranker, "get_reranker", lambda: _DummyCE([0.1, 5.0, -2.0]))

    winner = reranker.rerank("explain RAG", candidates)
    assert winner is not None
    assert winner.chunk_id == "VID_chunk_1"
    assert winner.start_time == 120
    assert winner.cross_encoder_score == pytest.approx(5.0)


def test_rerank_returns_none_for_empty_candidates() -> None:
    assert reranker.rerank("anything", []) is None


def test_rerank_falls_back_to_cosine_on_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [
        _candidate("VID_chunk_0", "A", cosine=0.5, start=0),
        _candidate("VID_chunk_1", "B", cosine=0.9, start=45),
    ]

    def _boom() -> Any:
        raise RuntimeError("model files missing")

    monkeypatch.setattr(reranker, "get_reranker", _boom)
    winner = reranker.rerank("query", candidates)
    assert winner is not None
    assert winner.chunk_id == "VID_chunk_1"  # higher cosine
    assert winner.cross_encoder_score == pytest.approx(0.9)
