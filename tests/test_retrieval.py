"""Tests for searcher result-parsing and snippet-URL building."""
from __future__ import annotations

import pytest

from src.retrieval.response import build_snippet_url
from src.retrieval.searcher import _parse_results


def test_build_snippet_url_basic() -> None:
    assert (
        build_snippet_url("dQw4w9WgXcQ", 135)
        == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=135s"
    )


def test_build_snippet_url_zero_start() -> None:
    assert (
        build_snippet_url("dQw4w9WgXcQ", 0)
        == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s"
    )


def test_build_snippet_url_rejects_negative_time() -> None:
    with pytest.raises(ValueError):
        build_snippet_url("dQw4w9WgXcQ", -1)


def test_build_snippet_url_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        build_snippet_url("", 10)


def test_parse_results_empty() -> None:
    assert _parse_results({"documents": [[]], "metadatas": [[]], "distances": [[]]}) == []
    assert _parse_results({}) == []


def test_parse_results_converts_distance_to_similarity() -> None:
    results = {
        "documents": [["chunk text A", "chunk text B"]],
        "metadatas": [
            [
                {"chunk_id": "v_chunk_0", "video_id": "v", "start_time": 0, "end_time": 45},
                {"chunk_id": "v_chunk_1", "video_id": "v", "start_time": 35, "end_time": 80},
            ]
        ],
        "distances": [[0.1, 0.4]],
    }
    parsed = _parse_results(results)
    assert len(parsed) == 2
    # cosine_score = 1 - distance
    assert parsed[0]["cosine_score"] == pytest.approx(0.9)
    assert parsed[1]["cosine_score"] == pytest.approx(0.6)
    assert parsed[0]["text"] == "chunk text A"
