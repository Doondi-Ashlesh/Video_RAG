"""Tests for the QAM decomposition + filter-builder skills.

The decomposer hits NVIDIA NIM (Nemotron Super) in production via an
OpenAI-compatible client; here we mock the client and verify our parsing,
fallback, and filter-translation logic.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.retrieval.qam import build_metadata_filter, decompose_query


def _mock_client(json_text: str) -> MagicMock:
    """Mock client returning an OpenAI-shaped chat-completion response."""
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json_text))]
    )
    return client


def test_decompose_query_extracts_lecture_filters() -> None:
    raw = """
    {
      "metadata_filters": {
        "course_id": "CS231",
        "instructor": "Andrej Karpathy",
        "lecture_number": 5,
        "topic": "backpropagation",
        "upload_year": null
      },
      "semantic_element": "A lecturer walking through the chain rule applied to neural network weights."
    }
    """
    qam = decompose_query(
        "Explain backprop in CS231 lecture 5 by Karpathy", client=_mock_client(raw)
    )
    assert qam.metadata_filters == {
        "course_id": "CS231",
        "instructor": "Andrej Karpathy",
        "lecture_number": 5,
        "topic": "backpropagation",
    }
    assert "chain rule" in qam.semantic_element
    assert qam.original_query.startswith("Explain backprop")


def test_decompose_query_handles_empty_filters() -> None:
    raw = '{"metadata_filters": {"course_id": null, "instructor": null, "lecture_number": null, "topic": null, "upload_year": null}, "semantic_element": "A lecturer giving an overview of overfitting."}'
    qam = decompose_query("what is overfitting", client=_mock_client(raw))
    assert qam.metadata_filters == {}
    assert "overfitting" in qam.semantic_element


def test_decompose_query_falls_back_on_invalid_json() -> None:
    qam = decompose_query("explain SGD", client=_mock_client("not json at all"))
    assert qam.metadata_filters == {}
    assert qam.semantic_element == "explain SGD"


def test_decompose_query_extracts_json_from_prose() -> None:
    raw = 'Here is the result:\n{"metadata_filters": {"topic": "Regularization"}, "semantic_element": "An instructor explaining L1 vs L2 regularisation."}'
    qam = decompose_query("explain regularization", client=_mock_client(raw))
    assert qam.metadata_filters == {"topic": "regularization"}  # lowercased


def test_build_metadata_filter_no_filters() -> None:
    assert build_metadata_filter({}) is None


def test_build_metadata_filter_single() -> None:
    assert build_metadata_filter({"course_id": "CS231"}) == {
        "course_id": {"$eq": "CS231"}
    }


def test_build_metadata_filter_multiple() -> None:
    out = build_metadata_filter(
        {"course_id": "CS231", "lecture_number": 5, "topic": "backpropagation"}
    )
    assert "$and" in out
    conds = out["$and"]
    assert {"course_id": {"$eq": "CS231"}} in conds
    assert {"lecture_number": {"$eq": 5}} in conds
    assert {"topic": {"$eq": "backpropagation"}} in conds


# --------------------------------------------------- canonicalisation tests
class _FakeStore:
    """Minimal stand-in for VectorStore that exposes the metadata listing path
    used by _index_canonical_values (store._collection.get(include=['metadatas']))."""

    def __init__(self, metadatas: list[dict]) -> None:
        self._collection = self  # let .get() resolve back to this object
        self._metadatas = metadatas

    def get(self, include):  # type: ignore[no-untyped-def]
        return {"metadatas": self._metadatas}


def test_canonicalise_substring_instructor_match() -> None:
    """QAM emits 'Strang', index has 'Gilbert Strang' — should canonicalise."""
    store = _FakeStore(
        [
            {"instructor": "Gilbert Strang", "course_id": "MIT-18.06"},
            {"instructor": "Andrew Ng", "course_id": "CS229"},
        ]
    )
    out = build_metadata_filter({"instructor": "Strang"}, store=store)
    assert out == {"instructor": {"$eq": "Gilbert Strang"}}


def test_canonicalise_case_insensitive_course_id_match() -> None:
    """QAM emits 'cs229', index has 'CS229' — should canonicalise."""
    store = _FakeStore([{"instructor": "Andrew Ng", "course_id": "CS229"}])
    out = build_metadata_filter({"course_id": "cs229"}, store=store)
    assert out == {"course_id": {"$eq": "CS229"}}


def test_canonicalise_drops_filter_when_no_match() -> None:
    """QAM emits an unknown instructor — filter should be dropped, not 0-result."""
    store = _FakeStore([{"instructor": "Andrew Ng", "course_id": "CS229"}])
    out = build_metadata_filter({"instructor": "Hinton"}, store=store)
    assert out is None  # the only filter dropped, none left


def test_canonicalise_drops_when_substring_match_is_ambiguous() -> None:
    """If 'Smith' matches both 'John Smith' and 'Jane Smith', drop the filter."""
    store = _FakeStore(
        [{"instructor": "John Smith"}, {"instructor": "Jane Smith"}]
    )
    out = build_metadata_filter({"instructor": "Smith"}, store=store)
    assert out is None


def test_canonicalise_preserves_exact_match_unchanged() -> None:
    store = _FakeStore([{"instructor": "Gilbert Strang"}])
    out = build_metadata_filter({"instructor": "Gilbert Strang"}, store=store)
    assert out == {"instructor": {"$eq": "Gilbert Strang"}}


def test_canonicalise_skipped_when_no_store_provided() -> None:
    """Without a store, build_metadata_filter passes through (no canonicalisation)."""
    out = build_metadata_filter({"instructor": "Strang"})  # no store
    assert out == {"instructor": {"$eq": "Strang"}}
