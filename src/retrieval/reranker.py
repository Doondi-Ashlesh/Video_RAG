"""Skill: cross_encode_rerank.

Re-score the top-k vector-search candidates with a cross-encoder against the
*original* user query (not the rephrased semantic element). The cross-encoder
reads query+chunk together, distinguishing chunks that genuinely explain the
queried concept from chunks that merely mention it in passing.

The reranker model is loaded once via lru_cache. Failure to load (offline,
disk full, etc.) falls back to cosine top-1 with a warning so the pipeline
still produces a usable answer.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from loguru import logger
from sentence_transformers import CrossEncoder

from src.config import settings
from src.models import RankedChunk


@lru_cache(maxsize=1)
def get_reranker(model_name: str | None = None) -> CrossEncoder:
    name = model_name or settings.reranker_model
    logger.info("Loading cross-encoder: {}", name)
    return CrossEncoder(name)


def _to_ranked_chunk(candidate: dict[str, Any], score: float) -> RankedChunk:
    md = candidate["metadata"]
    return RankedChunk(
        chunk_id=md["chunk_id"],
        video_id=md["video_id"],
        text=candidate["text"],
        start_time=int(md["start_time"]),
        end_time=int(md["end_time"]),
        metadata=md,
        cosine_score=float(candidate["cosine_score"]),
        cross_encoder_score=float(score),
    )


def rerank(query: str, candidates: list[dict[str, Any]]) -> RankedChunk | None:
    """Cross-encode each candidate against the query and return the winner.

    Returns None if no candidates were supplied. On model-load failure falls
    back to the cosine top-1.
    """
    ranked = rerank_topn(query, candidates, n=1)
    return ranked[0] if ranked else None


def rerank_topn(
    query: str, candidates: list[dict[str, Any]], n: int = 3
) -> list[RankedChunk]:
    """Cross-encode and return the top-N chunks sorted by score (descending).

    Used by the multi-chunk answer path: broad queries like "explain linear
    regression" don't have a single 45s chunk that fully answers them; the
    answer LLM gets the top-N as combined context.
    """
    if not candidates:
        return []

    try:
        model = get_reranker()
    except Exception as e:
        logger.warning(
            "Cross-encoder unavailable ({}). Falling back to cosine top-{}.",
            e,
            n,
        )
        sorted_cosine = sorted(
            candidates, key=lambda c: c["cosine_score"], reverse=True
        )[:n]
        return [_to_ranked_chunk(c, score=c["cosine_score"]) for c in sorted_cosine]

    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs).tolist()

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:n]

    for c, s in ranked:
        md = c["metadata"]
        logger.debug(
            "rerank top: chunk_id={} start={}s score={:.4f}",
            md["chunk_id"],
            md["start_time"],
            s,
        )

    return [_to_ranked_chunk(c, score=s) for c, s in ranked]
