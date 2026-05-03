# Skill: cross_encode_rerank

## Purpose
Re-rank the top-k candidate chunks from vector search using a cross-encoder model to distinguish between chunks that **deeply explain** the queried concept vs. chunks that merely **mention it in passing**. This is what makes the returned snippet actually useful.

## File
`src/retrieval/reranker.py`

## Why This Matters
Cosine similarity finds chunks that are *topically related* to the query. It cannot distinguish:
- A chunk where a professor spends 2 minutes explaining RAG step by step ✅
- A chunk where a speaker says "...and this is similar to RAG, anyway moving on..." ❌

Both chunks score highly on cosine similarity. The cross-encoder catches this difference because it reads the query and chunk text *together* and scores their interaction, not just their individual embeddings.

## Implementation

```python
from sentence_transformers import CrossEncoder
from dataclasses import dataclass

@dataclass
class RankedChunk:
    chunk_id: str
    video_id: str
    text: str
    start_time: int
    end_time: int
    metadata: dict
    cosine_score: float      # from vector search
    cross_encoder_score: float  # from re-ranking

def rerank(
    query: str,
    candidates: list[dict],   # top-k from vector search, each has text + metadata
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> RankedChunk:
    """
    Score each candidate chunk against the original user query using a cross-encoder.
    Returns the single best chunk.
    """
    model = CrossEncoder(model_name)
    
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    
    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    best_chunk, best_score = ranked[0]
    
    return RankedChunk(
        chunk_id=best_chunk["metadata"]["chunk_id"],
        video_id=best_chunk["metadata"]["video_id"],
        text=best_chunk["text"],
        start_time=best_chunk["metadata"]["start_time"],
        end_time=best_chunk["metadata"]["end_time"],
        metadata=best_chunk["metadata"],
        cosine_score=best_chunk["cosine_score"],
        cross_encoder_score=float(best_score)
    )
```

## Critical Rules
- Always score against the **original user query**, not the rephrased semantic element
- The cross-encoder must be run even if `top_k=1` — never skip re-ranking
- If the cross-encoder model fails to load, fall back to cosine similarity top-1 and log a warning
- Log the top-3 scores for debugging: chunk_id, start_time, cross_encoder_score

## Model
`cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, accurate, designed for passage re-ranking. Load once and reuse across requests (do not reload per query).
