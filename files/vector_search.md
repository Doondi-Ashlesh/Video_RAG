# Skill: vector_search

## Purpose
Embed the semantic element from QAM decomposition and run a metadata-filtered cosine similarity search against the ChromaDB collection to retrieve the top-k candidate chunks.

## File
`src/retrieval/searcher.py`

## Implementation

```python
from sentence_transformers import SentenceTransformer
import chromadb

def vector_search(
    semantic_element: str,
    where_filter: dict | None,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int = 10
) -> list[dict]:
    """
    Embed the semantic element and search the vector DB.
    Falls back to no filter if filtered search returns 0 results.
    """
    query_embedding = model.encode([semantic_element]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    chunks = _parse_results(results)

    # Fallback: if filter returned nothing, retry without filter
    if len(chunks) == 0 and where_filter is not None:
        logger.warning("No results with metadata filter. Retrying without filter...")
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=None,
            include=["documents", "metadatas", "distances"]
        )
        chunks = _parse_results(results)

    return chunks

def _parse_results(results: dict) -> list[dict]:
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "cosine_score": 1 - results["distances"][0][i]  # convert distance to similarity
        })
    return chunks
```

## Notes
- Embed the `semantic_element` (rephrased), not the raw user query — this is what produces high-quality similarity matches
- Use the **same model** used at ingestion time — embedding models are not interchangeable
- `cosine_score` is stored in the returned chunk dict for logging/debugging; the cross-encoder will override this for final ranking
- Log: number of results returned and whether the filter fallback was triggered
