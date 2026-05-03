# Skill: embed_and_store

## Purpose
Generate vector embeddings for each transcript chunk and store them in ChromaDB with all metadata fields required for QAM-based filtering.

## File
`src/ingestion/embedder.py`

## Implementation

```python
from sentence_transformers import SentenceTransformer
import chromadb
from src.ingestion.chunker import TranscriptChunk

def embed_and_store(
    chunks: list[TranscriptChunk],
    video_meta: dict,
    model: SentenceTransformer,
    collection: chromadb.Collection
):
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True).tolist()

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "chunk_id": c.chunk_id,
            "video_id": c.video_id,
            "start_time": c.start_time,
            "end_time": c.end_time,
            "title": video_meta["title"],
            "channel_name": video_meta["channel_name"],
            "channel_id": video_meta["channel_id"],
            "upload_date": video_meta["upload_date"],
            "upload_year": int(video_meta["upload_date"][:4]),
            "topic_category": video_meta.get("topic_category", "")
        } for c in chunks]
    )
```

## Notes
- Embed only the `text` field — metadata is stored separately alongside the embedding
- `upload_year` must be stored as an integer for ChromaDB range filtering
- Use `batch_size=32` to avoid memory issues on large videos
- The `collection` object is created once at startup and injected — do not recreate per call
