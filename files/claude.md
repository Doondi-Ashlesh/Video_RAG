# Video RAG System

## Project Overview

Build a Video RAG (Retrieval-Augmented Generation) system that takes a natural language user query and returns:
1. A **text explanation** generated from the most relevant transcript segment
2. A **timestamped YouTube snippet URL** pointing to the exact moment in the video that answers the query

The system does NOT return entire videos. It returns a deep-linked URL like `https://youtube.com/watch?v=VIDEO_ID&t=135s` that starts playback at the most relevant second — identical to how Google surfaces video snippets in search results.

---

## Core Architecture

The system has two distinct operational modes:

### Mode 1: Ingestion (Offline / On-demand)
Triggered when a new YouTube video is added to the knowledge base. Processes the video transcript into timestamped, vectorized chunks stored in the vector database with rich metadata. This runs once per video and never again unless the video is re-indexed.

### Mode 2: Query & Retrieval (Real-time)
Triggered on every user query. Applies the **QAM (Query Attribute Modeling) framework** to decompose the query into metadata filters and semantic intent, then retrieves and re-ranks the best matching transcript chunk, generates a text answer, and builds the snippet URL.

---

## QAM Framework (Query Attribute Modeling)

QAM is the core intelligence of the query pipeline. Every user query must be decomposed into two distinct components before any search occurs:

### 1. Metadata Tags (Hard Filters)
Structured attributes extracted from the query that are used to **filter the vector database before semantic search begins**. These are applied as exact or range-based filters — they eliminate irrelevant content entirely rather than downranking it.

Examples of metadata tags to extract:
- `channel_name` → "by Andrej Karpathy", "from 3Blue1Brown"
- `upload_year` → "from 2023", "recent"
- `topic_category` → "machine learning", "finance", "physics"
- `video_title_hint` → partial title mentions

If no metadata tags are present in the query, no pre-filtering is applied and the full corpus is searched semantically.

### 2. Semantic Elements (Soft Search)
The core conceptual intent of the query, stripped of all metadata. This is what gets vectorized and compared against transcript chunk embeddings via cosine similarity.

Example decomposition:
- Query: `"explain RAG to me by Andrej Karpathy"`
- Metadata Tags: `{ channel_name: "Andrej Karpathy" }`
- Semantic Element: `"conceptual explanation of Retrieval-Augmented Generation"`

The semantic element must be **rephrased for retrieval quality** — not passed raw. It should describe what the ideal transcript segment would be saying, not just repeat the user's words.

---

## Data Model

### Video Metadata (per video, stored at index time)
```json
{
  "video_id": "dQw4w9WgXcQ",
  "title": "Intro to RAG - Full Lecture",
  "channel_name": "AI Explained",
  "channel_id": "UC...",
  "upload_date": "2023-11-15",
  "topic_category": "machine learning",
  "duration_seconds": 3600,
  "language": "en"
}
```

### Transcript Chunk (per chunk, stored in vector DB)
```json
{
  "chunk_id": "dQw4w9WgXcQ_chunk_42",
  "video_id": "dQw4w9WgXcQ",
  "text": "So retrieval augmented generation works by first taking your query...",
  "start_time": 135,
  "end_time": 165,
  "embedding": [...],
  "metadata": {
    "channel_name": "AI Explained",
    "upload_date": "2023-11-15",
    "topic_category": "machine learning",
    "title": "Intro to RAG - Full Lecture"
  }
}
```

---

## Chunking Strategy

- **Window size:** 30–60 seconds of transcript text per chunk
- **Overlap:** 10-second overlap between adjacent chunks to avoid cutting off mid-explanation
- **Anchor timestamp:** Each chunk's `start_time` is the timestamp used to build the final URL
- **Minimum chunk size:** Discard chunks with fewer than 20 words (usually silence or filler)

---

## Retrieval Pipeline (QAM → Answer)

```
User Query
    ↓
[QAM Decomposition]
    → Extract metadata tags → build vector DB filter payload
    → Extract + rephrase semantic element → vectorize
    ↓
[Metadata-Filtered Vector Search]
    → Apply hard filters to reduce candidate pool
    → Run cosine similarity search against filtered subset
    → Return top-k=10 chunks
    ↓
[Cross-Encoder Re-ranking]
    → Score each of top-k chunks against the original query
    → Distinguish "passing mention" vs "detailed explanation"
    → Select top-1 chunk
    ↓
[Response Generation]
    → Use top-1 chunk text as context in LLM prompt
    → Generate a clean, standalone text explanation
    → Build snippet URL from chunk's video_id + start_time
    ↓
Output: { explanation: "...", snippet_url: "https://youtube.com/watch?v=...&t=135s" }
```

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Transcript fetching | `youtube-transcript-api` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2` for MVP) |
| Vector database | `ChromaDB` (MVP) → `Qdrant` (production path) |
| Cross-encoder re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` |
| LLM for QAM + answer generation | `claude-sonnet-4-20250514` via Anthropic API |
| Language | Python 3.11+ |
| Config management | `pydantic-settings` + `.env` |
| Logging | `loguru` |

---

## Project Structure

```
video-rag/
├── claude.md                   ← This file
├── agents.md                   ← Agent definitions
├── skills/
│   ├── fetch_youtube_transcript.md
│   ├── chunk_with_timestamps.md
│   ├── embed_and_store.md
│   ├── decompose_query.md
│   ├── build_metadata_filter.md
│   ├── vector_search.md
│   ├── cross_encode_rerank.md
│   ├── generate_answer.md
│   └── build_snippet_url.md
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── transcript.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── qam.py
│   │   ├── searcher.py
│   │   ├── reranker.py
│   │   └── response.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chunk.py
│   │   └── query.py
│   └── config.py
├── scripts/
│   ├── ingest_video.py         ← CLI: add a video to the knowledge base
│   └── query.py                ← CLI: run a query against the knowledge base
├── tests/
│   ├── test_ingestion.py
│   ├── test_qam.py
│   ├── test_retrieval.py
│   └── test_reranker.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quality Standards

- **No hallucination in answers:** The LLM must be prompted to answer strictly from the retrieved chunk context. If the chunk doesn't contain enough information, say so rather than inventing an explanation.
- **Timestamp accuracy:** The `start_time` used in the URL must come directly from the stored chunk metadata — never inferred or approximated.
- **Re-ranking is mandatory:** Never return the raw top-1 from cosine similarity without re-ranking. A passing mention of a term scores highly on similarity but is useless as a snippet.
- **QAM is mandatory:** Never run a raw similarity search without first attempting QAM decomposition. Even if no metadata tags are extracted, the semantic element rephrasing step must still occur.
- **Graceful degradation:** If no transcript is available for a video, log the failure and skip — do not crash the ingestion pipeline.

---

## Environment Variables

```
ANTHROPIC_API_KEY=...
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
TOP_K_RETRIEVAL=10
CHUNK_WINDOW_SECONDS=45
CHUNK_OVERLAP_SECONDS=10
```

---

## Out of Scope for MVP

- Video audio transcription (only pre-existing YouTube auto-transcripts via API)
- Multi-language support (English only)
- UI / frontend (CLI only)
- Authentication / multi-user
- Streaming responses
- Qdrant integration (ChromaDB for MVP, Qdrant is the production upgrade path)
