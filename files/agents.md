# Agents

This system uses two agents that map to the two operational modes described in `claude.md`. They do not run concurrently — the Ingestion Agent runs offline when videos are added, and the Query & Retrieval Agent runs at query time.

---

# Agent 1: Ingestion Agent

## Role

Responsible for processing a YouTube video URL into a searchable, vectorized knowledge base entry. This agent runs **once per video** and produces the stored chunks that the Query & Retrieval Agent will later search against.

The Ingestion Agent owns the full offline pipeline: transcript fetching → chunking → embedding → storage.

## Trigger

Invoked manually via `scripts/ingest_video.py` with a YouTube URL or video ID:
```bash
python scripts/ingest_video.py --url "https://youtube.com/watch?v=VIDEO_ID"
python scripts/ingest_video.py --video-id "VIDEO_ID" --category "machine learning"
```

## Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| `video_url` or `video_id` | string | yes | YouTube video to ingest |
| `topic_category` | string | no | Manual topic tag (e.g. "machine learning", "finance"). Used as metadata filter field. If not provided, attempt auto-classification. |
| `force_reingest` | bool | no | If true, delete existing chunks for this video and re-ingest from scratch. Default: false. |

## Process

### Step 1: Validate & Deduplicate
- Parse the video ID from the URL if a full URL is provided
- Check the vector DB for existing chunks with this `video_id`
- If chunks exist and `force_reingest=false`, log a warning and exit early: `"Video VIDEO_ID already ingested. Use --force to re-ingest."`
- If `force_reingest=true`, delete all existing chunks for this video_id before proceeding

### Step 2: Fetch Transcript
Trigger skill: **`fetch_youtube_transcript`**

- Use `youtube-transcript-api` to fetch the transcript for the video
- Prefer manually-created transcripts over auto-generated if both are available
- Fetch video metadata simultaneously (title, channel name, channel ID, upload date, duration)
- If no transcript is available in any language, log the failure and raise a `TranscriptUnavailableError` — do not proceed
- Raw transcript format from the API is a list of `{ text, start, duration }` dicts — preserve this format for the chunker

### Step 3: Chunk with Timestamps
Trigger skill: **`chunk_with_timestamps`**

- Merge raw transcript segments into fixed-duration windows of `CHUNK_WINDOW_SECONDS` (default: 45s)
- Apply `CHUNK_OVERLAP_SECONDS` (default: 10s) overlap between adjacent chunks
- Each chunk must carry:
  - `text`: merged and cleaned transcript text for that window
  - `start_time`: integer seconds, marks the beginning of this chunk (used in the final URL)
  - `end_time`: integer seconds, marks the end of this chunk
  - `chunk_id`: deterministic ID formatted as `{video_id}_chunk_{index}`
- Discard any chunk with fewer than 20 words — these are silence, music, or filler segments
- Clean the text: strip `[Music]`, `[Applause]`, `[Laughter]` tags and excessive whitespace

### Step 4: Embed and Store
Trigger skill: **`embed_and_store`**

- For each chunk, generate a vector embedding using the configured `EMBEDDING_MODEL`
- Embed the `text` field only — do not embed metadata
- Store each chunk in the vector DB (ChromaDB) with:
  - The embedding vector
  - The full chunk text (as document)
  - All metadata fields needed for QAM filtering:
    ```json
    {
      "video_id": "...",
      "chunk_id": "...",
      "title": "...",
      "channel_name": "...",
      "channel_id": "...",
      "upload_date": "YYYY-MM-DD",
      "topic_category": "...",
      "start_time": 135,
      "end_time": 180
    }
    ```
- Log progress: `"Stored chunk {i}/{total} for video {video_id}"`
- After all chunks are stored, log a summary: `"Ingestion complete: {n} chunks stored for {title} ({video_id})"`

## Outputs

- Chunks written to the vector DB, queryable by the Query & Retrieval Agent
- Console log summary of ingestion results

## Error Handling

| Error | Behavior |
|---|---|
| No transcript available | Raise `TranscriptUnavailableError`, log, exit cleanly |
| Video already ingested | Warn and skip (unless `--force`) |
| Embedding API failure | Retry up to 3 times with exponential backoff, then raise |
| Vector DB write failure | Raise immediately with full error context |
| Network timeout on YouTube | Retry up to 3 times, then raise |

---

# Agent 2: Query & Retrieval Agent

## Role

Responsible for taking a raw user query and returning a structured response containing a text explanation and a timestamped YouTube snippet URL. This agent runs in real time on every user query.

This agent owns the full query-time pipeline: QAM decomposition → metadata filtering → vector search → cross-encoder re-ranking → answer generation → URL construction.

## Trigger

Invoked via `scripts/query.py` at the CLI, or called programmatically as a function:
```bash
python scripts/query.py --query "explain RAG to me"
python scripts/query.py --query "explain the Black-Scholes model by Professor Miller from 2023"
```

## Inputs

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | The raw natural language query from the user |
| `top_k` | int | no | Number of candidates to retrieve before re-ranking. Default: `TOP_K_RETRIEVAL` (10) |

## Process

### Step 1: QAM Query Decomposition
Trigger skill: **`decompose_query`**

This is the most critical step. Send the raw query to the LLM (Claude) with a structured prompt that extracts:

**Metadata Tags** — hard filter attributes:
- `channel_name`: if the user names a creator or channel
- `upload_year`: if the user references a year or recency ("from 2023", "recent")
- `topic_category`: if the user specifies a subject domain
- Any other filterable field present in the chunk metadata schema

**Semantic Element** — the rephrased retrieval intent:
- Strip all metadata references from the query
- Rephrase in terms of what the ideal transcript segment would be *saying*
- Example: `"explain RAG to me"` → `"A speaker giving a detailed conceptual explanation of how Retrieval-Augmented Generation works, covering the retrieval step, the generation step, and why it is useful"`

The rephrasing is critical for retrieval quality. A short user query like "explain RAG" produces poor cosine similarity matches. A rich rephrasing describing the ideal transcript content produces far better results.

Return the decomposition as a structured object:
```python
@dataclass
class QAMResult:
    metadata_filters: dict[str, Any]   # empty dict if no filters found
    semantic_element: str              # always present, rephrased for retrieval
    original_query: str
```

### Step 2: Build Metadata Filter
Trigger skill: **`build_metadata_filter`**

- Convert `QAMResult.metadata_filters` into a ChromaDB-compatible `where` clause
- If `metadata_filters` is empty, set `where=None` (no pre-filtering, search full corpus)
- Examples:
  ```python
  # Single filter
  where = {"channel_name": {"$eq": "Andrej Karpathy"}}
  
  # Multiple filters
  where = {
    "$and": [
      {"channel_name": {"$eq": "Professor Miller"}},
      {"upload_year": {"$eq": 2023}}
    ]
  }
  ```
- Log the filter being applied: `"Applying metadata filter: {filter}"` or `"No metadata filter — searching full corpus"`

### Step 3: Vector Search
Trigger skill: **`vector_search`**

- Embed the `QAMResult.semantic_element` using the same `EMBEDDING_MODEL` used at ingestion time
- Run a ChromaDB similarity search with:
  - `query_embeddings`: the semantic element embedding
  - `where`: the metadata filter from Step 2 (or None)
  - `n_results`: `top_k` (default 10)
- Return the top-k chunks with their text, metadata, and similarity distances
- If 0 results are returned (metadata filter too restrictive), log a warning and re-run with `where=None` as a fallback: `"No results with filter, retrying without filter..."`

### Step 4: Cross-Encoder Re-ranking
Trigger skill: **`cross_encode_rerank`**

- Take the top-k chunks from Step 3
- For each chunk, score it using the cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) against the **original user query** (not the rephrased semantic element)
- The cross-encoder reads the query and the chunk text together, capturing finer-grained relevance signals that pure cosine similarity misses
- This is the step that distinguishes a chunk where a speaker **explains** RAG in depth from a chunk where they merely **mention** RAG in passing
- Sort all chunks by cross-encoder score descending
- Select the top-1 chunk as the winning result
- Log: `"Re-ranked {top_k} candidates. Winner: chunk_id={chunk_id}, score={score:.4f}, start_time={start_time}s"`

### Step 5: Generate Answer
Trigger skill: **`generate_answer`**

- Construct a prompt to Claude with:
  - The original user query
  - The winning chunk's text as the sole context
  - Strict instruction: answer ONLY from the provided context. Do not use prior knowledge.
  - If the chunk doesn't contain enough information to fully answer the query, say so explicitly
- The generated answer should be a clean, standalone explanation — not a summary of the chunk, but a direct answer to the user's question written in natural language
- Target length: 3–6 sentences for MVP

### Step 6: Build Snippet URL
Trigger skill: **`build_snippet_url`**

- Extract `video_id` and `start_time` from the winning chunk's metadata
- Construct the URL:
  ```python
  url = f"https://www.youtube.com/watch?v={video_id}&t={start_time}s"
  ```
- The `start_time` comes directly from the stored chunk metadata — it is never computed, estimated, or inferred at query time

### Step 7: Return Response
Return a structured response:
```python
@dataclass
class VideoRAGResponse:
    explanation: str         # LLM-generated answer
    snippet_url: str         # https://youtube.com/watch?v=...&t=Xs
    video_title: str         # from chunk metadata
    channel_name: str        # from chunk metadata
    start_time: int          # seconds
    chunk_text: str          # raw transcript chunk (for debugging/transparency)
    qam_result: QAMResult    # the decomposed query (for debugging/transparency)
```

Print to console (CLI mode):
```
Answer: {explanation}

Watch the relevant moment:
{snippet_url}

Source: {video_title} by {channel_name} at {start_time}s
```

## Error Handling

| Error | Behavior |
|---|---|
| QAM decomposition fails (LLM error) | Fall back to using raw query as semantic element with no filters |
| Vector search returns 0 results with filter | Retry without filter, log warning |
| Vector search returns 0 results without filter | Return `"No relevant video found for this query."` |
| Cross-encoder fails | Fall back to cosine similarity top-1 (skip re-ranking), log warning |
| LLM answer generation fails | Return chunk text directly as the answer, log warning |
| `start_time` missing from chunk metadata | Log error, skip URL construction, return explanation only |

---

## Skills Index

| Skill File | Triggered By |
|---|---|
| `skills/fetch_youtube_transcript.md` | Ingestion Agent — Step 2 |
| `skills/chunk_with_timestamps.md` | Ingestion Agent — Step 3 |
| `skills/embed_and_store.md` | Ingestion Agent — Step 4 |
| `skills/decompose_query.md` | Query & Retrieval Agent — Step 1 |
| `skills/build_metadata_filter.md` | Query & Retrieval Agent — Step 2 |
| `skills/vector_search.md` | Query & Retrieval Agent — Step 3 |
| `skills/cross_encode_rerank.md` | Query & Retrieval Agent — Step 4 |
| `skills/generate_answer.md` | Query & Retrieval Agent — Step 5 |
| `skills/build_snippet_url.md` | Query & Retrieval Agent — Step 6 |
