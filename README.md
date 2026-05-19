# Video RAG 

Ask any question about a recorded lecture; get a grounded explanation
**plus** a YouTube player embedded at the exact moment of the lecture that
answers it. Version 2 updates in progress!

> 🚀 **Live demo:** https://huggingface.co/spaces/DoondiAshlesh/V_RAG

Built around the **QAM (Query Attribute Modeling) framework**, with full
metadata-filtered retrieval, cross-encoder re-ranking, sentence-level URL
anchoring, and YouTube-chapter-aware override. LLM calls go through
[Nemotron Super 49B](https://build.nvidia.com) on NVIDIA's free hosted NIM
endpoint — the rest runs locally (sentence-transformers + ChromaDB).

---

## What it does

```
Student types: "explain linear regression"
                        │
                        ▼
        ┌───────────────────────────────┐
        │ QAM decomposition (Nemotron)  │
        │  metadata_filters + rephrase  │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ Metadata + chapter pre-filter │
        │    (ChromaDB where clause)    │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │   Vector search (cosine)      │
        │     all-MiniLM-L6-v2          │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ Cross-encoder rerank top-30   │
        │    ms-marco-MiniLM-L-6-v2     │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ Chapter-aware promotion (LLM) │
        │  pick best matching chapter   │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Sentence-level anchor        │
        │ (refine URL inside chunk)     │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │ Strict-grounded answer (LLM)  │
        │   top-3 chunks as context     │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Embedded YouTube player      │
        │      at the right second      │
        └───────────────────────────────┘
```

---

## Demo corpus (4 lectures, 206 chunks)

| Course | Lecture | Instructor | Topic |
|---|---|---|---|
| MIT 18.06 | 1 — Geometry of Linear Equations | Gilbert Strang | linear-equations |
| MIT 18.06 | 3 — Multiplication and Inverse Matrices | Gilbert Strang | matrix-multiplication |
| Stanford CS229 | 1 — Welcome to Machine Learning | Andrew Ng | intro |
| Stanford CS229 | 2 — Linear Regression and Gradient Descent | Andrew Ng | linear-regression |

The 4-video corpus is bundled as `chroma_db.tar.gz` so the Space restores
state in seconds; for your own lectures, run `python -m scripts.ingest_video`.

---

## v2 Enhancements

Additions on top of the original MVP spec.

### Retrieval
- Lecturer-domain filter schema (`course_id` / `instructor` / `lecture_number` / `topic`) replaces the original generic `channel_name` / `topic_category`.
- Schema-aware QAM prompt — the LLM only emits filter fields the index actually supports.
- Index-driven canonicalisation — partial values like `"Strang"` resolve to `"Gilbert Strang"` before ChromaDB ever sees the filter.
- Top-3 multi-chunk answer synthesis — broad queries pull context from several moments at once.

### URL precision
- Sentence-level anchor: a second cross-encoder pass over ~15s sub-chunks inside the winning chunk so the URL lands on the actual sentence, not the chunk boundary.
- Padded playback window — the end timestamp extends forward so the student always gets ≥30s of context.
- `&end=` on the embed URL so the YouTube player auto-stops at the chunk boundary.

### YouTube chapter awareness (biggest single win)
- Chapter markers extracted at ingest (via yt-dlp); each chunk tagged with the chapter it falls in.
- Chapter-restricted vector search when the QAM topic matches one or more chapter titles.
- LLM chapter picker (reasoning mode ON) disambiguates titles — "Linear Regression Algorithm" beats "Motivate Linear Regression" without any hardcoded word lists.
- Chapter boundaries become the URL window when a chapter match fires.

### Chunking
- 90s chunks with 20s overlap (up from 45s/10s) — deep-explanation chunks now compete fairly with announcement chunks during rerank.
- Per-chunk segment list JSON-serialised in ChromaDB metadata for the sentence anchor (no transcript re-fetch at query time).

### UX
- Embedded YouTube player at the snippet timestamp; thumbnail-with-overlay fallback for videos whose owners disable iframe embedding.
- Secondary clips listed below the primary (top-2 alternate chunks, each with its own anchored window).
- `[!] Low-confidence match` flag when the rerank score falls below threshold.
- Debug accordion exposes QAM filters, semantic element, chosen chunk, and scores for full architectural transparency.

### Deployment
- One-command Hugging Face Spaces recipe (Gradio SDK, free CPU tier).
- Bundled corpus (`chroma_db.tar.gz`) extracted on cold start — no first-run re-ingest needed.
- HF log streamer (`scripts/space_logs.py`) pulls build + runtime logs via the Spaces API for debugging.

### Robustness
- Strict grounding — answer LLM explicitly forbidden from emitting "general explanation" addendums or using prior knowledge.
- Cascading fallbacks at every stage (cross-encoder → cosine, sentence anchor → chunk, filter → unfiltered, QAM → raw query, answer LLM → raw chunk).
- Cross-version `youtube-transcript-api` support (works on both 0.6.x and 1.x).

---

## Key features

### Retrieval pipeline
- **QAM decomposition** — every query is split into structured metadata
  filters (course/instructor/lecture/topic) and a rephrased semantic
  element optimised for embedding match.
- **Metadata pre-filtering** — ChromaDB `where` clause cuts the candidate
  pool before vector search, with automatic value canonicalisation
  ("Strang" → "Gilbert Strang").
- **Vector search** — local `sentence-transformers/all-MiniLM-L6-v2` against
  the filtered subset.
- **Cross-encoder rerank** — `ms-marco-MiniLM-L-6-v2` rescores top-30
  candidates against the query; returns top-3.
- **Multi-chunk synthesis** — the answer LLM gets all top-3 chunks as
  combined context (broad queries like "explain X" naturally span chunks).

### URL precision
- **Sentence-level anchor refinement** — second-pass cross-encoder over
  ~15s sub-chunks within the winning chunk, so the URL drops the student at
  the actual moment the answer is spoken (not at the chunk boundary).
- **YouTube chapter override** — when a chapter title matches the QAM
  topic, the chapter's official start/end becomes the URL window. The LLM
  picks the best chapter when multiple match (e.g. "Linear Regression
  Algorithm" wins over "Motivate Linear Regression").
- **Padded playback window** — `&end=` parameter on the embed URL so the
  player auto-stops at the chunk boundary.

### Quality safeguards
- **Strict grounding** — answer LLM is forbidden from using prior knowledge;
  must say what's missing rather than fabricate.
- **Confidence threshold** — cross-encoder score below threshold surfaces a
  `[!] Low-confidence` warning to the student.
- **Graceful fallbacks** — every LLM/model call has a fallback (raw chunk
  text, cosine top-1, chunk-level anchor) so the pipeline never crashes.

### UX
- **Embedded YouTube player** at the snippet timestamp + auto-stop end.
- **Thumbnail fallback** for videos that disable iframe embedding.
- **Debug accordion** showing QAM filters, semantic_element, chunk text —
  full architectural transparency.
- **Secondary clips** — the other top-3 chunks shown as additional jumps
  ("the answer was synthesised across these too").

---

## Tech stack

| Layer | Choice |
|---|---|
| LLM (QAM, chapter pick, answer) | Nemotron Super 49B via NVIDIA NIM (free) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Cross-encoder rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Vector DB | ChromaDB (local persistent) |
| Transcripts | `youtube-transcript-api` |
| Video metadata + chapters | `yt-dlp` |
| Web UI | Gradio 4.36 |
| Hosting | Hugging Face Spaces (free CPU) |
| LLM SDK | `openai` (NVIDIA NIM is OpenAI-compatible) |

Total cost to run: **$0** — NVIDIA NIM free tier, HF Spaces free tier,
everything else is local / open-source.

---

## Quick start (local)

```bash
git clone https://github.com/Doondi-Ashlesh/Video_RAG.git
cd Video_RAG

python -m venv .venv
.venv\Scripts\activate     # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

# Get a free NVIDIA NIM key at https://build.nvidia.com -> "Get API Key"
copy .env.example .env
# edit .env and set NVIDIA_API_KEY=nvapi-...

# Either restore the bundled demo corpus, or ingest your own lectures
python -c "import tarfile; tarfile.open('chroma_db.tar.gz').extractall('.')"

# CLI query
python -m scripts.query --query "explain matrix multiplication" --debug

# Or launch the Gradio UI locally
python app.py
# -> open http://localhost:7860
```

### Ingest your own lecture

```bash
python -m scripts.ingest_video \
    --url "https://www.youtube.com/watch?v=YOUR_VIDEO" \
    --course-id "CS101" \
    --instructor "Prof Smith" \
    --lecture-number 5 \
    --topic "backpropagation"
```

All flags are optional. The pipeline auto-extracts YouTube chapters via
yt-dlp and uses them as the URL anchor when the query topic matches.

---

## Deploy your own to Hugging Face Spaces

Full step-by-step in **[DEPLOY.md](DEPLOY.md)**. TL;DR:

1. Create a free [Hugging Face Space](https://huggingface.co/new-space)
   (Gradio SDK, CPU basic).
2. Add `NVIDIA_API_KEY` as a Space secret.
3. Push `app.py`, `requirements.txt`, `src/`, `scripts/`,
   `chroma_db.tar.gz`, and rename `SPACE_README.md` → `README.md` in the
   Space repo.
4. Done. The Space wakes in ~30s on the first request.

---

## Project structure

```
.
├── README.md                ← this file
├── DEPLOY.md                ← Hugging Face Spaces deploy recipe
├── DEMO.md                  ← on-stage walkthrough with showcase queries
├── SPACE_README.md          ← Spaces-flavoured README (frontmatter for HF)
├── app.py                   ← Gradio entry point (HF Spaces)
├── requirements.txt
├── .env.example
├── files/                   ← original design specs (claude.md, agents.md, skill specs)
├── src/
│   ├── config.py            ← pydantic-settings (one source of truth)
│   ├── models/
│   │   ├── chunk.py         ← TranscriptChunk, VideoMeta (with chapters)
│   │   └── query.py         ← QAMResult, RankedChunk, VideoRAGResponse
│   ├── db/
│   │   └── vector_store.py  ← ChromaDB facade (Qdrant-swappable)
│   ├── ingestion/
│   │   ├── transcript.py    ← skill: fetch_youtube_transcript (+ chapters)
│   │   ├── chunker.py       ← skill: chunk_with_timestamps (+ segments)
│   │   └── embedder.py      ← skill: embed_and_store (+ chapter metadata)
│   └── retrieval/
│       ├── qam.py           ← skills: decompose_query, build_metadata_filter, pick_best_chapter
│       ├── searcher.py      ← skill: vector_search (with chapter-restrict)
│       ├── reranker.py      ← skill: cross_encode_rerank (top-N)
│       ├── sentence_anchor.py  ← second-pass anchor refinement
│       └── response.py      ← skills: generate_answer (multi-chunk), build_snippet_url
├── scripts/
│   ├── ingest_video.py      ← single-video ingestion CLI
│   ├── ingest_demo.py       ← bulk-ingest the 4-video demo corpus
│   ├── ingest_demo.bat      ← Windows wrapper
│   ├── query.py             ← end-to-end query CLI
│   ├── bundle_chroma_db.py  ← tarball the populated DB for shipping
│   ├── copy_to_space.sh     ← copy deploy artefacts into a Space repo
│   └── space_logs.py        ← stream Space logs via HF API
├── tests/
│   ├── test_ingestion.py    ← chunker, video-id parsing, segments
│   ├── test_qam.py          ← QAM decomposition, filter builder, canonicalisation
│   ├── test_retrieval.py    ← snippet URL build, search-result parsing
│   ├── test_reranker.py     ← cross-encoder fallback paths
│   └── test_sentence_anchor.py  ← sub-chunk windowing, anchor refinement
└── chroma_db.tar.gz         ← bundled populated corpus (~5 MB, restored on cold start)
```

---

## Architecture decisions worth noting

- **QAM is the centrepiece.** The query is never embedded raw — the LLM
  decomposes it into structured filters + a richer rephrased semantic
  element that scores better against actual transcript content.
- **Chapter override beats heuristics.** When the YouTube chapter title
  matches the topic, the chapter's official boundaries are the URL anchor —
  this is dramatically better than chunk-level timestamps for broad
  explanatory queries.
- **No hardcoded word lists.** The chapter picker uses Nemotron's reasoning
  mode to compare chapter titles semantically — no curated "meta words"
  taxonomy that wouldn't generalise.
- **Locally-anchored compute.** Embedding, reranking, sentence-anchoring all
  run on CPU via sentence-transformers. The only network calls are to
  NVIDIA NIM for the 2-3 LLM hops per query.
- **Strict grounding.** The answer LLM is prompted to refuse to invent
  content, must say what's missing if the chunks don't fully cover the
  question. Verified on off-corpus queries (e.g. CNN question against a
  linear-algebra corpus).

---

## Roadmap

Things deliberately deferred for the MVP:

- Multi-language transcripts.
- Whisper re-transcription (currently uses YouTube auto-transcripts).
- Slide / whiteboard OCR ingest alongside transcript chunks.
- Hybrid search (BM25 + dense) for exact-term queries.
- Conversational query rewriting for follow-up questions.
- Eval harness (Recall@k, MRR, timestamp accuracy).
- Per-course collections + auth for multi-tenant deployment.
- Qdrant migration when hybrid search / payload indexing is needed.

---

## Tests

```bash
pytest -q
```

40+ tests covering chunker behaviour, QAM JSON parsing, filter
canonicalisation, search-result distance→similarity conversion, snippet URL
construction, cross-encoder fallback paths, and sentence-anchor sub-chunk
windowing. Mocks the LLM client and cross-encoder so the suite runs offline.

---

## License

MIT.

## Acknowledgements

- Lectures: MIT OpenCourseWare (Gilbert Strang, 18.06 Linear Algebra),
  Stanford Online (Andrew Ng, CS229 Machine Learning).
- Models: NVIDIA Nemotron Super 49B, sentence-transformers community.
- Hosting: Hugging Face Spaces.
