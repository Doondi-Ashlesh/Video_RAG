# Demo Walkthrough

A reproducible end-to-end demo on real lecture material. Every query below is
chosen to exercise a specific part of the architecture — taken together they
prove that QAM, metadata filtering, semantic rephrasing, cross-encoder
re-ranking, grounding, and confidence scoring are all working as designed.

---

## 1. Ingest the demo corpus (one command)

```bash
python -m scripts.ingest_demo
```

This ingests four lectures with deliberately diverse metadata so each QAM
filter dimension has contrast to show off:

| # | Course | Instructor | Lec | Topic | Source |
|---|---|---|---|---|---|
| 1 | MIT-18.06 | Gilbert Strang | 1 | linear-equations | [J7DzL2_Na80](https://www.youtube.com/watch?v=J7DzL2_Na80) |
| 2 | MIT-18.06 | Gilbert Strang | 3 | matrix-multiplication | [FX4C-JpTFgY](https://www.youtube.com/watch?v=FX4C-JpTFgY) |
| 3 | CS229 | Andrew Ng | 1 | intro | [jGwO_UgTS7I](https://www.youtube.com/watch?v=jGwO_UgTS7I) |
| 4 | CS229 | Andrew Ng | 2 | linear-regression | [4b4MUYve_U8](https://www.youtube.com/watch?v=4b4MUYve_U8) |

**Time budget:** ~2-4 minutes total. Most of it is embedding ~200-400 chunks
per video on CPU. Transcript fetch is fast.

If a single video fails (transcript region-locked, network blip), re-ingest
only that one:

```bash
python -m scripts.ingest_demo --videos 2
```

---

## 2. The five demo queries

Each query is paired with the architectural feature it proves. Run all five
with `--debug` so the audience sees the QAM decomposition in real time.

### Query A — pure semantic search (no filters)
```bash
python -m scripts.query --query "explain matrix multiplication" --debug
```

**What this proves:** QAM rephrasing rescues short queries. The user types
three words; QAM's `semantic_element` expands to a richer description of what
an ideal lecture segment would be saying, which is what gets embedded.

**Expected behaviour:**
- `metadata_filters: {}` → no pre-filter
- Cosine search ranks all 4 videos' chunks together
- Cross-encoder picks Strang's actual matrix-multiplication explanation in
  Lec 3 over any tangential mentions in other videos
- URL lands on MIT 18.06 Lec 3, somewhere mid-lecture

### Query B — multi-condition hard filter
```bash
python -m scripts.query --query "what does Strang say in lecture 3 about matrix multiplication?" --debug
```

**What this proves:** QAM extracts multiple structured filters from a single
natural-language query and `build_metadata_filter` translates them into a
ChromaDB `$and` clause.

**Expected behaviour:**
- `metadata_filters: {instructor: "Gilbert Strang", lecture_number: 3, topic: "matrix-multiplication"}`
- Where-clause: `{$and: [{instructor: ...}, {lecture_number: 3}, {topic: ...}]}`
- Filter narrows the candidate pool to chunks from Lec 3 only
- Re-ranker picks the most relevant chunk within that pool

### Query C — single-field topic filter
```bash
python -m scripts.query --query "explain linear regression" --debug
```

**What this proves:** QAM lifts a topical phrase ("linear regression") out of
the query as a structured filter (`topic="linear-regression"`) **before**
semantic search. The filter pre-narrows to one video; cosine + rerank find
the best moment within that video.

**Expected behaviour:**
- `metadata_filters: {topic: "linear-regression"}`
- Filter narrows to CS229 Lec 2 only
- URL points to a moment where Ng actually walks through the regression model

### Query D — course-level filter
```bash
python -m scripts.query --query "what is taught in CS229?" --debug
```

**What this proves:** Course-level filtering returns multiple candidate videos
and lets cosine + rerank choose between them.

**Expected behaviour:**
- `metadata_filters: {course_id: "CS229"}`
- Filter narrows to 2 videos (CS229 Lec 1 and Lec 2)
- Re-ranker picks Lec 1 (the welcome / overview lecture) since the query is
  about course content as a whole

### Query E — off-corpus question (low-confidence path)
```bash
python -m scripts.query --query "explain convolutional neural networks" --debug
```

**What this proves:** Confidence threshold + honest "I'm not sure" behaviour.
None of these four lectures cover CNNs. The system shouldn't fabricate.

**Expected behaviour:**
- `metadata_filters: {}` (no filter — CNN isn't a topic in the corpus)
- Cross-encoder score below threshold (`RERANK_CONFIDENCE_THRESHOLD=-2.0`)
- CLI prints **`⚠ Low-confidence match`** before the answer
- Answer text says explicitly that the excerpt doesn't fully cover the
  question (because the strict-grounding system prompt forbids inventing
  facts not in the chunk)

---

## 3. What to point at on stage

When walking through each query, the `--debug` panel shows three things side
by side that make the architecture concrete:

```
metadata_filters: {instructor: "Gilbert Strang", lecture_number: 3, ...}
semantic_element: "A lecturer demonstrating how matrices multiply by combining
                   rows and columns step by step, including the worked
                   computation and the geometric intuition behind it."
chunk_text:       <the actual transcript excerpt that grounded the answer>
```

Each row is one architectural piece doing its job:

| Row | Skill responsible | Architectural claim it proves |
|---|---|---|
| `metadata_filters` | `decompose_query` + `build_metadata_filter` | QAM extracts structured filters from natural language |
| `semantic_element` | `decompose_query` (rephrasing half) | The query the user typed is **not** what we embedded — we embedded a richer description |
| `chunk_text` | `vector_search` + `cross_encode_rerank` | The chunk that was actually retrieved + re-ranked |
| Answer panel | `generate_answer` | Strict grounding — the answer only says what the chunk says |
| URL line | `build_snippet_url` | `start_time` came untouched from chunk metadata |
| `⚠ Low-confidence` | confidence threshold | The system knows when not to trust itself |

---

## 4. Talking points for the architectural Q&A

If someone asks *"why pre-filter? cosine similarity already handles relevance"*:

> Because cosine ranks chunks that are **topically related** to the query.
> It can't tell a deep explanation from a passing mention. Pre-filtering with
> structured metadata cuts the candidate pool down to *only* lectures the
> student already implied they want, which both improves precision and
> reduces the work the cross-encoder has to do. Watch query B — without the
> filter we'd be ranking against ~600 chunks; with it we rank against ~150.

If someone asks *"why a cross-encoder on top of cosine?"*:

> Cosine similarity treats query and chunk independently — same chunk scores
> the same against any vaguely related query. The cross-encoder reads them
> together. That's how it tells the difference between Strang spending five
> minutes on matrix multiplication and Ng saying "matrix multiplication" once
> in passing during a probability review. Watch query A: cosine top-1 might
> well be a CS229 chunk that mentions matrices; rerank top-1 will be the
> 18.06 chunk that explains them.

If someone asks *"why rephrase the query before embedding?"*:

> Because students type three-word queries and lecturers don't speak in
> three-word soundbites. "Explain backprop" has weak cosine signal because
> it's not a sentence anyone in a lecture actually says. The rephrased
> semantic element — "a lecturer walking through how the chain rule applies
> to neural network weight updates step by step" — looks much more like real
> transcript content, so it matches real transcript content better.

If someone asks *"what happens when a student asks something off-syllabus?"*:

> Query E. The cross-encoder score falls below threshold and the CLI flags
> the match as low-confidence. The strict-grounding prompt also forces the
> answer LLM to say what the excerpt actually covers and what it doesn't,
> rather than blending in prior knowledge.
