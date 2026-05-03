---
title: Video RAG - Lecture Q&A
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
short_description: Ask a lecture question, get the precise YouTube moment.
---

# Video RAG — Lecture Q&A

Ask a natural-language question about any of the bundled lectures. The system
finds the exact moment in the right video that answers it and embeds the
YouTube player at that timestamp.

## How it works (architecture)

```
User query
  ↓ QAM decomposition  (Nemotron Super 49B via NVIDIA NIM)
    → metadata_filters (course/instructor/lecture/topic)
    → semantic_element (rephrased for retrieval)
  ↓ build_metadata_filter (with canonicalisation against indexed values)
  ↓ vector_search (cosine, ChromaDB, top-10)
  ↓ cross_encoder_rerank (ms-marco-MiniLM, against original query)
  ↓ sentence_anchor (sub-chunk rerank inside the winner)
  ↓ generate_answer (strict-grounded, no fabrication)
  ↓ build_snippet_url (start_time from chunk metadata)
Output: explanation + embedded YouTube player at the exact moment
```

## Demo corpus

| Course | Lecture | Instructor | Topic |
|---|---|---|---|
| MIT 18.06 | 1 | Gilbert Strang | linear-equations |
| MIT 18.06 | 3 | Gilbert Strang | matrix-multiplication |
| Stanford CS229 | 1 | Andrew Ng | intro |
| Stanford CS229 | 2 | Andrew Ng | linear-regression |

## Configuration

This Space requires one secret: **`NVIDIA_API_KEY`** (sign up free at
[build.nvidia.com](https://build.nvidia.com)).

Set it via Space → Settings → Repository secrets.

## Try these queries

- *"explain matrix multiplication"* — pure semantic search
- *"what does Strang say in lecture 3 about matrix multiplication?"* — multi-condition filter (instructor + lecture + topic)
- *"explain linear regression"* — single topic filter pre-narrows to the right video
- *"what is taught in CS229?"* — course filter + reranker
- *"explain convolutional neural networks"* — off-corpus, low-confidence flag fires
