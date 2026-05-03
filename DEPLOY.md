# Deploying to Hugging Face Spaces

Run the demo as a public web app at no cost. The whole pipeline lives in one
Space — embedder, reranker, ChromaDB, and the Gradio UI all run there. The
NVIDIA NIM API key sits in Space secrets.

## Why HF Spaces

- 16 GB RAM / 2 CPU on the free tier — 8× what we need.
- Models load **once** at container boot and stay warm; subsequent queries
  are sub-second (vs. ~30 s cold-load locally because every CLI invocation
  spawns a fresh Python process).
- Free, shareable URL. Public access, no install.
- ChromaDB is bundled in the repo as a tarball, extracted at boot.

## What gets deployed

```
.
├── app.py                  ← Gradio entry point (the file Spaces runs)
├── README.md               ← copy of SPACE_README.md (frontmatter tells Spaces this is a Gradio app)
├── requirements.txt        ← installed at container build
├── chroma_db.tar.gz        ← populated vector DB, ~80 MB compressed
├── src/                    ← the entire pipeline
└── scripts/                ← ingestion + query CLIs (used as fallback in app.py)
```

## One-time setup

### 1. Sign in / sign up on Hugging Face

[huggingface.co](https://huggingface.co) → free account, no card.

### 2. Create a Space

[huggingface.co/new-space](https://huggingface.co/new-space)

- **Owner:** your username
- **Space name:** `video-rag-demo` (or anything)
- **License:** MIT
- **SDK:** **Gradio**
- **Hardware:** **CPU basic** (free)
- **Visibility:** Public (or Private — same compute)

### 3. Add the NVIDIA API key as a secret

On your new Space:

> **Settings → Variables and secrets → New secret**
> - Name: `NVIDIA_API_KEY`
> - Value: `nvapi-...` (your key from build.nvidia.com)

This is stored encrypted; never commit `.env` to the Space repo.

### 4. Bundle the ChromaDB locally (one time)

You need a populated `chroma_db/` first. If you've already run
`python -m scripts.ingest_demo`, skip the ingest step; otherwise:

```bash
python -m scripts.ingest_demo            # fills ./chroma_db
python -m scripts.bundle_chroma_db       # writes chroma_db.tar.gz (~80 MB)
```

### 5. Push the deployable files to the Space repo

```bash
# Clone the empty Space repo
git clone https://huggingface.co/spaces/<your-username>/video-rag-demo space
cd space

# Copy the files Spaces actually needs
cp ../app.py .
cp ../requirements.txt .
cp ../SPACE_README.md README.md          # rename — Spaces reads frontmatter from README.md
cp -r ../src .
cp -r ../scripts .
cp ../chroma_db.tar.gz .

# Commit and push
git add .
git commit -m "Initial deploy — Video RAG demo with bundled ChromaDB"
git push
```

The Space will start building. First boot takes ~3-5 minutes (downloads
sentence-transformers model + cross-encoder, extracts ChromaDB). Subsequent
boots are ~30 s because models cache.

### 6. Test it

Open the Space URL. Click one of the demo queries. Within ~3 s you should
see:
- The grounded answer
- The embedded YouTube player at the snippet timestamp
- The Debug panel showing QAM filter + semantic_element + chunk text

## Rebuilds

When you edit code locally and want to ship the change:

```bash
cp app.py src/... scripts/... ../space/      # mirror the changed files
cd ../space
git add . && git commit -m "..." && git push
```

The Space rebuilds automatically. ~30 s downtime.

## Updating the corpus

If you re-ingest a video locally, regenerate the bundle and push it:

```bash
python -m scripts.ingest_video --url "..." --course-id "..." ...    # local
python -m scripts.bundle_chroma_db                                  # writes chroma_db.tar.gz
cp chroma_db.tar.gz ../space/
cd ../space && git add chroma_db.tar.gz && git commit -m "Update corpus" && git push
```

The Space's `chroma_db/` is wiped on each container restart, so the bundle
is always the source of truth.

## Cold-start fallback (if the bundle is missing)

If `chroma_db.tar.gz` isn't in the Space repo, `app.py` falls back to
ingesting the 4-video demo corpus from scratch. This takes ~3-4 minutes on
the first cold start (transcript fetch + embedding for each video). The
bundle approach is recommended because it's faster and deterministic.

## Local dry-run

Before pushing, sanity-check `app.py` runs locally:

```bash
python app.py
# Visit http://localhost:7860
```

Set `NVIDIA_API_KEY` in `.env` first.

## Troubleshooting

- **"NVIDIA_API_KEY is not set"** in Space logs — confirm you added it as a
  *secret* (not a variable) under Settings.
- **Cold start hangs at "Downloading model"** — first boot is slow on free
  tier; second boot uses the model cache. Patience.
- **No queries return** — check Space logs for the QAM decomposition log
  line; if it's missing, the API call failed (check the key, check NIM rate
  limits).
- **Player iframe blank** — YouTube blocks embedding for some videos in
  some regions; the snippet URL link still works.
