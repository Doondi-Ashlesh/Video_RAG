"""Gradio app — entry point for Hugging Face Spaces deployment.

Wraps the existing query pipeline (`scripts.query.run_query`) in a small UI
that:
  - takes a natural-language question
  - shows the grounded answer
  - embeds the YouTube player at the snippet timestamp (the headline output
    of the system)
  - exposes QAM filters, the rephrased semantic element, and the raw winning
    chunk in a Debug accordion so the architecture is visible on stage

Cold-start behaviour (Spaces filesystem is ephemeral on the free tier):
  1. If `chroma_db/` already has data, use it.
  2. Else if `chroma_db.tar.gz` is present in the repo, extract it.
  3. Else, ingest the bundled demo corpus on the fly (~3-4 minutes, slow but
     fully self-contained).

The NVIDIA NIM API key is read from the NVIDIA_API_KEY environment variable —
on Spaces, set it via Settings -> Repository secrets, never commit a .env.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

# Make `src` importable when this module is the entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from src.config import settings
from src.db import get_store


# ---------------------------------------------------------------------------
# Cold-start: make sure ChromaDB is populated before the first query lands.
# ---------------------------------------------------------------------------

_DEMO_CORPUS = [
    {
        "url": "https://www.youtube.com/watch?v=J7DzL2_Na80",
        "course_id": "MIT-18.06",
        "instructor": "Gilbert Strang",
        "lecture_number": "1",
        "topic": "linear-equations",
    },
    {
        "url": "https://www.youtube.com/watch?v=FX4C-JpTFgY",
        "course_id": "MIT-18.06",
        "instructor": "Gilbert Strang",
        "lecture_number": "3",
        "topic": "matrix-multiplication",
    },
    {
        "url": "https://www.youtube.com/watch?v=jGwO_UgTS7I",
        "course_id": "CS229",
        "instructor": "Andrew Ng",
        "lecture_number": "1",
        "topic": "intro",
    },
    {
        "url": "https://www.youtube.com/watch?v=4b4MUYve_U8",
        "course_id": "CS229",
        "instructor": "Andrew Ng",
        "lecture_number": "2",
        "topic": "linear-regression",
    },
]


def _ensure_corpus_loaded() -> None:
    """Idempotent: returns fast if the collection already has data."""
    store = get_store()
    if store.count() > 0:
        print(f"[startup] ChromaDB already populated: {store.count()} chunks.")
        return

    bundle = Path("chroma_db.tar.gz")
    if bundle.exists():
        print(f"[startup] Extracting bundled ChromaDB from {bundle}...")
        with tarfile.open(bundle, "r:gz") as tar:
            tar.extractall(path=".")
        # Re-open the store after extraction so it picks up the new files.
        get_store.cache_clear()  # type: ignore[attr-defined]
        store = get_store()
        print(f"[startup] Restored {store.count()} chunks from bundle.")
        return

    print("[startup] No ChromaDB found — ingesting demo corpus from scratch.")
    print("[startup] This takes ~3-4 minutes on first cold start.")
    for entry in _DEMO_CORPUS:
        cmd = [
            sys.executable, "-m", "scripts.ingest_video",
            "--url", entry["url"],
            "--course-id", entry["course_id"],
            "--instructor", entry["instructor"],
            "--lecture-number", entry["lecture_number"],
            "--topic", entry["topic"],
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[startup] WARNING: ingest failed for {entry['url']}")

    get_store.cache_clear()  # type: ignore[attr-defined]
    store = get_store()
    print(f"[startup] Ingestion complete: {store.count()} chunks loaded.")


# ---------------------------------------------------------------------------
# Pipeline call (delegates to scripts.query.run_query — single source of truth)
# ---------------------------------------------------------------------------
from scripts.query import run_query  # noqa: E402 — must come after sys.path tweak


_VIDEO_ID_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]{11})")

# Videos whose owners disabled iframe embedding on third-party sites.
# For these we render a thumbnail-with-link instead of an iframe that would
# show YouTube's "Video unavailable" message inline.
_NO_EMBED_VIDEO_IDS = {
    "jGwO_UgTS7I",  # Stanford CS229 Lec 1
    "4b4MUYve_U8",  # Stanford CS229 Lec 2
}


def _video_id_from_url(url: str) -> str | None:
    m = _VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def _thumbnail_player(vid: str, snippet_url: str, start_time: int) -> str:
    """Clickable thumbnail with a play overlay — for embed-disabled videos."""
    thumb = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
    return (
        f'<a href="{snippet_url}" target="_blank" style="display:block;position:relative;text-decoration:none;">'
        f'  <img src="{thumb}" alt="Watch on YouTube" '
        '       style="width:100%;display:block;border-radius:8px;" />'
        '  <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'
        '              background:rgba(0,0,0,0.75);color:white;padding:14px 22px;'
        '              border-radius:8px;font-size:17px;font-weight:600;">'
        f'    ▶ Watch on YouTube at {start_time}s'
        '  </div>'
        '</a>'
        '<p style="text-align:center;margin-top:8px;color:#888;font-size:13px;">'
        '  Embedding disabled by the video owner — opens in a new tab.'
        '</p>'
    )


def _embed_player(snippet_url: str, start_time: int, end_time: int | None = None) -> str:
    """Return an HTML iframe that plays the YouTube video at the snippet timestamp.

    If end_time is provided, YouTube auto-stops playback at that second so the
    student doesn't drift past the relevant chunk window. For videos whose
    owner disabled iframe embedding, returns a clickable thumbnail instead.
    """
    vid = _video_id_from_url(snippet_url)
    if not vid:
        return f'<a href="{snippet_url}" target="_blank">{snippet_url}</a>'

    if vid in _NO_EMBED_VIDEO_IDS:
        return _thumbnail_player(vid, snippet_url, start_time)

    embed_url = f"https://www.youtube.com/embed/{vid}?start={start_time}&autoplay=0"
    if end_time and end_time > start_time:
        embed_url += f"&end={end_time}"
    return (
        '<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;">'
        f'<iframe src="{embed_url}" '
        'style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;" '
        'allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
        'allowfullscreen></iframe>'
        '</div>'
    )


def _format_debug(qam, chunk_text: str, score: float) -> str:
    return (
        f"**metadata_filters:** `{qam.metadata_filters}`\n\n"
        f"**semantic_element:** {qam.semantic_element}\n\n"
        f"**cross-encoder score:** `{score:.3f}` "
        f"(threshold = `{settings.rerank_confidence_threshold}`)\n\n"
        f"**chunk_text (the segment grounding the answer):**\n\n```\n{chunk_text}\n```"
    )


def query_handler(question: str):
    if not question or not question.strip():
        return (
            "Please type a question.",
            "",
            "",
            "",
        )

    response = run_query(question.strip(), top_k=settings.top_k_retrieval)
    if response is None:
        return (
            "No relevant lecture content found for this query.",
            "",
            "",
            "",
        )

    confidence_banner = ""
    if response.low_confidence:
        confidence_banner = (
            "> ⚠ **Low-confidence match.** The closest chunk may not fully "
            "answer the query — see the source clip and chunk text below.\n\n"
        )

    # Show the YouTube-labelled chapter when chapter-override fired — that's
    # the headline architectural improvement: we anchored to the lecturer's
    # own chapter boundaries, not just our chunk boundaries.
    chapter_line = ""
    if response.chapter_title:
        chapter_line = f"**YouTube chapter:** *{response.chapter_title}*  \n"

    answer_md = (
        f"{confidence_banner}{response.explanation}\n\n"
        f"---\n"
        f"**Primary clip:** *{response.video_title}* — {response.instructor}  \n"
        f"{chapter_line}"
        f"**Window:** {response.start_time}s – {response.end_time}s "
        f"(~{response.end_time - response.start_time}s of context)  \n"
        f"[Open on YouTube ↗]({response.snippet_url})"
    )

    if response.additional_clips:
        answer_md += "\n\n**Other relevant moments** (the answer was synthesised across these too):\n"
        for clip in response.additional_clips:
            answer_md += (
                f"- [*{clip.video_title}* @ {clip.start_time}s–{clip.end_time}s "
                f"(score {clip.cross_encoder_score:.2f})]({clip.snippet_url})\n"
            )

    player_html = _embed_player(
        response.snippet_url, response.start_time, response.end_time
    )
    debug_md = _format_debug(
        response.qam_result, response.chunk_text, response.cross_encoder_score
    )
    return answer_md, player_html, response.snippet_url, debug_md


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_DEMO_QUERIES = [
    "explain linear regression",
    "explain matrix multiplication",
]

_DESCRIPTION = """\
A retrieval-augmented Q&A system over recorded lectures. Ask a natural-language
question; the system finds the precise moment in the right lecture that answers
it and embeds the YouTube player there.

**Demo corpus (4 lectures):**
- MIT 18.06 — *Geometry of Linear Equations* (Strang, Lecture 1)
- MIT 18.06 — *Multiplication and Inverse Matrices* (Strang, Lecture 3)
- Stanford CS229 — *Welcome to Machine Learning* (Ng, Lecture 1)
- Stanford CS229 — *Linear Regression and Gradient Descent* (Ng, Lecture 2)

**Architecture (visible in the Debug panel):** QAM decomposition → metadata
pre-filter → vector search → cross-encoder re-rank → sentence-level anchor →
strict-grounded answer + timestamped snippet URL.
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Video RAG — Lecture Q&A") as demo:
        gr.Markdown("# Video RAG — Lecture Q&A")
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. explain matrix multiplication in lecture 3",
                lines=2,
                scale=4,
            )
            submit = gr.Button("Ask", variant="primary", scale=1)

        gr.Examples(
            examples=[[q] for q in _DEMO_QUERIES],
            inputs=[question],
            label="Demo queries (click to load):",
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Answer")
                answer_out = gr.Markdown()
                snippet_url_out = gr.Textbox(
                    label="Snippet URL", interactive=False, show_copy_button=True
                )
            with gr.Column(scale=1):
                gr.Markdown("### Watch the relevant moment")
                player_out = gr.HTML()

        with gr.Accordion("Debug (QAM filters + semantic element + chunk text)", open=False):
            debug_out = gr.Markdown()

        submit.click(
            fn=query_handler,
            inputs=[question],
            outputs=[answer_out, player_out, snippet_url_out, debug_out],
        )
        question.submit(
            fn=query_handler,
            inputs=[question],
            outputs=[answer_out, player_out, snippet_url_out, debug_out],
        )

    return demo


if __name__ == "__main__":
    if not os.environ.get("NVIDIA_API_KEY") and not settings.nvidia_api_key:
        print(
            "WARNING: NVIDIA_API_KEY is not set. Queries will fail until you set "
            "it (locally via .env, on Spaces via repo secrets)."
        )
    _ensure_corpus_loaded()
    demo = build_ui()
    # On HF Spaces, GRADIO_SERVER_NAME / GRADIO_SERVER_PORT come from env;
    # passing server_name explicitly trips the localhost-accessibility check.
    demo.queue().launch()
