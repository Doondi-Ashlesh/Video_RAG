"""Query & Retrieval Agent CLI.

Run:
    python -m scripts.query --query "explain backpropagation"
    python -m scripts.query --query "what does Prof Ng say about regularization in lecture 5?"

Pipeline (per agents.md):
    1. QAM decomposition       (skill: decompose_query)
    2. Build metadata filter   (skill: build_metadata_filter)
    3. Vector search           (skill: vector_search)
    4. Cross-encoder rerank    (skill: cross_encode_rerank)
    5. Generate answer         (skill: generate_answer)
    6. Build snippet URL       (skill: build_snippet_url)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from src.config import settings
from src.db import get_store
from src.ingestion.embedder import get_embedding_model
from src.models import ClipReference, VideoRAGResponse
from src.retrieval import (
    anchor_to_segment,
    build_metadata_filter,
    build_snippet_url,
    decompose_query,
    generate_answer,
    pick_best_chapter,
    rerank_topn,
    vector_search,
)


console = Console()

# When sentence_anchor shifts the start forward (e.g. chunk 0-45s, refined
# start 38s), the original chunk end (45s) gives only 7s of playback. Pad the
# end_time so the student gets a sensible window of context.
MIN_PLAYBACK_SECONDS = 30


def run_query(query: str, top_k: int) -> VideoRAGResponse | None:
    # Step 1: QAM decomposition
    qam = decompose_query(query)

    # Step 2: Build metadata filter (with store for value canonicalisation —
    # rescues partial names like "Strang" -> "Gilbert Strang" before ChromaDB)
    store = get_store()
    where = build_metadata_filter(qam.metadata_filters, store=store)

    # Step 2b: Chapter-aware filter augmentation.
    # If QAM extracted a topic that matches one or more YouTube chapter titles
    # in the index, restrict the search to chunks within those chapters. This
    # forces chunks like "Linear Regression Algorithm" (884-1086s) into the
    # rerank pool even when their lexical density on the topic is lower than
    # the announcement chunks at the start of the lecture.
    qam_topic = (qam.metadata_filters.get("topic") or "").strip().lower()
    if qam_topic:
        topic_words = set(qam_topic.replace("-", " ").split())
        try:
            all_meta = store._collection.get(include=["metadatas"])["metadatas"] or []
            unique_chapter_titles = {
                m["chapter_title"] for m in all_meta if m.get("chapter_title")
            }
            matching = [
                ct for ct in unique_chapter_titles
                if topic_words.issubset(set(ct.lower().split()))
            ]
            if matching:
                logger.info(
                    "Topic '{}' matches {} chapter title(s): {}",
                    qam_topic,
                    len(matching),
                    matching,
                )
                chapter_clause = {"chapter_title": {"$in": matching}}
                if where is None:
                    where = chapter_clause
                elif "$and" in where:
                    where["$and"].append(chapter_clause)
                else:
                    where = {"$and": [where, chapter_clause]}
        except Exception as e:
            logger.warning("Chapter-title lookup failed ({}); continuing without it.", e)

    # Step 3: Vector search
    embed_model = get_embedding_model()
    candidates = vector_search(
        semantic_element=qam.semantic_element,
        where_filter=where,
        store=store,
        model=embed_model,
        top_k=top_k,
    )
    if not candidates:
        console.print("[red]No relevant lecture content found for this query.[/red]")
        return None

    # Step 4: Cross-encoder re-ranking against the original query.
    # We pull the top-3 chunks (not just top-1) because broad explanatory
    # queries — "explain linear regression" — span multiple chunks. The
    # answer LLM then synthesises across all three, and the UI surfaces all
    # three as jump-to clips so the student can navigate to whichever moment
    # matches their need.
    ranked = rerank_topn(qam.original_query, candidates, n=3)
    if not ranked:
        console.print("[red]Re-ranking returned no winner.[/red]")
        return None

    # Step 4b: Chapter-aware promotion via the LLM (no hardcoded heuristics).
    # When the QAM topic matches multiple chapters in the top-N reranked
    # chunks, ask the LLM which chapter title most likely contains the
    # substantive answer. This handles cases like "linear regression" matching
    # both "Motivate Linear Regression" and "Linear Regression Algorithm" —
    # the LLM picks the latter without us hand-coding what counts as a
    # "meta" word. Falls back gracefully to the top-1 chunk on any LLM fail.
    qam_topic = (qam.metadata_filters.get("topic") or "").strip().lower()
    chapter_winner = None
    if qam_topic:
        topic_words = set(qam_topic.replace("-", " ").split())
        # Find the unique chapter titles represented in the top-N reranked
        # chunks that match the topic (topic words ⊆ chapter title words).
        chunks_by_chapter: dict[str, Any] = {}
        for cand in ranked:
            ch_title = (cand.metadata.get("chapter_title") or "").strip()
            if not ch_title:
                continue
            if not topic_words.issubset(set(ch_title.lower().split())):
                continue
            # Keep the highest-cross-encoder-scored chunk per chapter
            if ch_title not in chunks_by_chapter or (
                cand.cross_encoder_score
                > chunks_by_chapter[ch_title].cross_encoder_score
            ):
                chunks_by_chapter[ch_title] = cand

        if chunks_by_chapter:
            picked = pick_best_chapter(qam.original_query, list(chunks_by_chapter.keys()))
            if picked and picked in chunks_by_chapter:
                chapter_winner = chunks_by_chapter[picked]
            else:
                # LLM failed or returned nothing usable — fall back to the
                # highest-cross-encoder-scored chunk among the matching chapters.
                chapter_winner = max(
                    chunks_by_chapter.values(),
                    key=lambda c: c.cross_encoder_score,
                )
            logger.info(
                "Chapter promotion -> '{}' ({}s-{}s)",
                chapter_winner.metadata.get("chapter_title"),
                chapter_winner.metadata.get("chapter_start"),
                chapter_winner.metadata.get("chapter_end"),
            )

    if chapter_winner is not None:
        winner = chapter_winner
        secondary = [c for c in ranked if c is not winner][:2]
        # Use the chapter boundaries directly — the lecturer's own segmentation.
        refined_start = int(winner.metadata.get("chapter_start", winner.start_time))
        extended_end = int(winner.metadata.get("chapter_end", winner.end_time))
        logger.info(
            "Chapter-aware override: matched topic '{}' to chapter '{}' ({}s-{}s)",
            qam_topic,
            winner.metadata.get("chapter_title"),
            refined_start,
            extended_end,
        )
    else:
        winner = ranked[0]
        secondary = ranked[1:]
        # Standard sentence-anchor path.
        refined_start = anchor_to_segment(qam.original_query, winner)
        extended_end = max(winner.end_time, refined_start + MIN_PLAYBACK_SECONDS)

    low_confidence = winner.cross_encoder_score < settings.rerank_confidence_threshold

    # Step 5: Generate the answer using all top-N chunks as combined context.
    explanation = generate_answer(
        query=qam.original_query,
        chunk_text=[winner.text] + [c.text for c in secondary],
    )

    # Step 6: Build the primary snippet URL.
    snippet_url = build_snippet_url(winner.video_id, refined_start)

    additional_clips: list[ClipReference] = []
    for clip in secondary:
        clip_start = anchor_to_segment(qam.original_query, clip)
        clip_end = max(clip.end_time, clip_start + MIN_PLAYBACK_SECONDS)
        additional_clips.append(
            ClipReference(
                snippet_url=build_snippet_url(clip.video_id, clip_start),
                video_title=clip.metadata.get("title", ""),
                instructor=clip.metadata.get("instructor", ""),
                start_time=clip_start,
                end_time=clip_end,
                cross_encoder_score=clip.cross_encoder_score,
            )
        )

    chapter_title_for_response = (
        winner.metadata.get("chapter_title", "")
        if chapter_winner is not None
        else ""
    )

    return VideoRAGResponse(
        explanation=explanation,
        snippet_url=snippet_url,
        video_title=winner.metadata.get("title", ""),
        instructor=winner.metadata.get("instructor", ""),
        start_time=refined_start,
        end_time=extended_end,
        chunk_text=winner.text,
        qam_result=qam,
        cross_encoder_score=winner.cross_encoder_score,
        low_confidence=low_confidence,
        additional_clips=additional_clips,
        chapter_title=chapter_title_for_response,
    )


def query(
    query: str = typer.Option(..., "--query", "-q", help="Natural language question"),
    top_k: int = typer.Option(None, "--top-k", help="Override TOP_K_RETRIEVAL"),
    show_debug: bool = typer.Option(False, "--debug", help="Print QAM decomposition + chunk text"),
) -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)

    k = top_k or settings.top_k_retrieval
    response = run_query(query, top_k=k)
    if response is None:
        raise typer.Exit(code=1)

    if response.low_confidence:
        # Plain ASCII marker so this prints correctly on cp1252 Windows
        # consoles without forcing the user to chcp 65001.
        console.print(
            "[yellow][!] Low-confidence match - the closest chunk may not fully answer the query.[/yellow]"
        )

    console.print(Panel(response.explanation, title="Answer", border_style="green"))
    console.print(
        f"[bold]Watch the relevant moment:[/bold] [cyan]{response.snippet_url}[/cyan]"
    )
    console.print(
        f"[dim]Source: {response.video_title} by {response.instructor} at {response.start_time}s "
        f"(rerank score: {response.cross_encoder_score:.3f})[/dim]"
    )

    if show_debug:
        console.print(
            Panel(
                f"metadata_filters: {response.qam_result.metadata_filters}\n"
                f"semantic_element: {response.qam_result.semantic_element}\n\n"
                f"chunk_text:\n{response.chunk_text}",
                title="Debug",
                border_style="magenta",
            )
        )


if __name__ == "__main__":
    typer.run(query)
