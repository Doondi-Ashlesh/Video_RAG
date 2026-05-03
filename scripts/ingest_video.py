"""Ingestion Agent CLI.

Run:
    python -m scripts.ingest_video --url <URL> --course-id CS231 --instructor "Andrej Karpathy" --lecture-number 5 --topic "backpropagation"
    python -m scripts.ingest_video --video-id dQw4w9WgXcQ --force

Pipeline (per agents.md):
    1. Validate & deduplicate
    2. Fetch transcript (skill: fetch_youtube_transcript)
    3. Chunk with timestamps (skill: chunk_with_timestamps)
    4. Embed and store (skill: embed_and_store)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/ingest_video.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from loguru import logger
from rich.console import Console

from src.config import settings
from src.db import get_store
from src.ingestion import (
    TranscriptUnavailableError,
    chunk_transcript,
    embed_and_store,
    extract_video_id,
    fetch_transcript,
    fetch_video_meta,
    get_embedding_model,
)


console = Console()


def ingest(
    url: str = typer.Option(None, "--url", help="Full YouTube URL"),
    video_id: str = typer.Option(None, "--video-id", help="11-char YouTube video ID"),
    course_id: str = typer.Option(None, "--course-id", help="e.g. CS231"),
    instructor: str = typer.Option(None, "--instructor", help="Override channel name as instructor"),
    lecture_number: int = typer.Option(None, "--lecture-number", help="Lecture number within the course"),
    topic: str = typer.Option(None, "--topic", help="Topical tag, lowercase (e.g. 'backpropagation')"),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already in the index"),
) -> None:
    if not url and not video_id:
        raise typer.BadParameter("Provide either --url or --video-id")

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)

    vid = extract_video_id(url) if url else video_id
    console.print(f"[bold]Ingesting[/bold] video_id=[cyan]{vid}[/cyan]")

    store = get_store()

    # Step 1: Validate & deduplicate
    if store.video_exists(vid):
        if not force:
            console.print(
                f"[yellow]Video {vid} already ingested. Use --force to re-ingest.[/yellow]"
            )
            raise typer.Exit(code=0)
        store.delete_video(vid)

    # Step 2: Fetch transcript
    try:
        segments = fetch_transcript(vid)
    except TranscriptUnavailableError as e:
        console.print(f"[red]Transcript unavailable:[/red] {e}")
        raise typer.Exit(code=1)

    meta = fetch_video_meta(
        vid,
        course_id=course_id,
        instructor_override=instructor,
        lecture_number=lecture_number,
        topic=topic.lower() if topic else None,
    )
    console.print(
        f"  title=[bold]{meta.title}[/bold]  instructor=[bold]{meta.instructor}[/bold]  "
        f"course=[bold]{meta.course_id}[/bold]  lecture=[bold]{meta.lecture_number}[/bold]  "
        f"topic=[bold]{meta.topic}[/bold]"
    )

    # Step 3: Chunk with timestamps
    chunks = chunk_transcript(
        segments,
        video_id=vid,
        window_seconds=settings.chunk_window_seconds,
        overlap_seconds=settings.chunk_overlap_seconds,
        min_words=settings.min_chunk_words,
    )
    if not chunks:
        console.print("[red]No usable chunks produced (transcript too short or empty).[/red]")
        raise typer.Exit(code=1)
    console.print(f"  built [bold]{len(chunks)}[/bold] chunks")

    # Step 4: Embed and store
    model = get_embedding_model()
    n_stored = embed_and_store(chunks, meta, store=store, model=model)

    console.print(
        f"[green]Ingestion complete:[/green] {n_stored} chunks stored "
        f"for [bold]{meta.title}[/bold] ({vid})"
    )


if __name__ == "__main__":
    typer.run(ingest)
