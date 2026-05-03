"""Bundle the populated ChromaDB into a tar.gz for shipping with the HF Space.

Hugging Face Spaces' free-tier storage is ephemeral (resets on container
restart). The cleanest workaround for a small demo corpus is to commit the
populated DB as a compressed tarball alongside the code; `app.py` extracts
it at startup if `chroma_db/` is empty.

Run from the project root:
    python -m scripts.bundle_chroma_db
    python -m scripts.bundle_chroma_db --output chroma_db.tar.gz
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console

from src.config import settings


console = Console()


def main(
    output: Path = typer.Option(
        Path("chroma_db.tar.gz"), "--output", "-o", help="Tarball destination"
    ),
) -> None:
    src_dir = settings.chroma_persist_dir
    if not src_dir.exists() or not any(src_dir.iterdir()):
        console.print(
            f"[red]Source ChromaDB directory {src_dir} is missing or empty.[/red] "
            "Run the ingestion scripts first."
        )
        raise typer.Exit(code=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"Compressing [cyan]{src_dir}[/cyan] -> [cyan]{output}[/cyan]...")
    with tarfile.open(output, "w:gz", compresslevel=6) as tar:
        tar.add(src_dir, arcname=src_dir.name)

    size_mb = output.stat().st_size / (1024 * 1024)
    console.print(
        f"[green]Bundle written:[/green] {output} ({size_mb:.1f} MB)\n"
        "Commit this file alongside app.py so Spaces can restore it on cold start."
    )


if __name__ == "__main__":
    typer.run(main)
