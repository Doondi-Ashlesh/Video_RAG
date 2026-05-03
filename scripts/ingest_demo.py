"""Ingest the 4-video demo corpus (2 MIT + 2 Stanford) in one shot.

Run from the project root:
    python -m scripts.ingest_demo
    python -m scripts.ingest_demo --force        # re-ingest all
    python -m scripts.ingest_demo --videos 1 3   # only ingest videos 1 and 3

Each entry maps to one --url + a deliberate set of QAM filter tags so the
demo queries in DEMO.md can prove every architectural feature end-to-end.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess
import typer
from rich.console import Console


# ----------------------------------------------------------- demo corpus
DEMO_CORPUS: list[dict[str, str | int]] = [
    {
        "url": "https://www.youtube.com/watch?v=J7DzL2_Na80",
        "course_id": "MIT-18.06",
        "instructor": "Gilbert Strang",
        "lecture_number": 1,
        "topic": "linear-equations",
        "label": "MIT 18.06 - Lec 1: Geometry of Linear Equations",
    },
    {
        "url": "https://www.youtube.com/watch?v=FX4C-JpTFgY",
        "course_id": "MIT-18.06",
        "instructor": "Gilbert Strang",
        "lecture_number": 3,
        "topic": "matrix-multiplication",
        "label": "MIT 18.06 - Lec 3: Multiplication and Inverse Matrices",
    },
    {
        "url": "https://www.youtube.com/watch?v=jGwO_UgTS7I",
        "course_id": "CS229",
        "instructor": "Andrew Ng",
        "lecture_number": 1,
        "topic": "intro",
        "label": "Stanford CS229 - Lec 1: Welcome",
    },
    {
        "url": "https://www.youtube.com/watch?v=4b4MUYve_U8",
        "course_id": "CS229",
        "instructor": "Andrew Ng",
        "lecture_number": 2,
        "topic": "linear-regression",
        "label": "Stanford CS229 - Lec 2: Linear Regression and Gradient Descent",
    },
]


console = Console()


def _ingest_one(entry: dict[str, str | int], force: bool) -> int:
    cmd = [
        sys.executable,
        "-m",
        "scripts.ingest_video",
        "--url",
        str(entry["url"]),
        "--course-id",
        str(entry["course_id"]),
        "--instructor",
        str(entry["instructor"]),
        "--lecture-number",
        str(entry["lecture_number"]),
        "--topic",
        str(entry["topic"]),
    ]
    if force:
        cmd.append("--force")

    console.rule(f"[bold cyan]{entry['label']}[/bold cyan]")
    return subprocess.call(cmd)


def main(
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already in the index"),
    videos: list[int] = typer.Option(
        None, "--videos", help="1-indexed video numbers to ingest (default: all)"
    ),
) -> None:
    indices = videos if videos else list(range(1, len(DEMO_CORPUS) + 1))
    selected = [DEMO_CORPUS[i - 1] for i in indices if 1 <= i <= len(DEMO_CORPUS)]

    if not selected:
        console.print("[red]No videos selected.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Ingesting {len(selected)} video(s)[/bold]\n")
    failures: list[str] = []
    for entry in selected:
        rc = _ingest_one(entry, force=force)
        if rc != 0:
            failures.append(str(entry["label"]))

    console.rule("[bold]Demo ingestion summary[/bold]")
    if failures:
        console.print(f"[yellow]Completed with {len(failures)} failure(s):[/yellow]")
        for f in failures:
            console.print(f"  [yellow]- {f}[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[green]All {len(selected)} videos ingested successfully.[/green]")
    console.print(
        "\nTry a demo query:\n"
        "  [cyan]python -m scripts.query --query \"explain matrix multiplication\" --debug[/cyan]\n"
        "See [bold]DEMO.md[/bold] for the full showcase."
    )


if __name__ == "__main__":
    typer.run(main)
