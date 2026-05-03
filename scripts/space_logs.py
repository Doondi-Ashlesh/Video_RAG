"""Fetch HF Space build + runtime logs without leaving the terminal.

Usage:
    set HF_TOKEN=hf_...                       # PowerShell: $env:HF_TOKEN="hf_..."
    python -m scripts.space_logs DoondiAshlesh/V_RAG --tail 200
    python -m scripts.space_logs DoondiAshlesh/V_RAG --type build
    python -m scripts.space_logs DoondiAshlesh/V_RAG --type run --follow

Requires a Hugging Face token with read access to the Space repo. Reads from
the env var HF_TOKEN (or HUGGING_FACE_HUB_TOKEN).
"""
from __future__ import annotations

import os
import sys
import time

import typer
import urllib.request


app = typer.Typer(add_completion=False)


def _fetch(space_id: str, kind: str, token: str) -> str:
    url = f"https://huggingface.co/api/spaces/{space_id}/logs/{kind}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main(
    space_id: str = typer.Argument(..., help="user/space (e.g. DoondiAshlesh/V_RAG)"),
    kind: str = typer.Option("run", "--type", "-t", help="build | run"),
    tail: int = typer.Option(200, "--tail", "-n", help="Last N lines"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Poll for new lines"),
) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN env var to your Hugging Face access token.", file=sys.stderr)
        raise typer.Exit(code=1)

    seen = 0
    while True:
        try:
            raw = _fetch(space_id, kind, token)
        except Exception as e:
            print(f"[logs] fetch failed: {e}", file=sys.stderr)
            if not follow:
                raise typer.Exit(code=1)
            time.sleep(3)
            continue

        lines = [ln for ln in raw.splitlines() if ln and not ln.startswith(":")]
        if follow:
            new = lines[seen:]
            if new:
                for ln in new:
                    print(ln)
                seen = len(lines)
            time.sleep(2)
        else:
            for ln in lines[-tail:]:
                print(ln)
            break


if __name__ == "__main__":
    typer.run(main)
