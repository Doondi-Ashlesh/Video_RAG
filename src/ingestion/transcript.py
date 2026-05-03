"""Skill: fetch_youtube_transcript.

Resolves a YouTube URL or ID, fetches the transcript via youtube-transcript-api
(preferring manually-created transcripts over auto-generated ones), and pulls
basic video metadata via yt-dlp.

Compatible with both youtube-transcript-api 0.6.x (static `list_transcripts`,
list[dict] segments) and 1.x (instance `list`/`fetch`, FetchedTranscript with
`.snippets`). The pipeline below normalises both shapes into our internal
RawTranscriptSegment dataclass.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from loguru import logger
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from src.models import RawTranscriptSegment, VideoMeta


class TranscriptUnavailableError(Exception):
    """Raised when no usable transcript exists for the requested video."""


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id: str) -> str:
    """Accept either a full YouTube URL or a bare 11-char video id."""
    s = url_or_id.strip()
    if _VIDEO_ID_RE.match(s):
        return s

    parsed = urlparse(s)
    if parsed.hostname in {"youtu.be"}:
        candidate = parsed.path.lstrip("/")
        if _VIDEO_ID_RE.match(candidate):
            return candidate
    if parsed.hostname and "youtube.com" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if "v" in qs and _VIDEO_ID_RE.match(qs["v"][0]):
            return qs["v"][0]
        # /embed/<id> or /shorts/<id>
        m = re.match(r"^/(embed|shorts|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(2)

    raise ValueError(f"Could not extract a YouTube video id from {url_or_id!r}")


def _segment_field(seg: Any, name: str) -> Any:
    """Tolerate both dict (0.6.x) and dataclass-like (1.x) segment shapes."""
    if isinstance(seg, dict):
        return seg[name]
    return getattr(seg, name)


def _list_transcripts(video_id: str) -> Any:
    """Cross-version transcript-listing call.

    1.x:   YouTubeTranscriptApi().list(video_id)         (instance method)
    0.6.x: YouTubeTranscriptApi.list_transcripts(...)    (static)
    """
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        return YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore[attr-defined]
    return YouTubeTranscriptApi().list(video_id)


def _iter_snippets(fetched: Any) -> list[Any]:
    """1.x returns FetchedTranscript with .snippets; 0.6.x returns list[dict]."""
    if hasattr(fetched, "snippets"):
        return list(fetched.snippets)
    return list(fetched)


def fetch_transcript(video_id: str) -> list[RawTranscriptSegment]:
    """Fetch transcript segments for a YouTube video.

    Prefers manually-created English transcripts; falls back to auto-generated.
    Raises TranscriptUnavailableError if neither exists.
    """
    try:
        listing = _list_transcripts(video_id)
        try:
            transcript = listing.find_manually_created_transcript(["en"])
            source = "manual"
        except NoTranscriptFound:
            transcript = listing.find_generated_transcript(["en"])
            source = "auto-generated"

        fetched = transcript.fetch()
        raw_segments = _iter_snippets(fetched)
        segments = [
            RawTranscriptSegment(
                text=str(_segment_field(s, "text")),
                start=float(_segment_field(s, "start")),
                duration=float(_segment_field(s, "duration")),
            )
            for s in raw_segments
        ]
        logger.info(
            "Fetched {} transcript segments ({}) for video_id={}",
            len(segments),
            source,
            video_id,
        )
        return segments

    except (TranscriptsDisabled, NoTranscriptFound) as e:
        raise TranscriptUnavailableError(
            f"No transcript available for {video_id}: {e}"
        ) from e


def fetch_video_meta(
    video_id: str,
    *,
    course_id: str | None = None,
    instructor_override: str | None = None,
    lecture_number: int | None = None,
    topic: str | None = None,
) -> VideoMeta:
    """Pull title, channel, upload date, and duration via yt-dlp.

    The lecturer-specific fields (course_id, lecture_number, topic) are passed
    through from the CLI — yt-dlp doesn't know about them. instructor defaults
    to the channel name unless overridden.
    """
    from yt_dlp import YoutubeDL

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            f"https://www.youtube.com/watch?v={video_id}",
            download=False,
        )

    upload_date_raw = info.get("upload_date") or "19700101"  # YYYYMMDD
    upload_date = (
        f"{upload_date_raw[:4]}-{upload_date_raw[4:6]}-{upload_date_raw[6:8]}"
    )

    # Pull YouTube chapter markers (lecturer-defined topical segmentation).
    # Some videos have none; that's fine — chapter-aware retrieval is an
    # opportunistic enhancement, the rest of the pipeline still works.
    chapters_raw = info.get("chapters") or []
    chapters = [
        {
            "start": int(c.get("start_time", 0)),
            "end": int(c.get("end_time", 0)),
            "title": str(c.get("title", "")).strip(),
        }
        for c in chapters_raw
        if c.get("title")
    ]
    if chapters:
        logger.info("Found {} YouTube chapters for {}", len(chapters), video_id)

    return VideoMeta(
        video_id=video_id,
        title=info.get("title", "Unknown title"),
        instructor=instructor_override or info.get("channel", "Unknown instructor"),
        upload_date=upload_date,
        duration_seconds=int(info.get("duration") or 0),
        course_id=course_id,
        lecture_number=lecture_number,
        topic=topic,
        chapters=chapters,
    )
