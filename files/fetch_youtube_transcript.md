# Skill: fetch_youtube_transcript

## Purpose
Fetch the transcript and video metadata for a given YouTube video using the `youtube-transcript-api` library.

## File
`src/ingestion/transcript.py`

## Implementation

```python
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class RawTranscriptSegment:
    text: str
    start: float      # seconds
    duration: float   # seconds

@dataclass
class VideoMeta:
    video_id: str
    title: str
    channel_name: str
    upload_date: str  # YYYY-MM-DD
    duration_seconds: int

def fetch_transcript(video_id: str) -> list[RawTranscriptSegment]:
    """
    Fetch transcript segments for a YouTube video.
    Prefers manually-created transcripts over auto-generated.
    Raises TranscriptUnavailableError if none exist.
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manual over auto-generated
        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(['en'])
        
        raw = transcript.fetch()
        return [RawTranscriptSegment(text=s['text'], start=s['start'], duration=s['duration']) for s in raw]
    
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        raise TranscriptUnavailableError(f"No transcript available for {video_id}: {e}")

class TranscriptUnavailableError(Exception):
    pass
```

## Notes
- The `start` field in each segment is the key — it maps to the timestamp used in the final YouTube URL
- Do not strip or modify `start` values at this stage — the chunker needs raw floats
- Video metadata (title, channel, upload_date) is fetched separately using `pytube` or `yt-dlp` and merged before storage
