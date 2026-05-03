# Skill: chunk_with_timestamps

## Purpose
Merge raw transcript segments into overlapping fixed-duration chunks, each carrying a precise `start_time` timestamp used to construct the final YouTube snippet URL.

## File
`src/ingestion/chunker.py`

## Implementation

```python
from dataclasses import dataclass
from src.ingestion.transcript import RawTranscriptSegment
import re

@dataclass
class TranscriptChunk:
    chunk_id: str       # "{video_id}_chunk_{index}"
    video_id: str
    text: str
    start_time: int     # integer seconds — used directly in YouTube URL
    end_time: int

FILLER_PATTERN = re.compile(r'\[.*?\]')  # removes [Music], [Applause], etc.

def chunk_transcript(
    segments: list[RawTranscriptSegment],
    video_id: str,
    window_seconds: int = 45,
    overlap_seconds: int = 10,
    min_words: int = 20
) -> list[TranscriptChunk]:
    """
    Merge segments into overlapping time-windowed chunks.
    Each chunk covers `window_seconds` of content with `overlap_seconds` overlap.
    """
    chunks = []
    total_duration = segments[-1].start + segments[-1].duration
    step = window_seconds - overlap_seconds
    chunk_index = 0
    t = 0.0

    while t < total_duration:
        window_end = t + window_seconds
        window_segs = [s for s in segments if s.start >= t and s.start < window_end]
        
        if not window_segs:
            t += step
            continue

        raw_text = " ".join(s.text for s in window_segs)
        clean_text = FILLER_PATTERN.sub('', raw_text).strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)

        if len(clean_text.split()) >= min_words:
            chunks.append(TranscriptChunk(
                chunk_id=f"{video_id}_chunk_{chunk_index}",
                video_id=video_id,
                text=clean_text,
                start_time=int(t),
                end_time=int(min(window_end, total_duration))
            ))
            chunk_index += 1

        t += step

    return chunks
```

## Critical Rules
- `start_time` must be cast to `int` — the YouTube URL parameter `&t=` does not accept floats
- The `start_time` of the first segment in the window is the chunk's anchor — this is what points the user to the right moment
- Never modify `start_time` after it is set here — all downstream components use it as-is
- Overlap ensures that explanations spanning a chunk boundary are captured in at least one chunk
