# Skill: build_snippet_url

## Purpose
Construct a timestamped YouTube URL from the winning chunk's `video_id` and `start_time` metadata. This is the deeplink that takes the user directly to the relevant moment in the video.

## File
`src/retrieval/response.py`

## Implementation

```python
def build_snippet_url(video_id: str, start_time: int) -> str:
    """
    Build a timestamped YouTube URL.
    
    Args:
        video_id: YouTube video ID (e.g. "dQw4w9WgXcQ")
        start_time: Start time in integer seconds (e.g. 135)
    
    Returns:
        URL string (e.g. "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=135s")
    """
    if not video_id:
        raise ValueError("video_id is required to build snippet URL")
    if start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time}")
    
    return f"https://www.youtube.com/watch?v={video_id}&t={start_time}s"
```

## Rules
- `start_time` is always taken from chunk metadata — **never computed, estimated, or modified at this stage**
- `start_time` must be an integer — the `&t=` parameter does not accept floats or formatted strings like `2m15s` (though both are valid YouTube syntax, use integer seconds for simplicity and consistency)
- If `start_time` is missing from metadata, raise a `ValueError` and log — do not silently return a URL without a timestamp

## Output Examples

| video_id | start_time | URL |
|---|---|---|
| `dQw4w9WgXcQ` | `0` | `https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s` |
| `dQw4w9WgXcQ` | `135` | `https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=135s` |
| `dQw4w9WgXcQ` | `3600` | `https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3600s` |
