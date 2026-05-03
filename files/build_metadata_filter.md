# Skill: build_metadata_filter

## Purpose
Convert the metadata tags extracted by QAM decomposition into a ChromaDB-compatible `where` filter clause that pre-filters the vector search to only relevant videos.

## File
`src/retrieval/qam.py` (alongside decompose_query)

## Implementation

```python
def build_metadata_filter(metadata_filters: dict) -> dict | None:
    """
    Convert QAM metadata tags to a ChromaDB where clause.
    Returns None if no filters (search full corpus).
    """
    if not metadata_filters:
        return None

    conditions = []

    if "channel_name" in metadata_filters:
        conditions.append({"channel_name": {"$eq": metadata_filters["channel_name"]}})

    if "upload_year" in metadata_filters:
        conditions.append({"upload_year": {"$eq": int(metadata_filters["upload_year"])}})

    if "topic_category" in metadata_filters:
        conditions.append({"topic_category": {"$eq": metadata_filters["topic_category"]}})

    if len(conditions) == 1:
        return conditions[0]
    elif len(conditions) > 1:
        return {"$and": conditions}
    return None
```

## Filter Examples

Single filter:
```python
{"channel_name": {"$eq": "Andrej Karpathy"}}
```

Multiple filters:
```python
{"$and": [
    {"channel_name": {"$eq": "Professor Miller"}},
    {"upload_year": {"$eq": 2023}},
    {"topic_category": {"$eq": "finance"}}
]}
```

No filters (full corpus search):
```python
None
```

## Notes
- Log the filter applied for every query — this aids debugging when results are unexpectedly empty
- If a filter returns 0 results, the `vector_search` skill handles the fallback (not this skill)
- String comparisons are case-sensitive in ChromaDB — the QAM decomposer and ingestion metadata must use consistent casing conventions
