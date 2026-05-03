# Skill: decompose_query

## Purpose
Apply the QAM (Query Attribute Modeling) framework to decompose a raw user query into (1) structured metadata filter tags and (2) a rephrased semantic retrieval element. This is the most important skill in the query pipeline — its output quality directly determines retrieval precision.

## File
`src/retrieval/qam.py`

## LLM Prompt

```python
QAM_SYSTEM_PROMPT = """
You are a query decomposition engine for a video RAG system.

Your job is to analyze a user's query and extract two things:

1. METADATA_FILTERS: Structured attributes that can be used to hard-filter a video database.
   Look for:
   - channel_name: mentions of a specific YouTube creator, professor, or channel
   - upload_year: mentions of a year ("from 2023", "recent" = current year)
   - topic_category: a subject domain ("machine learning", "finance", "physics", "history")
   
   Only extract a filter if the user explicitly mentions it. Do not infer filters that aren't stated.
   If no filters are present, return an empty object {}.

2. SEMANTIC_ELEMENT: The core retrieval intent of the query, rephrased to describe what the
   ideal transcript segment would be SAYING — not just the user's words.
   
   Rules for rephrasing:
   - Remove all metadata references (channel names, years, topics) from the semantic element
   - Expand the query into a description of an ideal speaker explanation
   - Add context that would appear in a real transcript (e.g., "a speaker explaining step by step...")
   - Longer and more descriptive is better — aim for 2-3 sentences

Return ONLY valid JSON in this format:
{
  "metadata_filters": {
    "channel_name": "string or null",
    "upload_year": "integer or null",
    "topic_category": "string or null"
  },
  "semantic_element": "string"
}
"""

def decompose_query(query: str, anthropic_client) -> QAMResult:
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=QAM_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": query}]
    )
    raw = response.content[0].text
    parsed = json.loads(raw)
    
    filters = {k: v for k, v in parsed["metadata_filters"].items() if v is not None}
    
    return QAMResult(
        metadata_filters=filters,
        semantic_element=parsed["semantic_element"],
        original_query=query
    )
```

## Example Decompositions

**Query:** `"explain RAG to me"`
```json
{
  "metadata_filters": {},
  "semantic_element": "A speaker giving a detailed conceptual explanation of Retrieval-Augmented Generation, covering what retrieval is, how it connects to generation, and why adding a retrieval step improves LLM outputs"
}
```

**Query:** `"explain the Black-Scholes model by Professor Miller from 2023 Finance series"`
```json
{
  "metadata_filters": {
    "channel_name": "Professor Miller",
    "upload_year": 2023,
    "topic_category": "finance"
  },
  "semantic_element": "A professor giving a detailed step-by-step explanation of the Black-Scholes options pricing model, covering its assumptions, formula components, and how it is used to price financial derivatives"
}
```

## Fallback
If the LLM call fails or JSON parsing fails, return:
```python
QAMResult(
    metadata_filters={},
    semantic_element=query,  # use raw query as fallback
    original_query=query
)
```
Log the failure: `"QAM decomposition failed, falling back to raw query"`
