# Skill: generate_answer

## Purpose
Use the winning re-ranked transcript chunk as context to generate a clean, grounded text explanation that directly answers the user's original query.

## File
`src/retrieval/response.py`

## LLM Prompt

```python
ANSWER_SYSTEM_PROMPT = """
You are an expert at explaining concepts clearly and concisely.

You will be given:
1. A user's question
2. A transcript excerpt from a video that is relevant to their question

Your job is to write a clear, direct answer to the user's question using ONLY the information in the transcript excerpt.

Rules:
- Answer directly — do not say "According to the transcript..." or "In this video..."
- Write in your own words, but stay strictly grounded in the provided context
- If the transcript doesn't fully answer the question, say what it does cover and note the gap
- Do not use prior knowledge — only use what is in the excerpt
- Target length: 3–5 sentences
"""

def generate_answer(
    query: str,
    chunk_text: str,
    anthropic_client
) -> str:
    user_message = f"""Question: {query}

Transcript excerpt:
\"\"\"
{chunk_text}
\"\"\"

Please answer the question based on this excerpt."""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=ANSWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text.strip()
```

## Fallback
If the LLM call fails, return the raw `chunk_text` directly as the answer and log:
`"Answer generation failed, returning raw chunk text"`

## Notes
- The strict grounding instruction is essential — without it the model will blend its own knowledge with the chunk, making the snippet URL misleading (the video may not actually say what the LLM added)
- Do not pass the full conversation history — each query is stateless
