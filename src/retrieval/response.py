"""Skills: generate_answer + build_snippet_url.

generate_answer prompts Nemotron Super (via NVIDIA NIM) with the winning chunk
as the SOLE context and asks for a clean, grounded explanation. If the chunk
doesn't fully cover the question the model is told to say so explicitly —
better an honest gap than a hallucinated answer that the snippet URL won't
actually back up.

The system prompt opens with "detailed thinking off" to suppress Nemotron's
reasoning trace; we want the user-facing answer, not the chain of thought.

build_snippet_url is the only place we ever read start_time, and we read it
straight from chunk metadata. start_time is never computed or modified at
query time.
"""
from __future__ import annotations

from loguru import logger
from openai import OpenAI

from src.config import settings
from src.retrieval.qam import get_llm_client


_ANSWER_SYSTEM_PROMPT = """detailed thinking off

You are an expert at explaining lecture concepts clearly and concisely to a student.

You will be given:
1. The student's question.
2. A transcript excerpt from a lecture video that the retrieval system selected as most relevant.

Write a direct answer to the student's question using ONLY the information in the transcript excerpt.

ABSOLUTE RULES (these override every other instinct you have):
1. Use ONLY information that appears in the transcript excerpt. Do NOT use any prior knowledge whatsoever.
2. If the excerpt does not fully answer the question, you MUST say so. State briefly what the excerpt does cover and what is missing. Stop there.
3. NEVER add a "general explanation", "for context", "background", "additional information", "optional", "outside the excerpt", or any similar section that introduces information not present in the excerpt. Do not offer to provide one. Do not gesture toward one.
4. NEVER say things like "if you'd like, I can explain X in general" or "here is a brief overview not based on the excerpt". Do not include disclaimers that invite further generation outside the excerpt.
5. Do NOT say "according to the transcript", "in this video", "the speaker says", or similar meta-references. Just answer.
6. Write in your own words but every factual claim must trace to a specific phrase in the excerpt.

Target length: 3-5 sentences. Shorter is fine if the excerpt only supports a short answer.
"""


def generate_answer(
    query: str,
    chunk_text: str | list[str],
    client: OpenAI | None = None,
) -> str:
    """Generate a grounded explanation.

    `chunk_text` may be a single string (single-chunk path) or a list of
    strings (multi-chunk synthesis path — top-N chunks from rerank). When
    multiple chunks are provided, they're concatenated with clear separators
    so the LLM can synthesise across them while staying grounded.
    """
    client = client or get_llm_client()

    if isinstance(chunk_text, str):
        excerpts = chunk_text
    else:
        excerpts = "\n\n".join(
            f"--- Excerpt {i+1} ---\n{t}" for i, t in enumerate(chunk_text)
        )

    user_message = (
        f"Question: {query}\n\n"
        f"Transcript excerpt(s):\n\"\"\"\n{excerpts}\n\"\"\"\n\n"
        "Please answer the question based ONLY on the excerpt(s) above."
    )

    try:
        response = client.chat.completions.create(
            model=settings.answer_model,
            max_tokens=512,
            temperature=0.4,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(
            "Answer generation failed ({}). Returning raw chunk text.", e
        )
        return excerpts


def build_snippet_url(video_id: str, start_time: int) -> str:
    """Construct the timestamped YouTube deeplink.

    start_time MUST come from chunk metadata. Never inferred, never adjusted.
    """
    if not video_id:
        raise ValueError("video_id is required to build snippet URL")
    if start_time < 0:
        raise ValueError(f"start_time must be >= 0, got {start_time}")
    return f"https://www.youtube.com/watch?v={video_id}&t={start_time}s"
