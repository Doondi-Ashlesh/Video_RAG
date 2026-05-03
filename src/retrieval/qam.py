"""Skills: decompose_query (QAM) + build_metadata_filter.

QAM (Query Attribute Modeling) is the centrepiece of the query pipeline. The
LLM decomposes a raw user query into:

1. metadata_filters — structured attributes used for ChromaDB hard filtering
   *before* vector search. Lecturer-corpus fields:
       course_id, instructor, lecture_number, topic, upload_year
2. semantic_element — a rephrased description of what the ideal transcript
   segment would be SAYING. This is what gets embedded for cosine search.
   Rephrasing is what rescues short queries like "explain backprop" from
   producing weak similarity matches.

LLM backend: Nemotron Super 49B via NVIDIA NIM (OpenAI-compatible). The system
prompt opens with "detailed thinking off" to suppress the model's reasoning
trace — we want clean structured JSON, not a chain-of-thought.

The prompt is schema-aware — the available filter fields and their expected
types are spelled out so the LLM doesn't invent fields the index can't filter
on. If the LLM call or JSON parse fails, we fall back to using the raw query
as the semantic element with no filters.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from loguru import logger
from openai import OpenAI

from src.config import settings
from src.models import QAMResult


_QAM_SYSTEM_PROMPT = """detailed thinking off

You are a query decomposition engine for a Video RAG system over a corpus of recorded lectures.

Your job is to analyse a student's natural-language query and emit STRICT JSON with two fields.

1. metadata_filters
   Structured attributes used to hard-filter the lecture index before semantic search runs.
   ONLY include fields the user explicitly mentions or strongly implies. Do not invent values.
   Available fields (omit any that don't apply — set them to null):
     - course_id        (string) e.g. "CS231", "MATH101"
     - instructor       (string) name of the lecturer or professor
     - lecture_number   (integer) e.g. "in lecture 5" -> 5
     - topic            (string, lowercase, hyphen-separated) e.g. "backpropagation", "linear-regression", "matrix-multiplication" — NEVER use underscores or spaces
     - upload_year      (integer) e.g. "from 2024" -> 2024

2. semantic_element
   A rephrased description of what the ideal transcript segment would be SAYING.
   Rules:
     - Strip every metadata reference (course names, lecture numbers, instructor names, years, topic words used as filters) from this string.
     - Expand the query into 2-3 sentences describing an ideal speaker explanation.
     - Add the kind of context that would actually appear in a real lecture transcript ("a lecturer walking through the derivation step by step", "an instructor giving a worked example", etc.).
     - Longer and more descriptive is better.

Return ONLY a JSON object with exactly this shape and no surrounding prose:
{
  "metadata_filters": {
    "course_id": null,
    "instructor": null,
    "lecture_number": null,
    "topic": null,
    "upload_year": null
  },
  "semantic_element": "..."
}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """OpenAI-compatible client pointed at NVIDIA NIM."""
    if not settings.nvidia_api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set — required for QAM and answer generation"
        )
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
    )


def _extract_json(raw: str) -> dict[str, Any]:
    """Tolerant JSON extraction. Models occasionally wrap output in prose."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(raw)
        if not m:
            raise
        return json.loads(m.group(0))


def decompose_query(query: str, client: OpenAI | None = None) -> QAMResult:
    """Run QAM decomposition. Falls back to raw-query / no-filter on any error."""
    client = client or get_llm_client()

    try:
        response = client.chat.completions.create(
            model=settings.qam_model,
            max_tokens=512,
            temperature=0.2,  # low temp — this is a structured extraction task
            messages=[
                {"role": "system", "content": _QAM_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        raw = response.choices[0].message.content or ""
        parsed = _extract_json(raw)

        raw_filters = parsed.get("metadata_filters") or {}
        filters = {k: v for k, v in raw_filters.items() if v not in (None, "", [])}

        # Coerce types — the LLM occasionally returns "5" instead of 5 etc.
        if "lecture_number" in filters:
            filters["lecture_number"] = int(filters["lecture_number"])
        if "upload_year" in filters:
            filters["upload_year"] = int(filters["upload_year"])
        if "topic" in filters and isinstance(filters["topic"], str):
            # Normalise to the canonical form used at ingestion time:
            # lowercase, hyphen-separated, no spaces or underscores.
            t = filters["topic"].lower().strip()
            t = t.replace("_", "-").replace(" ", "-")
            filters["topic"] = t

        semantic_element = (parsed.get("semantic_element") or "").strip() or query

        result = QAMResult(
            metadata_filters=filters,
            semantic_element=semantic_element,
            original_query=query,
        )
        logger.info(
            "QAM decomposition: filters={}, semantic_element='{}...'",
            filters,
            semantic_element[:80],
        )
        return result

    except Exception as e:
        logger.warning("QAM decomposition failed ({}). Falling back to raw query.", e)
        return QAMResult(
            metadata_filters={},
            semantic_element=query,
            original_query=query,
        )


def _index_canonical_values(store: Any, field: str) -> set[str]:
    """Distinct values of a metadata field across the entire collection.

    Used to canonicalise QAM-extracted filter values that don't exactly match
    what the ingester wrote. Cheap on small demo corpora; for larger indices
    this would be backed by a per-collection cache invalidated on writes.
    """
    try:
        all_meta = store._collection.get(include=["metadatas"])["metadatas"] or []
    except Exception as e:
        logger.warning("Could not enumerate {} values from index: {}", field, e)
        return set()
    return {str(m[field]) for m in all_meta if m.get(field) not in (None, "")}


def _canonicalise_filters(
    metadata_filters: dict[str, Any], store: Any | None
) -> dict[str, Any]:
    """Replace partial / non-canonical filter values with canonical ones.

    Examples this rescues:
      - QAM extracts "Strang", index has "Gilbert Strang" -> last-name match
      - QAM extracts "cs229" lowercase, index has "CS229" -> case-insensitive match

    If a filter value can't be canonicalised, we DROP it rather than send a
    guaranteed-zero filter to ChromaDB (the unfiltered fallback would still
    fire downstream, but dropping here keeps the audit log honest about which
    filters actually applied).
    """
    if not metadata_filters or store is None:
        return metadata_filters

    out = dict(metadata_filters)

    for field in ("instructor", "course_id"):
        if field not in out:
            continue
        extracted = str(out[field]).strip()
        if not extracted:
            del out[field]
            continue

        canonical = _index_canonical_values(store, field)
        if not canonical or extracted in canonical:
            continue

        # Case-insensitive exact match.
        ci_match = next(
            (c for c in canonical if c.lower() == extracted.lower()), None
        )
        if ci_match:
            logger.info("Canonicalised {}: {!r} -> {!r}", field, extracted, ci_match)
            out[field] = ci_match
            continue

        # Substring match — handles "Strang" -> "Gilbert Strang".
        e_lc = extracted.lower()
        substring_matches = [c for c in canonical if e_lc in c.lower()]
        if len(substring_matches) == 1:
            logger.info(
                "Canonicalised {} (substring): {!r} -> {!r}",
                field,
                extracted,
                substring_matches[0],
            )
            out[field] = substring_matches[0]
        elif len(substring_matches) > 1:
            logger.warning(
                "Ambiguous {} {!r} matches {} — dropping filter",
                field,
                extracted,
                substring_matches,
            )
            del out[field]
        else:
            logger.warning(
                "Unknown {} {!r} (index has {}) — dropping filter",
                field,
                extracted,
                sorted(canonical),
            )
            del out[field]

    return out


def pick_best_chapter(
    query: str,
    candidate_titles: list[str],
    client: OpenAI | None = None,
) -> str | None:
    """Ask the LLM which chapter title best represents the substantive answer.

    Used when the QAM topic matches multiple chapters (e.g. "linear regression"
    matches both "Motivate Linear Regression" and "Linear Regression Algorithm").
    The LLM picks based on semantic understanding — no hardcoded word lists,
    no curated meta-word taxonomy. Generalises to any corpus and naming style.

    Returns one of the input titles (case-corrected to match the candidate
    list) or None if the call fails / returns an unrecognised string.
    """
    titles = [t for t in candidate_titles if t]
    if not titles:
        return None
    if len(titles) == 1:
        return titles[0]

    client = client or get_llm_client()
    titles_block = "\n".join(f"- {t}" for t in titles)
    user_msg = (
        f"User question: {query}\n\n"
        f"Available chapter titles in the lecture:\n{titles_block}\n\n"
        "Which chapter is most likely to contain the substantive answer to the question — "
        "the actual explanation, definition, derivation, or worked example — rather than "
        "a meta-introduction, motivation, or recap?\n\n"
        "Reply with ONLY the exact chapter title from the list above. "
        "No quotes, no extra text, no formatting."
    )
    try:
        resp = client.chat.completions.create(
            model=settings.qam_model,
            max_tokens=2048,  # reasoning needs headroom for the <think> trace
            temperature=0.0,
            messages=[
                # Reasoning ON — this is exactly the kind of comparative
                # judgement Nemotron Super was trained for. Trade-off: ~1-2s
                # extra latency, but this call only fires when 2+ chapters
                # match the topic, so it doesn't hit every query.
                {"role": "system", "content": "detailed thinking on"},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip the reasoning trace (<think>...</think>) — the actual answer
        # is whatever follows.
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1].strip()
        picked = raw.strip().strip('"').strip("'").strip()
        # Take only the first non-empty line in case the model added a
        # one-sentence justification.
        if "\n" in picked:
            picked = picked.split("\n", 1)[0].strip().strip('"').strip("'").strip()

        for t in titles:
            if t.lower() == picked.lower():
                logger.info("LLM picked chapter: '{}' (from {} candidates)", t, len(titles))
                return t

        # Fallback: substring match in case the model returned a slight variant
        for t in titles:
            if t.lower() in picked.lower() or picked.lower() in t.lower():
                logger.info(
                    "LLM picked '{}' (matched to candidate '{}' by substring)",
                    picked, t,
                )
                return t

        logger.warning(
            "LLM picked '{}' which isn't a candidate; available={}", picked, titles
        )
    except Exception as e:
        logger.warning("LLM chapter pick failed ({}); caller will fall back.", e)
    return None


def build_metadata_filter(
    metadata_filters: dict[str, Any],
    store: Any | None = None,
) -> dict[str, Any] | None:
    """Translate QAM filters into a ChromaDB `where` clause.

    If `store` is provided, partial / non-canonical filter values are
    canonicalised against the indexed metadata before the where-clause is
    built. Returns None when there are no filters (full-corpus search).
    """
    if not metadata_filters:
        return None

    metadata_filters = _canonicalise_filters(metadata_filters, store)
    if not metadata_filters:
        return None

    conditions: list[dict[str, Any]] = []

    if "course_id" in metadata_filters:
        conditions.append({"course_id": {"$eq": metadata_filters["course_id"]}})
    if "instructor" in metadata_filters:
        conditions.append({"instructor": {"$eq": metadata_filters["instructor"]}})
    if "lecture_number" in metadata_filters:
        conditions.append(
            {"lecture_number": {"$eq": int(metadata_filters["lecture_number"])}}
        )
    if "topic" in metadata_filters:
        conditions.append({"topic": {"$eq": metadata_filters["topic"]}})
    if "upload_year" in metadata_filters:
        conditions.append(
            {"upload_year": {"$eq": int(metadata_filters["upload_year"])}}
        )

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
