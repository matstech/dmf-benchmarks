"""Canonical usage and timing report helpers."""

from __future__ import annotations

from statistics import mean
from typing import Any


TIMING_SCOPE_QUESTION = "question"
TIMING_SCOPE_CONVERSATION = "conversation"
PIPELINE_TIMING_FIELDS: dict[str, tuple[str, str]] = {
    "ingestion": ("ingestion_ms", "ingestion_scope"),
    "qa": ("qa_ms", "qa_scope"),
    "retrieval_pipeline": ("retrieval_pipeline_ms", "retrieval_pipeline_scope"),
    "backend_search": ("backend_search_ms", "backend_search_scope"),
    "answer_generation": ("answer_generation_ms", "answer_generation_scope"),
    "judge": ("judge_ms", "judge_scope"),
}


def normalize_answerer_usage(data: dict[str, Any] | None) -> dict[str, int]:
    """Normalize one answerer usage payload to the report contract."""
    if not isinstance(data, dict):
        data = {}
    return {
        "prompt_tokens_total": int(data.get("prompt_tokens_total", 0) or 0),
        "completion_tokens": int(data.get("completion_tokens", 0) or 0),
        "total_tokens": int(data.get("total_tokens", 0) or 0),
    }


def build_timing_report(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-item pipeline timing into per-stage summaries."""
    report: dict[str, Any] = {}

    for stage_name, (value_key, scope_key) in PIPELINE_TIMING_FIELDS.items():
        stage_values: dict[str, float] = {}
        stage_scope: str | None = None

        for index, evaluation in enumerate(evaluations):
            timing = _normalize_pipeline_timing(evaluation.get("pipeline_timing"))
            if value_key not in timing:
                continue
            scope = str(timing.get(scope_key, TIMING_SCOPE_QUESTION))
            stage_scope = scope
            identity = _timing_identity(
                evaluation,
                scope=scope,
                fallback_index=index,
            )
            stage_values[identity] = float(timing[value_key])

        if not stage_values:
            continue

        values = list(stage_values.values())
        report[stage_name] = {
            "scope": stage_scope or TIMING_SCOPE_QUESTION,
            "count": len(values),
            "total_ms": sum(values),
            "avg_ms": mean(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }

    return report


def _normalize_pipeline_timing(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    normalized: dict[str, Any] = {}
    for value_key, scope_key in PIPELINE_TIMING_FIELDS.values():
        raw_value = data.get(value_key)
        if raw_value is None:
            continue
        normalized[value_key] = max(float(raw_value or 0.0), 0.0)
        scope = str(data.get(scope_key, TIMING_SCOPE_QUESTION)).strip().lower()
        if scope not in {TIMING_SCOPE_QUESTION, TIMING_SCOPE_CONVERSATION}:
            scope = TIMING_SCOPE_QUESTION
        normalized[scope_key] = scope
    return normalized


def _timing_identity(
    evaluation: dict[str, Any],
    *,
    scope: str,
    fallback_index: int,
) -> str:
    if scope == TIMING_SCOPE_CONVERSATION:
        conversation_idx = evaluation.get("conversation_idx")
        if conversation_idx is not None:
            return f"conversation:{conversation_idx}"
    question_id = evaluation.get("question_id")
    if question_id is not None:
        return f"question:{question_id}"
    return f"index:{fallback_index}"
