"""Deterministic per-query v3 analysis rows for offline comparison."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from dmf_bench.adapters.base import EvaluationReference
from dmf_bench.evaluation.registry import EvaluationRequirement


ANALYSIS_ROW_SCHEMA_VERSION = 3


def _record(value: Any, name: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _required(record: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in record:
        raise ValueError(f"{name} is missing {key}.")
    return record[key]


def _exact_keys(values: Mapping[str, Any], expected: tuple[str, ...], name: str) -> None:
    if not isinstance(values, Mapping) or set(values) != set(expected):
        raise ValueError(f"{name} must contain exactly the expected query IDs.")


def _finite_score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric score.")
    return float(value)


def _json_safe(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be JSON-serializable.") from exc


def build_analysis_rows(
    *,
    run_id: str,
    scientific_fingerprint: str,
    benchmark: str,
    framework: str,
    expected_query_ids: tuple[str, ...],
    expected_judge_ids: tuple[str, ...],
    primary_judge_id: str,
    predictions: Mapping[str, Any],
    retrievals: Mapping[str, Any],
    packed_contexts: Mapping[str, Any],
    judgments: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, EvaluationReference | Mapping[str, Any]],
    retrieval_metrics: Mapping[str, Any] | None = None,
    retrieval_requirement: EvaluationRequirement | None = None,
    ingestion: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Join complete query artifacts in manifest order, rejecting missing inputs."""
    for name, value in (("run_id", run_id), ("scientific_fingerprint", scientific_fingerprint),
                        ("benchmark", benchmark), ("framework", framework)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty.")
    if not isinstance(expected_query_ids, tuple) or not expected_query_ids or any(
        not isinstance(query_id, str) or not query_id.strip() for query_id in expected_query_ids
    ) or len(set(expected_query_ids)) != len(expected_query_ids):
        raise ValueError("expected_query_ids must be ordered, non-empty, and unique.")
    if not isinstance(expected_judge_ids, tuple) or not expected_judge_ids or any(
        not isinstance(judge_id, str) or not judge_id.strip() for judge_id in expected_judge_ids
    ) or len(set(expected_judge_ids)) != len(expected_judge_ids):
        raise ValueError("expected_judge_ids must be non-empty and unique.")
    if primary_judge_id not in expected_judge_ids:
        raise ValueError("primary_judge_id must be an expected judge.")
    for name, values in (("predictions", predictions), ("retrievals", retrievals),
                         ("packed_contexts", packed_contexts), ("judgments", judgments),
                         ("references", references)):
        _exact_keys(values, expected_query_ids, name)
    if retrieval_metrics is not None:
        _exact_keys(retrieval_metrics, expected_query_ids, "retrieval_metrics")
    if ingestion is not None:
        _exact_keys(ingestion, expected_query_ids, "ingestion")
    if retrieval_requirement is not None and retrieval_requirement.name != "retrieval_report":
        raise ValueError("retrieval_requirement must describe retrieval_report.")
    if retrieval_requirement is not None and retrieval_requirement.status != "NOT_APPLICABLE" and retrieval_metrics is None:
        raise ValueError("Applicable retrieval metrics are missing.")
    if retrieval_requirement is not None and retrieval_requirement.status == "NOT_APPLICABLE" and retrieval_metrics is not None:
        raise ValueError("Retrieval metrics were supplied for a NOT_APPLICABLE requirement.")

    rows: list[dict[str, Any]] = []
    for query_id in expected_query_ids:
        prediction = _record(predictions[query_id], "prediction")
        if prediction.get("question_id") != query_id:
            raise ValueError(f"Prediction query ID mismatch for {query_id}.")
        retrieval = _record(retrievals[query_id], "retrieval")
        packed = _record(packed_contexts[query_id], "packed context")
        items = _required(retrieval, "items", "retrieval")
        if not isinstance(items, list):
            raise ValueError("Retrieval items must be a list.")
        returned_ids = [_required(_record(item, "retrieved item"), "memory_id", "retrieved item")
                        for item in items]
        if any(not isinstance(memory_id, str) or not memory_id.strip() for memory_id in returned_ids):
            raise ValueError("Retrieved memory IDs must be non-empty strings.")
        if len(returned_ids) != len(set(returned_ids)):
            raise ValueError("Retrieval contains duplicate memory IDs.")
        packed_ids = _required(packed, "returned_ids", "packed context")
        included_ids = _required(packed, "included_ids", "packed context")
        excluded_ids = _required(packed, "excluded_ids", "packed context")
        if packed_ids != returned_ids or not isinstance(included_ids, list) or not isinstance(excluded_ids, list) or included_ids + excluded_ids != returned_ids:
            raise ValueError("Packed IDs do not match ranked retrieval IDs.")
        original_tokens = _required(packed, "original_tokens", "packed context")
        included_tokens = _required(packed, "included_tokens", "packed context")
        if any(type(value) is not int or value < 0 for value in (original_tokens, included_tokens)) or included_tokens > original_tokens:
            raise ValueError("Packed token counts are invalid.")
        judge_rows = judgments[query_id]
        if not isinstance(judge_rows, Mapping) or set(judge_rows) != set(expected_judge_ids):
            raise ValueError(f"Judgments for {query_id} must contain every expected judge.")
        judge_scores: dict[str, float] = {}
        judge_usage: dict[str, Any] = {}
        judge_timing: dict[str, Any] = {}
        for judge_id in expected_judge_ids:
            judgment = _record(judge_rows[judge_id], "judgment")
            if judgment.get("question_id") != query_id:
                raise ValueError(f"Judgment query ID mismatch for {query_id}/{judge_id}.")
            if "judge_id" in judgment and judgment["judge_id"] != judge_id:
                raise ValueError(f"Judgment judge ID mismatch for {query_id}/{judge_id}.")
            judge_scores[judge_id] = _finite_score(_required(judgment, "score", "judgment"), "judgment score")
            judge_usage[judge_id] = _json_safe(judgment.get("judge_usage", {}), "judge usage")
            judge_timing[judge_id] = _json_safe(judgment.get("timing"), "judge timing")
        reference = references[query_id]
        if isinstance(reference, EvaluationReference):
            if reference.query_id != query_id:
                raise ValueError("Reference query ID mismatch.")
            strata = reference.strata
        else:
            ref_record = _record(reference, "reference")
            if ref_record.get("query_id") != query_id:
                raise ValueError("Reference query ID mismatch.")
            strata = _required(ref_record, "strata", "reference")
        if not isinstance(strata, Mapping):
            raise ValueError("Reference strata must be an object.")
        if retrieval_requirement is not None and retrieval_requirement.not_applicable_reason:
            metric_row: Any = {"status": "NOT_APPLICABLE", "reason": retrieval_requirement.not_applicable_reason}
        elif retrieval_metrics is not None:
            metric_row = _record(retrieval_metrics[query_id], "retrieval metrics")
        else:
            metric_row = {"status": "NOT_APPLICABLE", "reason": "No retrieval metric plan or metric was supplied."}
        ingest_row = _record(ingestion[query_id], "ingestion") if ingestion is not None else {}
        row = {
            "schema_version": ANALYSIS_ROW_SCHEMA_VERSION,
            "run_id": run_id,
            "scientific_fingerprint": scientific_fingerprint,
            "benchmark": benchmark,
            "framework": framework,
            "query_id": query_id,
            "strata": _json_safe(dict(strata), "strata"),
            "primary_judge_id": primary_judge_id,
            "primary_score": judge_scores[primary_judge_id],
            "secondary_scores": {judge_id: judge_scores[judge_id] for judge_id in expected_judge_ids if judge_id != primary_judge_id},
            "judge_scores": judge_scores,
            "retrieval_metrics": _json_safe(metric_row, "retrieval metrics"),
            "returned_memory_count": len(returned_ids),
            "included_memory_count": len(included_ids),
            "excluded_memory_count": len(excluded_ids),
            "retrieved_tokens": original_tokens,
            "included_tokens": included_tokens,
            "answerer_usage": _json_safe(_required(prediction, "answerer_usage", "prediction"), "answerer usage"),
            "framework_usage": {
                "ingestion": _json_safe(ingest_row.get("ingestion_usage"), "ingestion usage"),
                "retrieval": _json_safe(retrieval.get("usage", {}), "retrieval usage"),
            },
            "judge_usage": judge_usage,
            "timing": {
                "ingestion": _json_safe(ingest_row.get("ingestion_timing"), "ingestion timing"),
                "retrieval": _json_safe(retrieval.get("timing"), "retrieval timing"),
                "answer": _json_safe(prediction.get("answer_timing"), "answer timing"),
                "judging": judge_timing,
            },
            "estimated_cost": _json_safe(prediction.get("estimated_cost"), "estimated cost"),
        }
        rows.append(row)
    return tuple(rows)


def analysis_rows_jsonl(rows: tuple[Mapping[str, Any], ...]) -> bytes:
    """Encode rows as stable UTF-8 JSON Lines with one terminal newline."""
    if not isinstance(rows, tuple):
        raise ValueError("rows must be a tuple.")
    return b"".join(
        (json.dumps(_json_safe(dict(row), "analysis row"), ensure_ascii=False,
                    sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        for row in rows
    )
