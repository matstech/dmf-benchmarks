"""Pure retrieval-stage ablation evaluator for LoCoMo v2 outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


_RETRIEVAL_METRICS = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k")


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids) & gold_ids) / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return sum(1 for item in retrieved_ids if item in gold_ids) / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for index, item in enumerate(retrieved_ids, start=1):
        if item in gold_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(retrieved_ids, start=1)
        if item in gold_ids
    )
    ideal_hits = min(len(gold_ids), len(retrieved_ids))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_source_ids_from_canonical(
    candidates: list[dict[str, Any]],
) -> list[str]:
    ids: list[str] = []
    for item in candidates:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        single_id = metadata.get("source_unit_id")
        if isinstance(single_id, str) and single_id:
            ids.append(single_id)
        multiple_ids = metadata.get("source_unit_ids")
        if isinstance(multiple_ids, list):
            ids.extend(
                value
                for value in multiple_ids
                if isinstance(value, str) and value
            )
    return ids


def _extract_source_ids_from_raw(
    raw_candidates: list[dict[str, Any]],
    record_id_to_source: dict[str, str],
) -> list[str]:
    ids: list[str] = []
    for candidate in raw_candidates:
        record = candidate.get("record", {})
        record_id = str(record.get("record_id", ""))
        source_id = record_id_to_source.get(record_id)
        if source_id:
            ids.append(source_id)
    return ids


def _build_record_id_to_source_mapping(item: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    diagnostics = (item.get("retrieval") or {}).get("recall_diagnostics") or {}
    for source_key in ("ranked_candidates_canonical", "final_candidates_canonical"):
        candidates = diagnostics.get(source_key, [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            candidate_id = candidate.get("id", "")
            metadata = candidate.get("metadata") or {}
            source_id = metadata.get("source_unit_id")
            if candidate_id and source_id:
                mapping[candidate_id] = source_id
    return mapping


def _compute_retrieval_metrics(
    retrieved_ids: list[str],
    gold_ids: set[str],
) -> dict[str, float]:
    return {
        "recall_at_k": recall_at_k(retrieved_ids, gold_ids),
        "precision_at_k": precision_at_k(retrieved_ids, gold_ids),
        "mrr": reciprocal_rank(retrieved_ids, gold_ids),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, gold_ids),
    }


def evaluate_ablation(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute raw, ranked, and final retrieval metrics."""
    stages = ("raw", "ranked", "final")
    overall: dict[str, dict[str, list[float]]] = {
        stage: {metric: [] for metric in _RETRIEVAL_METRICS}
        for stage in stages
    }
    by_group: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {
            stage: {metric: [] for metric in _RETRIEVAL_METRICS}
            for stage in stages
        }
    )
    stats: dict[str, Any] = {
        "total_questions": 0,
        "questions_with_diagnostics": 0,
        "questions_without_diagnostics": 0,
        "raw_unmappable_candidates": 0,
        "avg_raw_count": [],
        "avg_ranked_count": [],
        "avg_final_count": [],
    }

    for item in evaluations:
        group = str(item.get("category_name", item.get("category", "unknown")))
        gold_ids = {
            evidence
            for evidence in item.get("evidence", [])
            if isinstance(evidence, str) and evidence
        }
        if not gold_ids:
            continue
        stats["total_questions"] += 1
        retrieval = item.get("retrieval") or {}
        search_results = retrieval.get("search_results") or []
        diagnostics = retrieval.get("recall_diagnostics") or {}

        final_ids = _extract_source_ids_from_canonical(search_results)
        final_metrics = _compute_retrieval_metrics(final_ids, gold_ids)
        for metric, value in final_metrics.items():
            overall["final"][metric].append(value)
            by_group[group]["final"][metric].append(value)
        stats["avg_final_count"].append(len(final_ids))

        ranked_canonical = diagnostics.get("ranked_candidates_canonical") or []
        raw_candidates = diagnostics.get("raw_candidates") or []
        diagnostics_available = diagnostics.get("diagnostics_available") is True
        if not diagnostics_available and not ranked_canonical and not raw_candidates:
            stats["questions_without_diagnostics"] += 1
            for metric, value in final_metrics.items():
                overall["raw"][metric].append(value)
                overall["ranked"][metric].append(value)
                by_group[group]["raw"][metric].append(value)
                by_group[group]["ranked"][metric].append(value)
            stats["avg_raw_count"].append(len(final_ids))
            stats["avg_ranked_count"].append(len(final_ids))
            continue

        stats["questions_with_diagnostics"] += 1
        ranked_ids = _extract_source_ids_from_canonical(ranked_canonical)
        ranked_metrics = _compute_retrieval_metrics(ranked_ids, gold_ids)
        for metric, value in ranked_metrics.items():
            overall["ranked"][metric].append(value)
            by_group[group]["ranked"][metric].append(value)
        stats["avg_ranked_count"].append(len(ranked_ids))

        raw_ids = _extract_source_ids_from_raw(
            raw_candidates,
            _build_record_id_to_source_mapping(item),
        )
        stats["raw_unmappable_candidates"] += len(raw_candidates) - len(raw_ids)
        raw_metrics = (
            _compute_retrieval_metrics(raw_ids, gold_ids)
            if raw_ids
            else ranked_metrics
        )
        for metric, value in raw_metrics.items():
            overall["raw"][metric].append(value)
            by_group[group]["raw"][metric].append(value)
        stats["avg_raw_count"].append(len(raw_ids) if raw_ids else len(ranked_ids))

    result: dict[str, Any] = {"stages": {}}
    for stage in stages:
        result["stages"][stage] = {
            "overall": {
                metric: average(values)
                for metric, values in overall[stage].items()
            },
            "by_group": {
                group: {
                    metric: average(values)
                    for metric, values in group_stages[stage].items()
                }
                for group, group_stages in sorted(by_group.items())
            },
        }
    result["stats"] = {
        "total_questions": stats["total_questions"],
        "questions_with_diagnostics": stats["questions_with_diagnostics"],
        "questions_without_diagnostics": stats["questions_without_diagnostics"],
        "raw_unmappable_candidates_total": stats["raw_unmappable_candidates"],
        "avg_candidates_raw": average(stats["avg_raw_count"]),
        "avg_candidates_ranked": average(stats["avg_ranked_count"]),
        "avg_candidates_final": average(stats["avg_final_count"]),
    }
    return result
