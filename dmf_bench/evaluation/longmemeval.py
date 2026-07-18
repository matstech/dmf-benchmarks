"""Pure LongMemEval evaluator adapters."""

from __future__ import annotations

from typing import Any

from longmemeval.evaluate_ablation import evaluate_ablation
from longmemeval.evaluate_rigorous import evaluate_flat, ensure_uniform_schema


EVALUATOR_VERSION = "longmemeval-evaluator-v1"


def rigorous_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    ensure_uniform_schema(evaluations)
    cutoff_label, metrics = evaluate_flat(evaluations, metadata)
    return {
        "schema_version": 1,
        "benchmark": "longmemeval",
        "evaluator": "rigorous_report",
        "evaluator_version": EVALUATOR_VERSION,
        "status": "COMPLETED",
        "cutoff_label": cutoff_label,
        "metrics": metrics,
    }


def ablation_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    ensure_uniform_schema(evaluations)
    return {
        "schema_version": 1,
        "benchmark": "longmemeval",
        "evaluator": "ablation_report",
        "evaluator_version": EVALUATOR_VERSION,
        "status": "COMPLETED",
        "framework": metadata.get("framework", "unknown"),
        "metrics": evaluate_ablation(evaluations),
    }
