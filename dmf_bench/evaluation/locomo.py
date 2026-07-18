"""Pure LoCoMo evaluator adapters."""

from __future__ import annotations

from typing import Any

from locomo.evaluate_ablation import evaluate_ablation
from locomo.evaluate_rigorous import evaluate_flat, ensure_locomo_uniform_schema


EVALUATOR_VERSION = "locomo-evaluator-v1"


def rigorous_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    ensure_locomo_uniform_schema(evaluations)
    cutoff_label, metrics = evaluate_flat(evaluations, metadata)
    return {
        "schema_version": 1,
        "benchmark": "locomo",
        "evaluator": "rigorous_report",
        "evaluator_version": EVALUATOR_VERSION,
        "status": "COMPLETED",
        "cutoff_label": cutoff_label,
        "metrics": metrics,
    }


def ablation_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    ensure_locomo_uniform_schema(evaluations)
    return {
        "schema_version": 1,
        "benchmark": "locomo",
        "evaluator": "ablation_report",
        "evaluator_version": EVALUATOR_VERSION,
        "status": "COMPLETED",
        "framework": metadata.get("framework", "unknown"),
        "metrics": evaluate_ablation(evaluations),
    }
