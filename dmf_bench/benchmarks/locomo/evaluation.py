"""Pure LoCoMo evaluator adapters."""

from __future__ import annotations

from typing import Any

from dmf_bench.contracts import EVALUATION_SCHEMA_VERSION
from .ablation import evaluate_ablation
from .rigorous import evaluate_flat


EVALUATOR_VERSION = "locomo-evaluator-v2"


def ensure_v2_evaluations(evaluations: list[dict[str, Any]]) -> None:
    """Reject historical cutoff-nested or incomplete LoCoMo results."""
    if any(
        not ("generated_answer" in item or "prediction" in item)
        or isinstance(item.get("cutoff_results"), dict)
        for item in evaluations
    ):
        raise ValueError(
            "Unsupported LOCOMO evaluation payload: expected flat v2 prediction items."
        )


def rigorous_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    ensure_v2_evaluations(evaluations)
    cutoff_label, metrics = evaluate_flat(evaluations, metadata)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": "locomo",
        "evaluator": "rigorous_report",
        "evaluator_version": EVALUATOR_VERSION,
        "status": "COMPLETED",
        "cutoff_label": cutoff_label,
        "metrics": metrics,
    }


def ablation_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    ensure_v2_evaluations(evaluations)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": "locomo",
        "evaluator": "ablation_report",
        "evaluator_version": EVALUATOR_VERSION,
        "status": "COMPLETED",
        "framework": metadata.get("framework", "unknown"),
        "metrics": evaluate_ablation(evaluations),
    }
