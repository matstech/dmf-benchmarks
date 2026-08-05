# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""Native benchmark primary quality and secondary reporting helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .paths import native_benchmark_results_dir


PRIMARY_QUALITY_METRIC = "judge_score"
PRIMARY_QUALITY_DESCRIPTION = (
    "Primary native quality metric: common LLM judge score over the generated "
    "answer, where 1.0 is CORRECT and 0.0 is WRONG."
)
SECONDARY_RIGOROUS_DESCRIPTION = (
    "Secondary rigorous metrics are deterministic answer/retrieval diagnostics "
    "computed from the flat-compatible native bundle. They are not mixed into "
    "the primary native judge score."
)


def apply_native_primary_judge_score(
    item: dict[str, Any],
    *,
    judgment: str,
    score: float,
    reason: str,
    judge_provider: str,
    judge_model: str,
) -> dict[str, Any]:
    """Attach the common judge score as the primary native quality signal."""
    normalized_label = judgment.strip().upper()
    normalized_score = float(score)

    item["judgment"] = normalized_label
    item["score"] = normalized_score
    item["reason"] = reason
    item["judge_provider"] = judge_provider
    item["judge_model"] = judge_model

    item["judge_score"] = normalized_score
    item["primary_quality"] = {
        "metric": PRIMARY_QUALITY_METRIC,
        "score": normalized_score,
        "label": normalized_label,
        "reason": reason,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
    }

    return item


def _group_key(item: dict[str, Any], *, benchmark_name: str) -> str:
    if benchmark_name == "longmemeval":
        return str(item.get("question_type", "unknown"))
    if benchmark_name == "locomo":
        return str(item.get("category_name", item.get("category", "unknown")))
    return str(item.get("group", "unknown"))


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_native_primary_quality(
    evaluations: list[dict[str, Any]],
    *,
    benchmark_name: str,
) -> dict[str, Any]:
    """Aggregate only the native primary judge metric."""
    overall_scores: list[float] = []
    overall_passes = 0
    by_group_scores: dict[str, list[float]] = defaultdict(list)
    by_group_passes: dict[str, int] = defaultdict(int)

    for item in evaluations:
        raw_score = item.get(PRIMARY_QUALITY_METRIC)
        if raw_score is None:
            continue

        score = float(raw_score)
        group = _group_key(item, benchmark_name=benchmark_name)
        label = str(item.get("judgment", "")).strip().upper()
        passed = label in {"CORRECT", "PASS"} or score >= 1.0

        overall_scores.append(score)
        by_group_scores[group].append(score)
        if passed:
            overall_passes += 1
            by_group_passes[group] += 1

    judged_count = len(overall_scores)
    overall = {
        "metric": PRIMARY_QUALITY_METRIC,
        "description": PRIMARY_QUALITY_DESCRIPTION,
        "judged_count": judged_count,
        "missing_count": max(len(evaluations) - judged_count, 0),
        "avg_judge_score": _average(overall_scores),
        "judge_pass_rate": (overall_passes / judged_count) if judged_count else 0.0,
    }

    grouped = {}
    for group, scores in sorted(by_group_scores.items()):
        grouped[group] = {
            "judged_count": len(scores),
            "avg_judge_score": _average(scores),
            "judge_pass_rate": by_group_passes[group] / len(scores),
        }

    return {
        "primary_metric": PRIMARY_QUALITY_METRIC,
        "overall": overall,
        "by_group": grouped,
    }


def native_bundle_path(*, benchmark_name: str, project_name: str) -> Path:
    """Return the canonical native flat bundle path for one run."""
    return (
        native_benchmark_results_dir(benchmark_name)
        / f"{benchmark_name}_native_{project_name}.json"
    )


def native_primary_report_path(*, benchmark_name: str, project_name: str) -> Path:
    """Return the canonical primary native report path for one run."""
    return (
        native_benchmark_results_dir(benchmark_name)
        / f"{benchmark_name}_native_{project_name}.primary_quality.json"
    )


def native_secondary_manifest_path(*, benchmark_name: str, project_name: str) -> Path:
    """Return the canonical secondary rigorous manifest path for one run."""
    return (
        native_benchmark_results_dir(benchmark_name)
        / f"{benchmark_name}_native_{project_name}.secondary_rigorous.json"
    )


def build_native_secondary_rigorous_manifest(
    *,
    benchmark_name: str,
    input_path: str | Path,
    project_name: str | None = None,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build launchable commands for rigorous secondary native reports."""
    input_path = Path(input_path)
    if project_name:
        stem = f"{benchmark_name}_native_{project_name}"
    else:
        stem = input_path.stem
    output_dir = Path(report_dir) if report_dir is not None else input_path.parent
    rigorous_stdout = output_dir / f"{stem}.rigorous.txt"
    ablation_json = output_dir / f"{stem}.ablation.json"

    return {
        "report_type": "secondary_rigorous",
        "benchmark": benchmark_name,
        "description": SECONDARY_RIGOROUS_DESCRIPTION,
        "input_path": str(input_path),
        "artifacts": {
            "evaluate_rigorous_stdout": str(rigorous_stdout),
            "evaluate_ablation_json": str(ablation_json),
        },
        "commands": {
            "evaluate_rigorous": [
                "python",
                "-m",
                f"{benchmark_name}.evaluate_rigorous",
                "--input",
                str(input_path),
            ],
            "evaluate_ablation": [
                "python",
                "-m",
                f"{benchmark_name}.evaluate_ablation",
                "--input",
                str(input_path),
                "--json-output",
                str(ablation_json),
            ],
        },
        "stdout_capture": {
            "evaluate_rigorous": str(rigorous_stdout),
        },
    }


def build_native_primary_quality_report(
    *,
    bundle: dict[str, Any],
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the primary native report without secondary rigorous metrics."""
    metadata = dict(bundle.get("metadata") or {})
    benchmark_name = str(metadata.get("benchmark", "unknown"))
    evaluations = bundle.get("evaluations", [])
    if not isinstance(evaluations, list):
        evaluations = []

    return {
        "report_type": "native_primary_quality",
        "metadata": metadata,
        "methodology": {
            "primary_metric": PRIMARY_QUALITY_METRIC,
            "primary_quality_description": PRIMARY_QUALITY_DESCRIPTION,
            "secondary_rigorous_description": SECONDARY_RIGOROUS_DESCRIPTION,
        },
        "primary_quality": aggregate_native_primary_quality(
            evaluations,
            benchmark_name=benchmark_name,
        ),
        "token_accounting": bundle.get("token_accounting"),
        "latency_breakdown": bundle.get("latency_breakdown"),
        "timing_report": bundle.get("timing_report"),
        "resource_report": bundle.get("resource_report"),
        "source_bundle": str(input_path) if input_path is not None else None,
    }
