"""Offline analysis of verified, committed v3 benchmark runs."""

from __future__ import annotations

import csv
import io
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import RunManifest, hash_canonical_json, sha256_file
from dmf_bench.state import StateError


ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_ROW_SCHEMA_VERSION = 3
ANALYSIS_ROW_KEYS = frozenset({
    "schema_version", "run_id", "scientific_fingerprint", "benchmark", "framework",
    "query_id", "strata", "primary_judge_id", "primary_score", "secondary_scores",
    "judge_scores", "retrieval_metrics", "returned_memory_count", "included_memory_count",
    "excluded_memory_count", "retrieved_tokens", "included_tokens", "answerer_usage",
    "framework_usage", "judge_usage", "timing", "estimated_cost",
})
COMMON_CONTROLS = (
    "schema_version", "scientific_profile", "benchmark", "benchmark_adapter_version",
    "normalization_version", "dataset", "selection", "expected_question_ids",
    "retrieval", "context_budget", "answerer", "judges", "evaluation",
)
MATRIX_BENCHMARKS = ("locomo", "longmemeval", "memoryagentbench", "beam")
MATRIX_FRAMEWORKS = ("model-only", "full-context", "vector-rag", "mem0", "dmf", "lightmem", "amem")


@dataclass(frozen=True)
class VerifiedRun:
    manifest: RunManifest
    rows: tuple[dict[str, Any], ...]
    unit_by_query: dict[str, str]
    publication_sha256: str
    latency_ms_by_query: dict[str, float]
    bounded_full_context: bool

    @property
    def inputs(self) -> dict[str, Any]:
        return self.manifest.fingerprint_inputs


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _member(store: LocalArtifactStore, run_id: str, paths: set[str], relative: str) -> Path:
    if relative not in paths:
        raise ValueError(f"Missing committed artifact: {run_id}/{relative}.")
    return store.run_dir(run_id) / "final" / relative


def load_verified_run(store: LocalArtifactStore, run_id: str) -> VerifiedRun:
    """Verify publication and cross-check predictions, judgments and analysis rows."""
    verification = store.verify_committed(run_id)
    paths = {item["path"] for item in verification["artifacts"]}
    manifest = RunManifest.from_dict(_object(read_json(_member(store, run_id, paths, "run-manifest.json")), "manifest"))
    if manifest.schema_version != 3 or manifest.fingerprint_inputs.get("schema_version") != 3:
        raise ValueError(f"{run_id}: only v3 runs are supported.")
    if manifest.run_id != run_id or manifest.scientific_fingerprint != hash_canonical_json(manifest.fingerprint_inputs):
        raise ValueError(f"{run_id}: manifest identity or scientific fingerprint mismatch.")
    if read_json(store.run_dir(run_id) / "run-manifest.json") != manifest.to_dict():
        raise ValueError(f"{run_id}: working and committed manifests differ.")
    inputs = manifest.fingerprint_inputs
    expected = inputs.get("expected_question_ids")
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(query, str) or not query for query in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError(f"{run_id}: invalid expected query set.")
    judges = inputs.get("judges")
    primary = _object(inputs.get("evaluation"), "evaluation").get("primary_judge_id")
    if not isinstance(judges, list) or not judges or not isinstance(primary, str):
        raise ValueError(f"{run_id}: invalid judge plan.")
    judge_by_id = {_object(judge, "judge").get("id"): judge for judge in judges}
    if len(judge_by_id) != len(judges) or primary not in judge_by_id:
        raise ValueError(f"{run_id}: invalid judge IDs or primary judge.")
    for judge in judges:
        identity = {key: value for key, value in judge.items() if key != "fingerprint"}
        if judge.get("fingerprint") != hash_canonical_json(identity):
            raise ValueError(f"{run_id}: judge identity digest mismatch.")

    summary = _object(read_json(_member(store, run_id, paths, "evaluations/analysis_rows.json")), "analysis summary")
    rows_path = _member(store, run_id, paths, "evaluations/analysis_rows.jsonl")
    if (summary.get("schema_version") != 3 or summary.get("status") != "COMPLETED"
            or summary.get("evaluator_version") != "analysis-rows-v3"
            or summary.get("path") != "evaluations/analysis_rows.jsonl"
            or summary.get("sha256") != sha256_file(rows_path)
            or summary.get("row_count") != len(expected)):
        raise ValueError(f"{run_id}: analysis rows summary mismatch.")
    rows = tuple(_object(json.loads(line), "analysis row") for line in rows_path.read_text(encoding="utf-8").splitlines())
    if [row.get("query_id") for row in rows] != expected:
        raise ValueError(f"{run_id}: missing, extra or reordered paired observations.")
    evaluations = read_json(_member(store, run_id, paths, "evaluations/evaluations.json"))
    if not isinstance(evaluations, list) or [item.get("question_id") for item in evaluations] != expected:
        raise ValueError(f"{run_id}: evaluation query set mismatch.")
    evaluation_by_query = {item["question_id"]: _object(item, "evaluation") for item in evaluations}
    retrieval_report = _object(read_json(_member(store, run_id, paths, "evaluations/retrieval_report.json")), "retrieval report")
    if retrieval_report.get("schema_version") != 3 or retrieval_report.get("status") not in {"COMPLETED", "NOT_APPLICABLE"}:
        raise ValueError(f"{run_id}: retrieval report schema or status mismatch.")
    retrieval_queries = retrieval_report.get("queries")
    if retrieval_report["status"] == "COMPLETED" and (not isinstance(retrieval_queries, dict) or set(retrieval_queries) != set(expected)):
        raise ValueError(f"{run_id}: retrieval report query set mismatch.")
    if retrieval_report["status"] == "NOT_APPLICABLE" and not retrieval_report.get("reason"):
        raise ValueError(f"{run_id}: retrieval NOT_APPLICABLE requires a reason.")

    unit_by_query: dict[str, str] = {}
    latency: dict[str, float] = {}
    prediction_by_query: dict[str, dict[str, Any]] = {}
    bounded_full_context = False
    for unit_id in manifest.expected_item_ids:
        relative = f"items/{unit_id}/predictions.json"
        aggregate = _object(read_json(_member(store, run_id, paths, relative)), "prediction aggregate")
        predictions = aggregate.get("predictions")
        question_ids = aggregate.get("question_ids")
        if (
            aggregate.get("schema_version") != 3
            or not isinstance(predictions, list)
            or not isinstance(question_ids, list)
            or len(predictions) != len(question_ids)
        ):
            raise ValueError(f"{run_id}: invalid prediction aggregate for {unit_id}.")
        for query_id, prediction in zip(question_ids, predictions):
            prediction = _object(prediction, "prediction")
            if (
                query_id in unit_by_query
                or query_id not in expected
                or prediction.get("schema_version") != 3
                or prediction.get("question_id") != query_id
                or prediction.get("benchmark") != inputs.get("benchmark")
                or prediction.get("framework") != inputs.get("framework")
            ):
                raise ValueError(f"{run_id}: prediction identity mismatch for {query_id}.")
            unit_by_query[query_id] = unit_id
            prediction_by_query[query_id] = prediction
            if inputs.get("framework") == "full-context":
                retrieval_path = f"items/{unit_id}/questions/{query_id}.retrieval.json"
                retrieval = _object(read_json(_member(store, run_id, paths, retrieval_path)), "retrieval")
                family = _object(retrieval.get("diagnostics"), "retrieval diagnostics").get("baseline_family")
                if family not in {"full-context", "bounded-full-context"}:
                    raise ValueError(f"{run_id}: full-context diagnostic is missing for {query_id}.")
                bounded_full_context |= family == "bounded-full-context"
            evaluation = evaluation_by_query[query_id]
            if (
                evaluation.get("schema_version") != 3
                or evaluation.get("primary_judge_id") != primary
                or list(_object(evaluation.get("judges"), "judges")) != list(judge_by_id)
            ):
                raise ValueError(f"{run_id}: evaluation judge plan mismatch for {query_id}.")
            for judge_id, judge in judge_by_id.items():
                judgment = _object(read_json(_member(store, run_id, paths, f"judgments/{judge_id}/{query_id}.json")), "judgment")
                judge_result = _object(evaluation["judges"][judge_id], "evaluation judgment")
                if (judgment.get("schema_version") != 3 or judgment.get("question_id") != query_id
                        or judgment.get("judge_id") != judge_id or judgment.get("scientific_fingerprint") != manifest.scientific_fingerprint
                        or judgment.get("prediction_sha256") != hash_canonical_json(prediction)
                        or judgment.get("judge_identity_fingerprint") != judge["fingerprint"]
                        or judgment.get("judge_fingerprint") != hash_canonical_json(judge["contract"])
                        or judgment.get("judge_provider") != judge["model"]["provider"]
                        or judgment.get("judge_requested_model") != judge["model"]["requested_model"]
                        or _number(judgment.get("score"), "judgment score") != _number(judge_result.get("score"), "evaluation score")):
                    raise ValueError(f"{run_id}: judgment mismatch for {query_id}/{judge_id}.")
            timing_relative = f"items/{unit_id}/timing/{query_id}.json"
            if timing_relative in paths:
                timing = _object(read_json(_member(store, run_id, paths, timing_relative)), "timing")
                qa_ms = _object(timing.get("pipeline_timing"), "pipeline timing").get("qa_ms")
                if qa_ms is not None:
                    latency[query_id] = _number(qa_ms, "qa_ms")
            elif isinstance(prediction.get("answer_timing"), dict) and prediction["answer_timing"].get("qa_ms") is not None:
                latency[query_id] = _number(prediction["answer_timing"]["qa_ms"], "qa_ms")
    if set(unit_by_query) != set(expected) or len(set(manifest.expected_item_ids)) != len(manifest.expected_item_ids):
        raise ValueError(f"{run_id}: incomplete or duplicate atomic units.")

    for row in rows:
        query_id = row["query_id"]
        evaluation = evaluation_by_query[query_id]
        prediction = prediction_by_query[query_id]
        if (set(row) != ANALYSIS_ROW_KEYS or row.get("schema_version") != ANALYSIS_ROW_SCHEMA_VERSION
                or row.get("run_id") != run_id or row.get("scientific_fingerprint") != manifest.scientific_fingerprint
                or row.get("benchmark") != inputs.get("benchmark") or row.get("framework") != inputs.get("framework")
                or row.get("primary_judge_id") != primary or not isinstance(row.get("strata"), dict)):
            raise ValueError(f"{run_id}: analysis row identity mismatch for {query_id}.")
        scores = _object(row.get("judge_scores"), "judge scores")
        if set(scores) != set(judge_by_id) or set(_object(row.get("secondary_scores"), "secondary scores")) != set(judge_by_id) - {primary}:
            raise ValueError(f"{run_id}: analysis row judge set mismatch for {query_id}.")
        for judge_id in judge_by_id:
            if _number(scores[judge_id], "row judge score") != _number(evaluation["judges"][judge_id]["score"], "evaluation judge score"):
                raise ValueError(f"{run_id}: analysis row score mismatch for {query_id}/{judge_id}.")
            if judge_id != primary and _number(row["secondary_scores"][judge_id], "secondary score") != scores[judge_id]:
                raise ValueError(f"{run_id}: secondary score mismatch for {query_id}.")
        if _number(row.get("primary_score"), "primary score") != scores[primary]:
            raise ValueError(f"{run_id}: primary score mismatch for {query_id}.")
        metric = _object(row.get("retrieval_metrics"), "retrieval metrics")
        if metric.get("status") == "NOT_APPLICABLE" and not metric.get("reason"):
            raise ValueError(f"{run_id}: retrieval NOT_APPLICABLE requires a reason.")
        if retrieval_report["status"] == "COMPLETED":
            if metric != retrieval_queries[query_id]:
                raise ValueError(f"{run_id}: retrieval metric mismatch for {query_id}.")
        elif metric.get("status") != "NOT_APPLICABLE" or metric.get("reason") != retrieval_report["reason"]:
            raise ValueError(f"{run_id}: retrieval NOT_APPLICABLE mismatch for {query_id}.")
        if row["answerer_usage"] != prediction.get("answerer_usage") or row["estimated_cost"] != prediction.get("estimated_cost"):
            raise ValueError(f"{run_id}: analysis row prediction usage or cost mismatch for {query_id}.")
        if row["timing"]["answer"] != prediction.get("answer_timing"):
            raise ValueError(f"{run_id}: analysis row answer timing mismatch for {query_id}.")
        counts = [row[key] for key in (
            "returned_memory_count", "included_memory_count",
            "excluded_memory_count", "retrieved_tokens", "included_tokens",
        )]
        if any(type(value) is not int or value < 0 for value in counts) or counts[0] != counts[1] + counts[2] or counts[4] > counts[3]:
            raise ValueError(f"{run_id}: invalid retrieval counts for {query_id}.")
        if (set(_object(row["framework_usage"], "framework usage")) != {"ingestion", "retrieval"}
                or set(_object(row["judge_usage"], "judge usage")) != set(judge_by_id)
                or set(_object(row["timing"], "timing")) != {"ingestion", "retrieval", "answer", "judging"}):
            raise ValueError(f"{run_id}: analysis row usage or timing schema mismatch for {query_id}.")
        expected_strata = {
            "locomo": ("category",),
            "longmemeval": ("question_type", "is_abstention"),
            "memoryagentbench": ("competence", "task", "source", "answer_format"),
            "beam": ("scale", "ability", "difficulty"),
        }.get(inputs.get("benchmark"))
        if expected_strata is None or any(row["strata"].get(key) != evaluation.get(key) for key in expected_strata):
            raise ValueError(f"{run_id}: analysis row strata mismatch for {query_id}.")
    return VerifiedRun(manifest, rows, unit_by_query, verification["final_manifest_sha256"], latency, bounded_full_context)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _unit_means(run: VerifiedRun, judge_id: str, queries: set[str] | None = None) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in run.rows:
        query_id = row["query_id"]
        if queries is None or query_id in queries:
            grouped[run.unit_by_query[query_id]].append(float(row["judge_scores"][judge_id]))
    return {unit: _mean(scores) for unit, scores in grouped.items()}


def _bootstrap_ci(values: list[float], *, seed: int = 7, samples: int = 10000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(_mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [means[int(.025 * samples)], means[int(.975 * samples)]]


def _wilcoxon(values: list[float]) -> dict[str, Any]:
    nonzero = sorted(((abs(value), value > 0) for value in values if value != 0), key=lambda pair: pair[0])
    n = len(nonzero)
    if not n:
        return {"statistic": 0.0, "p_two_sided": 1.0, "method": "all-zero", "nonzero_units": 0}
    ranks: list[int] = []
    index = 0
    while index < n:
        end = index + 1
        while end < n and nonzero[end][0] == nonzero[index][0]:
            end += 1
        ranks.extend([index + 1 + end] * (end - index))  # Twice the average rank.
        index = end
    positive = sum(rank for rank, (_, sign) in zip(ranks, nonzero) if sign)
    total = sum(ranks)
    statistic = min(positive, total - positive) / 2
    if n <= 25:
        counts = {0: 1}
        for rank in ranks:
            next_counts = counts.copy()
            for subtotal, count in counts.items():
                next_counts[subtotal + rank] = next_counts.get(subtotal + rank, 0) + count
            counts = next_counts
        p_value = min(1.0, 2 * sum(count for subtotal, count in counts.items() if subtotal <= min(positive, total - positive)) / 2**n)
        method = "exact-sign-enumeration"
    else:
        variance = sum(rank * rank for rank in ranks) / 4
        z = max(0.0, abs(positive - total / 2) - 1) / math.sqrt(variance)
        p_value = math.erfc(z / math.sqrt(2))
        method = "normal-tie-corrected-continuity"
    return {"statistic": statistic, "p_two_sided": p_value, "method": method, "nonzero_units": n}


def _paired(left: VerifiedRun, right: VerifiedRun, judge_id: str, queries: set[str] | None = None) -> dict[str, Any]:
    left_means = _unit_means(left, judge_id, queries)
    right_means = _unit_means(right, judge_id, queries)
    if left_means.keys() != right_means.keys() or not left_means:
        raise ValueError("Missing paired atomic units.")
    units = [unit for unit in left.manifest.expected_item_ids if unit in left_means]
    differences = [right_means[unit] - left_means[unit] for unit in units]
    mean = _mean(differences)
    variance = (
        sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
        if len(differences) > 1 else 0.0
    )
    return {
        "unit_count": len(units),
        "query_count": sum(row["query_id"] in queries for row in left.rows)
        if queries is not None else len(left.rows),
        "mean_delta_right_minus_left": mean, "cohen_dz": mean / math.sqrt(variance) if variance else None,
        "bootstrap_95_percentile_ci": _bootstrap_ci(differences), "wilcoxon": _wilcoxon(differences),
    }


def _compatible(left: VerifiedRun, right: VerifiedRun) -> None:
    for field in COMMON_CONTROLS:
        if left.inputs.get(field) != right.inputs.get(field):
            raise ValueError(f"Uncontrolled cross-run difference: {field}.")
    if (
        left.manifest.atomic_unit != right.manifest.atomic_unit
        or left.manifest.expected_item_ids != right.manifest.expected_item_ids
        or left.unit_by_query != right.unit_by_query
    ):
        raise ValueError("Atomic unit or query pairing mismatch.")
    left_ablation = left.inputs.get("ablation")
    right_ablation = right.inputs.get("ablation")
    if bool(left_ablation) != bool(right_ablation):
        raise ValueError("Ablation and non-ablation runs cannot be paired.")
    if left_ablation and (
        left.inputs.get("framework") != "dmf"
        or right.inputs.get("framework") != "dmf"
        or left_ablation.get("full_config_sha256") != right_ablation.get("full_config_sha256")
        or left_ablation.get("scientific_controls_sha256")
        != right_ablation.get("scientific_controls_sha256")
    ):
        raise ValueError("DMF ablation controls mismatch.")


def _profile(run: VerifiedRun) -> str:
    framework = run.inputs["framework"]
    if run.inputs.get("ablation"):
        return str(run.inputs["ablation"]["profile"])
    if framework == "full-context":
        return "bounded-full-context" if run.bounded_full_context else "full-context"
    return framework


def _summary(run: VerifiedRun) -> dict[str, Any]:
    primary = run.inputs["evaluation"]["primary_judge_id"]
    unit_scores = list(_unit_means(run, primary).values())
    secondary = {
        judge["id"]: _mean(list(_unit_means(run, judge["id"]).values()))
        for judge in run.inputs["judges"] if judge["id"] != primary
    }
    strata: dict[str, dict[str, dict[str, Any]]] = {}
    for name in ("category", "competence", "task", "question_type", "ability", "scale"):
        values = {str(row["strata"][name]) for row in run.rows if row["strata"].get(name) is not None}
        if values:
            strata[name] = {}
            for value in sorted(values):
                queries = {row["query_id"] for row in run.rows if str(row["strata"].get(name)) == value}
                scores = list(_unit_means(run, primary, queries).values())
                strata[name][value] = {
                    "query_count": len(queries), "unit_count": len(scores),
                    "mean_unit_score": _mean(scores),
                }
    costs = [row["estimated_cost"] for row in run.rows]
    cost = (
        sum(_number(value, "estimated cost") for value in costs)
        if all(value is not None and isinstance(value, (int, float)) for value in costs)
        else None
    )
    latency = _mean(list(run.latency_ms_by_query.values())) if len(run.latency_ms_by_query) == len(run.rows) else None
    retrieval = [row["retrieval_metrics"] for row in run.rows]
    unavailable = [metric["reason"] for metric in retrieval if metric.get("status") == "NOT_APPLICABLE"]
    return {
        "run_id": run.manifest.run_id, "benchmark": run.inputs["benchmark"], "framework": run.inputs["framework"],
        "profile": _profile(run), "scientific_profile": run.inputs.get("scientific_profile"),
        "protocol": "retrieval-controlled" if run.inputs["benchmark"] == "memoryagentbench" else "benchmark-retrieval-qa",
        "atomic_unit": run.manifest.atomic_unit, "query_count": len(run.rows), "unit_count": len(unit_scores),
        "primary_judge_id": primary, "primary_mean_unit_score": _mean(unit_scores),
        "secondary_mean_unit_scores": secondary, "strata": strata,
        "retrieval": {
            "status": "NOT_APPLICABLE" if len(unavailable) == len(retrieval)
            else "PARTIAL" if unavailable else "AVAILABLE",
            "not_applicable_count": len(unavailable), "reasons": sorted(set(unavailable)),
        },
        "estimated_cost_total": cost, "mean_qa_latency_ms": latency,
        "publication_manifest_sha256": run.publication_sha256,
    }


def _matrix(cells: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    declared: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        benchmark = cell.get("benchmark")
        framework = cell.get("framework")
        if (
            benchmark not in MATRIX_BENCHMARKS
            or framework not in MATRIX_FRAMEWORKS
            or (benchmark, framework) in declared
        ):
            raise ValueError("Matrix cells require unique supported benchmark/framework pairs.")
        run_id = cell.get("run_id")
        status = cell.get("status")
        if run_id is not None:
            if (
                status is not None or run_id not in summaries
                or summaries[run_id]["benchmark"] != benchmark
                or summaries[run_id]["framework"] != framework
            ):
                raise ValueError("Matrix run cell does not match its verified run.")
            declared[(benchmark, framework)] = {"status": "VERIFIED", "run_id": run_id, "profile": summaries[run_id]["profile"]}
        elif (
            status in {"MISSING", "FAILED", "NOT_APPLICABLE"}
            and isinstance(cell.get("reason"), str)
            and cell["reason"].strip()
        ):
            declared[(benchmark, framework)] = {"status": status, "reason": cell["reason"]}
        else:
            raise ValueError("Matrix cell needs a verified run or explicit status and reason.")
    return [
        {
            "benchmark": benchmark,
            "framework": framework,
            **declared.get((benchmark, framework), {
                "status": "MISSING", "reason": "No run declared in analysis plan.",
            }),
        }
        for benchmark in MATRIX_BENCHMARKS
        for framework in MATRIX_FRAMEWORKS
    ]


def analyze(store: LocalArtifactStore, plan: dict[str, Any]) -> dict[str, Any]:
    """Build a deterministic report from an explicit analysis plan."""
    if plan.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("Analysis plan schema_version must be 1.")
    run_ids = plan.get("run_ids")
    comparisons = plan.get("comparisons", [])
    cells = plan.get("cells", [])
    if (
        not isinstance(run_ids, list) or not run_ids
        or len(set(run_ids)) != len(run_ids)
        or not isinstance(comparisons, list)
        or not isinstance(cells, list)
    ):
        raise ValueError("Analysis plan needs unique run_ids and lists of comparisons and cells.")
    runs = {run_id: load_verified_run(store, run_id) for run_id in run_ids}
    summaries = {run_id: _summary(run) for run_id, run in runs.items()}
    contrasts: list[dict[str, Any]] = []
    for comparison in comparisons:
        left_id, right_id = comparison.get("left"), comparison.get("right")
        if left_id not in runs or right_id not in runs or left_id == right_id:
            raise ValueError("Comparison must name two distinct verified runs.")
        left, right = runs[left_id], runs[right_id]
        _compatible(left, right)
        primary = left.inputs["evaluation"]["primary_judge_id"]
        result = {
            "left": left_id, "right": right_id, "benchmark": left.inputs["benchmark"],
            "atomic_unit": left.manifest.atomic_unit, "primary_judge_id": primary,
            "primary": _paired(left, right, primary),
            "secondary_sensitivity": {}, "strata": {},
        }
        for judge in left.inputs["judges"]:
            judge_id = judge["id"]
            if judge_id != primary:
                result["secondary_sensitivity"][judge_id] = _paired(left, right, judge_id)
        for name in ("category", "competence", "task", "question_type", "ability", "scale"):
            values = {str(row["strata"][name]) for row in left.rows if row["strata"].get(name) is not None}
            if values:
                result["strata"][name] = {}
                for value in sorted(values):
                    queries = {row["query_id"] for row in left.rows if str(row["strata"].get(name)) == value}
                    right_queries = {row["query_id"] for row in right.rows if str(row["strata"].get(name)) == value}
                    if queries != right_queries:
                        raise ValueError(f"Stratum pairing mismatch: {name}={value}.")
                    result["strata"][name][value] = _paired(left, right, primary, queries)
        contrasts.append(result)
    ranked = sorted(contrasts, key=lambda contrast: contrast["primary"]["wilcoxon"]["p_two_sided"])
    adjusted = 0.0
    for index, contrast in enumerate(ranked):
        p_value = contrast["primary"]["wilcoxon"]["p_two_sided"]
        adjusted = max(adjusted, min(1.0, (len(ranked) - index) * p_value))
        contrast["primary"]["holm_adjusted_p"] = adjusted
    points = [
        {
            "run_id": run_id, "benchmark": summary["benchmark"],
            "profile": summary["profile"],
            "frontier_family": "full-context" if summary["framework"] == "full-context"
            else "fixed-retrieval-budget",
            "score": summary["primary_mean_unit_score"],
            "estimated_cost_total": summary["estimated_cost_total"],
            "mean_qa_latency_ms": summary["mean_qa_latency_ms"],
        }
        for run_id, summary in summaries.items()
    ]
    frontier_controls = {
        run_id: hash_canonical_json({key: run.inputs.get(key) for key in COMMON_CONTROLS})
        for run_id, run in runs.items()
    }
    for point in points:
        comparable = [
            other for other in points
            if other["frontier_family"] == point["frontier_family"]
            and frontier_controls[other["run_id"]] == frontier_controls[point["run_id"]]
        ]
        if point["estimated_cost_total"] is None or point["mean_qa_latency_ms"] is None:
            point["frontier_status"] = "UNAVAILABLE"
        else:
            dominated = any(
                other is not point
                and other["estimated_cost_total"] is not None
                and other["mean_qa_latency_ms"] is not None
                and other["score"] >= point["score"]
                and other["estimated_cost_total"] <= point["estimated_cost_total"]
                and other["mean_qa_latency_ms"] <= point["mean_qa_latency_ms"]
                and (
                    other["score"] > point["score"]
                    or other["estimated_cost_total"] < point["estimated_cost_total"]
                    or other["mean_qa_latency_ms"] < point["mean_qa_latency_ms"]
                )
                for other in comparable
            )
            point["frontier_status"] = "DOMINATED" if dominated else "FRONTIER"
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "DEMONSTRATION_ANALYSIS", "scientific_result": False,
        "statistical_unit": "manifest atomic unit; query scores averaged within unit",
        "bootstrap": "paired percentile, 10000 resamples, seed 7",
        "multiple_comparisons": "Holm over declared primary contrasts",
        "runs": summaries, "matrix": _matrix(cells, summaries),
        "comparisons": contrasts, "frontier_points": points,
    }


def write_analysis(result: dict[str, Any], output_dir: Path) -> None:
    """Write JSON tables and CSV plot data after all validation succeeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "analysis.json", result)
    comparison_rows = []
    for item in result["comparisons"]:
        primary = item["primary"]
        comparison_rows.append({
            "left": item["left"], "right": item["right"],
            "benchmark": item["benchmark"],
            "unit_count": primary["unit_count"],
            "query_count": primary["query_count"],
            "mean_delta_right_minus_left": primary["mean_delta_right_minus_left"],
            "ci_low": primary["bootstrap_95_percentile_ci"][0],
            "ci_high": primary["bootstrap_95_percentile_ci"][1],
            "wilcoxon_p": primary["wilcoxon"]["p_two_sided"],
            "holm_adjusted_p": primary["holm_adjusted_p"],
        })
    for name, rows, fields in (
        (
            "matrix.csv", result["matrix"],
            ("benchmark", "framework", "status", "run_id", "profile", "reason"),
        ),
        (
            "frontier.csv", result["frontier_points"],
            ("run_id", "benchmark", "profile", "frontier_family", "score",
             "estimated_cost_total", "mean_qa_latency_ms", "frontier_status"),
        ),
        (
            "comparisons.csv", comparison_rows,
            ("left", "right", "benchmark", "unit_count", "query_count",
             "mean_delta_right_minus_left", "ci_low", "ci_high",
             "wilcoxon_p", "holm_adjusted_p"),
        ),
    ):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        (output_dir / name).write_text(buffer.getvalue(), encoding="utf-8")
