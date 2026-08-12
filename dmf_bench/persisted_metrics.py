"""Prometheus projection of the latest persisted run per benchmark/framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, generate_latest

from dmf_bench.atomic_io import read_json


RUN_LABELS = ("benchmark", "framework")


def render_persisted_run_metrics(runs_dir: str | Path) -> bytes:
    registry = CollectorRegistry()
    expected_records = Gauge(
        "dmf_bench_persisted_run_expected_input_records",
        "Expected source conversations or records in the latest persisted run.",
        RUN_LABELS,
        registry=registry,
    )
    completed_records = Gauge(
        "dmf_bench_persisted_run_completed_input_records",
        "Completed source conversations or records in the latest persisted run.",
        RUN_LABELS,
        registry=registry,
    )
    expected_items = Gauge(
        "dmf_bench_persisted_run_expected_evaluation_items",
        "Expected questions or scored records in the latest persisted run.",
        RUN_LABELS,
        registry=registry,
    )
    completed_items = Gauge(
        "dmf_bench_persisted_run_completed_evaluation_items",
        "Completed questions or scored records in the latest persisted run.",
        RUN_LABELS,
        registry=registry,
    )
    progress = Gauge(
        "dmf_bench_persisted_run_progress_ratio",
        "Completed evaluation items divided by expected evaluation items.",
        RUN_LABELS,
        registry=registry,
    )
    state = Gauge(
        "dmf_bench_persisted_run_state",
        "One for the current state of the latest persisted run.",
        (*RUN_LABELS, "state"),
        registry=registry,
    )
    updated = Gauge(
        "dmf_bench_persisted_run_last_update_timestamp_seconds",
        "Filesystem update time of the latest persisted run status.",
        RUN_LABELS,
        registry=registry,
    )
    tokens = Gauge(
        "dmf_bench_persisted_run_llm_tokens",
        "Committed LLM tokens in the latest persisted run.",
        (*RUN_LABELS, "role", "token_type"),
        registry=registry,
    )
    execution = Gauge(
        "dmf_bench_persisted_run_execution_seconds",
        "Measured execution time of the latest persisted attempt.",
        RUN_LABELS,
        registry=registry,
    )
    pipeline = Gauge(
        "dmf_bench_persisted_run_pipeline_duration_seconds",
        "Total measured pipeline duration by stage in the latest persisted run.",
        (*RUN_LABELS, "stage"),
        registry=registry,
    )
    process_cpu = Gauge(
        "dmf_bench_persisted_run_process_cpu_seconds",
        "Process CPU time consumed by the latest persisted run attempt.",
        RUN_LABELS,
        registry=registry,
    )
    average_cpu = Gauge(
        "dmf_bench_persisted_run_average_cpu_percent",
        "Average process CPU utilization in the latest persisted run attempt.",
        RUN_LABELS,
        registry=registry,
    )
    peak_rss = Gauge(
        "dmf_bench_persisted_run_peak_rss_bytes",
        "Peak process resident memory in the latest persisted run attempt.",
        RUN_LABELS,
        registry=registry,
    )
    result = Gauge(
        "dmf_bench_persisted_run_result",
        "Evaluation result metric from the latest persisted completed run.",
        (*RUN_LABELS, "evaluator", "metric"),
        registry=registry,
    )

    for run_path, manifest, status in _latest_runs(Path(runs_dir)):
        inputs = manifest.get("fingerprint_inputs") or {}
        labels = (str(inputs.get("benchmark", "unknown")), str(inputs.get("framework", "unknown")))
        records = status.get("units") or {}
        items = status.get("items") or {}
        expected_record_count = _integer(records.get("expected"))
        completed_record_count = _integer(records.get("committed"))
        expected_item_count = _integer(items.get("expected"))
        completed_item_count = _integer(items.get("committed"))

        expected_records.labels(*labels).set(expected_record_count)
        completed_records.labels(*labels).set(completed_record_count)
        expected_items.labels(*labels).set(expected_item_count)
        completed_items.labels(*labels).set(completed_item_count)
        progress.labels(*labels).set(
            completed_item_count / expected_item_count if expected_item_count else 0.0
        )
        state.labels(*labels, str(status.get("state", "UNKNOWN"))).set(1)
        updated.labels(*labels).set((run_path / "run-status.json").stat().st_mtime)

        usage = _read_optional_object(run_path / "reports" / "usage.json")
        if usage:
            _record_usage(tokens, labels, usage)
        timing = _read_optional_object(run_path / "reports" / "timing.json")
        if timing:
            _record_timing(execution, pipeline, labels, timing)
        resources = _read_optional_object(run_path / "reports" / "resources.json")
        if resources:
            _record_resources(process_cpu, average_cpu, peak_rss, labels, resources)
        _record_results(result, labels, run_path)

    return generate_latest(registry)


def _latest_runs(root: Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    latest: dict[tuple[str, str], tuple[float, Path, dict[str, Any], dict[str, Any]]] = {}
    if not root.is_dir():
        return []
    for run_path in root.iterdir():
        manifest_path = run_path / "run-manifest.json"
        status_path = run_path / "run-status.json"
        if not run_path.is_dir() or not manifest_path.is_file() or not status_path.is_file():
            continue
        try:
            manifest = _read_object(manifest_path)
            status = _read_object(status_path)
            inputs = manifest.get("fingerprint_inputs") or {}
            key = (str(inputs.get("benchmark", "unknown")), str(inputs.get("framework", "unknown")))
            modified = status_path.stat().st_mtime
        except (OSError, TypeError, ValueError):
            continue
        previous = latest.get(key)
        if previous is None or modified > previous[0]:
            latest[key] = (modified, run_path, manifest, status)
    return [
        (run_path, manifest, status)
        for _, run_path, manifest, status in sorted(latest.values(), key=lambda item: item[1].name)
    ]


def _record_usage(metric: Gauge, labels: tuple[str, str], usage: dict[str, Any]) -> None:
    components = usage.get("components") or {}
    for role in ("answerer", "judge", "memory_internal"):
        component = components.get(role) or {}
        for token_type, field_name in (
            ("prompt", "prompt_tokens"),
            ("completion", "completion_tokens"),
            ("total", "total_tokens"),
        ):
            metric.labels(*labels, role, token_type).set(_integer(component.get(field_name)))


def _record_timing(
    execution_metric: Gauge,
    pipeline_metric: Gauge,
    labels: tuple[str, str],
    timing: dict[str, Any],
) -> None:
    attempt = timing.get("attempt") or {}
    execution_metric.labels(*labels).set(_number(attempt.get("execution_seconds")))
    stages = timing.get("pipeline") or {}
    for stage, values in stages.items():
        if isinstance(values, dict):
            pipeline_metric.labels(*labels, str(stage)).set(
                _number(values.get("total_ms")) / 1000.0
            )


def _record_resources(
    process_cpu_metric: Gauge,
    average_cpu_metric: Gauge,
    peak_rss_metric: Gauge,
    labels: tuple[str, str],
    resources: dict[str, Any],
) -> None:
    process = resources.get("process") or {}
    process_cpu_metric.labels(*labels).set(_number(process.get("cpu_total_seconds")))
    average_cpu_metric.labels(*labels).set(_number(process.get("average_cpu_percent")))
    peak_rss_metric.labels(*labels).set(_number(process.get("peak_rss_bytes")))


def _record_results(metric: Gauge, labels: tuple[str, str], run_path: Path) -> None:
    rigorous_path = run_path / "evaluations" / "rigorous_report.json"
    derived_root = run_path.parent / ".derived-evaluations" / run_path.name
    if derived_root.is_dir():
        derived_reports = sorted(
            (
                path / "rigorous_report.json"
                for path in derived_root.iterdir()
                if path.is_dir() and (path / "rigorous_report.json").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
        )
        if derived_reports:
            rigorous_path = derived_reports[-1]
    reports = (
        ("judge", run_path / "evaluations" / "primary_judge_score.json"),
        ("rigorous", rigorous_path),
    )
    for prefix, path in reports:
        report = _read_optional_object(path)
        evaluator = str(report.get("evaluator_version", prefix))
        overall = (report.get("metrics") or {}).get("overall") or {}
        for name, value in overall.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric.labels(*labels, evaluator, str(name)).set(float(value))


def _read_optional_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _read_object(path)
    except (OSError, TypeError, ValueError):
        return {}


def _read_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
