"""Prometheus metrics for benchmark runtime boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
)
from prometheus_client import generate_latest, start_http_server

from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import RUN_MANIFEST_SCHEMA_VERSION, RUN_STATUS_SCHEMA_VERSION
from dmf_bench.state import StateError


PHASES = (
    "PREFLIGHT",
    "RUNNING",
    "JUDGING",
    "EVALUATING",
    "REPORTING",
    "PUBLISHING",
    "VERIFYING",
    "COMPLETED",
    "FAILED",
    "FAILED_EVALUATION",
)
RUN_STATES = (
    "CREATED",
    "PREFLIGHT",
    "RUNNING",
    "JUDGING",
    "EVALUATING",
    "REPORTING",
    "PUBLISHING",
    "VERIFYING",
    "COMPLETED",
    "PARTIAL",
    "INTERRUPTING",
    "INTERRUPTED",
    "FAILED_PREFLIGHT",
    "FAILED_RUNNING",
    "FAILED_JUDGING",
    "FAILED_EVALUATION",
    "FAILED_REPORTING",
    "FAILED_PUBLISHING",
    "FAILED_VERIFYING",
)
RUN_LABELS = ("benchmark", "framework")


@dataclass(frozen=True)
class MetricsServer:
    port: int
    thread: Any
    httpd: Any


class BenchmarkMetrics:
    """Bounded-cardinality metrics registry for one benchmark process."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.process_collector = ProcessCollector(registry=self.registry)
        self.expected_units = Gauge(
            "dmf_bench_run_expected_units",
            "Expected benchmark atomic units for the current run.",
            RUN_LABELS,
            registry=self.registry,
        )
        self.committed_units = Gauge(
            "dmf_bench_run_committed_units",
            "Committed benchmark atomic units for the current run.",
            RUN_LABELS,
            registry=self.registry,
        )
        self.progress_ratio = Gauge(
            "dmf_bench_run_progress_ratio",
            "Committed units divided by expected units.",
            RUN_LABELS,
            registry=self.registry,
        )
        self.expected_items = Gauge(
            "dmf_bench_run_expected_items",
            "Expected benchmark questions/items for the current run.",
            RUN_LABELS,
            registry=self.registry,
        )
        self.committed_items = Gauge(
            "dmf_bench_run_committed_items",
            "Committed benchmark questions/items for the current run.",
            RUN_LABELS,
            registry=self.registry,
        )
        self.item_progress_ratio = Gauge(
            "dmf_bench_run_item_progress_ratio",
            "Committed questions/items divided by expected questions/items.",
            RUN_LABELS,
            registry=self.registry,
        )
        self.phase_active = Gauge(
            "dmf_bench_run_phase_active",
            "One when the process is in the labelled phase, zero otherwise.",
            (*RUN_LABELS, "phase"),
            registry=self.registry,
        )
        self.state_active = Gauge(
            "dmf_bench_run_state_active",
            "One when the run is in the labelled state, zero otherwise.",
            (*RUN_LABELS, "state"),
            registry=self.registry,
        )
        self.last_progress_timestamp = Gauge(
            "dmf_bench_last_progress_timestamp_seconds",
            "Unix timestamp of the last observed benchmark progress.",
            RUN_LABELS,
            registry=self.registry,
        )
        self.attempts_total = Counter(
            "dmf_bench_attempts_total",
            "Benchmark attempts by bounded outcome.",
            (*RUN_LABELS, "outcome"),
            registry=self.registry,
        )
        self.phase_duration = Histogram(
            "dmf_bench_phase_duration_seconds",
            "Duration of benchmark lifecycle phases.",
            (*RUN_LABELS, "phase"),
            buckets=(0.1, 0.5, 1, 2.5, 5, 15, 30, 60, 300, 900, float("inf")),
            registry=self.registry,
        )
        self.llm_requests_total = Counter(
            "dmf_bench_llm_requests_total",
            "LLM requests by role/provider/outcome.",
            ("role", "provider", "outcome"),
            registry=self.registry,
        )
        self.llm_tokens_total = Counter(
            "dmf_bench_llm_tokens_total",
            "LLM token counts by role/provider/token type.",
            ("role", "provider", "token_type"),
            registry=self.registry,
        )
        self.llm_retries_total = Counter(
            "dmf_bench_llm_retries_total",
            "LLM retries by role/provider.",
            ("role", "provider"),
            registry=self.registry,
        )
        self.qdrant_operations_total = Counter(
            "dmf_bench_qdrant_client_operations_total",
            "Qdrant client operations by operation/outcome.",
            ("operation", "outcome"),
            registry=self.registry,
        )
        self.qdrant_operation_duration = Histogram(
            "dmf_bench_qdrant_client_operation_duration_seconds",
            "Qdrant client operation duration.",
            ("operation",),
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, float("inf")),
            registry=self.registry,
        )
        self.artifact_publish_total = Counter(
            "dmf_bench_artifact_publish_total",
            "Artifact publish operations by backend/outcome.",
            ("backend", "outcome"),
            registry=self.registry,
        )
        self.artifact_publish_duration = Histogram(
            "dmf_bench_artifact_publish_duration_seconds",
            "Artifact publish duration by backend.",
            ("backend",),
            buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 15, 60, float("inf")),
            registry=self.registry,
        )

    def record_run_status(
        self,
        *,
        benchmark: str,
        framework: str,
        phase: str,
        expected: int,
        committed: int,
        state: str | None = None,
        expected_units: int | None = None,
        committed_units: int | None = None,
    ) -> None:
        labels = _run_labels(benchmark, framework)
        resolved_expected_units = expected if expected_units is None else expected_units
        resolved_committed_units = committed if committed_units is None else committed_units
        self.expected_units.labels(*labels).set(resolved_expected_units)
        self.committed_units.labels(*labels).set(resolved_committed_units)
        self.progress_ratio.labels(*labels).set(
            (resolved_committed_units / resolved_expected_units)
            if resolved_expected_units
            else 0.0
        )
        self.expected_items.labels(*labels).set(expected)
        self.committed_items.labels(*labels).set(committed)
        self.item_progress_ratio.labels(*labels).set(
            (committed / expected) if expected else 0.0
        )
        for candidate in PHASES:
            self.phase_active.labels(*labels, candidate).set(1 if candidate == phase else 0)
        resolved_state = str(state or phase).upper()
        for candidate in RUN_STATES:
            self.state_active.labels(*labels, candidate).set(
                1 if candidate == resolved_state else 0
            )
        self.last_progress_timestamp.labels(*labels).set(time.time())

    def restore_run_status(self, run_dir: str | Path) -> None:
        run_path = Path(run_dir)
        manifest = _read_object(run_path / "run-manifest.json")
        status = _read_object(run_path / "run-status.json")
        if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
            raise StateError(
                f"Run manifest must declare schema_version={RUN_MANIFEST_SCHEMA_VERSION}."
            )
        if status.get("schema_version") != RUN_STATUS_SCHEMA_VERSION:
            raise StateError(
                f"Run status must declare schema_version={RUN_STATUS_SCHEMA_VERSION}."
            )
        inputs = manifest.get("fingerprint_inputs") or {}
        items = status.get("items") or {}
        units = status.get("units") or {}
        self.record_run_status(
            benchmark=str(inputs.get("benchmark", "")),
            framework=str(inputs.get("framework", "")),
            phase=str(status.get("phase", status.get("state", "PREFLIGHT"))),
            expected=int(items.get("expected", 0)),
            committed=int(items.get("committed", 0)),
            state=str(status.get("state", "PREFLIGHT")),
            expected_units=int(units.get("expected", items.get("expected", 0))),
            committed_units=int(units.get("committed", items.get("committed", 0))),
        )

    def record_attempt(self, *, benchmark: str, framework: str, outcome: str) -> None:
        self.attempts_total.labels(*_run_labels(benchmark, framework), _bounded(outcome)).inc()

    def observe_phase_duration(
        self,
        *,
        benchmark: str,
        framework: str,
        phase: str,
        seconds: float,
    ) -> None:
        self.phase_duration.labels(*_run_labels(benchmark, framework), _phase(phase)).observe(seconds)

    def record_llm_request(
        self,
        *,
        role: str,
        provider: str,
        outcome: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.llm_requests_total.labels(_role(role), _provider(provider), _bounded(outcome)).inc()
        self.llm_tokens_total.labels(_role(role), _provider(provider), "prompt").inc(max(prompt_tokens, 0))
        self.llm_tokens_total.labels(_role(role), _provider(provider), "completion").inc(max(completion_tokens, 0))

    def record_llm_retry(self, *, role: str, provider: str) -> None:
        self.llm_retries_total.labels(_role(role), _provider(provider)).inc()

    def record_qdrant_operation(self, *, operation: str, outcome: str, seconds: float) -> None:
        op = _operation(operation)
        self.qdrant_operations_total.labels(op, _bounded(outcome)).inc()
        self.qdrant_operation_duration.labels(op).observe(seconds)

    def record_artifact_publish(self, *, backend: str, outcome: str, seconds: float) -> None:
        safe_backend = _backend(backend)
        self.artifact_publish_total.labels(safe_backend, _bounded(outcome)).inc()
        self.artifact_publish_duration.labels(safe_backend).observe(seconds)

    def render(self) -> bytes:
        return generate_latest(self.registry)

    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST

    def write_snapshot(
        self,
        run_dir: str | Path,
        *,
        operational_summary: dict[str, Any] | None = None,
    ) -> Path:
        metrics_dir = Path(run_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = metrics_dir / "prometheus-snapshot.prom"
        snapshot_path.write_bytes(self.render())
        summary_path = metrics_dir / "operational-summary.json"
        existing = _read_object(summary_path) if summary_path.is_file() else {}
        payload = {
            **existing,
            "schema_version": 1,
            "metrics_snapshot_path": snapshot_path.relative_to(Path(run_dir)).as_posix(),
            "content_type": self.content_type(),
            "cardinality_policy": {
                "run_id_label": "forbidden",
                "item_id_label": "forbidden",
                "prompt_label": "forbidden",
                "collection_label": "forbidden",
            },
        }
        if operational_summary is not None:
            payload.update(operational_summary)
        write_json_atomic(
            summary_path,
            payload,
        )
        return snapshot_path


def start_metrics_endpoint(
    *,
    metrics: BenchmarkMetrics,
    port: int = 9464,
    address: str = "0.0.0.0",
) -> MetricsServer:
    httpd, thread = start_http_server(port, addr=address, registry=metrics.registry)
    return MetricsServer(port=int(httpd.server_port), thread=thread, httpd=httpd)


def stop_metrics_endpoint(server: MetricsServer) -> None:
    server.httpd.shutdown()
    server.httpd.server_close()
    server.thread.join(timeout=5)


def _read_object(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise StateError(f"Expected JSON object at {path}")
    return payload


def _run_labels(benchmark: str, framework: str) -> tuple[str, str]:
    return (_bounded(benchmark), _bounded(framework))


def _phase(value: str) -> str:
    normalized = str(value).upper()
    return normalized if normalized in PHASES else "FAILED"


def _role(value: str) -> str:
    normalized = str(value).lower()
    return normalized if normalized in {"answerer", "judge", "embedding", "reranker"} else "other"


def _provider(value: str) -> str:
    return _bounded(value)


def _operation(value: str) -> str:
    normalized = str(value).lower().replace("-", "_")
    allowed = {"create_collection", "delete_collection", "count", "upsert", "search", "query", "retrieve", "health"}
    return normalized if normalized in allowed else "other"


def _backend(value: str) -> str:
    normalized = str(value).lower()
    return normalized if normalized in {"local-only", "local", "s3-compatible"} else "other"


def _bounded(value: str) -> str:
    normalized = str(value).lower().replace("-", "_")
    if not normalized or len(normalized) > 48:
        return "other"
    if not all(char.isalnum() or char == "_" for char in normalized):
        return "other"
    return normalized
