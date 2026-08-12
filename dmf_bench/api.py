"""Read-only FastAPI application for committed local benchmark artifacts."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from dmf_bench import __version__
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json
from dmf_bench.derived_evaluation import derived_evaluation_dir
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.persisted_metrics import render_persisted_run_metrics
from dmf_bench.state import StateError


LANDING_PAGE_PATH = Path(__file__).with_name("static") / "index.html"


def create_app(runs_dir: str | Path, metrics: BenchmarkMetrics | None = None) -> FastAPI:
    runs_path = Path(runs_dir)
    store = LocalArtifactStore(runs_path)
    metrics_registry = metrics or BenchmarkMetrics()
    app = FastAPI(title="DMF Benchmark Artifact API", version=__version__)

    @app.get("/", include_in_schema=False)
    def landing_page() -> FileResponse:
        return FileResponse(LANDING_PAGE_PATH, media_type="text/html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "mode": "read-only", "writable": False}

    @app.get("/services")
    def services() -> dict[str, Any]:
        return service_catalog()

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(
            content=metrics_registry.render() + render_persisted_run_metrics(runs_path),
            media_type=metrics_registry.content_type(),
        )

    @app.get("/runs")
    def list_runs() -> dict[str, Any]:
        return {
            "runs": list_committed_runs(runs_path),
            "mode": "read-only",
        }

    @app.get("/runs/{run_id}")
    def inspect_run(run_id: str) -> dict[str, Any]:
        try:
            return store.inspect_committed(run_id)
        except (StateError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/artifacts")
    def list_artifacts(run_id: str) -> dict[str, Any]:
        try:
            inspected = store.inspect_committed(run_id)
        except (StateError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "run_id": inspected["run_id"],
            "backend": inspected["backend"],
            "artifacts": inspected["artifacts"],
        }

    @app.get("/runs/{run_id}/verify")
    def verify_run(run_id: str) -> dict[str, Any]:
        try:
            return store.verify_committed(run_id)
        except (StateError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/artifacts/{artifact_path:path}")
    def download_artifact(
        run_id: str,
        artifact_path: str,
        download: bool = False,
    ) -> FileResponse:
        try:
            file_path = store.committed_artifact_path(run_id, artifact_path)
        except (StateError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            file_path,
            media_type=content_type_for(file_path),
            filename=file_path.name if download else None,
        )

    @app.get("/runs/{run_id}/derived-evaluations")
    def list_derived_evaluations(run_id: str) -> dict[str, Any]:
        try:
            root = derived_evaluation_dir(runs_path, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        versions = []
        if root.is_dir():
            versions = sorted(
                path.name
                for path in root.iterdir()
                if path.is_dir() and (path / "derivation.json").is_file()
            )
        return {"run_id": run_id, "evaluator_versions": versions}

    @app.get("/runs/{run_id}/derived-evaluations/{version}/{artifact_name}")
    def download_derived_evaluation(
        run_id: str,
        version: str,
        artifact_name: str,
    ) -> FileResponse:
        allowed = {"derivation.json", "rigorous_report.json", "ablation_report.json"}
        if Path(version).name != version or artifact_name not in allowed:
            raise HTTPException(status_code=404, detail="derived evaluation artifact not found")
        try:
            path = derived_evaluation_dir(runs_path, run_id) / version / artifact_name
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="derived evaluation artifact not found")
        return FileResponse(path, media_type=content_type_for(path))

    return app


def create_default_app() -> FastAPI:
    return create_app(os.getenv("DMF_BENCH_RUNS_DIR", "/bench/runs"))


def list_committed_runs(runs_path: Path) -> list[dict[str, Any]]:
    """Return lightweight metadata for committed runs without rehashing artifacts."""

    if not runs_path.is_dir():
        return []
    runs: list[tuple[float, dict[str, Any]]] = []
    for run_path in runs_path.iterdir():
        if not run_path.is_dir() or run_path.name.startswith("."):
            continue
        completion_path = run_path / "final" / "COMPLETED.json"
        manifest_path = run_path / "run-manifest.json"
        if not completion_path.is_file() or not manifest_path.is_file():
            continue
        try:
            manifest = read_json(manifest_path)
            completion = read_json(completion_path)
            if not isinstance(manifest, dict) or not isinstance(completion, dict):
                continue
            inputs = manifest.get("fingerprint_inputs") or {}
            if not isinstance(inputs, dict):
                inputs = {}
            modified = completion_path.stat().st_mtime
        except (OSError, TypeError, ValueError):
            continue
        runs.append(
            (
                modified,
                {
                    "run_id": run_path.name,
                    "benchmark": str(inputs.get("benchmark", "unknown")),
                    "framework": str(inputs.get("framework", "unknown")),
                    "completed_at": datetime.fromtimestamp(modified, UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "artifact_count": int(
                        ((completion.get("verification") or {}).get("verified_artifact_count") or 0)
                    ),
                },
            )
        )
    return [payload for _modified, payload in sorted(runs, key=lambda item: item[0], reverse=True)]


def serve_artifact_api(
    *,
    runs_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    uvicorn.run(create_app(runs_dir), host=host, port=int(port))


def content_type_for(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return "application/json"
    if path.suffix.lower() in {".md", ".txt"}:
        return "text/plain; charset=utf-8"
    return "application/octet-stream"


def service_catalog() -> dict[str, Any]:
    """Describe the loopback services linked by the landing page."""
    return {
        "schema_version": 1,
        "mode": "local-only",
        "services": {
            "artifact_api": {
                "label": "Artifact API",
                "same_origin": True,
                "path": "/docs",
            },
            "qdrant": {
                "label": "Qdrant",
                "port": _environment_port("QDRANT_HTTP_PORT", 6333),
                "path": "/dashboard",
            },
            "prometheus": {
                "label": "Prometheus",
                "port": _environment_port("PROMETHEUS_PORT", 9090),
                "path": "/",
            },
            "grafana": {
                "label": "Grafana Operations",
                "port": _environment_port("GRAFANA_PORT", 3000),
                "path": "/d/dmf-benchmarks/dmf-benchmarks",
            },
            "grafana_evaluation": {
                "label": "Grafana Evaluation",
                "port": _environment_port("GRAFANA_PORT", 3000),
                "path": "/d/dmf-benchmarks-evaluation/dmf-benchmarks-evaluation",
            },
        },
    }


def _environment_port(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer port.") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535.")
    return port
