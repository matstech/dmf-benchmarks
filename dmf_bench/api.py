"""FastAPI application for local JSON artifact publication and access."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.artifacts.local import resolve_committed_artifact_path
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.state import StateError


class JsonArtifactWrite(BaseModel):
    path: str
    payload: Any


class PublishJsonArtifactsRequest(BaseModel):
    artifacts: list[JsonArtifactWrite] = Field(min_length=1)


def create_app(runs_dir: str | Path) -> FastAPI:
    store = LocalArtifactStore(runs_dir)
    app = FastAPI(title="DMF Benchmark Artifact API")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "writable": True, "write_contract": "json-artifact-batch"}

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
    def download_artifact(run_id: str, artifact_path: str) -> FileResponse:
        try:
            file_path = store.committed_artifact_path(run_id, artifact_path)
        except (StateError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(file_path, media_type=content_type_for(file_path))

    @app.post("/runs/{run_id}/artifacts", status_code=201)
    def publish_json_artifacts(
        run_id: str,
        request: PublishJsonArtifactsRequest,
    ) -> dict[str, Any]:
        try:
            return publish_json_batch(store, run_id, request.artifacts)
        except StateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def create_default_app() -> FastAPI:
    return create_app(os.getenv("DMF_BENCH_RUNS_DIR", "/bench/runs"))


def publish_json_batch(
    store: LocalArtifactStore,
    run_id: str,
    artifacts: list[JsonArtifactWrite],
) -> dict[str, Any]:
    run_dir = store.run_dir(run_id)
    if not run_dir.is_dir():
        raise StateError(f"Run directory does not exist: {run_dir}")

    seen: set[str] = set()
    ingest_dir = run_dir / "api-ingest" / uuid4().hex
    try:
        for artifact in artifacts:
            artifact_path = validate_json_artifact_path(artifact.path)
            if artifact_path in seen:
                raise ValueError(f"Duplicate artifact path: {artifact_path}")
            seen.add(artifact_path)
            target = resolve_committed_artifact_path(ingest_dir, artifact_path)
            write_json_atomic(target, artifact.payload)

        staged = store.stage(run_id, ingest_dir)
        receipt = store.verify(staged)
        store.commit(staged, receipt)
        verified = store.verify_committed(run_id)
        return {
            "run_id": run_id,
            "status": "published",
            "verified": verified,
        }
    finally:
        if ingest_dir.exists():
            shutil.rmtree(ingest_dir)


def validate_json_artifact_path(artifact_path: str) -> str:
    path = Path(artifact_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe artifact path: {artifact_path!r}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Artifact path must end with .json: {artifact_path!r}")
    if path.parts and path.parts[0] in {"api-ingest", "final", "publish-staging"}:
        raise ValueError(f"Artifact path uses a reserved prefix: {artifact_path!r}")
    if path.name in {"COMPLETED.json", "manifest.json"}:
        raise ValueError(f"Artifact path uses a reserved filename: {artifact_path!r}")
    return path.as_posix()


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
