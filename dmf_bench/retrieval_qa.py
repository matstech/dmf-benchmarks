"""Shared retrieval-QA prediction lifecycle for v3 benchmark cases."""

from __future__ import annotations

import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from .adapters.base import (
    BenchmarkCase,
    CanonicalRetrievalResult,
    FrameworkRunContext,
    OwnedResource,
    PreparedMemoryUnit,
)
from .artifacts import LocalArtifactStore
from .atomic_io import read_json, write_json_atomic
from .config import validate_v3_config
from .context import pack_context
from .contracts import ArtifactRef, RunManifest, RunStatus, UnitCheckpoint, V3_RUN_MANIFEST_SCHEMA_VERSION, scientific_fingerprint, sha256_file
from .fingerprints import build_v3_fingerprint_inputs
from .registry import BENCHMARKS, validate_combination
from .runner import InjectedInterrupt, PredictOnlyResult, _persist_resolved_config, _write_running_phase_checkpoint
from .state import (
    RunLock,
    StateError,
    finish_attempt,
    latest_attempt_id,
    load_manifest,
    new_attempt,
    plan_resume,
    write_attempt,
)


class RetrievalQAPredictOnlyRunner:
    """Execute normalized cases with a single atomic unit lifecycle."""

    def __init__(
        self,
        *,
        benchmark: Any,
        framework: Any,
        answerer: Any,
        artifact_store: LocalArtifactStore,
    ) -> None:
        self.benchmark = benchmark
        self.framework = framework
        self.answerer = answerer
        self.artifact_store = artifact_store

    def run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
        interrupt_at: str | None = None,
        cancel_check: Callable[[], None] | None = None,
        _lock_held: bool = False,
        _attempt: Any = None,
        _finish_attempt: bool = True,
        _on_run_ready: Callable[[Path, Any], None] | None = None,
    ) -> PredictOnlyResult:
        validate_v3_config(
            {key: value for key, value in config.items() if key != "source_path"},
            source_path=Path(str(config.get("source_path", "experiment.json"))),
        )
        benchmark_name = str(config["benchmark"])
        framework_name = str(config["framework"])
        compatibility = validate_combination(benchmark_name, framework_name)
        if self.benchmark.name != benchmark_name or self.framework.name != framework_name:
            raise ValueError("Configured benchmark/framework differs from runner adapters.")
        if BENCHMARKS[benchmark_name].lifecycle != "retrieval-qa-v1":
            raise ValueError("Unsupported benchmark lifecycle.")

        units = self.benchmark.enumerate_units(config)
        cases = {unit.unit_id: self.benchmark.load_case(unit, config) for unit in units}
        expected_units = tuple(unit.unit_id for unit in units)
        if len(cases) != len(units):
            raise ValueError("Benchmark unit IDs must be unique.")
        expected_queries = tuple(
            query.query_id for unit in units for query in cases[unit.unit_id].queries
        )
        if len(expected_queries) != len(set(expected_queries)):
            raise ValueError("Benchmark query IDs must be unique within a run.")
        fingerprint_inputs = build_v3_fingerprint_inputs(
            config, expected_question_ids=expected_queries
        )
        fingerprint = scientific_fingerprint(fingerprint_inputs)
        selected_run_id = str(run_id or config.get("experiment_id") or f"{benchmark_name}-{framework_name}-{fingerprint[:12]}")
        manifest = RunManifest(
            run_id=selected_run_id,
            scientific_fingerprint=fingerprint,
            fingerprint_inputs=fingerprint_inputs,
            expected_item_ids=expected_units,
            atomic_unit=BENCHMARKS[benchmark_name].atomic_unit,
            provenance={"compatibility_status": compatibility.status},
            schema_version=V3_RUN_MANIFEST_SCHEMA_VERSION,
        )
        run_path = self.artifact_store.run_dir(selected_run_id)
        run_context = FrameworkRunContext(
            run_id=selected_run_id,
            scientific_fingerprint=fingerprint,
            run_dir=run_path,
        )
        lock = nullcontext() if _lock_held else RunLock(
            self.artifact_store.runs_dir / ".locks" / f"{selected_run_id}.lock"
        )
        with lock:
            if run_path.exists():
                if not resume:
                    raise StateError(f"Run directory already exists: {run_path}.")
                previous = load_manifest(run_path)
                if previous.scientific_fingerprint != fingerprint or previous.expected_item_ids != expected_units:
                    raise StateError("Resume refused: v3 manifest mismatch.")
                plan = plan_resume(run_path)
                if plan.next_phase == "COMPLETED":
                    return PredictOnlyResult(selected_run_id, "COMPLETED", expected_units, expected_units, ())
                restart_ids = set(plan.restart_unit_ids)
            else:
                if resume:
                    raise StateError(f"Cannot resume missing run directory: {run_path}.")
                self.artifact_store.create_run(manifest)
                restart_ids = set(expected_units)

            attempt = _attempt or new_attempt(
                run_id=selected_run_id,
                entry_phase="RUNNING",
                resume=resume,
                resumed_from_attempt_id=latest_attempt_id(run_path),
            )
            write_attempt(run_path, attempt)
            _persist_resolved_config(run_path, config)
            if _on_run_ready:
                _on_run_ready(run_path, attempt)
            committed = tuple(unit_id for unit_id in expected_units if unit_id not in restart_ids)
            self._write_status(run_path, manifest, cases, committed, "RUNNING")
            restarted: list[str] = []
            try:
                for unit in units:
                    if unit.unit_id not in restart_ids:
                        continue
                    if cancel_check:
                        cancel_check()
                    case: BenchmarkCase = cases[unit.unit_id]
                    safe_unit = case.unit
                    resources = self.framework.resources_for_unit_v3(
                        safe_unit, config, run_context
                    )
                    if not isinstance(resources, tuple) or any(
                        not isinstance(resource, OwnedResource) for resource in resources
                    ):
                        raise StateError("Framework resources must be a tuple of OwnedResource.")
                    cleanup_evidence = self._restart_unit(
                        run_path, safe_unit, resources, config, run_context
                    )
                    cleanup_path = (
                        run_path / "attempts" / attempt.attempt_id / "cleanup"
                        / f"{safe_unit.unit_id}.json"
                    )
                    write_json_atomic(cleanup_path, cleanup_evidence)
                    restarted.append(safe_unit.unit_id)
                    self._write_checkpoint(run_path, manifest, safe_unit.unit_id, "PREPARING")
                    unit_dir = run_path / "items" / safe_unit.unit_id
                    resource_path = unit_dir / "resources.json"
                    write_json_atomic(resource_path, {
                        "schema_version": 3,
                        "resources": [resource.to_dict() for resource in resources],
                    })
                    self._write_checkpoint(run_path, manifest, safe_unit.unit_id, "INGESTING")
                    prepared = self.framework.prepare_unit_v3(
                        safe_unit, case.events, config, run_context
                    )
                    if not isinstance(prepared, PreparedMemoryUnit):
                        raise StateError("Framework did not return PreparedMemoryUnit.")
                    if prepared.resources != resources:
                        raise StateError("Prepared resources differ from the declared owned resources.")
                    prepared_path = unit_dir / "prepared.json"
                    write_json_atomic(prepared_path, prepared.to_dict())
                    evidence = self.framework.verify_prepared_v3(
                        safe_unit, prepared, config, run_context
                    )
                    if not isinstance(evidence, dict) or evidence.get("verified") is not True:
                        raise StateError("Framework commit barrier was not verified.")
                    barrier_path = unit_dir / "commit-barrier.json"
                    write_json_atomic(barrier_path, evidence)
                    if interrupt_at == "after-preparation":
                        raise InjectedInterrupt("Interrupted after preparation.")

                    self._write_checkpoint(run_path, manifest, safe_unit.unit_id, "PREDICTING")
                    artifact_paths = [cleanup_path, resource_path, prepared_path, barrier_path]
                    predictions: list[dict[str, Any]] = []
                    for query_index, query in enumerate(case.queries):
                        if cancel_check:
                            cancel_check()
                        retrieval = self.framework.retrieve_v3(
                            safe_unit, query, prepared, config, run_context
                        )
                        if not isinstance(retrieval, CanonicalRetrievalResult):
                            raise StateError("Framework did not return CanonicalRetrievalResult.")
                        max_results = int(config["retrieval"]["max_results"])
                        if len(retrieval.items) > max_results:
                            raise StateError("Framework exceeded retrieval.max_results.")
                        context_config = config["context_budget"]
                        packed = pack_context(
                            retrieval, context_config["max_tokens"],
                            tokenizer_id=context_config["tokenizer"],
                            renderer_id=context_config["renderer"],
                            packing_id=context_config["packing"],
                        )
                        request = self.benchmark.build_answerer_request_v3(
                            query, packed, config
                        )
                        answer = self.answerer.generate(request)
                        prediction = self.benchmark.build_prediction_v3(
                            case, query, retrieval, packed, answer
                        )
                        prediction_retrieval = prediction["retrieval"]
                        prediction_retrieval["search_results"] = [
                            {
                                "id": item.memory_id,
                                "memory": item.content,
                                "metadata": {
                                    "source_unit_ids": list(
                                        item.metadata.get("source_refs", [])
                                    ),
                                },
                            }
                            for item in retrieval.items
                        ]
                        prediction_retrieval["memories_evaluated"] = len(retrieval.items)
                        prediction["cutoff_label"] = "ranked-whole-items-v1"
                        prediction["memory_internal_usage"] = retrieval.usage.get(
                            "memory_internal", {}
                        )
                        prediction["framework"] = framework_name
                        prediction["scientific_fingerprint"] = fingerprint
                        question_dir = unit_dir / "questions"
                        retrieval_path = question_dir / f"{query.query_id}.retrieval.json"
                        context_path = question_dir / f"{query.query_id}.context.json"
                        prediction_path = question_dir / f"{query.query_id}.json"
                        write_json_atomic(retrieval_path, retrieval.to_dict())
                        write_json_atomic(context_path, packed.to_dict())
                        write_json_atomic(prediction_path, prediction)
                        artifact_paths.extend((retrieval_path, context_path, prediction_path))
                        predictions.append(prediction)
                        if interrupt_at == "after-first-prediction" and query_index == 0:
                            raise InjectedInterrupt("Interrupted after first prediction.")

                    aggregate_path = unit_dir / "predictions.json"
                    write_json_atomic(aggregate_path, {
                        "schema_version": 3,
                        "benchmark": benchmark_name,
                        "atomic_unit": manifest.atomic_unit,
                        "question_ids": [query.query_id for query in case.queries],
                        "predictions": predictions,
                    })
                    artifact_paths.append(aggregate_path)
                    if len(predictions) == 1:
                        single_path = unit_dir / "prediction.json"
                        write_json_atomic(single_path, predictions[0])
                        artifact_paths.append(single_path)
                    retention = config["storage"]["retention"]
                    retention_result = {"policy": retention, "outcome": "kept"}
                    if retention == "delete-on-success":
                        retention_result["cleanup"] = self.framework.cleanup_unit_v3(
                            safe_unit, resources, config, run_context
                        )
                        retention_result["outcome"] = "deleted"
                    retention_path = unit_dir / "retention.json"
                    write_json_atomic(retention_path, retention_result)
                    artifact_paths.append(retention_path)
                    if interrupt_at == "before-commit":
                        raise InjectedInterrupt("Interrupted before unit commit.")
                    self._write_checkpoint(
                        run_path, manifest, safe_unit.unit_id, "COMMITTED",
                        tuple(artifact_paths),
                    )
                    committed = tuple(
                        unit_id for unit_id in expected_units
                        if unit_id in set(committed) | {safe_unit.unit_id}
                    )
                    self._write_status(run_path, manifest, cases, committed, "RUNNING")
                    if interrupt_at == "after-commit":
                        raise InjectedInterrupt("Interrupted after unit commit.")
                if len(committed) == len(expected_units):
                    _write_running_phase_checkpoint(run_path, manifest, attempt)
                self._write_status(run_path, manifest, cases, committed, "PARTIAL")
                if _finish_attempt:
                    finish_attempt(run_path, attempt, status="PARTIAL")
                return PredictOnlyResult(
                    selected_run_id, "PARTIAL", expected_units, committed,
                    tuple(restarted),
                )
            except InjectedInterrupt:
                finish_attempt(run_path, attempt, status="INTERRUPTED")
                raise
            except Exception:
                self._write_status(run_path, manifest, cases, committed, "FAILED_RUNNING")
                finish_attempt(run_path, attempt, status="FAILED")
                raise

    def resolve_run_id(self, config: dict[str, Any], *, run_id: str | None = None) -> str:
        units = self.benchmark.enumerate_units(config)
        expected_questions = tuple(
            query.query_id
            for unit in units
            for query in self.benchmark.load_case(unit, config).queries
        )
        fingerprint = scientific_fingerprint(build_v3_fingerprint_inputs(
            config, expected_question_ids=expected_questions,
        ))
        return str(run_id or config.get("experiment_id") or f"{config['benchmark']}-{config['framework']}-{fingerprint[:12]}")

    def _restart_unit(
        self, run_path: Path, unit: Any, resources: tuple[OwnedResource, ...],
        config: dict[str, Any], run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        unit_id = unit.unit_id
        unit_dir = run_path / "items" / unit_id
        manifest_path = unit_dir / "resources.json"
        if manifest_path.exists():
            payload = read_json(manifest_path)
            if payload.get("resources") != [resource.to_dict() for resource in resources]:
                raise StateError("Resume refused: owned resource manifest mismatch.")
        cleanup = self.framework.cleanup_unit_v3(
            unit, resources, config, run_context
        )
        if not isinstance(cleanup, dict) or cleanup.get("verified") is not True:
            raise StateError("Framework cleanup was not verified.")
        if unit_dir.exists():
            shutil.rmtree(unit_dir)
        checkpoint_dir = run_path / "checkpoints" / unit_id
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        return cleanup

    @staticmethod
    def _write_checkpoint(
        run_path: Path, manifest: RunManifest, unit_id: str, status: str,
        artifact_paths: tuple[Path, ...] = (),
    ) -> None:
        artifacts = tuple(
            ArtifactRef(
                path=path.relative_to(run_path).as_posix(),
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
            )
            for path in artifact_paths
        )
        checkpoint = UnitCheckpoint(
            run_id=manifest.run_id,
            unit_id=unit_id,
            status=status,
            scientific_fingerprint=manifest.scientific_fingerprint,
            artifacts=artifacts,
        )
        write_json_atomic(
            run_path / "checkpoints" / unit_id / "checkpoint.json",
            checkpoint.to_dict(),
        )

    @staticmethod
    def _write_status(
        run_path: Path, manifest: RunManifest,
        cases: dict[str, BenchmarkCase], committed: tuple[str, ...], state: str,
    ) -> None:
        completed_queries = sum(len(cases[unit_id].queries) for unit_id in committed)
        total_queries = sum(len(case.queries) for case in cases.values())
        status = RunStatus(
            run_id=manifest.run_id,
            state=state,
            phase="RUNNING",
            expected=total_queries,
            committed=completed_queries,
            expected_units=len(manifest.expected_item_ids),
            committed_units=len(committed),
        )
        write_json_atomic(run_path / "run-status.json", status.to_dict())
