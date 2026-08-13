"""Conversation- and question-atomic prediction runner primitives.

The runners create authoritative prediction and checkpoint artifacts for the
full lifecycle orchestrator.
"""

from __future__ import annotations

import shutil
import hashlib
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from dmf_bench.adapters.base import (
    AnswererAdapter,
    AnswererRequest,
    BenchmarkUnit,
    FrameworkRunContext,
    ProgressUpdate,
    RetrievalResult,
)
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter, LoCoMoQuestion
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.contracts import (
    ArtifactRef,
    Attempt,
    LifecycleCheckpoint,
    PREDICTION_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    RunManifest,
    RunStatus,
    SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
    UnitCheckpoint,
    scientific_fingerprint,
    hash_canonical_json,
    sha256_file,
)
from dmf_bench.fingerprints import build_scientific_fingerprint_inputs
from dmf_bench.execution import RunInterrupted
from dmf_bench.logging_config import JsonEventLogger
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.provenance import build_run_provenance
from dmf_bench.state import (
    RunLock,
    StateError,
    UnitState,
    artifact_ref_for,
    expected_terminal_item_ids,
    finish_attempt,
    latest_attempt_id,
    load_lifecycle_checkpoint,
    load_manifest,
    load_unit_checkpoint,
    new_attempt,
    plan_resume,
    validate_artifact_refs,
    write_attempt,
    write_lifecycle_checkpoint,
)


class InjectedInterrupt(RuntimeError):
    """Raised by tests to stop a run at a deterministic fault-injection point."""


def _progress_context(
    base: FrameworkRunContext,
    reporter: Callable[[ProgressUpdate], None],
) -> FrameworkRunContext:
    return FrameworkRunContext(
        run_id=base.run_id,
        scientific_fingerprint=base.scientific_fingerprint,
        run_dir=base.run_dir,
        progress_reporter=reporter,
    )


def _current_activity(
    update: ProgressUpdate,
    *,
    framework: str,
    unit_index: int,
    unit_total: int,
) -> dict[str, Any]:
    percentage = (update.completed / update.total * 100.0) if update.total else 0.0
    return {
        "stage": update.stage,
        "label": update.label,
        "memory_system": framework,
        "completed": update.completed,
        "total": update.total,
        "percentage": round(min(100.0, max(0.0, percentage)), 1),
        "item_label": update.item_label,
        "input_record": {
            "number": unit_index + 1,
            "total": unit_total,
        },
        "updated_at": datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
    }


class LongMemEvalQuestionFramework(Protocol):
    name: str

    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        """Delete framework resources for a non-committed unit before restart."""

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        """Ingest one question's memory resources and return barrier metadata."""

    def retrieve(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        """Return framework retrieval and context payload."""


class LoCoMoConversationFramework(Protocol):
    name: str

    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        """Delete framework resources for a non-committed conversation before restart."""

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        """Ingest one conversation once and return barrier metadata."""

    def retrieve_question(
        self,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        question: LoCoMoQuestion,
        config: dict[str, Any],
        prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        """Return framework retrieval and context for one question."""


@dataclass(frozen=True)
class PredictOnlyResult:
    run_id: str
    state: str
    expected_unit_ids: tuple[str, ...]
    committed_unit_ids: tuple[str, ...]
    restarted_unit_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "expected_unit_ids": list(self.expected_unit_ids),
            "committed_unit_ids": list(self.committed_unit_ids),
            "restarted_unit_ids": list(self.restarted_unit_ids),
        }


class LongMemEvalPredictOnlyRunner:
    """Prediction-only LongMemEval runner with unit-level atomic resume."""

    def __init__(
        self,
        *,
        benchmark: LongMemEvalAdapter | None = None,
        artifact_store: LocalArtifactStore,
        framework: LongMemEvalQuestionFramework,
        answerer: AnswererAdapter,
        metrics: BenchmarkMetrics | None = None,
        events: JsonEventLogger | None = None,
    ) -> None:
        self.benchmark = benchmark or LongMemEvalAdapter()
        self.artifact_store = artifact_store
        self.framework = framework
        self.answerer = answerer
        self.metrics = metrics
        self.events = events

    def run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
        interrupt_at: str | None = None,
        cancel_check: Callable[[], None] | None = None,
        _lock_held: bool = False,
        _attempt: Attempt | None = None,
        _finish_attempt: bool = True,
        _on_run_ready: Callable[[Path, Attempt], None] | None = None,
    ) -> PredictOnlyResult:
        if config.get("benchmark") != "longmemeval":
            raise ValueError("LongMemEvalPredictOnlyRunner only supports benchmark='longmemeval'.")
        if str(config.get("framework", "")) != self.framework.name:
            raise ValueError(
                f"Config framework {config.get('framework')!r} does not match "
                f"runner framework {self.framework.name!r}."
            )

        units = self.benchmark.enumerate_units(config)
        questions_by_id = self.benchmark.selected_questions_by_id(config)
        expected_unit_ids = tuple(unit.unit_id for unit in units)
        manifest = self._build_manifest(config, expected_unit_ids, run_id=run_id)
        run_path = self.artifact_store.run_dir(manifest.run_id)
        run_context = FrameworkRunContext(
            run_id=manifest.run_id,
            scientific_fingerprint=manifest.scientific_fingerprint,
            run_dir=run_path,
        )

        lock_path = self.artifact_store.runs_dir / ".locks" / f"{manifest.run_id}.lock"
        lock = nullcontext() if _lock_held else RunLock(lock_path)
        with lock:
            active_attempt = _attempt
            attempt_persisted = False
            try:
                if run_path.exists():
                    if not resume:
                        raise StateError(
                            f"Run directory already exists: {run_path}. Use resume=True to continue."
                        )
                    self._validate_resume_manifest(run_path, manifest)
                    resume_plan = plan_resume(run_path)
                    if resume_plan.next_phase == "COMPLETED":
                        return PredictOnlyResult(
                            run_id=manifest.run_id,
                            state="COMPLETED",
                            expected_unit_ids=expected_unit_ids,
                            committed_unit_ids=expected_unit_ids,
                            restarted_unit_ids=(),
                        )
                    restart_ids = set(resume_plan.restart_unit_ids)
                else:
                    if resume:
                        raise StateError(f"Cannot resume missing run directory: {run_path}")
                    self.artifact_store.create_run(manifest)
                    restart_ids = set(expected_unit_ids)

                if active_attempt is None:
                    previous_attempt = latest_attempt_id(run_path)
                    active_attempt = new_attempt(
                        run_id=manifest.run_id,
                        entry_phase="RUNNING",
                        resume=resume,
                        resumed_from_attempt_id=previous_attempt,
                    )
                write_attempt(run_path, active_attempt)
                attempt_persisted = True
                _persist_resolved_config(run_path, config)
                if _on_run_ready is not None:
                    _on_run_ready(run_path, active_attempt)
                _check_cancel(cancel_check)

                committed = self._committed_units(run_path, expected_unit_ids)
                self._write_status(run_path, manifest, committed_units=committed)
                restarted: list[str] = []
                for unit_index, unit in enumerate(units):
                    if unit.unit_id not in restart_ids:
                        continue
                    unit_run_context = _progress_context(
                        run_context,
                        lambda update, committed_units=committed, current_index=unit_index: self._write_status(
                            run_path,
                            manifest,
                            committed_units=committed_units,
                            current_activity=_current_activity(
                                update,
                                framework=self.framework.name,
                                unit_index=current_index,
                                unit_total=len(units),
                            ),
                        ),
                    )
                    _check_cancel(cancel_check)
                    _emit_unit_event(
                        self.events,
                        "unit.prediction.started",
                        manifest,
                        active_attempt,
                        unit,
                        len(restarted),
                    )
                    question = questions_by_id[unit.unit_id]
                    self._restart_unit(
                        run_path,
                        unit,
                        question,
                        config,
                        manifest,
                        unit_run_context,
                    )
                    restarted.append(unit.unit_id)
                    if interrupt_at == "before-prediction-commit":
                        raise InjectedInterrupt("Interrupted before prediction commit.")

                    prediction_paths = self._write_prediction(
                        run_path,
                        unit,
                        question,
                        config,
                        manifest,
                        unit_run_context,
                        cancel_check,
                    )
                    self._write_committed_checkpoint(
                        run_path,
                        manifest,
                        unit,
                        prediction_paths,
                    )
                    committed = self._committed_units(run_path, expected_unit_ids)
                    self._write_status(run_path, manifest, committed_units=committed)
                    _emit_unit_event(
                        self.events,
                        "unit.prediction.committed",
                        manifest,
                        active_attempt,
                        unit,
                        len(restarted) - 1,
                    )
                    _check_cancel(cancel_check)
                    if interrupt_at == "after-prediction-commit":
                        raise InjectedInterrupt("Interrupted after prediction commit.")

                committed = self._committed_units(run_path, expected_unit_ids)
                if committed == expected_unit_ids:
                    _write_running_phase_checkpoint(
                        run_path,
                        manifest,
                        active_attempt,
                    )
                self._write_status(
                    run_path,
                    manifest,
                    committed_units=committed,
                    state="PARTIAL" if _finish_attempt else "RUNNING",
                )
                if self.metrics is not None:
                    self.metrics.write_snapshot(run_path)
                if _finish_attempt:
                    finish_attempt(run_path, active_attempt, status="PARTIAL")
                return PredictOnlyResult(
                    run_id=manifest.run_id,
                    state="PARTIAL",
                    expected_unit_ids=expected_unit_ids,
                    committed_unit_ids=committed,
                    restarted_unit_ids=tuple(restarted),
                )
            except RunInterrupted:
                if attempt_persisted and active_attempt is not None:
                    committed = self._committed_units(run_path, expected_unit_ids)
                    self._write_status(
                        run_path,
                        manifest,
                        committed_units=committed,
                        state="INTERRUPTING",
                    )
                    self._write_status(
                        run_path,
                        manifest,
                        committed_units=committed,
                        state="INTERRUPTED",
                    )
                    finish_attempt(run_path, active_attempt, status="INTERRUPTED")
                raise
            except InjectedInterrupt:
                if attempt_persisted and active_attempt is not None:
                    finish_attempt(run_path, active_attempt, status="INTERRUPTED")
                raise
            except Exception as exc:
                if attempt_persisted:
                    committed = self._committed_units(run_path, expected_unit_ids)
                    self._write_status(
                        run_path,
                        manifest,
                        committed_units=committed,
                        state="FAILED_RUNNING",
                    )
                    if active_attempt is not None:
                        _write_failed_running_checkpoint(
                            run_path,
                            manifest,
                            active_attempt,
                            error_type=type(exc).__name__,
                        )
                        finish_attempt(run_path, active_attempt, status="FAILED")
                raise

    def resolve_run_id(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> str:
        expected = tuple(unit.unit_id for unit in self.benchmark.enumerate_units(config))
        resolved = self._build_manifest(config, expected, run_id=run_id).run_id
        self.artifact_store.run_dir(resolved)
        return resolved

    def _build_manifest(
        self,
        config: dict[str, Any],
        expected_unit_ids: tuple[str, ...],
        *,
        run_id: str | None,
    ) -> RunManifest:
        fingerprint_inputs = build_scientific_fingerprint_inputs(config)
        fingerprint = scientific_fingerprint(fingerprint_inputs)
        resolved_run_id = str(
            run_id
            or config.get("run_id")
            or config.get("experiment_id")
            or f"longmemeval-{config.get('framework')}-{fingerprint[:12]}"
        )
        return RunManifest(
            run_id=resolved_run_id,
            scientific_fingerprint=fingerprint,
            fingerprint_inputs=fingerprint_inputs,
            expected_item_ids=expected_unit_ids,
            atomic_unit=self.benchmark.atomic_unit,
            resume_policy="restart-unit",
            provenance=build_run_provenance(config, fingerprint_inputs=fingerprint_inputs),
        )

    def _validate_resume_manifest(self, run_path: Path, expected: RunManifest) -> None:
        existing = load_manifest(run_path)
        if existing.scientific_fingerprint != expected.scientific_fingerprint:
            raise StateError("Resume refused: scientific fingerprint mismatch.")
        if existing.expected_item_ids != expected.expected_item_ids:
            raise StateError("Resume refused: expected item set mismatch.")
        if existing.atomic_unit != expected.atomic_unit:
            raise StateError("Resume refused: atomic unit mismatch.")

    def _restart_unit(
        self,
        run_path: Path,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        manifest: RunManifest,
        run_context: FrameworkRunContext,
    ) -> None:
        self.framework.cleanup_unit(
            unit,
            question,
            config,
            run_context=run_context,
        )
        _remove_child_dir(run_path / "items", unit.unit_id)
        _remove_child_dir(run_path / "checkpoints", unit.unit_id)
        self._write_checkpoint(run_path, manifest, unit, UnitState.PREPARING)
        self._write_checkpoint(run_path, manifest, unit, UnitState.INGESTING)

    def _write_prediction(
        self,
        run_path: Path,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        manifest: RunManifest,
        run_context: FrameworkRunContext,
        cancel_check: Callable[[], None] | None,
    ) -> tuple[Path, ...]:
        ingestion_started_at = time.perf_counter()
        prepared = self.framework.prepare_unit(
            unit,
            question,
            config,
            run_context=run_context,
        )
        ingestion_ms = (time.perf_counter() - ingestion_started_at) * 1000
        _check_cancel(cancel_check)
        self._write_checkpoint(run_path, manifest, unit, UnitState.PREDICTING)
        run_context.report_progress(
            ProgressUpdate(
                stage="answer_generation",
                label="Generating benchmark answer",
                completed=0,
                total=1,
                item_label="question",
            )
        )
        qa_started_at = time.perf_counter()
        retrieval = self.framework.retrieve(
            unit,
            question,
            config,
            prepared,
            run_context=run_context,
        ).to_dict()
        _check_cancel(cancel_check)
        answerer_input = self.benchmark.build_answerer_input(
            question=question,
            framework_name=self.framework.name,
            retrieval=retrieval,
        )
        answerer_started_at = time.perf_counter()
        answerer_output = self.answerer.generate(
            AnswererRequest(
                system_prompt=answerer_input.system_prompt,
                user_prompt=answerer_input.user_prompt,
                metadata={
                    **answerer_input.metadata,
                    "unit_id": unit.unit_id,
                },
            )
        )
        answer_generation_ms = (time.perf_counter() - answerer_started_at) * 1000
        pipeline_timing = (
            dict(retrieval.get("timing", {}))
            if isinstance(retrieval.get("timing", {}), dict)
            else {}
        )
        pipeline_timing.update(
            {
                "ingestion_ms": ingestion_ms,
                "ingestion_scope": "question",
                "qa_ms": (time.perf_counter() - qa_started_at) * 1000,
                "qa_scope": "question",
                "answer_generation_ms": answer_generation_ms,
                "answer_generation_scope": "question",
            }
        )
        _check_cancel(cancel_check)
        prediction = self.benchmark.build_prediction(
            question=question,
            framework_name=self.framework.name,
            retrieval=retrieval,
            answerer_input=answerer_input,
            answerer_output=answerer_output,
        )
        prediction_path = run_path / "items" / unit.unit_id / "prediction.json"
        write_json_atomic(prediction_path, prediction)
        timing_path = run_path / "items" / unit.unit_id / "timing.json"
        write_json_atomic(
            timing_path,
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "benchmark": str(prediction.get("benchmark", "longmemeval")),
                "question_id": str(prediction.get("question_id", unit.unit_id)),
                "pipeline_timing": pipeline_timing,
            },
        )
        run_context.report_progress(
            ProgressUpdate(
                stage="answer_generation",
                label="Generating benchmark answer",
                completed=1,
                total=1,
                item_label="question",
            )
        )
        cleanup_manifest_path = _persist_cleanup_manifest(run_path, unit, prepared)
        retention_path = _apply_unit_retention(
            run_path=run_path,
            unit=unit,
            item=question,
            config=config,
            framework=self.framework,
            run_context=run_context,
        )
        return tuple(
            path
            for path in (
                prediction_path,
                timing_path,
                cleanup_manifest_path,
                retention_path,
            )
            if path is not None
        )

    def _write_committed_checkpoint(
        self,
        run_path: Path,
        manifest: RunManifest,
        unit: BenchmarkUnit,
        artifact_paths: tuple[Path, ...],
    ) -> None:
        artifacts = tuple(
            ArtifactRef(
                path=path.relative_to(run_path).as_posix(),
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
            )
            for path in artifact_paths
        )
        self._write_checkpoint(
            run_path,
            manifest,
            unit,
            UnitState.COMMITTED,
            artifacts=artifacts,
        )

    def _write_checkpoint(
        self,
        run_path: Path,
        manifest: RunManifest,
        unit: BenchmarkUnit,
        state: UnitState,
        *,
        artifacts: tuple[ArtifactRef, ...] = (),
    ) -> None:
        checkpoint = UnitCheckpoint(
            run_id=manifest.run_id,
            unit_id=unit.unit_id,
            status=state.value,
            scientific_fingerprint=manifest.scientific_fingerprint,
            artifacts=artifacts,
        )
        write_json_atomic(
            run_path / "checkpoints" / unit.unit_id / "checkpoint.json",
            checkpoint.to_dict(),
        )

    def _committed_units(self, run_path: Path, expected_unit_ids: tuple[str, ...]) -> tuple[str, ...]:
        resume_plan = plan_resume(run_path)
        return tuple(
            unit_id
            for unit_id in expected_unit_ids
            if unit_id in set(resume_plan.committed_unit_ids)
        )

    def _write_status(
        self,
        run_path: Path,
        manifest: RunManifest,
        *,
        committed_units: tuple[str, ...],
        state: str = "RUNNING",
        current_activity: dict[str, Any] | None = None,
    ) -> None:
        expected = len(expected_terminal_item_ids(manifest))
        committed = len(committed_units)
        status = RunStatus(
            run_id=manifest.run_id,
            state=state,
            phase="RUNNING",
            expected=expected,
            committed=committed,
            expected_units=len(manifest.expected_item_ids),
            committed_units=len(committed_units),
            current_activity=current_activity,
        )
        write_json_atomic(run_path / "run-status.json", status.to_dict())
        if self.metrics is not None:
            self.metrics.record_run_status(
                benchmark=str(manifest.fingerprint_inputs.get("benchmark", "")),
                framework=str(manifest.fingerprint_inputs.get("framework", "")),
                phase="RUNNING",
                expected=expected,
                committed=committed,
                state=state,
                expected_units=len(manifest.expected_item_ids),
                committed_units=len(committed_units),
                current_activity=current_activity,
            )


def _remove_child_dir(parent: Path, child_name: str) -> None:
    child = Path(child_name)
    if child.is_absolute() or ".." in child.parts:
        raise StateError(f"Refusing to clean unsafe unit path: {child_name}")
    target = parent / child_name
    if target.exists():
        shutil.rmtree(target)


class LoCoMoPredictOnlyRunner:
    """Prediction-only LoCoMo runner with conversation-level atomic resume."""

    def __init__(
        self,
        *,
        benchmark: LoCoMoAdapter | None = None,
        artifact_store: LocalArtifactStore,
        framework: LoCoMoConversationFramework,
        answerer: AnswererAdapter,
        metrics: BenchmarkMetrics | None = None,
        events: JsonEventLogger | None = None,
    ) -> None:
        self.benchmark = benchmark or LoCoMoAdapter()
        self.artifact_store = artifact_store
        self.framework = framework
        self.answerer = answerer
        self.metrics = metrics
        self.events = events

    def run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
        interrupt_at: str | None = None,
        cancel_check: Callable[[], None] | None = None,
        _lock_held: bool = False,
        _attempt: Attempt | None = None,
        _finish_attempt: bool = True,
        _on_run_ready: Callable[[Path, Attempt], None] | None = None,
    ) -> PredictOnlyResult:
        if config.get("benchmark") != "locomo":
            raise ValueError("LoCoMoPredictOnlyRunner only supports benchmark='locomo'.")
        if str(config.get("framework", "")) != self.framework.name:
            raise ValueError(
                f"Config framework {config.get('framework')!r} does not match "
                f"runner framework {self.framework.name!r}."
            )

        units = self.benchmark.enumerate_units(config)
        conversations_by_id = self.benchmark.selected_conversations_by_id(config)
        expected_unit_ids = tuple(unit.unit_id for unit in units)
        manifest = self._build_manifest(config, expected_unit_ids, run_id=run_id)
        run_path = self.artifact_store.run_dir(manifest.run_id)
        run_context = FrameworkRunContext(
            run_id=manifest.run_id,
            scientific_fingerprint=manifest.scientific_fingerprint,
            run_dir=run_path,
        )

        lock_path = self.artifact_store.runs_dir / ".locks" / f"{manifest.run_id}.lock"
        lock = nullcontext() if _lock_held else RunLock(lock_path)
        with lock:
            active_attempt = _attempt
            attempt_persisted = False
            try:
                if run_path.exists():
                    if not resume:
                        raise StateError(
                            f"Run directory already exists: {run_path}. Use resume=True to continue."
                        )
                    self._validate_resume_manifest(run_path, manifest)
                    resume_plan = plan_resume(run_path)
                    if resume_plan.next_phase == "COMPLETED":
                        return PredictOnlyResult(
                            run_id=manifest.run_id,
                            state="COMPLETED",
                            expected_unit_ids=expected_unit_ids,
                            committed_unit_ids=expected_unit_ids,
                            restarted_unit_ids=(),
                        )
                    restart_ids = set(resume_plan.restart_unit_ids)
                else:
                    if resume:
                        raise StateError(f"Cannot resume missing run directory: {run_path}")
                    self.artifact_store.create_run(manifest)
                    restart_ids = set(expected_unit_ids)

                if active_attempt is None:
                    previous_attempt = latest_attempt_id(run_path)
                    active_attempt = new_attempt(
                        run_id=manifest.run_id,
                        entry_phase="RUNNING",
                        resume=resume,
                        resumed_from_attempt_id=previous_attempt,
                    )
                write_attempt(run_path, active_attempt)
                attempt_persisted = True
                _persist_resolved_config(run_path, config)
                if _on_run_ready is not None:
                    _on_run_ready(run_path, active_attempt)
                _check_cancel(cancel_check)

                committed = self._committed_units(run_path, expected_unit_ids)
                self._write_status(run_path, manifest, committed_units=committed)
                restarted: list[str] = []
                for unit_index, unit in enumerate(units):
                    if unit.unit_id not in restart_ids:
                        continue
                    unit_run_context = _progress_context(
                        run_context,
                        lambda update, committed_units=committed, current_index=unit_index: self._write_status(
                            run_path,
                            manifest,
                            committed_units=committed_units,
                            current_activity=_current_activity(
                                update,
                                framework=self.framework.name,
                                unit_index=current_index,
                                unit_total=len(units),
                            ),
                        ),
                    )
                    _check_cancel(cancel_check)
                    _emit_unit_event(
                        self.events,
                        "unit.prediction.started",
                        manifest,
                        active_attempt,
                        unit,
                        len(restarted),
                    )
                    _conversation_idx, conversation, questions = conversations_by_id[unit.unit_id]
                    self._restart_unit(
                        run_path,
                        unit,
                        conversation,
                        config,
                        manifest,
                        unit_run_context,
                    )
                    restarted.append(unit.unit_id)
                    ingestion_started_at = time.perf_counter()
                    prepared = self.framework.prepare_unit(
                        unit,
                        conversation,
                        config,
                        run_context=unit_run_context,
                    )
                    ingestion_ms = (time.perf_counter() - ingestion_started_at) * 1000
                    _check_cancel(cancel_check)
                    cleanup_manifest_path = _persist_cleanup_manifest(
                        run_path,
                        unit,
                        prepared,
                    )
                    self._write_checkpoint(run_path, manifest, unit, UnitState.PREDICTING)
                    if interrupt_at == "after-ingestion-before-first-prediction":
                        raise InjectedInterrupt("Interrupted after ingestion before first prediction.")

                    prediction_paths: list[Path] = []
                    timing_paths: list[Path] = []
                    unit_run_context.report_progress(
                        ProgressUpdate(
                            stage="answer_generation",
                            label="Generating benchmark answers",
                            completed=0,
                            total=len(questions),
                            item_label="questions",
                        )
                    )
                    for question_offset, question in enumerate(questions):
                        prediction_path, timing_path = self._write_question_prediction(
                            run_path,
                            unit,
                            conversation,
                            question,
                            config,
                            prepared,
                            ingestion_ms,
                            unit_run_context,
                            cancel_check,
                        )
                        prediction_paths.append(prediction_path)
                        timing_paths.append(timing_path)
                        unit_run_context.report_progress(
                            ProgressUpdate(
                                stage="answer_generation",
                                label="Generating benchmark answers",
                                completed=question_offset + 1,
                                total=len(questions),
                                item_label="questions",
                            )
                        )
                        if interrupt_at == "after-first-prediction" and question_offset == 0:
                            raise InjectedInterrupt("Interrupted after first LoCoMo prediction.")

                    aggregate_path = self._write_conversation_aggregate(run_path, unit, prediction_paths)
                    retention_path = _apply_unit_retention(
                        run_path=run_path,
                        unit=unit,
                        item=conversation,
                        config=config,
                        framework=self.framework,
                        run_context=unit_run_context,
                    )
                    self._write_committed_checkpoint(
                        run_path,
                        manifest,
                        unit,
                        tuple(
                            path
                            for path in (
                                *prediction_paths,
                                *timing_paths,
                                aggregate_path,
                                cleanup_manifest_path,
                                retention_path,
                            )
                            if path is not None
                        ),
                    )
                    committed = self._committed_units(run_path, expected_unit_ids)
                    self._write_status(run_path, manifest, committed_units=committed)
                    _emit_unit_event(
                        self.events,
                        "unit.prediction.committed",
                        manifest,
                        active_attempt,
                        unit,
                        len(restarted) - 1,
                    )
                    _check_cancel(cancel_check)

                committed = self._committed_units(run_path, expected_unit_ids)
                if committed == expected_unit_ids:
                    _write_running_phase_checkpoint(
                        run_path,
                        manifest,
                        active_attempt,
                    )
                self._write_status(
                    run_path,
                    manifest,
                    committed_units=committed,
                    state="PARTIAL" if _finish_attempt else "RUNNING",
                )
                if self.metrics is not None:
                    self.metrics.write_snapshot(run_path)
                if _finish_attempt:
                    finish_attempt(run_path, active_attempt, status="PARTIAL")
                return PredictOnlyResult(
                    run_id=manifest.run_id,
                    state="PARTIAL",
                    expected_unit_ids=expected_unit_ids,
                    committed_unit_ids=committed,
                    restarted_unit_ids=tuple(restarted),
                )
            except RunInterrupted:
                if attempt_persisted and active_attempt is not None:
                    committed = self._committed_units(run_path, expected_unit_ids)
                    self._write_status(
                        run_path,
                        manifest,
                        committed_units=committed,
                        state="INTERRUPTING",
                    )
                    self._write_status(
                        run_path,
                        manifest,
                        committed_units=committed,
                        state="INTERRUPTED",
                    )
                    finish_attempt(run_path, active_attempt, status="INTERRUPTED")
                raise
            except InjectedInterrupt:
                if attempt_persisted and active_attempt is not None:
                    finish_attempt(run_path, active_attempt, status="INTERRUPTED")
                raise
            except Exception as exc:
                if attempt_persisted:
                    committed = self._committed_units(run_path, expected_unit_ids)
                    self._write_status(
                        run_path,
                        manifest,
                        committed_units=committed,
                        state="FAILED_RUNNING",
                    )
                    if active_attempt is not None:
                        _write_failed_running_checkpoint(
                            run_path,
                            manifest,
                            active_attempt,
                            error_type=type(exc).__name__,
                        )
                        finish_attempt(run_path, active_attempt, status="FAILED")
                raise

    def resolve_run_id(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> str:
        expected = tuple(unit.unit_id for unit in self.benchmark.enumerate_units(config))
        resolved = self._build_manifest(config, expected, run_id=run_id).run_id
        self.artifact_store.run_dir(resolved)
        return resolved

    def _build_manifest(
        self,
        config: dict[str, Any],
        expected_unit_ids: tuple[str, ...],
        *,
        run_id: str | None,
    ) -> RunManifest:
        fingerprint_inputs = build_scientific_fingerprint_inputs(
            config,
            expected_question_ids=self.benchmark.expected_question_ids(config),
        )
        fingerprint = scientific_fingerprint(fingerprint_inputs)
        resolved_run_id = str(
            run_id
            or config.get("run_id")
            or config.get("experiment_id")
            or f"locomo-{config.get('framework')}-{fingerprint[:12]}"
        )
        return RunManifest(
            run_id=resolved_run_id,
            scientific_fingerprint=fingerprint,
            fingerprint_inputs=fingerprint_inputs,
            expected_item_ids=expected_unit_ids,
            atomic_unit=self.benchmark.atomic_unit,
            resume_policy="restart-unit",
            provenance=build_run_provenance(config, fingerprint_inputs=fingerprint_inputs),
        )

    def _validate_resume_manifest(self, run_path: Path, expected: RunManifest) -> None:
        existing = load_manifest(run_path)
        if existing.scientific_fingerprint != expected.scientific_fingerprint:
            raise StateError("Resume refused: scientific fingerprint mismatch.")
        if existing.expected_item_ids != expected.expected_item_ids:
            raise StateError("Resume refused: expected conversation set mismatch.")
        if existing.atomic_unit != expected.atomic_unit:
            raise StateError("Resume refused: atomic unit mismatch.")

    def _restart_unit(
        self,
        run_path: Path,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        config: dict[str, Any],
        manifest: RunManifest,
        run_context: FrameworkRunContext,
    ) -> None:
        self.framework.cleanup_unit(
            unit,
            conversation,
            config,
            run_context=run_context,
        )
        _remove_child_dir(run_path / "items", unit.unit_id)
        _remove_child_dir(run_path / "checkpoints", unit.unit_id)
        self._write_checkpoint(run_path, manifest, unit, UnitState.PREPARING)
        self._write_checkpoint(run_path, manifest, unit, UnitState.INGESTING)

    def _write_question_prediction(
        self,
        run_path: Path,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        question: LoCoMoQuestion,
        config: dict[str, Any],
        prepared: dict[str, Any],
        ingestion_ms: float,
        run_context: FrameworkRunContext,
        cancel_check: Callable[[], None] | None,
    ) -> tuple[Path, Path]:
        qa_started_at = time.perf_counter()
        retrieval = self.framework.retrieve_question(
            unit,
            conversation,
            question,
            config,
            prepared,
            run_context=run_context,
        ).to_dict()
        _check_cancel(cancel_check)
        answerer_input = self.benchmark.build_answerer_input(
            conversation=conversation,
            question=question,
            framework_name=self.framework.name,
            retrieval=retrieval,
        )
        answerer_started_at = time.perf_counter()
        answerer_output = self.answerer.generate(
            AnswererRequest(
                system_prompt=answerer_input.system_prompt,
                user_prompt=answerer_input.user_prompt,
                metadata={
                    **answerer_input.metadata,
                    "unit_id": unit.unit_id,
                },
            )
        )
        answer_generation_ms = (time.perf_counter() - answerer_started_at) * 1000
        pipeline_timing = (
            dict(retrieval.get("timing", {}))
            if isinstance(retrieval.get("timing", {}), dict)
            else {}
        )
        pipeline_timing.update(
            {
                "ingestion_ms": ingestion_ms,
                "ingestion_scope": "conversation",
                "qa_ms": (time.perf_counter() - qa_started_at) * 1000,
                "qa_scope": "question",
                "answer_generation_ms": answer_generation_ms,
                "answer_generation_scope": "question",
            }
        )
        _check_cancel(cancel_check)
        prediction = self.benchmark.build_prediction(
            conversation=conversation,
            question=question,
            framework_name=self.framework.name,
            retrieval=retrieval,
            answerer_input=answerer_input,
            answerer_output=answerer_output,
        )
        prediction_path = run_path / "items" / unit.unit_id / "questions" / f"{question.question_id}.json"
        write_json_atomic(prediction_path, prediction)
        timing_path = run_path / "items" / unit.unit_id / "timing" / f"{question.question_id}.json"
        write_json_atomic(
            timing_path,
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "benchmark": str(prediction.get("benchmark", "locomo")),
                "conversation_idx": question.conversation_idx,
                "question_id": question.question_id,
                "pipeline_timing": pipeline_timing,
            },
        )
        return prediction_path, timing_path

    def _write_conversation_aggregate(
        self,
        run_path: Path,
        unit: BenchmarkUnit,
        prediction_paths: list[Path],
    ) -> Path:
        predictions = [load_json_dict(path) for path in prediction_paths]
        aggregate_path = run_path / "items" / unit.unit_id / "predictions.json"
        write_json_atomic(
            aggregate_path,
            {
                "schema_version": PREDICTION_SCHEMA_VERSION,
                "benchmark": "locomo",
                "atomic_unit": self.benchmark.atomic_unit,
                "conversation_id": unit.unit_id,
                "question_ids": [prediction["question_id"] for prediction in predictions],
                "predictions": predictions,
            },
        )
        return aggregate_path

    def _write_committed_checkpoint(
        self,
        run_path: Path,
        manifest: RunManifest,
        unit: BenchmarkUnit,
        artifact_paths: tuple[Path, ...],
    ) -> None:
        artifacts = tuple(
            ArtifactRef(
                path=path.relative_to(run_path).as_posix(),
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
            )
            for path in artifact_paths
        )
        self._write_checkpoint(
            run_path,
            manifest,
            unit,
            UnitState.COMMITTED,
            artifacts=artifacts,
        )

    def _write_checkpoint(
        self,
        run_path: Path,
        manifest: RunManifest,
        unit: BenchmarkUnit,
        state: UnitState,
        *,
        artifacts: tuple[ArtifactRef, ...] = (),
    ) -> None:
        checkpoint = UnitCheckpoint(
            run_id=manifest.run_id,
            unit_id=unit.unit_id,
            status=state.value,
            scientific_fingerprint=manifest.scientific_fingerprint,
            artifacts=artifacts,
        )
        write_json_atomic(
            run_path / "checkpoints" / unit.unit_id / "checkpoint.json",
            checkpoint.to_dict(),
        )

    def _committed_units(self, run_path: Path, expected_unit_ids: tuple[str, ...]) -> tuple[str, ...]:
        resume_plan = plan_resume(run_path)
        committed_ids = set(resume_plan.committed_unit_ids)
        return tuple(unit_id for unit_id in expected_unit_ids if unit_id in committed_ids)

    def _write_status(
        self,
        run_path: Path,
        manifest: RunManifest,
        *,
        committed_units: tuple[str, ...],
        state: str = "RUNNING",
        current_activity: dict[str, Any] | None = None,
    ) -> None:
        terminal_ids = expected_terminal_item_ids(manifest)
        committed = _committed_terminal_item_count(run_path, manifest, committed_units)
        status = RunStatus(
            run_id=manifest.run_id,
            state=state,
            phase="RUNNING",
            expected=len(terminal_ids),
            committed=committed,
            expected_units=len(manifest.expected_item_ids),
            committed_units=len(committed_units),
            current_activity=current_activity,
        )
        write_json_atomic(run_path / "run-status.json", status.to_dict())
        if self.metrics is not None:
            self.metrics.record_run_status(
                benchmark=str(manifest.fingerprint_inputs.get("benchmark", "")),
                framework=str(manifest.fingerprint_inputs.get("framework", "")),
                phase="RUNNING",
                expected=len(terminal_ids),
                committed=committed,
                state=state,
                expected_units=len(manifest.expected_item_ids),
                committed_units=len(committed_units),
                current_activity=current_activity,
            )


def load_json_dict(path: Path) -> dict[str, Any]:
    from dmf_bench.atomic_io import read_json

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise StateError(f"Expected JSON object at {path}")
    return payload


def _persist_cleanup_manifest(
    run_path: Path,
    unit: BenchmarkUnit,
    prepared: dict[str, Any],
) -> Path | None:
    cleanup_manifest = prepared.get("cleanup_manifest")
    if cleanup_manifest is None:
        return None
    if not isinstance(cleanup_manifest, dict):
        raise StateError("Framework cleanup_manifest must be a JSON object.")
    path = run_path / "items" / unit.unit_id / "cleanup-manifest.json"
    write_json_atomic(path, cleanup_manifest)
    return path


def _apply_unit_retention(
    *,
    run_path: Path,
    unit: BenchmarkUnit,
    item: dict[str, Any],
    config: dict[str, Any],
    framework: Any,
    run_context: FrameworkRunContext,
) -> Path:
    qdrant = config.get("qdrant")
    policy = str(qdrant.get("retention", "keep")) if isinstance(qdrant, dict) else "keep"
    if policy not in {"keep", "delete-on-success"}:
        raise StateError(f"Unsupported Qdrant retention policy: {policy!r}.")

    outcome = "kept"
    if policy == "delete-on-success":
        framework.cleanup_unit(
            unit,
            item,
            config,
            run_context=run_context,
        )
        outcome = "deleted"

    path = run_path / "items" / unit.unit_id / "retention.json"
    write_json_atomic(
        path,
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "policy": policy,
            "outcome": outcome,
        },
    )
    return path


def _write_running_phase_checkpoint(
    run_path: Path,
    manifest: RunManifest,
    attempt: Attempt,
) -> LifecycleCheckpoint:
    unit_checkpoints: list[tuple[Path, UnitCheckpoint]] = []
    artifact_paths: list[Path] = []
    for unit_id in manifest.expected_item_ids:
        checkpoint_path = run_path / "checkpoints" / unit_id / "checkpoint.json"
        checkpoint = load_unit_checkpoint(checkpoint_path)
        if checkpoint.status != UnitState.COMMITTED.value:
            raise StateError(f"Cannot commit RUNNING phase with incomplete unit: {unit_id}")
        unit_checkpoints.append((checkpoint_path, checkpoint))
        artifact_paths.append(checkpoint_path)
        artifact_paths.extend(run_path / artifact.path for artifact in checkpoint.artifacts)

    input_fingerprint = hash_canonical_json(
        {
            "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
            "expected_unit_ids": list(manifest.expected_item_ids),
            "unit_checkpoint_digests": {
                checkpoint.unit_id: checkpoint.checkpoint_digest
                for _path, checkpoint in unit_checkpoints
            },
        }
    )
    checkpoint_path = run_path / "phase-checkpoints" / "RUNNING" / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            existing = load_lifecycle_checkpoint(checkpoint_path)
            if (
                existing.run_id == manifest.run_id
                and existing.phase == "RUNNING"
                and existing.status == "COMMITTED"
                and existing.scientific_fingerprint == manifest.scientific_fingerprint
                and existing.input_fingerprint == input_fingerprint
                and existing.expected_item_ids == manifest.expected_item_ids
                and existing.predecessor_digest is None
            ):
                validate_artifact_refs(run_path, existing.artifacts)
                return existing
        except (KeyError, TypeError, ValueError, StateError):
            pass

    checkpoint = LifecycleCheckpoint(
        run_id=manifest.run_id,
        attempt_id=attempt.attempt_id,
        phase="RUNNING",
        status="COMMITTED",
        scientific_fingerprint=manifest.scientific_fingerprint,
        input_fingerprint=input_fingerprint,
        expected_item_ids=manifest.expected_item_ids,
        artifacts=tuple(artifact_ref_for(run_path, path) for path in artifact_paths),
        predecessor_digest=None,
        metadata={
            "atomic_unit": manifest.atomic_unit,
            "unit_checkpoint_digests": {
                checkpoint.unit_id: checkpoint.checkpoint_digest
                for _path, checkpoint in unit_checkpoints
            },
        },
    )
    path = write_lifecycle_checkpoint(run_path, checkpoint)
    return load_lifecycle_checkpoint(path)


def _write_failed_running_checkpoint(
    run_path: Path,
    manifest: RunManifest,
    attempt: Attempt,
    *,
    error_type: str,
) -> None:
    checkpoint = LifecycleCheckpoint(
        run_id=manifest.run_id,
        attempt_id=attempt.attempt_id,
        phase="RUNNING",
        status="FAILED",
        scientific_fingerprint=manifest.scientific_fingerprint,
        input_fingerprint=hash_canonical_json(
            {
                "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                "expected_unit_ids": list(manifest.expected_item_ids),
            }
        ),
        expected_item_ids=manifest.expected_item_ids,
        predecessor_digest=None,
        metadata={"error_type": error_type},
    )
    write_lifecycle_checkpoint(run_path, checkpoint)


def _committed_terminal_item_count(
    run_path: Path,
    manifest: RunManifest,
    committed_units: tuple[str, ...],
) -> int:
    if manifest.fingerprint_inputs.get("benchmark") != "locomo":
        return len(committed_units)
    committed = 0
    for unit_id in committed_units:
        aggregate = load_json_dict(run_path / "items" / unit_id / "predictions.json")
        question_ids = aggregate.get("question_ids")
        if not isinstance(question_ids, list):
            raise StateError(f"LoCoMo aggregate missing question_ids: {unit_id}")
        committed += len(question_ids)
    return committed


def _persist_resolved_config(run_path: Path, config: dict[str, Any]) -> Path:
    from dmf_bench.config import redact_secrets

    path = run_path / "resolved-config.json"
    if not path.exists():
        write_json_atomic(path, redact_secrets(config))
    return path


def _check_cancel(cancel_check: Callable[[], None] | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _emit_unit_event(
    events: JsonEventLogger | None,
    event: str,
    manifest: RunManifest,
    attempt: Attempt,
    unit: BenchmarkUnit,
    unit_index: int,
) -> None:
    if events is None:
        return
    inputs = manifest.fingerprint_inputs
    events.event(
        event,
        "Prediction unit boundary reached.",
        run_id=manifest.run_id,
        attempt_id=attempt.attempt_id,
        benchmark=str(inputs.get("benchmark", "")),
        framework=str(inputs.get("framework", "")),
        phase="RUNNING",
        unit_index=unit_index,
        unit_id_hash=hashlib.sha256(unit.unit_id.encode("utf-8")).hexdigest()[:16],
        outcome="committed" if event.endswith("committed") else "started",
    )
