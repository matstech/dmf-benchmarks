"""Local benchmark runner primitives.

Phase 5 introduced LongMemEval prediction-only execution. Phase 6 adds LoCoMo
conversation-atomic prediction-only execution. These runners create
authoritative run/checkpoint artifacts, but do not claim full
judging/evaluation/report completion yet.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dmf_bench.adapters.base import AnswererAdapter, BenchmarkUnit
from dmf_bench.adapters.locomo import LoCoMoAdapter, LoCoMoQuestion
from dmf_bench.adapters.longmemeval import LongMemEvalAdapter
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.contracts import (
    ArtifactRef,
    RunManifest,
    RunStatus,
    UnitCheckpoint,
    scientific_fingerprint,
    sha256_file,
)
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.provenance import build_run_provenance
from dmf_bench.state import RunLock, StateError, UnitState, load_manifest, plan_resume


class InjectedInterrupt(RuntimeError):
    """Raised by tests to stop a run at a deterministic fault-injection point."""


class LongMemEvalQuestionFramework(Protocol):
    name: str

    def cleanup_unit(self, unit: BenchmarkUnit, question: dict[str, Any], config: dict[str, Any]) -> None:
        """Delete framework resources for a non-committed unit before restart."""

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Ingest one question's memory resources and return barrier metadata."""

    def retrieve(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        """Return protocol-specific retrieval/native context payload."""


class LoCoMoConversationFramework(Protocol):
    name: str

    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Delete framework resources for a non-committed conversation before restart."""

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Ingest one conversation once and return barrier metadata."""

    def retrieve_question(
        self,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        question: LoCoMoQuestion,
        config: dict[str, Any],
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        """Return protocol-specific retrieval/native context for one question."""


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
    ) -> None:
        self.benchmark = benchmark or LongMemEvalAdapter()
        self.artifact_store = artifact_store
        self.framework = framework
        self.answerer = answerer
        self.metrics = metrics

    def run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
        interrupt_at: str | None = None,
    ) -> PredictOnlyResult:
        if config.get("benchmark") != "longmemeval":
            raise ValueError("LongMemEvalPredictOnlyRunner only supports benchmark='longmemeval'.")
        protocol = str(config.get("protocol", ""))
        if protocol not in self.benchmark.supported_protocols:
            raise ValueError(f"Unsupported LongMemEval protocol: {protocol!r}.")
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

        lock_path = self.artifact_store.runs_dir / ".locks" / f"{manifest.run_id}.lock"
        with RunLock(lock_path):
            if run_path.exists():
                if not resume:
                    raise StateError(
                        f"Run directory already exists: {run_path}. Use resume=True to continue."
                    )
                self._validate_resume_manifest(run_path, manifest)
                resume_plan = plan_resume(run_path)
                restart_ids = set(resume_plan.restart_unit_ids)
            else:
                if resume:
                    raise StateError(f"Cannot resume missing run directory: {run_path}")
                self.artifact_store.create_run(manifest)
                restart_ids = set(expected_unit_ids)

            committed = self._committed_units(run_path, expected_unit_ids)
            self._write_status(
                run_path,
                manifest,
                expected=len(expected_unit_ids),
                committed=len(committed),
            )
            restarted: list[str] = []
            for unit in units:
                if unit.unit_id not in restart_ids:
                    continue
                question = questions_by_id[unit.unit_id]
                self._restart_unit(run_path, unit, question, config, manifest)
                restarted.append(unit.unit_id)
                if interrupt_at == "before-prediction-commit":
                    raise InjectedInterrupt("Interrupted before prediction commit.")

                prediction_path = self._write_prediction(
                    run_path,
                    unit,
                    question,
                    config,
                    manifest,
                    protocol,
                )
                self._write_committed_checkpoint(run_path, manifest, unit, prediction_path)
                committed = self._committed_units(run_path, expected_unit_ids)
                self._write_status(
                    run_path,
                    manifest,
                    expected=len(expected_unit_ids),
                    committed=len(committed),
                )
                if interrupt_at == "after-prediction-commit":
                    raise InjectedInterrupt("Interrupted after prediction commit.")

            committed = self._committed_units(run_path, expected_unit_ids)
            self._write_status(run_path, manifest, expected=len(expected_unit_ids), committed=len(committed))
            if self.metrics is not None:
                self.metrics.write_snapshot(run_path)
            return PredictOnlyResult(
                run_id=manifest.run_id,
                state="PARTIAL",
                expected_unit_ids=expected_unit_ids,
                committed_unit_ids=committed,
                restarted_unit_ids=tuple(restarted),
            )

    def _build_manifest(
        self,
        config: dict[str, Any],
        expected_unit_ids: tuple[str, ...],
        *,
        run_id: str | None,
    ) -> RunManifest:
        fingerprint_inputs = {
            "schema_version": 1,
            "benchmark": config.get("benchmark"),
            "framework": config.get("framework"),
            "protocol": config.get("protocol"),
            "dataset": _fingerprint_dataset(config),
            "selection": config.get("selection", {}),
            "models": {"answerer": (config.get("models") or {}).get("answerer", {})},
        }
        fingerprint = scientific_fingerprint(fingerprint_inputs)
        resolved_run_id = str(
            run_id
            or config.get("run_id")
            or config.get("experiment_id")
            or f"longmemeval-{config.get('framework')}-{config.get('protocol')}-{fingerprint[:12]}"
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
    ) -> None:
        self.framework.cleanup_unit(unit, question, config)
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
        protocol: str,
    ) -> Path:
        prepared = self.framework.prepare_unit(unit, question, config)
        self._write_checkpoint(run_path, manifest, unit, UnitState.PREDICTING)
        retrieval = self.framework.retrieve(unit, question, config, prepared)
        answerer_input = self.benchmark.build_answerer_input(
            question=question,
            protocol=protocol,
            framework_name=self.framework.name,
            retrieval=retrieval,
        )
        answerer_output = self.answerer.generate(
            answerer_input.user_prompt,
            {
                **answerer_input.metadata,
                "system_prompt": answerer_input.system_prompt,
                "unit_id": unit.unit_id,
            },
        )
        prediction = self.benchmark.build_prediction(
            question=question,
            protocol=protocol,
            framework_name=self.framework.name,
            retrieval=retrieval,
            answerer_input=answerer_input,
            answerer_output=answerer_output,
        )
        prediction_path = run_path / "items" / unit.unit_id / "prediction.json"
        write_json_atomic(prediction_path, prediction)
        return prediction_path

    def _write_committed_checkpoint(
        self,
        run_path: Path,
        manifest: RunManifest,
        unit: BenchmarkUnit,
        prediction_path: Path,
    ) -> None:
        artifact = ArtifactRef(
            path=prediction_path.relative_to(run_path).as_posix(),
            sha256=sha256_file(prediction_path),
            bytes=prediction_path.stat().st_size,
        )
        self._write_checkpoint(
            run_path,
            manifest,
            unit,
            UnitState.COMMITTED,
            artifacts=(artifact,),
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
        expected: int,
        committed: int,
    ) -> None:
        status = RunStatus(
            run_id=manifest.run_id,
            state="PARTIAL",
            phase="RUNNING",
            expected=expected,
            committed=committed,
        )
        write_json_atomic(run_path / "run-status.json", status.to_dict())
        if self.metrics is not None:
            self.metrics.record_run_status(
                benchmark=str(manifest.fingerprint_inputs.get("benchmark", "")),
                framework=str(manifest.fingerprint_inputs.get("framework", "")),
                protocol=str(manifest.fingerprint_inputs.get("protocol", "")),
                phase="RUNNING",
                expected=expected,
                committed=committed,
            )


def _fingerprint_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config.get("dataset") or {}
    if not isinstance(dataset, dict):
        return {}
    return {
        "name": dataset.get("name"),
        "path": dataset.get("path"),
        "sha256": dataset.get("sha256"),
        "source": dataset.get("source"),
        "revision": dataset.get("revision"),
    }


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
    ) -> None:
        self.benchmark = benchmark or LoCoMoAdapter()
        self.artifact_store = artifact_store
        self.framework = framework
        self.answerer = answerer
        self.metrics = metrics

    def run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
        interrupt_at: str | None = None,
    ) -> PredictOnlyResult:
        if config.get("benchmark") != "locomo":
            raise ValueError("LoCoMoPredictOnlyRunner only supports benchmark='locomo'.")
        protocol = str(config.get("protocol", ""))
        if protocol not in self.benchmark.supported_protocols:
            raise ValueError(f"Unsupported LoCoMo protocol: {protocol!r}.")
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

        lock_path = self.artifact_store.runs_dir / ".locks" / f"{manifest.run_id}.lock"
        with RunLock(lock_path):
            if run_path.exists():
                if not resume:
                    raise StateError(
                        f"Run directory already exists: {run_path}. Use resume=True to continue."
                    )
                self._validate_resume_manifest(run_path, manifest)
                restart_ids = set(plan_resume(run_path).restart_unit_ids)
            else:
                if resume:
                    raise StateError(f"Cannot resume missing run directory: {run_path}")
                self.artifact_store.create_run(manifest)
                restart_ids = set(expected_unit_ids)

            committed = self._committed_units(run_path, expected_unit_ids)
            self._write_status(
                run_path,
                manifest,
                expected=len(expected_unit_ids),
                committed=len(committed),
            )
            restarted: list[str] = []
            for unit in units:
                if unit.unit_id not in restart_ids:
                    continue
                _conversation_idx, conversation, questions = conversations_by_id[unit.unit_id]
                self._restart_unit(run_path, unit, conversation, config, manifest)
                restarted.append(unit.unit_id)
                prepared = self.framework.prepare_unit(unit, conversation, config)
                self._write_checkpoint(run_path, manifest, unit, UnitState.PREDICTING)
                if interrupt_at == "after-ingestion-before-first-prediction":
                    raise InjectedInterrupt("Interrupted after ingestion before first prediction.")

                prediction_paths: list[Path] = []
                for question_offset, question in enumerate(questions):
                    prediction_paths.append(
                        self._write_question_prediction(
                            run_path,
                            unit,
                            conversation,
                            question,
                            config,
                            protocol,
                            prepared,
                        )
                    )
                    if interrupt_at == "after-first-prediction" and question_offset == 0:
                        raise InjectedInterrupt("Interrupted after first LoCoMo prediction.")

                aggregate_path = self._write_conversation_aggregate(run_path, unit, prediction_paths)
                self._write_committed_checkpoint(
                    run_path,
                    manifest,
                    unit,
                    (*prediction_paths, aggregate_path),
                )
                committed = self._committed_units(run_path, expected_unit_ids)
                self._write_status(
                    run_path,
                    manifest,
                    expected=len(expected_unit_ids),
                    committed=len(committed),
                )

            committed = self._committed_units(run_path, expected_unit_ids)
            self._write_status(run_path, manifest, expected=len(expected_unit_ids), committed=len(committed))
            if self.metrics is not None:
                self.metrics.write_snapshot(run_path)
            return PredictOnlyResult(
                run_id=manifest.run_id,
                state="PARTIAL",
                expected_unit_ids=expected_unit_ids,
                committed_unit_ids=committed,
                restarted_unit_ids=tuple(restarted),
            )

    def _build_manifest(
        self,
        config: dict[str, Any],
        expected_unit_ids: tuple[str, ...],
        *,
        run_id: str | None,
    ) -> RunManifest:
        fingerprint_inputs = {
            "schema_version": 1,
            "benchmark": config.get("benchmark"),
            "framework": config.get("framework"),
            "protocol": config.get("protocol"),
            "dataset": _fingerprint_dataset(config),
            "selection": config.get("selection", {}),
            "models": {"answerer": (config.get("models") or {}).get("answerer", {})},
            "expected_question_ids": self.benchmark.expected_question_ids(config),
        }
        fingerprint = scientific_fingerprint(fingerprint_inputs)
        resolved_run_id = str(
            run_id
            or config.get("run_id")
            or config.get("experiment_id")
            or f"locomo-{config.get('framework')}-{config.get('protocol')}-{fingerprint[:12]}"
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
    ) -> None:
        self.framework.cleanup_unit(unit, conversation, config)
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
        protocol: str,
        prepared: dict[str, Any],
    ) -> Path:
        retrieval = self.framework.retrieve_question(
            unit,
            conversation,
            question,
            config,
            prepared,
        )
        answerer_input = self.benchmark.build_answerer_input(
            conversation=conversation,
            question=question,
            protocol=protocol,
            framework_name=self.framework.name,
            retrieval=retrieval,
        )
        answerer_output = self.answerer.generate(
            answerer_input.user_prompt,
            {
                **answerer_input.metadata,
                "system_prompt": answerer_input.system_prompt,
                "unit_id": unit.unit_id,
            },
        )
        prediction = self.benchmark.build_prediction(
            conversation=conversation,
            question=question,
            protocol=protocol,
            framework_name=self.framework.name,
            retrieval=retrieval,
            answerer_input=answerer_input,
            answerer_output=answerer_output,
        )
        prediction_path = run_path / "items" / unit.unit_id / "questions" / f"{question.question_id}.json"
        write_json_atomic(prediction_path, prediction)
        return prediction_path

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
                "schema_version": 1,
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
        expected: int,
        committed: int,
    ) -> None:
        status = RunStatus(
            run_id=manifest.run_id,
            state="PARTIAL",
            phase="RUNNING",
            expected=expected,
            committed=committed,
        )
        write_json_atomic(run_path / "run-status.json", status.to_dict())
        if self.metrics is not None:
            self.metrics.record_run_status(
                benchmark=str(manifest.fingerprint_inputs.get("benchmark", "")),
                framework=str(manifest.fingerprint_inputs.get("framework", "")),
                protocol=str(manifest.fingerprint_inputs.get("protocol", "")),
                phase="RUNNING",
                expected=expected,
                committed=committed,
            )


def load_json_dict(path: Path) -> dict[str, Any]:
    from dmf_bench.atomic_io import read_json

    payload = read_json(path)
    if not isinstance(payload, dict):
        raise StateError(f"Expected JSON object at {path}")
    return payload
