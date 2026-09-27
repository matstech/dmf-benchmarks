"""Offline judge/evaluate/report/publish lifecycle finalization."""

from __future__ import annotations

import os
import re
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from dmf_bench.frameworks.mem0_runtime import normalize_memory_internal_usage
from dmf_bench.reporting.reports import build_timing_report, normalize_answerer_usage
from dmf_bench.reporting.resources import ResourceUsageTracker
from dmf_bench.adapters.base import JudgeAdapter, JudgeRequest
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import fsync_directory, read_json, write_json_atomic
from dmf_bench.contracts import (
    Attempt,
    EVALUATION_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    LifecycleCheckpoint,
    PREDICTION_SCHEMA_VERSION,
    V3_PREDICTION_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    RunManifest,
    RunStatus,
    SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
    hash_canonical_json,
    sha256_file,
)
from dmf_bench.evaluation.registry import (
    EvaluationRequirement,
    evaluation_plan_for,
    evaluation_plan_v3,
)
from dmf_bench.reporting.analysis import analysis_rows_jsonl, build_analysis_rows
from dmf_bench.execution import RunInterrupted
from dmf_bench.logging_config import JsonEventLogger
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.state import (
    RunLock,
    StateError,
    artifact_ref_for,
    expected_terminal_item_ids,
    finish_attempt,
    latest_attempt_id,
    lifecycle_checkpoint_path,
    load_lifecycle_checkpoint,
    load_manifest,
    new_attempt,
    plan_resume,
    validate_artifact_refs,
    write_attempt,
    write_lifecycle_checkpoint,
)

from dmf_bench.benchmarks.locomo import evaluation as locomo_evaluation
from dmf_bench.benchmarks.longmemeval import evaluation as longmemeval_evaluation


PredictionLoader = Callable[[Path, RunManifest], list[dict[str, Any]]]
Evaluator = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class FinalizationResult:
    run_id: str
    state: str
    judged_count: int
    evaluated_count: int
    excluded_count: int
    failed_count: int
    final_completion_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state,
            "judged_count": self.judged_count,
            "evaluated_count": self.evaluated_count,
            "excluded_count": self.excluded_count,
            "failed_count": self.failed_count,
            "final_completion_path": str(self.final_completion_path),
        }


class OfflineLifecycleFinalizer:
    """Complete terminal phases with digest-protected, resumable checkpoints."""

    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore,
        judge: JudgeAdapter | None = None,
        judges: Mapping[str, JudgeAdapter] | None = None,
        primary_judge_id: str | None = None,
        metrics: BenchmarkMetrics | None = None,
        events: JsonEventLogger | None = None,
        evaluation_plans: dict[tuple[str, str], tuple[EvaluationRequirement, ...]] | None = None,
        prediction_loaders: dict[str, PredictionLoader] | None = None,
        evaluators: dict[str, Evaluator] | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        if judges is not None:
            if not judges or not primary_judge_id or primary_judge_id not in judges:
                raise ValueError("judges must contain the primary_judge_id.")
            self.judges = dict(judges)
            self.primary_judge_id = primary_judge_id
            self.judge = self.judges[primary_judge_id]
        else:
            if judge is None:
                raise ValueError("A judge or ordered judges map is required.")
            self.judges = None
            self.primary_judge_id = None
            self.judge = judge
        self.metrics = metrics
        self.evaluation_plans = evaluation_plans
        self.prediction_loaders = dict(prediction_loaders or {})
        self.evaluators = dict(evaluators or {})
        self.events = events

    def finalize(
        self,
        run_id: str,
        *,
        interrupt_at: str | None = None,
        cancel_check: Callable[[], None] | None = None,
        _lock_held: bool = False,
        _attempt: Attempt | None = None,
        _resource_tracker: ResourceUsageTracker | None = None,
    ) -> FinalizationResult:
        resource_tracker = _resource_tracker or ResourceUsageTracker.start()
        self.artifact_store.run_dir(run_id)
        lock_path = self.artifact_store.runs_dir / ".locks" / f"{run_id}.lock"
        lock = nullcontext() if _lock_held else RunLock(lock_path)
        with lock:
            return self._finalize_locked(
                run_id,
                interrupt_at=interrupt_at,
                cancel_check=cancel_check,
                attempt=_attempt,
                resource_tracker=resource_tracker,
            )

    def _finalize_locked(
        self,
        run_id: str,
        *,
        interrupt_at: str | None,
        cancel_check: Callable[[], None] | None,
        attempt: Attempt | None,
        resource_tracker: ResourceUsageTracker,
    ) -> FinalizationResult:
        run_dir = self.artifact_store.run_dir(run_id)
        manifest = load_manifest(run_dir)
        resume_plan = plan_resume(run_dir)
        if resume_plan.next_phase == "COMPLETED":
            return self._completed_result(run_dir, manifest)
        if resume_plan.next_phase == "RUNNING" or resume_plan.restart_unit_ids:
            raise StateError(
                "Evaluation refused: expected prediction set is incomplete; "
                f"restart units: {list(resume_plan.restart_unit_ids)}"
            )

        terminal_ids = expected_terminal_item_ids(manifest)
        predictions = self._load_predictions(run_dir, manifest)
        self._validate_prediction_set(predictions, terminal_ids, manifest)
        metadata = self._metadata_from_manifest(manifest)
        active_attempt = attempt or new_attempt(
            run_id=run_id,
            entry_phase=resume_plan.next_phase,
            resume=latest_attempt_id(run_dir) is not None,
            resumed_from_attempt_id=latest_attempt_id(run_dir),
        )
        write_attempt(run_dir, active_attempt)
        _check_cancel(cancel_check)

        phase = "JUDGING"
        phase_input = hash_canonical_json(
            {
                "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                "phase": phase,
                "run_id": manifest.run_id,
            }
        )
        phase_predecessor: str | None = None
        committed_items = 0
        publish_started_at: float | None = None
        try:
            judge_input = self._judge_input_fingerprint(manifest, predictions)
            phase_input = judge_input
            judging_checkpoint = self._load_valid_phase_checkpoint(
                run_dir,
                manifest,
                phase="JUDGING",
                input_fingerprint=judge_input,
                expected_item_ids=terminal_ids,
                predecessor_digest=None,
            )
            if judging_checkpoint is None:
                self._write_status(
                    run_dir,
                    manifest,
                    state="JUDGING",
                    phase="JUDGING",
                    committed=0,
                )

                def record_judging_progress(completed: int) -> None:
                    nonlocal committed_items
                    committed_items = completed
                    self._write_status(
                        run_dir,
                        manifest,
                        state="JUDGING",
                        phase="JUDGING",
                        committed=completed,
                    )

                evaluations, judgment_paths, item_digests = self._judge_predictions(
                    run_dir,
                    manifest,
                    predictions,
                    metadata,
                    attempt=active_attempt,
                    interrupt_at=interrupt_at,
                    cancel_check=cancel_check,
                    progress=record_judging_progress,
                )
                evaluations_path = run_dir / "evaluations" / "evaluations.json"
                write_json_atomic(evaluations_path, evaluations)
                judging_checkpoint = self._commit_phase_checkpoint(
                    run_dir,
                    manifest,
                    attempt=active_attempt,
                    phase="JUDGING",
                    input_fingerprint=judge_input,
                    expected_item_ids=terminal_ids,
                    artifacts=(
                        evaluations_path,
                        *judgment_paths,
                        *self._judging_item_checkpoint_paths(run_dir, manifest, terminal_ids),
                    ),
                    predecessor_digest=None,
                    metadata={
                        "judge_requested_model": self._judge_identity(manifest)[1],
                        "judge_fingerprint": self._judge_identity(manifest)[2],
                        "item_checkpoint_digests": item_digests,
                    },
                )
                if interrupt_at == "after-judge":
                    raise InjectedTerminalInterrupt("Interrupted after judge phase.")
            else:
                evaluations = _load_json_list(run_dir / "evaluations" / "evaluations.json")
                self._validate_evaluation_set(evaluations, terminal_ids, manifest)
            committed_items = len(evaluations)

            phase = "EVALUATING"
            phase_predecessor = judging_checkpoint.checkpoint_digest
            evaluator_identities = self._evaluation_identities(metadata)
            evaluation_input = hash_canonical_json(
                {
                    "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                    "judging_checkpoint": judging_checkpoint.checkpoint_digest,
                    "evaluations_sha256": sha256_file(
                        run_dir / "evaluations" / "evaluations.json"
                    ),
                    "evaluation": manifest.fingerprint_inputs.get("evaluation", {}),
                    "evaluators": evaluator_identities,
                }
            )
            phase_input = evaluation_input
            evaluation_checkpoint = self._load_valid_phase_checkpoint(
                run_dir,
                manifest,
                phase="EVALUATING",
                input_fingerprint=evaluation_input,
                expected_item_ids=terminal_ids,
                predecessor_digest=judging_checkpoint.checkpoint_digest,
            )
            if evaluation_checkpoint is None:
                self._write_status(
                    run_dir,
                    manifest,
                    state="EVALUATING",
                    phase="EVALUATING",
                    committed=committed_items,
                )
                reports, evaluator_paths, evaluator_digests = self._evaluate(
                    run_dir,
                    manifest,
                    evaluations,
                    metadata,
                    attempt=active_attempt,
                    interrupt_at=interrupt_at,
                    cancel_check=cancel_check,
                )
                summary_path = run_dir / "evaluations" / "evaluation-summary.json"
                write_json_atomic(summary_path, reports)
                evaluation_checkpoint = self._commit_phase_checkpoint(
                    run_dir,
                    manifest,
                    attempt=active_attempt,
                    phase="EVALUATING",
                    input_fingerprint=evaluation_input,
                    expected_item_ids=terminal_ids,
                    artifacts=(
                        summary_path,
                        *evaluator_paths,
                        *(
                            lifecycle_checkpoint_path(
                                run_dir,
                                "EVALUATING",
                                item_id=identity["name"],
                            )
                            for identity in evaluator_identities
                        ),
                    ),
                    predecessor_digest=judging_checkpoint.checkpoint_digest,
                    metadata={
                        "evaluator_versions": _evaluator_versions(reports),
                        "item_checkpoint_digests": evaluator_digests,
                    },
                )
                if interrupt_at == "after-evaluate":
                    raise InjectedTerminalInterrupt("Interrupted after evaluation phase.")
            else:
                reports = _load_json_dict(
                    run_dir / "evaluations" / "evaluation-summary.json"
                )
                self._validate_evaluation_summary(reports, manifest)

            phase = "REPORTING"
            phase_predecessor = evaluation_checkpoint.checkpoint_digest
            report_input = hash_canonical_json(
                {
                    "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                    "evaluation_checkpoint": evaluation_checkpoint.checkpoint_digest,
                    "evaluation_summary_sha256": sha256_file(
                        run_dir / "evaluations" / "evaluation-summary.json"
                    ),
                }
            )
            phase_input = report_input
            reporting_checkpoint = self._load_valid_phase_checkpoint(
                run_dir,
                manifest,
                phase="REPORTING",
                input_fingerprint=report_input,
                expected_item_ids=terminal_ids,
                predecessor_digest=evaluation_checkpoint.checkpoint_digest,
            )
            if reporting_checkpoint is None:
                self._write_status(
                    run_dir,
                    manifest,
                    state="REPORTING",
                    phase="REPORTING",
                    committed=committed_items,
                )
                _check_cancel(cancel_check)
                report_paths = self._write_reports(
                    run_dir,
                    manifest,
                    metadata,
                    evaluations,
                    reports,
                    attempt=active_attempt,
                    resource_tracker=resource_tracker,
                )
                _check_cancel(cancel_check)
                _emit_phase_event(
                    self.events,
                    "report.written",
                    manifest,
                    active_attempt,
                    "REPORTING",
                )
                if self.metrics is not None:
                    self.metrics.write_snapshot(
                        run_dir,
                        operational_summary={
                            "usage": _load_json_dict(run_dir / "reports" / "usage.json"),
                            "timing": _load_json_dict(run_dir / "reports" / "timing.json"),
                            "resources": _load_json_dict(
                                run_dir / "reports" / "resources.json"
                            ),
                        },
                    )
                reporting_checkpoint = self._commit_phase_checkpoint(
                    run_dir,
                    manifest,
                    attempt=active_attempt,
                    phase="REPORTING",
                    input_fingerprint=report_input,
                    expected_item_ids=terminal_ids,
                    artifacts=report_paths,
                    predecessor_digest=evaluation_checkpoint.checkpoint_digest,
                    metadata={"report_schema_version": 3 if self._is_v3(manifest) else REPORT_SCHEMA_VERSION},
                )
                if interrupt_at == "after-report":
                    raise InjectedTerminalInterrupt("Interrupted after report phase.")

            phase = "PUBLISHING"
            phase_predecessor = reporting_checkpoint.checkpoint_digest
            publishing_input = hash_canonical_json(
                {
                    "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                    "reporting_checkpoint": reporting_checkpoint.checkpoint_digest,
                    "backend": "local-only",
                }
            )
            phase_input = publishing_input
            publishing_checkpoint = self._load_valid_phase_checkpoint(
                run_dir,
                manifest,
                phase="PUBLISHING",
                input_fingerprint=publishing_input,
                expected_item_ids=terminal_ids,
                predecessor_digest=reporting_checkpoint.checkpoint_digest,
            )
            publish_started_at = time.perf_counter()
            if publishing_checkpoint is None:
                self._write_status(
                    run_dir,
                    manifest,
                    state="PUBLISHING",
                    phase="PUBLISHING",
                    committed=committed_items,
                )
                _check_cancel(cancel_check)
                _emit_phase_event(
                    self.events,
                    "artifact.publish.started",
                    manifest,
                    active_attempt,
                    "PUBLISHING",
                )
                staged = self.artifact_store.stage(
                    run_id,
                    run_dir,
                    publication_id=reporting_checkpoint.checkpoint_digest,
                )
                _check_cancel(cancel_check)
                if interrupt_at == "after-publish-stage":
                    raise InjectedTerminalInterrupt("Interrupted after publish stage.")
                publishing_checkpoint = self._commit_phase_checkpoint(
                    run_dir,
                    manifest,
                    attempt=active_attempt,
                    phase="PUBLISHING",
                    input_fingerprint=publishing_input,
                    expected_item_ids=terminal_ids,
                    artifacts=(staged.manifest_path,),
                    predecessor_digest=reporting_checkpoint.checkpoint_digest,
                    metadata={
                        "staging_dir": staged.staging_dir.relative_to(run_dir).as_posix(),
                        "manifest_sha256": sha256_file(staged.manifest_path),
                    },
                )
            else:
                staging_relative = str(publishing_checkpoint.metadata.get("staging_dir", ""))
                staged = self.artifact_store.load_staged(run_id, run_dir / staging_relative)

            phase = "VERIFYING"
            phase_predecessor = publishing_checkpoint.checkpoint_digest
            verifying_input = hash_canonical_json(
                {
                    "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                    "publishing_checkpoint": publishing_checkpoint.checkpoint_digest,
                    "staged_manifest_sha256": sha256_file(staged.manifest_path),
                }
            )
            phase_input = verifying_input
            verifying_checkpoint = self._load_valid_phase_checkpoint(
                run_dir,
                manifest,
                phase="VERIFYING",
                input_fingerprint=verifying_input,
                expected_item_ids=terminal_ids,
                predecessor_digest=publishing_checkpoint.checkpoint_digest,
            )
            self._write_status(
                run_dir,
                manifest,
                state="VERIFYING",
                phase="VERIFYING",
                committed=committed_items,
            )
            _check_cancel(cancel_check)
            receipt = self.artifact_store.verify(staged)
            _check_cancel(cancel_check)
            if interrupt_at == "after-verify" and verifying_checkpoint is None:
                raise InjectedTerminalInterrupt("Interrupted after publication verification.")
            if verifying_checkpoint is None:
                verifying_checkpoint = self._commit_phase_checkpoint(
                    run_dir,
                    manifest,
                    attempt=active_attempt,
                    phase="VERIFYING",
                    input_fingerprint=verifying_input,
                    expected_item_ids=terminal_ids,
                    artifacts=(staged.manifest_path, receipt.receipt_path),
                    predecessor_digest=publishing_checkpoint.checkpoint_digest,
                    metadata={
                        "manifest_sha256": receipt.manifest_sha256,
                        "artifact_count": receipt.artifact_count,
                    },
                )
            _check_cancel(cancel_check)
            marker = self.artifact_store.commit(staged, receipt)
            _emit_phase_event(
                self.events,
                "artifact.publish.completed",
                manifest,
                active_attempt,
                "VERIFYING",
            )
            if interrupt_at == "after-publish-commit":
                raise InjectedTerminalInterrupt("Interrupted after publication commit.")
            self._write_status(
                run_dir,
                manifest,
                state="COMPLETED",
                phase="VERIFYING",
                committed=committed_items,
            )
            if self.metrics is not None:
                self.metrics.record_artifact_publish(
                    backend="local-only",
                    outcome="completed",
                    seconds=time.perf_counter() - publish_started_at,
                )
            finish_attempt(run_dir, active_attempt, status="COMPLETED")
        except RunInterrupted:
            self._write_status(
                run_dir,
                manifest,
                state="INTERRUPTING",
                phase=phase,
                committed=committed_items,
            )
            self._write_status(
                run_dir,
                manifest,
                state="INTERRUPTED",
                phase=phase,
                committed=committed_items,
            )
            finish_attempt(run_dir, active_attempt, status="INTERRUPTED")
            raise
        except InjectedTerminalInterrupt:
            finish_attempt(run_dir, active_attempt, status="INTERRUPTED")
            raise
        except Exception as exc:
            failed_state = {
                "JUDGING": "FAILED_JUDGING",
                "EVALUATING": "FAILED_EVALUATION",
                "REPORTING": "FAILED_REPORTING",
                "PUBLISHING": "FAILED_PUBLISHING",
                "VERIFYING": "FAILED_VERIFYING",
            }[phase]
            self._write_status(
                run_dir,
                manifest,
                state=failed_state,
                phase=phase,
                committed=committed_items,
            )
            self._write_failed_phase_checkpoint(
                run_dir,
                manifest,
                attempt=active_attempt,
                phase=phase,
                input_fingerprint=phase_input,
                expected_item_ids=terminal_ids,
                predecessor_digest=phase_predecessor,
                error_type=type(exc).__name__,
            )
            if self.metrics is not None and phase in {"PUBLISHING", "VERIFYING"}:
                self.metrics.record_artifact_publish(
                    backend="local-only",
                    outcome="failed",
                    seconds=(time.perf_counter() - publish_started_at) if publish_started_at else 0.0,
                )
            if phase in {"PUBLISHING", "VERIFYING"}:
                _emit_phase_event(
                    self.events,
                    "artifact.publish.failed",
                    manifest,
                    active_attempt,
                    phase,
                )
            finish_attempt(run_dir, active_attempt, status="FAILED")
            if isinstance(exc, StateError):
                raise
            raise StateError(f"Finalization failed during {phase}: {exc}") from exc

        return FinalizationResult(
            run_id=run_id,
            state=marker.state,
            judged_count=len(evaluations),
            evaluated_count=len(evaluations),
            excluded_count=0,
            failed_count=0,
            final_completion_path=run_dir / "final" / "COMPLETED.json",
        )

    def _completed_result(self, run_dir: Path, manifest: RunManifest) -> FinalizationResult:
        self.artifact_store.verify_committed(manifest.run_id)
        evaluations = _load_json_list(
            run_dir / "final" / "evaluations" / "evaluations.json",
            default=[],
        )
        self._write_status(
            run_dir,
            manifest,
            state="COMPLETED",
            phase="VERIFYING",
            committed=len(evaluations),
        )
        return FinalizationResult(
            run_id=manifest.run_id,
            state="COMPLETED",
            judged_count=sum(1 for item in evaluations if item.get("judgment")),
            evaluated_count=len(evaluations),
            excluded_count=0,
            failed_count=0,
            final_completion_path=run_dir / "final" / "COMPLETED.json",
        )

    def _metadata_from_manifest(self, manifest: RunManifest) -> dict[str, Any]:
        inputs = manifest.fingerprint_inputs
        return {
            "schema_version": 3 if inputs.get("schema_version") == 3 else REPORT_SCHEMA_VERSION,
            "experiment_schema_version": inputs.get("schema_version", 2),
            "judge_ids": [str(item.get("id", "")) for item in inputs.get("judges", [])]
            if isinstance(inputs.get("judges"), list) else [],
            "run_id": manifest.run_id,
            "benchmark": str(inputs.get("benchmark", "")),
            "framework": str(inputs.get("framework", "")),
            "scientific_fingerprint": manifest.scientific_fingerprint,
        }

    def _load_predictions(self, run_dir: Path, manifest: RunManifest) -> list[dict[str, Any]]:
        benchmark = str(manifest.fingerprint_inputs.get("benchmark", ""))
        if benchmark in self.prediction_loaders:
            return self.prediction_loaders[benchmark](run_dir, manifest)
        predictions: list[dict[str, Any]] = []
        if self._is_v3(manifest):
            for unit_id in manifest.expected_item_ids:
                aggregate = _load_json_dict(run_dir / "items" / unit_id / "predictions.json")
                rows = aggregate.get("predictions")
                if not isinstance(rows, list):
                    raise StateError(f"V3 aggregate missing predictions list: {unit_id}")
                predictions.extend(_ensure_dict_list(rows))
            return predictions
        if benchmark == "longmemeval":
            for unit_id in manifest.expected_item_ids:
                predictions.append(_load_json_dict(run_dir / "items" / unit_id / "prediction.json"))
            return predictions
        if benchmark == "locomo":
            for unit_id in manifest.expected_item_ids:
                aggregate = _load_json_dict(run_dir / "items" / unit_id / "predictions.json")
                raw_predictions = aggregate.get("predictions")
                if not isinstance(raw_predictions, list):
                    raise StateError(f"LoCoMo aggregate missing predictions list: {unit_id}")
                predictions.extend(_ensure_dict_list(raw_predictions))
            return predictions
        raise StateError(f"Unsupported benchmark for prediction loading: {benchmark!r}")

    def _validate_prediction_set(
        self,
        predictions: list[dict[str, Any]],
        expected_item_ids: tuple[str, ...],
        manifest: RunManifest,
    ) -> None:
        expected_schema = (
            V3_PREDICTION_SCHEMA_VERSION
            if self._is_v3(manifest) else PREDICTION_SCHEMA_VERSION
        )
        if any(prediction.get("schema_version") != expected_schema for prediction in predictions):
            raise StateError(
                "Evaluation refused: prediction schema_version must be "
                f"{expected_schema}; v1 state is not supported."
            )
        observed = tuple(str(item.get("question_id", "")) for item in predictions)
        if not all(observed) or observed != expected_item_ids or len(set(observed)) != len(observed):
            raise StateError(
                "Evaluation refused: prediction question set/order does not match manifest."
            )

    @staticmethod
    def _is_v3(manifest: RunManifest) -> bool:
        return manifest.fingerprint_inputs.get("schema_version") == 3

    @staticmethod
    def _evaluation_schema(manifest: RunManifest) -> int:
        return 3 if manifest.fingerprint_inputs.get("schema_version") == 3 else EVALUATION_SCHEMA_VERSION

    def _v3_judge_definitions(self, manifest: RunManifest) -> list[tuple[str, JudgeAdapter, dict[str, Any]]]:
        definitions = manifest.fingerprint_inputs.get("judges")
        if not isinstance(definitions, list) or not definitions:
            raise StateError("V3 manifest is missing ordered judge identities.")
        primary_id = manifest.fingerprint_inputs.get("evaluation", {}).get("primary_judge_id")
        if not isinstance(primary_id, str) or not primary_id:
            raise StateError("V3 manifest is missing primary judge ID.")
        if self.judges is None or self.primary_judge_id != primary_id:
            raise StateError("V3 finalizer judges differ from the manifest primary judge.")
        ids = [str(item.get("id", "")) for item in definitions if isinstance(item, dict)]
        if ids != list(self.judges) or len(ids) != len(definitions):
            raise StateError("V3 finalizer judge order differs from the manifest.")
        result: list[tuple[str, JudgeAdapter, dict[str, Any]]] = []
        for item in definitions:
            judge_id = str(item["id"])
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", judge_id) is None:
                raise StateError(f"V3 judge ID is unsafe for artifact paths: {judge_id!r}")
            identity = {"id": judge_id, "model": item.get("model"), "contract": item.get("contract")}
            if item.get("fingerprint") != hash_canonical_json(identity):
                raise StateError(f"V3 manifest judge fingerprint mismatch: {judge_id}")
            adapter = self.judges[judge_id]
            rubric = hash_canonical_json(item["contract"])
            if getattr(adapter, "judge_fingerprint", rubric) != rubric:
                raise StateError(f"V3 judge rubric fingerprint mismatch: {judge_id}")
            result.append((judge_id, adapter, item))
        return result

    @staticmethod
    def _v3_item_key(question_id: str, judge_id: str) -> str:
        return hash_canonical_json({"query_id": question_id, "judge_id": judge_id})[:32]

    def _judging_item_checkpoint_paths(
        self, run_dir: Path, manifest: RunManifest, terminal_ids: tuple[str, ...],
    ) -> tuple[Path, ...]:
        if self._is_v3(manifest):
            return tuple(
                lifecycle_checkpoint_path(
                    run_dir, "JUDGING", item_id=self._v3_item_key(query_id, judge_id),
                )
                for query_id in terminal_ids
                for judge_id, _adapter, _identity in self._v3_judge_definitions(manifest)
            )
        return tuple(
            lifecycle_checkpoint_path(run_dir, "JUDGING", item_id=item_id)
            for item_id in terminal_ids
        )

    def _judge_identity(self, manifest: RunManifest) -> tuple[str, str, str]:
        if self._is_v3(manifest):
            definitions = self._v3_judge_definitions(manifest)
            _judge_id, _adapter, item = next(
                row for row in definitions if row[0] == self.primary_judge_id
            )
            model = item["model"]
            return (
                str(model["provider"]), str(model["requested_model"]),
                hash_canonical_json(item["contract"]),
            )
        models = manifest.fingerprint_inputs.get("models")
        judge_model = models.get("judge", {}) if isinstance(models, dict) else {}
        judge_contract = manifest.fingerprint_inputs.get("judge_contract")
        if not isinstance(judge_model, dict) or not isinstance(judge_contract, dict):
            raise StateError("Manifest is missing judge model or judge contract identity.")
        provider = str(judge_model.get("provider", ""))
        requested_model = str(judge_model.get("requested_model", ""))
        rubric_fingerprint = hash_canonical_json(judge_contract)
        adapter_fingerprint = str(
            getattr(self.judge, "judge_fingerprint", rubric_fingerprint)
        )
        if not provider or not requested_model:
            raise StateError("Manifest judge provider/requested_model cannot be empty.")
        if adapter_fingerprint != rubric_fingerprint:
            raise StateError("Judge adapter rubric fingerprint does not match run manifest.")
        return provider, requested_model, rubric_fingerprint

    def _judge_input_fingerprint(
        self,
        manifest: RunManifest,
        predictions: list[dict[str, Any]],
    ) -> str:
        if self._is_v3(manifest):
            return hash_canonical_json({
                "schema_version": 3,
                "scientific_fingerprint": manifest.scientific_fingerprint,
                "predictions": [
                    {"question_id": str(item["question_id"]),
                     "sha256": hash_canonical_json(item)} for item in predictions
                ],
                "judges": [item["fingerprint"] for _id, _adapter, item in self._v3_judge_definitions(manifest)],
                "primary_judge_id": self.primary_judge_id,
            })
        provider, requested_model, rubric_fingerprint = self._judge_identity(manifest)
        return hash_canonical_json(
            {
                "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                "predictions": [
                    {
                        "question_id": str(item["question_id"]),
                        "sha256": hash_canonical_json(item),
                    }
                    for item in predictions
                ],
                "judge_provider": provider,
                "judge_requested_model": requested_model,
                "judge_fingerprint": rubric_fingerprint,
            }
        )

    def _judge_predictions(
        self,
        run_dir: Path,
        manifest: RunManifest,
        predictions: list[dict[str, Any]],
        metadata: dict[str, Any],
        *,
        attempt: Attempt,
        interrupt_at: str | None,
        cancel_check: Callable[[], None] | None,
        progress: Callable[[int], None],
    ) -> tuple[list[dict[str, Any]], tuple[Path, ...], dict[str, str]]:
        if self._is_v3(manifest):
            return self._judge_predictions_v3(
                run_dir, manifest, predictions, metadata, attempt=attempt,
                interrupt_at=interrupt_at, cancel_check=cancel_check, progress=progress,
            )
        judged: list[dict[str, Any]] = []
        judgment_paths: list[Path] = []
        item_digests: dict[str, str] = {}
        provider, requested_model, rubric_fingerprint = self._judge_identity(manifest)
        for offset, prediction in enumerate(predictions):
            _check_cancel(cancel_check)
            question_id = str(prediction["question_id"])
            prediction_digest = hash_canonical_json(prediction)
            item_input = hash_canonical_json(
                {
                    "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                    "question_id": question_id,
                    "prediction_sha256": prediction_digest,
                    "judge_provider": provider,
                    "judge_requested_model": requested_model,
                    "judge_fingerprint": rubric_fingerprint,
                }
            )
            judgment_path = run_dir / "judgments" / f"{question_id}.json"
            timing_path = run_dir / "judgments" / "timing" / f"{question_id}.json"
            item_checkpoint = self._load_valid_item_checkpoint(
                run_dir,
                manifest,
                phase="JUDGING",
                item_id=question_id,
                input_fingerprint=item_input,
                expected_item_ids=(question_id,),
                predecessor_digest=prediction_digest,
            )
            judged_item: dict[str, Any] | None = None
            if item_checkpoint is not None:
                try:
                    candidate = _load_json_dict(judgment_path)
                    self._validate_judgment(
                        candidate,
                        question_id=question_id,
                        prediction_digest=prediction_digest,
                        manifest=manifest,
                    )
                    judged_item = candidate
                except (KeyError, TypeError, ValueError, StateError):
                    item_checkpoint = None

            if item_checkpoint is None:
                _emit_phase_event(
                    self.events,
                    "judge.started",
                    manifest,
                    attempt,
                    "JUDGING",
                )
                judge_started_at = time.perf_counter()
                judgment = self.judge.judge(
                    JudgeRequest(prediction=dict(prediction), metadata=dict(metadata))
                )
                judge_ms = (time.perf_counter() - judge_started_at) * 1000
                _check_cancel(cancel_check)
                judged_item = {
                    **prediction,
                    "schema_version": JUDGMENT_SCHEMA_VERSION,
                    "scientific_fingerprint": manifest.scientific_fingerprint,
                    "prediction_sha256": prediction_digest,
                    "judge_input_fingerprint": item_input,
                    "judgment": _normalized_judgment(judgment),
                    "score": float(judgment.get("score", 0.0)),
                    "reason": str(judgment.get("reason", "")),
                    "judge_provider": str(judgment.get("judge_provider", "")),
                    "judge_requested_model": str(judgment.get("judge_requested_model", "")),
                    "judge_model": str(judgment.get("judge_model", "")),
                    "judge_finish_reason": judgment.get("judge_finish_reason"),
                    "judge_usage": (
                        dict(judgment.get("judge_usage", {}))
                        if isinstance(judgment.get("judge_usage", {}), dict)
                        else {}
                    ),
                    "judge_fingerprint": str(judgment.get("judge_fingerprint", "")),
                }
                self._validate_judgment(
                    judged_item,
                    question_id=question_id,
                    prediction_digest=prediction_digest,
                    manifest=manifest,
                )
                write_json_atomic(judgment_path, judged_item)
                write_json_atomic(
                    timing_path,
                    {
                        "schema_version": REPORT_SCHEMA_VERSION,
                        "benchmark": str(prediction.get("benchmark", "")),
                        "conversation_idx": prediction.get("conversation_idx"),
                        "question_id": question_id,
                        "pipeline_timing": {
                            "judge_ms": judge_ms,
                            "judge_scope": "question",
                        },
                    },
                )
                item_checkpoint = self._commit_phase_checkpoint(
                    run_dir,
                    manifest,
                    attempt=attempt,
                    phase="JUDGING",
                    input_fingerprint=item_input,
                    expected_item_ids=(question_id,),
                    artifacts=(judgment_path, timing_path),
                    predecessor_digest=prediction_digest,
                    metadata={"question_id": question_id},
                    item_id=question_id,
                )
                _emit_phase_event(
                    self.events,
                    "judge.completed",
                    manifest,
                    attempt,
                    "JUDGING",
                )
            assert judged_item is not None
            judged.append(judged_item)
            judgment_paths.append(judgment_path)
            if timing_path.is_file():
                judgment_paths.append(timing_path)
            item_digests[question_id] = item_checkpoint.checkpoint_digest
            progress(len(judged))
            if interrupt_at == "after-first-judgment" and offset == 0:
                raise InjectedTerminalInterrupt("Interrupted after first judgment commit.")
        return judged, tuple(judgment_paths), item_digests

    def _judge_predictions_v3(
        self,
        run_dir: Path,
        manifest: RunManifest,
        predictions: list[dict[str, Any]],
        metadata: dict[str, Any],
        *,
        attempt: Attempt,
        interrupt_at: str | None,
        cancel_check: Callable[[], None] | None,
        progress: Callable[[int], None],
    ) -> tuple[list[dict[str, Any]], tuple[Path, ...], dict[str, str]]:
        definitions = self._v3_judge_definitions(manifest)
        evaluations: list[dict[str, Any]] = []
        paths: list[Path] = []
        digests: dict[str, str] = {}
        for prediction in predictions:
            query_id = str(prediction["question_id"])
            prediction_digest = hash_canonical_json(prediction)
            by_judge: dict[str, dict[str, Any]] = {}
            for judge_id, adapter, identity in definitions:
                _check_cancel(cancel_check)
                model = identity["model"]
                rubric_fingerprint = hash_canonical_json(identity["contract"])
                item_input = hash_canonical_json({
                    "schema_version": 3,
                    "scientific_fingerprint": manifest.scientific_fingerprint,
                    "query_id": query_id,
                    "judge_id": judge_id,
                    "prediction_sha256": prediction_digest,
                    "judge_fingerprint": identity["fingerprint"],
                })
                item_key = self._v3_item_key(query_id, judge_id)
                judgment_path = run_dir / "judgments" / judge_id / f"{query_id}.json"
                timing_path = run_dir / "judgments" / "timing" / judge_id / f"{query_id}.json"
                checkpoint = self._load_valid_item_checkpoint(
                    run_dir, manifest, phase="JUDGING", item_id=item_key,
                    input_fingerprint=item_input, expected_item_ids=(query_id,),
                    predecessor_digest=prediction_digest,
                )
                candidate: dict[str, Any] | None = None
                if checkpoint is not None:
                    try:
                        candidate = _load_json_dict(judgment_path)
                        self._validate_judgment_v3(
                            candidate, manifest=manifest, query_id=query_id,
                            judge_id=judge_id, prediction_digest=prediction_digest,
                            identity=identity,
                        )
                    except (KeyError, TypeError, ValueError, StateError):
                        checkpoint = None
                if checkpoint is None:
                    _emit_phase_event(self.events, "judge.started", manifest, attempt, "JUDGING")
                    started = time.perf_counter()
                    result = adapter.judge(JudgeRequest(
                        prediction=dict(prediction),
                        metadata={**metadata, "judge_id": judge_id},
                    ))
                    judge_ms = (time.perf_counter() - started) * 1000
                    _check_cancel(cancel_check)
                    candidate = {
                        **prediction,
                        "schema_version": 3,
                        "scientific_fingerprint": manifest.scientific_fingerprint,
                        "prediction_sha256": prediction_digest,
                        "judge_input_fingerprint": item_input,
                        "judge_id": judge_id,
                        "judge_identity_fingerprint": identity["fingerprint"],
                        "judgment": _normalized_judgment(result),
                        "score": float(result.get("score", 0.0)),
                        "reason": str(result.get("reason", "")),
                        "judge_provider": str(result.get("judge_provider", "")),
                        "judge_requested_model": str(result.get("judge_requested_model", "")),
                        "judge_model": str(result.get("judge_model", "")),
                        "judge_finish_reason": result.get("judge_finish_reason"),
                        "judge_usage": dict(result.get("judge_usage", {})) if isinstance(result.get("judge_usage", {}), dict) else {},
                        "judge_fingerprint": str(result.get("judge_fingerprint", "")),
                    }
                    self._validate_judgment_v3(
                        candidate, manifest=manifest, query_id=query_id,
                        judge_id=judge_id, prediction_digest=prediction_digest,
                        identity=identity,
                    )
                    write_json_atomic(judgment_path, candidate)
                    write_json_atomic(timing_path, {
                        "schema_version": 3,
                        "benchmark": str(prediction.get("benchmark", "")),
                        "question_id": query_id,
                        "judge_id": judge_id,
                        "pipeline_timing": {"judge_ms": judge_ms, "judge_scope": "question"},
                    })
                    checkpoint = self._commit_phase_checkpoint(
                        run_dir, manifest, attempt=attempt, phase="JUDGING",
                        input_fingerprint=item_input, expected_item_ids=(query_id,),
                        artifacts=(judgment_path, timing_path),
                        predecessor_digest=prediction_digest,
                        metadata={"question_id": query_id, "judge_id": judge_id,
                                  "judge_fingerprint": identity["fingerprint"]},
                        item_id=item_key,
                    )
                    _emit_phase_event(self.events, "judge.completed", manifest, attempt, "JUDGING")
                assert candidate is not None and checkpoint is not None
                by_judge[judge_id] = candidate
                paths.extend((judgment_path, timing_path))
                digests[f"{query_id}:{judge_id}"] = checkpoint.checkpoint_digest
                if interrupt_at == "after-first-judgment" and len(digests) == 1:
                    raise InjectedTerminalInterrupt("Interrupted after first judgment commit.")
            primary = by_judge[self.primary_judge_id]
            evaluations.append({
                **primary,
                "judges": {
                    judge_id: {
                        "judgment": item["judgment"], "score": item["score"],
                        "judge_fingerprint": item["judge_identity_fingerprint"],
                        "judge_usage": item["judge_usage"],
                        "judge_provider": item["judge_provider"],
                        "judge_model": item["judge_model"],
                    }
                    for judge_id, item in by_judge.items()
                },
                "primary_judge_id": self.primary_judge_id,
            })
            progress(len(evaluations))
        return evaluations, tuple(paths), digests

    def _validate_judgment_v3(
        self,
        judgment: dict[str, Any],
        *,
        manifest: RunManifest,
        query_id: str,
        judge_id: str,
        prediction_digest: str,
        identity: dict[str, Any],
    ) -> None:
        model = identity["model"]
        if (
            judgment.get("schema_version") != 3
            or judgment.get("question_id") != query_id
            or judgment.get("judge_id") != judge_id
            or judgment.get("scientific_fingerprint") != manifest.scientific_fingerprint
            or judgment.get("prediction_sha256") != prediction_digest
            or judgment.get("judge_identity_fingerprint") != identity["fingerprint"]
            or judgment.get("judge_provider") != model["provider"]
            or judgment.get("judge_requested_model") != model["requested_model"]
            or judgment.get("judge_fingerprint") != hash_canonical_json(identity["contract"])
            or not str(judgment.get("judge_model", ""))
        ):
            raise StateError(f"V3 judgment identity mismatch: {query_id}/{judge_id}")
        _normalized_judgment(judgment)
        float(judgment["score"])

    def _validate_judgment(
        self,
        judgment: dict[str, Any],
        *,
        question_id: str,
        prediction_digest: str,
        manifest: RunManifest,
    ) -> None:
        provider, requested_model, rubric_fingerprint = self._judge_identity(manifest)
        if judgment.get("schema_version") != JUDGMENT_SCHEMA_VERSION:
            raise StateError("Judgment schema_version mismatch.")
        if str(judgment.get("question_id", "")) != question_id:
            raise StateError("Judgment question_id mismatch.")
        if judgment.get("scientific_fingerprint") != manifest.scientific_fingerprint:
            raise StateError("Judgment scientific fingerprint mismatch.")
        if judgment.get("prediction_sha256") != prediction_digest:
            raise StateError("Judgment prediction digest mismatch.")
        if judgment.get("judge_provider") != provider:
            raise StateError("Judgment judge provider mismatch.")
        if judgment.get("judge_requested_model") != requested_model:
            raise StateError("Judgment requested judge model mismatch.")
        if judgment.get("judge_fingerprint") != rubric_fingerprint:
            raise StateError("Judgment rubric fingerprint mismatch.")
        if not str(judgment.get("judge_model", "")):
            raise StateError("Judgment returned model cannot be empty.")
        _normalized_judgment(judgment)
        float(judgment["score"])

    def _validate_evaluation_set(
        self,
        evaluations: list[dict[str, Any]],
        expected_item_ids: tuple[str, ...],
        manifest: RunManifest,
    ) -> None:
        observed = tuple(str(item.get("question_id", "")) for item in evaluations)
        if observed != expected_item_ids:
            raise StateError("Judgment aggregate expected item set mismatch.")
        if self._is_v3(manifest):
            judge_ids = [judge_id for judge_id, _adapter, _identity in self._v3_judge_definitions(manifest)]
            for item in evaluations:
                by_judge = item.get("judges")
                if not isinstance(by_judge, dict) or list(by_judge) != judge_ids:
                    raise StateError("V3 judgment aggregate judge set/order mismatch.")
                if item.get("primary_judge_id") != self.primary_judge_id:
                    raise StateError("V3 judgment aggregate primary judge mismatch.")
                primary = by_judge[self.primary_judge_id]
                if primary.get("judgment") != item.get("judgment") or primary.get("score") != item.get("score"):
                    raise StateError("V3 judgment aggregate primary score mismatch.")
            return
        for item in evaluations:
            self._validate_judgment(
                item,
                question_id=str(item["question_id"]),
                prediction_digest=str(item.get("prediction_sha256", "")),
                manifest=manifest,
            )

    def _evaluate(
        self,
        run_dir: Path,
        manifest: RunManifest,
        evaluations: list[dict[str, Any]],
        metadata: dict[str, Any],
        *,
        attempt: Attempt,
        interrupt_at: str | None,
        cancel_check: Callable[[], None] | None,
    ) -> tuple[dict[str, Any], tuple[Path, ...], dict[str, str]]:
        requirements = self._requirements(manifest, metadata)
        reports: dict[str, Any] = {
            "schema_version": self._evaluation_schema(manifest),
            "benchmark": metadata["benchmark"],
            "framework": metadata["framework"],
            "requirements": [requirement_to_dict(item) for item in requirements],
            "artifacts": {},
            "evaluator_versions": {},
        }
        output_paths: list[Path] = []
        item_digests: dict[str, str] = {}
        evaluations_digest = sha256_file(run_dir / "evaluations" / "evaluations.json")
        terminal_ids = expected_terminal_item_ids(manifest)
        for offset, requirement in enumerate(requirements):
            _check_cancel(cancel_check)
            evaluator_version = self._expected_evaluator_version(requirement, metadata)
            item_input = hash_canonical_json(
                {
                    "schema_version": SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
                    "evaluations_sha256": evaluations_digest,
                    "requirement": requirement_to_dict(requirement),
                    "evaluator_version": evaluator_version,
                }
            )
            output_path = run_dir / "evaluations" / f"{requirement.name}.json"
            item_checkpoint = self._load_valid_item_checkpoint(
                run_dir,
                manifest,
                phase="EVALUATING",
                item_id=requirement.name,
                input_fingerprint=item_input,
                expected_item_ids=terminal_ids,
                predecessor_digest=evaluations_digest,
            )
            artifact: dict[str, Any] | None = None
            if item_checkpoint is not None:
                try:
                    candidate = _load_json_dict(output_path)
                    self._validate_evaluator_output(
                        candidate,
                        requirement=requirement,
                        evaluator_version=evaluator_version,
                        schema_version=self._evaluation_schema(manifest),
                    )
                    artifact = candidate
                except (KeyError, TypeError, ValueError, StateError):
                    item_checkpoint = None
            if item_checkpoint is None:
                _emit_phase_event(
                    self.events,
                    "evaluation.started",
                    manifest,
                    attempt,
                    "EVALUATING",
                )
                artifact = self._run_evaluator(
                    requirement, evaluations, metadata,
                    run_dir=run_dir, manifest=manifest,
                )
                if self._is_v3(manifest):
                    artifact = {**artifact, "schema_version": 3}
                _check_cancel(cancel_check)
                self._validate_evaluator_output(
                    artifact,
                    requirement=requirement,
                    evaluator_version=evaluator_version,
                    schema_version=self._evaluation_schema(manifest),
                )
                write_json_atomic(output_path, artifact)
                extra_paths = (
                    (run_dir / "evaluations" / "analysis_rows.jsonl",)
                    if requirement.name == "analysis_rows" and artifact["status"] == "COMPLETED"
                    else ()
                )
                item_checkpoint = self._commit_phase_checkpoint(
                    run_dir,
                    manifest,
                    attempt=attempt,
                    phase="EVALUATING",
                    input_fingerprint=item_input,
                    expected_item_ids=terminal_ids,
                    artifacts=(output_path, *extra_paths),
                    predecessor_digest=evaluations_digest,
                    metadata={
                        "evaluator": requirement.name,
                        "evaluator_version": evaluator_version,
                    },
                    item_id=requirement.name,
                )
                _emit_phase_event(
                    self.events,
                    "evaluation.completed",
                    manifest,
                    attempt,
                    "EVALUATING",
                )
            assert artifact is not None
            reports["artifacts"][requirement.name] = {
                "path": output_path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(output_path),
                "status": artifact["status"],
            }
            reports["evaluator_versions"][requirement.name] = evaluator_version
            output_paths.append(output_path)
            if requirement.name == "analysis_rows" and artifact["status"] == "COMPLETED":
                output_paths.append(run_dir / "evaluations" / "analysis_rows.jsonl")
            item_digests[requirement.name] = item_checkpoint.checkpoint_digest
            if requirement.required and artifact["status"] != "COMPLETED":
                raise StateError(f"Required evaluator failed: {requirement.name}")
            if interrupt_at == "after-first-evaluator" and offset == 0:
                raise InjectedTerminalInterrupt("Interrupted after first evaluator commit.")
        return reports, tuple(output_paths), item_digests

    def _expected_evaluator_version(
        self,
        requirement: EvaluationRequirement,
        metadata: dict[str, Any],
    ) -> str:
        if requirement.not_applicable_reason:
            return "not-applicable-v1"
        if requirement.name == "primary_judge_score":
            return "primary-judge-v1"
        if requirement.name in {"analysis_rows", "retrieval_report", "judge_agreement"}:
            return {
                "analysis_rows": "analysis-rows-v3",
                "retrieval_report": "retrieval-rank-v3",
                "judge_agreement": "judge-agreement-v3",
            }[requirement.name]
        if requirement.name in {"rigorous_report", "ablation_report"}:
            if metadata["benchmark"] == "locomo":
                return locomo_evaluation.EVALUATOR_VERSION
            if metadata["benchmark"] == "longmemeval":
                return longmemeval_evaluation.EVALUATOR_VERSION
        evaluator = self.evaluators.get(requirement.name)
        version = getattr(evaluator, "evaluator_version", None)
        if not version:
            raise StateError(
                f"Custom evaluator must declare evaluator_version: {requirement.name}"
            )
        return str(version)

    def _evaluation_identities(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        requirements = self._requirements_from_metadata(metadata)
        return [
            {
                "name": requirement.name,
                "requirement": requirement_to_dict(requirement),
                "evaluator_version": self._expected_evaluator_version(
                    requirement,
                    metadata,
                ),
            }
            for requirement in requirements
        ]

    def _requirements_from_metadata(
        self, metadata: dict[str, Any],
    ) -> tuple[EvaluationRequirement, ...]:
        pair = (metadata["benchmark"], metadata["framework"])
        if self.evaluation_plans is not None:
            return evaluation_plan_for(
                benchmark=pair[0], framework=pair[1], plans=self.evaluation_plans,
            )
        if metadata.get("experiment_schema_version") == 3:
            return evaluation_plan_v3(pair[0], pair[1], tuple(metadata["judge_ids"]))
        return evaluation_plan_for(benchmark=pair[0], framework=pair[1])

    def _requirements(
        self, manifest: RunManifest, metadata: dict[str, Any],
    ) -> tuple[EvaluationRequirement, ...]:
        del manifest
        return self._requirements_from_metadata(metadata)

    def _validate_evaluator_output(
        self,
        artifact: dict[str, Any],
        *,
        requirement: EvaluationRequirement,
        evaluator_version: str,
        schema_version: int,
    ) -> None:
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema_version") != schema_version
        ):
            raise StateError(f"Evaluator schema mismatch: {requirement.name}")
        if artifact.get("evaluator") != requirement.name:
            raise StateError(f"Evaluator identity mismatch: {requirement.name}")
        status = str(artifact.get("status", ""))
        if status not in {"COMPLETED", "FAILED", "NOT_APPLICABLE"}:
            raise StateError(f"Evaluator status is invalid: {requirement.name}")
        if artifact.get("evaluator_version") != evaluator_version:
            raise StateError(f"Evaluator version mismatch: {requirement.name}")
        if requirement.not_applicable_reason and status != "NOT_APPLICABLE":
            raise StateError(f"Evaluator must be NOT_APPLICABLE: {requirement.name}")

    def _run_evaluator(
        self,
        requirement: EvaluationRequirement,
        evaluations: list[dict[str, Any]],
        metadata: dict[str, Any],
        *,
        run_dir: Path | None = None,
        manifest: RunManifest | None = None,
    ) -> dict[str, Any]:
        if requirement.not_applicable_reason:
            return {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "evaluator": requirement.name,
                "evaluator_version": "not-applicable-v1",
                "status": "NOT_APPLICABLE",
                "reason": requirement.not_applicable_reason,
            }
        if requirement.name in self.evaluators:
            return self.evaluators[requirement.name](evaluations, metadata)
        if manifest is not None and self._is_v3(manifest):
            if requirement.name == "judge_agreement":
                return self._judge_agreement_artifact(evaluations, metadata)
            if requirement.name == "retrieval_report":
                return self._retrieval_artifact(evaluations, metadata, run_dir, manifest)
            if requirement.name == "analysis_rows":
                return self._analysis_rows_artifact(evaluations, metadata, run_dir, manifest)
        if requirement.name == "primary_judge_score":
            return primary_judge_report(evaluations, metadata)
        if requirement.name == "rigorous_report":
            if metadata["benchmark"] == "locomo":
                return locomo_evaluation.rigorous_report(evaluations, metadata)
            if metadata["benchmark"] == "longmemeval":
                return longmemeval_evaluation.rigorous_report(evaluations, metadata)
        if requirement.name == "ablation_report":
            if metadata["benchmark"] == "locomo":
                return locomo_evaluation.ablation_report(evaluations, metadata)
            if metadata["benchmark"] == "longmemeval":
                return longmemeval_evaluation.ablation_report(evaluations, metadata)
        raise StateError(f"Evaluator is not implemented: {requirement.name}")

    @staticmethod
    def _query_artifacts_v3(
        run_dir: Path, manifest: RunManifest,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        retrievals: dict[str, dict[str, Any]] = {}
        packed_contexts: dict[str, dict[str, Any]] = {}
        ingestion: dict[str, dict[str, Any]] = {}
        for unit_id in manifest.expected_item_ids:
            unit_dir = run_dir / "items" / unit_id
            aggregate = _load_json_dict(unit_dir / "predictions.json")
            prepared = _load_json_dict(unit_dir / "prepared.json")
            query_ids = aggregate.get("question_ids")
            if not isinstance(query_ids, list):
                raise StateError(f"V3 aggregate missing question IDs: {unit_id}")
            for query_id in query_ids:
                query_id = str(query_id)
                if query_id in retrievals:
                    raise StateError(f"Duplicate v3 query ID: {query_id}")
                question_dir = unit_dir / "questions"
                retrievals[query_id] = _load_json_dict(question_dir / f"{query_id}.retrieval.json")
                packed_contexts[query_id] = _load_json_dict(question_dir / f"{query_id}.context.json")
                ingestion[query_id] = prepared
        return retrievals, packed_contexts, ingestion

    def _retrieval_artifact(
        self, evaluations: list[dict[str, Any]], metadata: dict[str, Any],
        run_dir: Path | None, manifest: RunManifest,
    ) -> dict[str, Any]:
        if run_dir is None:
            raise StateError("V3 retrieval evaluation requires a run directory.")
        retrievals, _packed, _ingestion = self._query_artifacts_v3(run_dir, manifest)
        rows: dict[str, dict[str, Any]] = {}
        for item in evaluations:
            query_id = str(item["question_id"])
            gold = (
                item.get("evidence") if metadata["benchmark"] == "locomo"
                else item.get("answer_session_ids")
            )
            if not isinstance(gold, list) or not gold:
                rows[query_id] = {"status": "NOT_APPLICABLE", "reason": "No official evidence IDs."}
                continue
            retrieved = retrievals[query_id].get("items")
            if not isinstance(retrieved, list):
                raise StateError(f"V3 retrieval items missing for {query_id}.")
            source_lists: list[list[str]] = []
            for memory in retrieved:
                if not isinstance(memory, dict):
                    raise StateError("V3 retrieved item must be an object.")
                item_meta = memory.get("metadata")
                refs = item_meta.get("source_refs") if isinstance(item_meta, dict) else None
                if not isinstance(refs, list) and metadata["benchmark"] == "locomo":
                    refs = memory.get("source_event_ids")
                if not isinstance(refs, list):
                    refs = []
                source_lists.append([str(value) for value in refs])
            if retrieved and not any(source_lists):
                rows[query_id] = {"status": "NOT_APPLICABLE", "reason": "Retrieved memories have no source provenance."}
                continue
            expected = set(str(value) for value in gold)
            seen: set[str] = set()
            first_rank: int | None = None
            for rank, refs in enumerate(source_lists, start=1):
                matches = expected.intersection(refs)
                if matches and first_rank is None:
                    first_rank = rank
                seen.update(matches)
            rows[query_id] = {
                "status": "COMPLETED",
                "recall_at_k": len(seen) / len(expected),
                "reciprocal_rank": 1 / first_rank if first_rank is not None else 0.0,
                "retrieved_count": len(retrieved),
            }
        completed = [row for row in rows.values() if row["status"] == "COMPLETED"]
        if not completed:
            return {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "evaluator": "retrieval_report", "evaluator_version": "retrieval-rank-v3",
                "status": "NOT_APPLICABLE",
                "reason": "No query has both official evidence and retrieved source provenance.",
                "queries": rows,
            }
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluator": "retrieval_report", "evaluator_version": "retrieval-rank-v3",
            "status": "COMPLETED",
            "queries": rows,
            "metrics": {
                "evaluated_count": len(completed),
                "mean_recall_at_k": sum(row["recall_at_k"] for row in completed) / len(completed),
                "mean_reciprocal_rank": sum(row["reciprocal_rank"] for row in completed) / len(completed),
            },
        }

    def _judge_agreement_artifact(
        self, evaluations: list[dict[str, Any]], metadata: dict[str, Any],
    ) -> dict[str, Any]:
        del metadata
        agreement = _multi_judge_report(evaluations, self.primary_judge_id)["agreement"]
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluator": "judge_agreement", "evaluator_version": "judge-agreement-v3",
            **agreement,
        }

    def _analysis_rows_artifact(
        self, evaluations: list[dict[str, Any]], metadata: dict[str, Any],
        run_dir: Path | None, manifest: RunManifest,
    ) -> dict[str, Any]:
        if run_dir is None:
            raise StateError("V3 analysis rows require a run directory.")
        retrievals, packed, ingestion = self._query_artifacts_v3(run_dir, manifest)
        requirement = next(
            req for req in self._requirements(manifest, metadata)
            if req.name == "retrieval_report"
        )
        retrieval_artifact = (
            {"status": "NOT_APPLICABLE", "reason": requirement.not_applicable_reason}
            if requirement.not_applicable_reason
            else self._retrieval_artifact(evaluations, metadata, run_dir, manifest)
        )
        retrieval_metrics = (
            retrieval_artifact.get("queries")
            if retrieval_artifact["status"] == "COMPLETED"
            else None
        )
        if retrieval_metrics is None:
            requirement = EvaluationRequirement(
                "retrieval_report", required=False,
                not_applicable_reason=str(retrieval_artifact.get("reason", "Retrieval metric unavailable.")),
            )
        predictions = {str(item["question_id"]): item for item in evaluations}
        references = {
            query_id: {
                "query_id": query_id,
                "strata": (
                    {"category": item.get("category")}
                    if metadata["benchmark"] == "locomo"
                    else {"question_type": item.get("question_type"),
                          "is_abstention": item.get("is_abstention")}
                ),
            }
            for query_id, item in predictions.items()
        }
        judgments = {
            query_id: {
                judge_id: {"question_id": query_id, "judge_id": judge_id, **value}
                for judge_id, value in item["judges"].items()
            }
            for query_id, item in predictions.items()
        }
        rows = build_analysis_rows(
            run_id=manifest.run_id,
            scientific_fingerprint=manifest.scientific_fingerprint,
            benchmark=metadata["benchmark"], framework=metadata["framework"],
            expected_query_ids=expected_terminal_item_ids(manifest),
            expected_judge_ids=tuple(metadata["judge_ids"]),
            primary_judge_id=self.primary_judge_id,
            predictions=predictions, retrievals=retrievals, packed_contexts=packed,
            judgments=judgments, references=references,
            retrieval_metrics=retrieval_metrics,
            retrieval_requirement=requirement,
            ingestion=ingestion,
        )
        jsonl_path = run_dir / "evaluations" / "analysis_rows.jsonl"
        _write_bytes_atomic(jsonl_path, analysis_rows_jsonl(rows))
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluator": "analysis_rows", "evaluator_version": "analysis-rows-v3",
            "status": "COMPLETED", "row_count": len(rows),
            "path": jsonl_path.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(jsonl_path),
        }

    def _validate_evaluation_summary(self, reports: dict[str, Any], manifest: RunManifest) -> None:
        if reports.get("schema_version") != self._evaluation_schema(manifest):
            raise StateError("Evaluation summary schema mismatch.")
        if not isinstance(reports.get("artifacts"), dict):
            raise StateError("Evaluation summary artifacts must be an object.")
        if not isinstance(reports.get("evaluator_versions"), dict):
            raise StateError("Evaluation summary evaluator_versions must be an object.")

    def _write_reports(
        self,
        run_dir: Path,
        manifest: RunManifest,
        metadata: dict[str, Any],
        evaluations: list[dict[str, Any]],
        reports: dict[str, Any],
        *,
        attempt: Attempt,
        resource_tracker: ResourceUsageTracker,
    ) -> tuple[Path, ...]:
        usage = _build_usage_report(evaluations)
        timing = _build_execution_timing_report(run_dir, evaluations, attempt)
        resources = resource_tracker.report(
            attempt_id=attempt.attempt_id,
            resume=attempt.resume,
        )
        if self._is_v3(manifest):
            usage["schema_version"] = 3
            timing["schema_version"] = 3
            resources["schema_version"] = 3
        summary = {
            "schema_version": 3 if self._is_v3(manifest) else REPORT_SCHEMA_VERSION,
            "run_id": manifest.run_id,
            "scientific_fingerprint": manifest.scientific_fingerprint,
            "benchmark": metadata["benchmark"],
            "framework": metadata["framework"],
            "counts": {
                "expected": len(evaluations),
                "evaluated": len(evaluations),
                "excluded": 0,
                "failed": 0,
            },
            "evaluator_versions": _evaluator_versions(reports),
            "evaluation_artifacts": reports["artifacts"],
            "usage": usage,
            "timing": timing,
            "report_artifacts": {
                "usage": "reports/usage.json",
                "timing": "reports/timing.json",
                "resources": "reports/resources.json",
            },
        }
        if self._is_v3(manifest):
            summary["judges"] = _multi_judge_report(evaluations, self.primary_judge_id)
        summary_path = run_dir / "reports" / "summary.json"
        markdown_path = run_dir / "reports" / "summary.md"
        usage_path = run_dir / "reports" / "usage.json"
        timing_path = run_dir / "reports" / "timing.json"
        resources_path = run_dir / "reports" / "resources.json"
        write_json_atomic(usage_path, usage)
        write_json_atomic(timing_path, timing)
        write_json_atomic(resources_path, resources)
        write_json_atomic(summary_path, summary)
        markdown = (
            f"# dmf-bench run {manifest.run_id}\n\n"
            f"- benchmark: {metadata['benchmark']}\n"
            f"- framework: {metadata['framework']}\n"
            f"- expected: {len(evaluations)}\n"
            f"- evaluated: {len(evaluations)}\n"
            f"- excluded: 0\n"
            f"- failed: 0\n"
            f"- benchmark tokens: {usage['totals']['benchmark']['total_tokens']}\n"
            f"- judge tokens: {usage['totals']['evaluation']['total_tokens']}\n"
            f"- total tokens: {usage['totals']['overall']['total_tokens']}\n"
            f"- execution through reporting: {timing['attempt']['execution_seconds']:.6f} s\n"
            f"- process CPU: {resources['process']['cpu_total_seconds']:.6f} s\n"
            f"- peak RSS: {resources['process']['peak_rss_bytes']} bytes\n"
        )
        markdown_path.parent.mkdir(exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        return summary_path, markdown_path, usage_path, timing_path, resources_path

    def _load_valid_phase_checkpoint(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        phase: str,
        input_fingerprint: str,
        expected_item_ids: tuple[str, ...],
        predecessor_digest: str | None,
    ) -> LifecycleCheckpoint | None:
        return self._load_valid_item_checkpoint(
            run_dir,
            manifest,
            phase=phase,
            item_id=None,
            input_fingerprint=input_fingerprint,
            expected_item_ids=expected_item_ids,
            predecessor_digest=predecessor_digest,
        )

    def _load_valid_item_checkpoint(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        phase: str,
        item_id: str | None,
        input_fingerprint: str,
        expected_item_ids: tuple[str, ...],
        predecessor_digest: str | None,
    ) -> LifecycleCheckpoint | None:
        path = lifecycle_checkpoint_path(run_dir, phase, item_id=item_id)
        if not path.exists():
            return None
        try:
            checkpoint = load_lifecycle_checkpoint(path)
            if (
                checkpoint.run_id != manifest.run_id
                or checkpoint.phase != phase
                or checkpoint.status != "COMMITTED"
                or checkpoint.scientific_fingerprint != manifest.scientific_fingerprint
                or checkpoint.input_fingerprint != input_fingerprint
                or checkpoint.expected_item_ids != expected_item_ids
                or checkpoint.predecessor_digest != predecessor_digest
            ):
                return None
            validate_artifact_refs(run_dir, checkpoint.artifacts)
            return checkpoint
        except (KeyError, TypeError, ValueError, StateError):
            return None

    def _commit_phase_checkpoint(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        attempt: Attempt | None,
        phase: str,
        input_fingerprint: str,
        expected_item_ids: tuple[str, ...],
        artifacts: tuple[Path, ...],
        predecessor_digest: str | None,
        metadata: dict[str, Any],
        item_id: str | None = None,
    ) -> LifecycleCheckpoint:
        checkpoint = LifecycleCheckpoint(
            run_id=manifest.run_id,
            attempt_id=attempt.attempt_id if attempt is not None else "item-checkpoint",
            phase=phase,
            status="COMMITTED",
            scientific_fingerprint=manifest.scientific_fingerprint,
            input_fingerprint=input_fingerprint,
            expected_item_ids=expected_item_ids,
            artifacts=tuple(artifact_ref_for(run_dir, path) for path in artifacts),
            predecessor_digest=predecessor_digest,
            metadata=metadata,
        )
        path = write_lifecycle_checkpoint(run_dir, checkpoint, item_id=item_id)
        return load_lifecycle_checkpoint(path)

    def _write_failed_phase_checkpoint(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        attempt: Attempt,
        phase: str,
        input_fingerprint: str,
        expected_item_ids: tuple[str, ...],
        predecessor_digest: str | None,
        error_type: str,
    ) -> None:
        checkpoint = LifecycleCheckpoint(
            run_id=manifest.run_id,
            attempt_id=attempt.attempt_id,
            phase=phase,
            status="FAILED",
            scientific_fingerprint=manifest.scientific_fingerprint,
            input_fingerprint=input_fingerprint,
            expected_item_ids=expected_item_ids,
            predecessor_digest=predecessor_digest,
            metadata={"error_type": error_type},
        )
        write_lifecycle_checkpoint(run_dir, checkpoint)

    def _write_status(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        state: str,
        phase: str,
        committed: int,
    ) -> None:
        terminal_ids = expected_terminal_item_ids(manifest)
        status = RunStatus(
            run_id=manifest.run_id,
            state=state,
            phase=phase,
            expected=len(terminal_ids),
            committed=committed,
            expected_units=len(manifest.expected_item_ids),
            committed_units=len(manifest.expected_item_ids),
        )
        write_json_atomic(run_dir / "run-status.json", status.to_dict())
        if self.metrics is not None:
            inputs = manifest.fingerprint_inputs
            self.metrics.record_run_status(
                benchmark=str(inputs.get("benchmark", "")),
                framework=str(inputs.get("framework", "")),
                phase=phase,
                expected=len(terminal_ids),
                committed=committed,
                state=state,
                expected_units=len(manifest.expected_item_ids),
                committed_units=len(manifest.expected_item_ids),
            )


class OfflineFullLifecycleRunner:
    """Compose prediction and terminal phases while retaining one run lock."""

    def __init__(
        self,
        *,
        prediction_runner: Any,
        artifact_store: LocalArtifactStore,
        judge: JudgeAdapter | None = None,
        judges: Mapping[str, JudgeAdapter] | None = None,
        primary_judge_id: str | None = None,
        metrics: BenchmarkMetrics | None = None,
        events: JsonEventLogger | None = None,
    ) -> None:
        self.prediction_runner = prediction_runner
        self.artifact_store = artifact_store
        self.metrics = metrics
        self.finalizer = OfflineLifecycleFinalizer(
            artifact_store=artifact_store,
            judge=judge,
            judges=judges,
            primary_judge_id=primary_judge_id,
            metrics=metrics,
            events=events,
        )

    def run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
        prediction_interrupt_at: str | None = None,
        terminal_interrupt_at: str | None = None,
        cancel_check: Callable[[], None] | None = None,
        on_run_ready: Callable[[Path, Attempt], None] | None = None,
    ) -> FinalizationResult:
        resolved_run_id = self.prediction_runner.resolve_run_id(config, run_id=run_id)
        lock_path = self.artifact_store.runs_dir / ".locks" / f"{resolved_run_id}.lock"
        with RunLock(lock_path):
            run_dir = self.artifact_store.run_dir(resolved_run_id)
            resumed_from = latest_attempt_id(run_dir) if run_dir.exists() else None
            entry_phase = plan_resume(run_dir).next_phase if run_dir.exists() else "RUNNING"
            attempt = new_attempt(
                run_id=resolved_run_id,
                entry_phase=entry_phase,
                resume=resume,
                resumed_from_attempt_id=resumed_from,
            )
            started_at = time.perf_counter()
            resource_tracker = ResourceUsageTracker.start()
            try:
                prediction_result = self.prediction_runner.run(
                    config,
                    run_id=resolved_run_id,
                    resume=resume,
                    interrupt_at=prediction_interrupt_at,
                    cancel_check=cancel_check,
                    _lock_held=True,
                    _attempt=attempt,
                    _finish_attempt=False,
                    _on_run_ready=on_run_ready,
                )
                result = self.finalizer.finalize(
                    prediction_result.run_id,
                    interrupt_at=terminal_interrupt_at,
                    cancel_check=cancel_check,
                    _lock_held=True,
                    _attempt=attempt,
                    _resource_tracker=resource_tracker,
                )
            except RunInterrupted:
                self._record_attempt(config, "interrupted", started_at)
                raise
            except Exception:
                self._record_attempt(config, "failed", started_at)
                raise
            self._record_attempt(config, "completed", started_at)
            return result

    def _record_attempt(
        self,
        config: dict[str, Any],
        outcome: str,
        started_at: float,
    ) -> None:
        if self.metrics is None:
            return
        self.metrics.record_attempt(
            benchmark=str(config.get("benchmark", "")),
            framework=str(config.get("framework", "")),
            outcome=outcome,
        )
        self.metrics.observe_phase_duration(
            benchmark=str(config.get("benchmark", "")),
            framework=str(config.get("framework", "")),
            phase="RUNNING",
            seconds=time.perf_counter() - started_at,
        )


class InjectedTerminalInterrupt(RuntimeError):
    """Raised by tests to stop finalization at a terminal phase boundary."""


def _build_usage_report(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    answerer = _aggregate_provider_usage(evaluations, role="answerer")
    judge_by_id: dict[str, dict[str, Any]] = {}
    if evaluations and all(isinstance(item.get("judges"), dict) for item in evaluations):
        judge_rows = [
            judgment
            for item in evaluations
            for judgment in item["judges"].values()
        ]
        judge = _aggregate_provider_usage(judge_rows, role="judge")
        judge_by_id = {
            judge_id: _aggregate_provider_usage(
                [item["judges"][judge_id] for item in evaluations], role="judge"
            )
            for judge_id in evaluations[0]["judges"]
        }
    else:
        judge = _aggregate_provider_usage(evaluations, role="judge")
    memory_internal = _aggregate_memory_usage(evaluations)
    benchmark_total = _sum_usage(memory_internal, answerer)
    evaluation_total = _sum_usage(judge)
    overall_total = _sum_usage(benchmark_total, evaluation_total)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": "committed_successful_items",
        "item_count": len(evaluations),
        "components": {
            "memory_internal": memory_internal,
            "answerer": answerer,
            "judge": judge,
        },
        "totals": {
            "benchmark": benchmark_total,
            "evaluation": evaluation_total,
            "overall": overall_total,
        },
        "notes": {
            "benchmark": "memory_internal plus answerer",
            "evaluation": "judge only",
            "overall": "benchmark plus evaluation",
            "attempts": (
                "Failed or interrupted attempt cost is operational and is not "
                "included in committed scientific-item totals."
            ),
        },
    }
    if judge_by_id:
        report["components"]["judges"] = judge_by_id
    return report


def _aggregate_provider_usage(
    evaluations: list[dict[str, Any]],
    *,
    role: str,
) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    calls = 0
    reported_calls = 0
    missing_usage_calls = 0
    providers: set[str] = set()
    models: set[str] = set()
    usage_key = f"{role}_usage"
    provider_key = f"{role}_provider"
    model_key = f"{role}_model"
    for evaluation in evaluations:
        calls += 1
        raw_usage = evaluation.get(usage_key)
        provider = str(evaluation.get(provider_key, "")).strip()
        model = str(evaluation.get(model_key, "")).strip()
        if provider:
            providers.add(provider)
        if model:
            models.add(model)
        if not isinstance(raw_usage, dict) or not any(
            key in raw_usage
            for key in ("prompt_tokens_total", "completion_tokens", "total_tokens")
        ):
            missing_usage_calls += 1
            continue
        reported_calls += 1
        usage = normalize_answerer_usage(raw_usage)
        prompt_tokens += usage["prompt_tokens_total"]
        completion_tokens += usage["completion_tokens"]
        observed_total = usage["total_tokens"]
        total_tokens += observed_total or (
            usage["prompt_tokens_total"] + usage["completion_tokens"]
        )
    return {
        "available": reported_calls > 0,
        "complete": missing_usage_calls == 0,
        "calls": calls,
        "reported_calls": reported_calls,
        "missing_usage_calls": missing_usage_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "providers": sorted(providers),
        "models": sorted(models),
    }


def _aggregate_memory_usage(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    available = False
    frameworks: set[str] = set()
    for evaluation in evaluations:
        usage = normalize_memory_internal_usage(
            evaluation.get("memory_internal_usage")
        )
        available = available or bool(usage["available"])
        calls += int(usage["calls"])
        prompt_tokens += int(usage["prompt_tokens"])
        completion_tokens += int(usage["completion_tokens"])
        total_tokens += int(usage["total_tokens"])
        framework = str(usage.get("framework") or "").strip()
        if framework:
            frameworks.add(framework)
    return {
        "available": available,
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "frameworks": sorted(frameworks),
    }


def _sum_usage(*components: dict[str, Any]) -> dict[str, int]:
    return {
        key: sum(int(component.get(key, 0) or 0) for component in components)
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")
    }


def _build_execution_timing_report(
    run_dir: Path,
    evaluations: list[dict[str, Any]],
    attempt: Attempt,
) -> dict[str, Any]:
    measured_at = datetime.now(timezone.utc)
    started_at = datetime.fromisoformat(attempt.started_at)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    execution_seconds = max(0.0, (measured_at - started_at).total_seconds())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": {
            "name": "active_attempt_through_reporting",
            "includes": "prediction, judging, evaluation and report generation",
            "excludes": "CLI preflight and post-report artifact publication/verification",
        },
        "attempt": {
            "attempt_id": attempt.attempt_id,
            "resume": attempt.resume,
            "started_at": started_at.isoformat(),
            "measured_at": measured_at.isoformat(),
            "execution_seconds": round(execution_seconds, 6),
        },
        "pipeline": build_timing_report(_load_pipeline_timing_records(run_dir, evaluations)),
    }


def _load_pipeline_timing_records(
    run_dir: Path,
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prediction_timing_paths = sorted((run_dir / "items").glob("*/timing.json"))
    prediction_timing_paths.extend(
        sorted((run_dir / "items").glob("*/timing/*.json"))
    )
    judgment_timing_paths = sorted((run_dir / "judgments" / "timing").glob("*.json"))
    judgment_timing_paths.extend(sorted((run_dir / "judgments" / "timing").glob("*/*.json")))

    records = (
        [_load_json_dict(path) for path in prediction_timing_paths]
        if prediction_timing_paths
        else list(evaluations)
    )
    records.extend(_load_json_dict(path) for path in judgment_timing_paths)
    return records


def primary_judge_report(
    evaluations: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    scores = [float(item.get("score", 0.0)) for item in evaluations if item.get("judgment")]
    passes = sum(
        1
        for item in evaluations
        if str(item.get("judgment", "")).upper() in {"CORRECT", "PASS"}
    )
    count = len(scores)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": metadata["benchmark"],
        "evaluator": "primary_judge_score",
        "evaluator_version": "primary-judge-v1",
        "status": "COMPLETED" if count == len(evaluations) else "FAILED",
        "metrics": {
            "overall": {
                "judged_count": count,
                "avg_judge_score": (sum(scores) / count) if count else 0.0,
                "judge_pass_rate": (passes / count) if count else 0.0,
            }
        },
    }


def _multi_judge_report(
    evaluations: list[dict[str, Any]], primary_judge_id: str | None,
) -> dict[str, Any]:
    if not evaluations or primary_judge_id is None:
        raise StateError("V3 multi-judge report requires evaluated items and a primary judge.")
    judge_ids = tuple(evaluations[0]["judges"])
    scores: dict[str, dict[str, Any]] = {}
    for judge_id in judge_ids:
        values = [float(item["judges"][judge_id]["score"]) for item in evaluations]
        passes = sum(
            item["judges"][judge_id]["judgment"] == "CORRECT"
            for item in evaluations
        )
        scores[judge_id] = {
            "judged_count": len(values),
            "avg_judge_score": sum(values) / len(values),
            "judge_pass_rate": passes / len(values),
        }
    discordant_ids = [
        str(item["question_id"])
        for item in evaluations
        if len({entry["judgment"] for entry in item["judges"].values()}) > 1
    ]
    agreement = (
        {"status": "COMPLETED", "exact_agreement_rate":
         (len(evaluations) - len(discordant_ids)) / len(evaluations),
         "discordant_query_ids": discordant_ids}
        if len(judge_ids) > 1
        else {"status": "NOT_APPLICABLE", "reason": "Only one judge is configured."}
    )
    return {
        "primary_judge_id": primary_judge_id,
        "primary": scores[primary_judge_id],
        "secondary": {judge_id: scores[judge_id] for judge_id in judge_ids if judge_id != primary_judge_id},
        "agreement": agreement,
    }


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def requirement_to_dict(requirement: EvaluationRequirement) -> dict[str, Any]:
    return {
        "name": requirement.name,
        "required": requirement.required,
        "status": requirement.status,
        "not_applicable_reason": requirement.not_applicable_reason,
    }


def _normalized_judgment(judgment: dict[str, Any]) -> str:
    label = str(judgment.get("judgment", judgment.get("label", ""))).strip().upper()
    if label in {"CORRECT", "PASS"}:
        return "CORRECT"
    if label in {"WRONG", "FAIL"}:
        return "WRONG"
    raise StateError(f"Invalid judge label: {label!r}")


def _evaluator_versions(reports: dict[str, Any]) -> dict[str, str]:
    raw_versions = reports.get("evaluator_versions")
    if isinstance(raw_versions, dict):
        return {str(key): str(value) for key, value in raw_versions.items()}
    return {}


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise StateError(f"Expected JSON object at {path}")
    return payload


def _load_json_list(
    path: Path,
    *,
    default: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not path.exists() and default is not None:
        return default
    payload = read_json(path)
    if not isinstance(payload, list):
        raise StateError(f"Expected JSON array at {path}")
    return _ensure_dict_list(payload)


def _ensure_dict_list(payload: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise StateError("Expected list of JSON objects.")
        result.append(item)
    return result


def _check_cancel(cancel_check: Callable[[], None] | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _emit_phase_event(
    events: JsonEventLogger | None,
    event: str,
    manifest: RunManifest,
    attempt: Attempt,
    phase: str,
) -> None:
    if events is None:
        return
    inputs = manifest.fingerprint_inputs
    events.event(
        event,
        "Lifecycle phase boundary reached.",
        run_id=manifest.run_id,
        attempt_id=attempt.attempt_id,
        benchmark=str(inputs.get("benchmark", "")),
        framework=str(inputs.get("framework", "")),
        phase=phase,
        outcome=(
            "failed"
            if event.endswith("failed")
            else "completed"
            if event.endswith(("completed", "written"))
            else "started"
        ),
    )
