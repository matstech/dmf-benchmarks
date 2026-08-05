"""Offline judge/evaluate/report/publish lifecycle finalization."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dmf_bench.frameworks.mem0_runtime import normalize_memory_internal_usage
from dmf_bench.reporting.results import build_timing_report, normalize_answerer_usage
from dmf_bench.adapters.base import JudgeAdapter, JudgeRequest
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import (
    Attempt,
    EVALUATION_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    LifecycleCheckpoint,
    PREDICTION_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    RunManifest,
    RunStatus,
    SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION,
    hash_canonical_json,
    sha256_file,
)
from dmf_bench.evaluation.registry import EvaluationRequirement, evaluation_plan_for
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
from . import longmemeval as longmemeval_evaluation


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
        judge: JudgeAdapter,
        metrics: BenchmarkMetrics | None = None,
        events: JsonEventLogger | None = None,
        evaluation_plans: dict[tuple[str, str], tuple[EvaluationRequirement, ...]] | None = None,
        prediction_loaders: dict[str, PredictionLoader] | None = None,
        evaluators: dict[str, Evaluator] | None = None,
    ) -> None:
        self.artifact_store = artifact_store
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
    ) -> FinalizationResult:
        self.artifact_store.run_dir(run_id)
        lock_path = self.artifact_store.runs_dir / ".locks" / f"{run_id}.lock"
        lock = nullcontext() if _lock_held else RunLock(lock_path)
        with lock:
            return self._finalize_locked(
                run_id,
                interrupt_at=interrupt_at,
                cancel_check=cancel_check,
                attempt=_attempt,
            )

    def _finalize_locked(
        self,
        run_id: str,
        *,
        interrupt_at: str | None,
        cancel_check: Callable[[], None] | None,
        attempt: Attempt | None,
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
        self._validate_prediction_set(predictions, terminal_ids)
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
                evaluations, judgment_paths, item_digests = self._judge_predictions(
                    run_dir,
                    manifest,
                    predictions,
                    metadata,
                    attempt=active_attempt,
                    interrupt_at=interrupt_at,
                    cancel_check=cancel_check,
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
                        *(
                            lifecycle_checkpoint_path(
                                run_dir,
                                "JUDGING",
                                item_id=item_id,
                            )
                            for item_id in terminal_ids
                        ),
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
                self._validate_evaluation_summary(reports)

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
                    metadata={"report_schema_version": REPORT_SCHEMA_VERSION},
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
            "schema_version": REPORT_SCHEMA_VERSION,
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
    ) -> None:
        if any(
            prediction.get("schema_version") != PREDICTION_SCHEMA_VERSION
            for prediction in predictions
        ):
            raise StateError(
                "Evaluation refused: prediction schema_version must be "
                f"{PREDICTION_SCHEMA_VERSION}; v1 state is not supported."
            )
        observed = tuple(str(item.get("question_id", "")) for item in predictions)
        if not all(observed) or observed != expected_item_ids or len(set(observed)) != len(observed):
            raise StateError(
                "Evaluation refused: prediction question set/order does not match manifest."
            )

    def _judge_identity(self, manifest: RunManifest) -> tuple[str, str, str]:
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
    ) -> tuple[list[dict[str, Any]], tuple[Path, ...], dict[str, str]]:
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
            if interrupt_at == "after-first-judgment" and offset == 0:
                raise InjectedTerminalInterrupt("Interrupted after first judgment commit.")
        return judged, tuple(judgment_paths), item_digests

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
        requirements = evaluation_plan_for(
            benchmark=metadata["benchmark"],
            framework=metadata["framework"],
            plans=self.evaluation_plans,
        )
        reports: dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
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
                artifact = self._run_evaluator(requirement, evaluations, metadata)
                _check_cancel(cancel_check)
                self._validate_evaluator_output(
                    artifact,
                    requirement=requirement,
                    evaluator_version=evaluator_version,
                )
                write_json_atomic(output_path, artifact)
                item_checkpoint = self._commit_phase_checkpoint(
                    run_dir,
                    manifest,
                    attempt=attempt,
                    phase="EVALUATING",
                    input_fingerprint=item_input,
                    expected_item_ids=terminal_ids,
                    artifacts=(output_path,),
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
        requirements = evaluation_plan_for(
            benchmark=metadata["benchmark"],
            framework=metadata["framework"],
            plans=self.evaluation_plans,
        )
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

    def _validate_evaluator_output(
        self,
        artifact: dict[str, Any],
        *,
        requirement: EvaluationRequirement,
        evaluator_version: str,
    ) -> None:
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema_version") != EVALUATION_SCHEMA_VERSION
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

    def _validate_evaluation_summary(self, reports: dict[str, Any]) -> None:
        if reports.get("schema_version") != EVALUATION_SCHEMA_VERSION:
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
    ) -> tuple[Path, ...]:
        usage = _build_usage_report(evaluations)
        timing = _build_execution_timing_report(run_dir, evaluations, attempt)
        summary = {
            "schema_version": REPORT_SCHEMA_VERSION,
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
            },
        }
        summary_path = run_dir / "reports" / "summary.json"
        markdown_path = run_dir / "reports" / "summary.md"
        usage_path = run_dir / "reports" / "usage.json"
        timing_path = run_dir / "reports" / "timing.json"
        write_json_atomic(usage_path, usage)
        write_json_atomic(timing_path, timing)
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
        )
        markdown_path.parent.mkdir(exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
        return summary_path, markdown_path, usage_path, timing_path

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
        judge: JudgeAdapter,
        metrics: BenchmarkMetrics | None = None,
        events: JsonEventLogger | None = None,
    ) -> None:
        self.prediction_runner = prediction_runner
        self.artifact_store = artifact_store
        self.metrics = metrics
        self.finalizer = OfflineLifecycleFinalizer(
            artifact_store=artifact_store,
            judge=judge,
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
    judge = _aggregate_provider_usage(evaluations, role="judge")
    memory_internal = _aggregate_memory_usage(evaluations)
    benchmark_total = _sum_usage(memory_internal, answerer)
    evaluation_total = _sum_usage(judge)
    overall_total = _sum_usage(benchmark_total, evaluation_total)
    return {
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
