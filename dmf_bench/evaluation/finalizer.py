"""Offline judge/evaluate/report/publish lifecycle finalization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dmf_bench.adapters.base import JudgeAdapter
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import RunManifest, RunStatus, sha256_file
from dmf_bench.evaluation.registry import EvaluationRequirement, evaluation_plan_for
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.state import StateError, load_manifest, plan_resume

from . import locomo as locomo_evaluation
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
    """Complete terminal phases for a local prediction run without subprocess calls."""

    def __init__(
        self,
        *,
        artifact_store: LocalArtifactStore,
        judge: JudgeAdapter,
        metrics: BenchmarkMetrics | None = None,
        evaluation_plans: dict[tuple[str, str, str], tuple[EvaluationRequirement, ...]] | None = None,
        prediction_loaders: dict[str, PredictionLoader] | None = None,
        evaluators: dict[str, Evaluator] | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.judge = judge
        self.metrics = metrics
        self.evaluation_plans = evaluation_plans
        self.prediction_loaders = dict(prediction_loaders or {})
        self.evaluators = dict(evaluators or {})

    def finalize(
        self,
        run_id: str,
        *,
        interrupt_at: str | None = None,
    ) -> FinalizationResult:
        run_dir = self.artifact_store.run_dir(run_id)
        manifest = load_manifest(run_dir)
        if (run_dir / "final" / "COMPLETED.json").exists():
            evaluations = _load_json_list(run_dir / "evaluations" / "evaluations.json", default=[])
            return FinalizationResult(
                run_id=run_id,
                state="COMPLETED",
                judged_count=sum(1 for item in evaluations if item.get("judgment")),
                evaluated_count=len(evaluations),
                excluded_count=0,
                failed_count=0,
                final_completion_path=run_dir / "final" / "COMPLETED.json",
            )

        self._ensure_predictions_complete(run_dir, manifest)
        metadata = self._metadata_from_manifest(manifest)

        try:
            self._write_status(run_dir, manifest, state="JUDGING", phase="JUDGING", committed=0)
            predictions = self._load_predictions(run_dir, manifest)
            evaluations = self._judge_predictions(run_dir, predictions, metadata)
            if interrupt_at == "after-judge":
                raise InjectedTerminalInterrupt("Interrupted after judge phase.")

            self._write_status(
                run_dir,
                manifest,
                state="EVALUATING",
                phase="EVALUATING",
                committed=len(evaluations),
            )
            reports = self._evaluate(run_dir, evaluations, metadata)
            if interrupt_at == "after-evaluate":
                raise InjectedTerminalInterrupt("Interrupted after evaluation phase.")

            self._write_status(
                run_dir,
                manifest,
                state="REPORTING",
                phase="REPORTING",
                committed=len(evaluations),
            )
            self._write_reports(run_dir, manifest, metadata, evaluations, reports)
            if self.metrics is not None:
                self.metrics.write_snapshot(run_dir)
            if interrupt_at == "after-report":
                raise InjectedTerminalInterrupt("Interrupted after report phase.")

            self._write_status(
                run_dir,
                manifest,
                state="PUBLISHING",
                phase="PUBLISHING",
                committed=len(evaluations),
            )
            publish_started_at = time.perf_counter()
            staged = self.artifact_store.stage(run_id, run_dir)
            if interrupt_at == "after-publish-stage":
                raise InjectedTerminalInterrupt("Interrupted after publish stage.")

            self._write_status(
                run_dir,
                manifest,
                state="VERIFYING",
                phase="VERIFYING",
                committed=len(evaluations),
            )
            receipt = self.artifact_store.verify(staged)
            marker = self.artifact_store.commit(staged, receipt)
            if self.metrics is not None:
                self.metrics.record_artifact_publish(
                    backend="local-only",
                    outcome="completed",
                    seconds=time.perf_counter() - publish_started_at,
                )
        except InjectedTerminalInterrupt:
            raise
        except StateError:
            raise
        except Exception as exc:
            if self.metrics is not None:
                self.metrics.record_artifact_publish(
                    backend="local-only",
                    outcome="failed",
                    seconds=0.0,
                )
            self._write_status(run_dir, manifest, state="FAILED_EVALUATION", phase="EVALUATING", committed=0)
            raise StateError(f"Finalization failed: {exc}") from exc

        return FinalizationResult(
            run_id=run_id,
            state=marker.state,
            judged_count=sum(1 for item in evaluations if item.get("judgment")),
            evaluated_count=len(evaluations),
            excluded_count=0,
            failed_count=0,
            final_completion_path=run_dir / "final" / "COMPLETED.json",
        )

    def _ensure_predictions_complete(self, run_dir: Path, manifest: RunManifest) -> None:
        resume_plan = plan_resume(run_dir)
        if resume_plan.next_phase == "COMPLETED":
            return
        if resume_plan.next_phase != "JUDGING":
            raise StateError(
                "Evaluation refused: expected prediction set is incomplete; "
                f"restart units: {list(resume_plan.restart_unit_ids)}"
            )

    def _metadata_from_manifest(self, manifest: RunManifest) -> dict[str, Any]:
        inputs = manifest.fingerprint_inputs
        return {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "benchmark": str(inputs.get("benchmark", "")),
            "framework": str(inputs.get("framework", "")),
            "protocol": str(inputs.get("protocol", "")),
            "protocol_mode": str(inputs.get("protocol", "")),
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
            expected_question_ids = tuple(
                str(item)
                for item in manifest.fingerprint_inputs.get("expected_question_ids", ())
            )
            if expected_question_ids and tuple(item["question_id"] for item in predictions) != expected_question_ids:
                raise StateError("Evaluation refused: LoCoMo expected question set mismatch.")
            return predictions
        raise StateError(f"Unsupported benchmark for prediction loading: {benchmark!r}")

    def _judge_predictions(
        self,
        run_dir: Path,
        predictions: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        judged: list[dict[str, Any]] = []
        for prediction in predictions:
            question_id = str(prediction["question_id"])
            judgment_path = run_dir / "judgments" / f"{question_id}.json"
            if judgment_path.exists():
                judged.append(_load_json_dict(judgment_path))
                continue

            judgment = self.judge.judge(
                str(prediction.get("question", "")),
                str(prediction.get("ground_truth_answer", "")),
                str(prediction.get("generated_answer", "")),
            )
            judged_item = {
                **prediction,
                "protocol_mode": metadata["protocol"],
                "judgment": _normalized_judgment(judgment),
                "score": float(judgment.get("score", 0.0)),
                "reason": str(judgment.get("reason", "")),
                "judge_provider": str(judgment.get("judge_provider", getattr(self.judge, "name", "unknown"))),
                "judge_model": str(judgment.get("judge_model", getattr(self.judge, "name", "unknown"))),
                "judge_fingerprint": str(judgment.get("judge_fingerprint", getattr(self.judge, "name", "unknown"))),
            }
            write_json_atomic(judgment_path, judged_item)
            judged.append(judged_item)

        write_json_atomic(run_dir / "evaluations" / "evaluations.json", judged)
        return judged

    def _evaluate(
        self,
        run_dir: Path,
        evaluations: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        benchmark = metadata["benchmark"]
        framework = metadata["framework"]
        protocol = metadata["protocol"]
        requirements = evaluation_plan_for(
            benchmark=benchmark,
            protocol=protocol,
            framework=framework,
            plans=self.evaluation_plans,
        )
        reports: dict[str, Any] = {
            "schema_version": 1,
            "benchmark": benchmark,
            "protocol": protocol,
            "framework": framework,
            "requirements": [requirement_to_dict(item) for item in requirements],
            "artifacts": {},
            "evaluator_versions": {},
        }
        for requirement in requirements:
            artifact = self._run_evaluator(requirement, evaluations, metadata)
            output_path = run_dir / "evaluations" / f"{requirement.name}.json"
            write_json_atomic(output_path, artifact)
            reports["artifacts"][requirement.name] = {
                "path": output_path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(output_path),
                "status": artifact["status"],
            }
            if artifact.get("evaluator_version"):
                reports["evaluator_versions"][requirement.name] = str(artifact["evaluator_version"])
            if requirement.required and artifact["status"] != "COMPLETED":
                self._write_status(
                    run_dir,
                    load_manifest(run_dir),
                    state="FAILED_EVALUATION",
                    phase="EVALUATING",
                    committed=len(evaluations),
                )
                raise StateError(f"Required evaluator failed: {requirement.name}")

        write_json_atomic(run_dir / "evaluations" / "evaluation-summary.json", reports)
        return reports

    def _run_evaluator(
        self,
        requirement: EvaluationRequirement,
        evaluations: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if requirement.not_applicable_reason:
            return {
                "schema_version": 1,
                "evaluator": requirement.name,
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
        return {
            "schema_version": 1,
            "evaluator": requirement.name,
            "status": "NOT_APPLICABLE",
            "reason": "Evaluator is not implemented for this benchmark/protocol.",
        }

    def _write_reports(
        self,
        run_dir: Path,
        manifest: RunManifest,
        metadata: dict[str, Any],
        evaluations: list[dict[str, Any]],
        reports: dict[str, Any],
    ) -> None:
        summary = {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "scientific_fingerprint": manifest.scientific_fingerprint,
            "benchmark": metadata["benchmark"],
            "protocol": metadata["protocol"],
            "framework": metadata["framework"],
            "counts": {
                "expected": len(evaluations),
                "evaluated": len(evaluations),
                "excluded": 0,
                "failed": 0,
            },
            "evaluator_versions": _evaluator_versions(reports),
            "evaluation_artifacts": reports["artifacts"],
        }
        write_json_atomic(run_dir / "reports" / "summary.json", summary)
        markdown = (
            f"# dmf-bench run {manifest.run_id}\n\n"
            f"- benchmark: {metadata['benchmark']}\n"
            f"- framework: {metadata['framework']}\n"
            f"- protocol: {metadata['protocol']}\n"
            f"- expected: {len(evaluations)}\n"
            f"- evaluated: {len(evaluations)}\n"
            f"- excluded: 0\n"
            f"- failed: 0\n"
        )
        (run_dir / "reports").mkdir(exist_ok=True)
        (run_dir / "reports" / "summary.md").write_text(markdown, encoding="utf-8")

    def _write_status(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        state: str,
        phase: str,
        committed: int,
    ) -> None:
        status = RunStatus(
            run_id=manifest.run_id,
            state=state,
            phase=phase,
            expected=len(manifest.expected_item_ids),
            committed=committed,
        )
        write_json_atomic(run_dir / "run-status.json", status.to_dict())
        if self.metrics is not None:
            inputs = manifest.fingerprint_inputs
            self.metrics.record_run_status(
                benchmark=str(inputs.get("benchmark", "")),
                framework=str(inputs.get("framework", "")),
                protocol=str(inputs.get("protocol", "")),
                phase=phase,
                expected=len(manifest.expected_item_ids),
                committed=committed,
            )


class OfflineFullLifecycleRunner:
    """Compose a prediction runner and the terminal finalizer as one internal run."""

    def __init__(
        self,
        *,
        prediction_runner: Any,
        artifact_store: LocalArtifactStore,
        judge: JudgeAdapter,
        metrics: BenchmarkMetrics | None = None,
    ) -> None:
        self.prediction_runner = prediction_runner
        self.finalizer = OfflineLifecycleFinalizer(
            artifact_store=artifact_store,
            judge=judge,
            metrics=metrics,
        )

    def run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
    ) -> FinalizationResult:
        prediction_result = self.prediction_runner.run(
            config,
            run_id=run_id,
            resume=resume,
        )
        return self.finalizer.finalize(prediction_result.run_id)


class InjectedTerminalInterrupt(RuntimeError):
    """Raised by tests to stop finalization at a terminal phase boundary."""


def primary_judge_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    scores = [float(item.get("score", 0.0)) for item in evaluations if item.get("judgment")]
    passes = sum(1 for item in evaluations if str(item.get("judgment", "")).upper() in {"CORRECT", "PASS"})
    count = len(scores)
    return {
        "schema_version": 1,
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


def _load_json_list(path: Path, *, default: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
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
