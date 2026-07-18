import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import BenchmarkUnit
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json
from dmf_bench.contracts import sha256_file
from dmf_bench.evaluation import locomo as locomo_eval
from dmf_bench.evaluation import longmemeval as longmemeval_eval
from dmf_bench.evaluation import OfflineFullLifecycleRunner, OfflineLifecycleFinalizer
from dmf_bench.evaluation.finalizer import InjectedTerminalInterrupt
from dmf_bench.runner import (
    InjectedInterrupt,
    LoCoMoPredictOnlyRunner,
    LongMemEvalPredictOnlyRunner,
)
from dmf_bench.state import StateError
from locomo.evaluate_rigorous import evaluate_flat as legacy_locomo_flat
from longmemeval.evaluate_rigorous import evaluate_flat as legacy_longmemeval_flat


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


class FakeLongMemEvalFramework:
    name = "dmf"

    def cleanup_unit(self, _unit: BenchmarkUnit, _question: dict[str, Any], _config: dict[str, Any]) -> None:
        return None

    def prepare_unit(
        self,
        _unit: BenchmarkUnit,
        _question: dict[str, Any],
        _config: dict[str, Any],
    ) -> dict[str, Any]:
        return {"barrier": True}

    def retrieve(
        self,
        _unit: BenchmarkUnit,
        question: dict[str, Any],
        _config: dict[str, Any],
        _prepared: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "search_results": [{"metadata": {"source_unit_ids": list(question["answer_session_ids"])}}],
            "cutoff_label": "top_1",
        }


class FakeLoCoMoFramework:
    name = "mem0"

    def cleanup_unit(
        self,
        _unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        _config: dict[str, Any],
    ) -> None:
        return None

    def prepare_unit(
        self,
        _unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        _config: dict[str, Any],
    ) -> dict[str, Any]:
        return {"barrier": True}

    def retrieve_question(
        self,
        _unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        question: Any,
        _config: dict[str, Any],
        _prepared: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "search_results": [{"metadata": {"source_unit_ids": list(question.qa_item.get("evidence", []))}}],
            "cutoff_label": "top_1",
        }


class FakeAnswerer:
    name = "fake-answerer"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)

    def generate(self, _prompt: str, _metadata: dict[str, Any]) -> dict[str, Any]:
        answer = self.answers.pop(0) if self.answers else "fallback"
        return {
            "generated_answer": answer,
            "answerer_provider": "fake",
            "answerer_model": "fixture-model",
            "answerer_usage": {"total_tokens": 1},
        }


class FakeJudge:
    name = "fake-judge"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge(self, question: str, expected: str, generated: str) -> dict[str, Any]:
        self.calls.append(question)
        return {
            "judgment": "CORRECT" if expected.lower() in generated.lower() or generated.lower() in expected.lower() else "WRONG",
            "score": 1.0 if expected.lower() in generated.lower() or generated.lower() in expected.lower() else 0.0,
            "reason": "offline fixture judgment",
            "judge_provider": "fake",
            "judge_model": "fixture-judge",
            "judge_fingerprint": "fixture-judge-v1",
        }


def make_longmemeval_config(tmp_path: Path) -> dict[str, Any]:
    dataset_path = tmp_path / "longmemeval-mini.json"
    shutil.copy2(FIXTURE_DIR / "longmemeval-mini.json", dataset_path)
    return {
        "schema_version": 1,
        "experiment_id": "lifecycle-longmemeval",
        "benchmark": "longmemeval",
        "framework": "dmf",
        "protocol": "strict",
        "runtime": {"root": str(tmp_path), "runs_dir": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")},
        "dataset": {"name": "longmemeval", "path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "selection": {"ordered_item_ids": ["lme-001", "lme-002"], "filters": {}, "seed": 7},
        "models": {
            "answerer": {"provider": "fake", "requested_model": "fixture-model", "parameters": {"temperature": 0}},
            "judge": {"provider": "fake", "requested_model": "fixture-judge", "parameters": {"temperature": 0}},
        },
        "evaluation": {"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
        "artifact_store": {"type": "local", "uri": str(tmp_path / "runs")},
    }


def make_locomo_config(tmp_path: Path) -> dict[str, Any]:
    dataset_path = tmp_path / "locomo-mini.json"
    shutil.copy2(FIXTURE_DIR / "locomo-mini.json", dataset_path)
    return {
        "schema_version": 1,
        "experiment_id": "lifecycle-locomo",
        "benchmark": "locomo",
        "framework": "mem0",
        "protocol": "strict",
        "runtime": {"root": str(tmp_path), "runs_dir": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")},
        "dataset": {"name": "locomo", "path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "selection": {"ordered_item_ids": ["conversation-0001"], "filters": {"categories": [1, 2]}, "seed": 7},
        "models": {
            "answerer": {"provider": "fake", "requested_model": "fixture-model", "parameters": {"temperature": 0}},
            "judge": {"provider": "fake", "requested_model": "fixture-judge", "parameters": {"temperature": 0}},
        },
        "evaluation": {"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
        "artifact_store": {"type": "local", "uri": str(tmp_path / "runs")},
    }


def test_full_lifecycle_offline_longmemeval_completes_and_publishes(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(["almonds", "the Rome trip"]),
    ).run(config)
    judge = FakeJudge()

    result = OfflineLifecycleFinalizer(artifact_store=store, judge=judge).finalize("lifecycle-longmemeval")

    run_dir = tmp_path / "runs" / "lifecycle-longmemeval"
    assert result.state == "COMPLETED"
    assert result.judged_count == 2
    assert len(judge.calls) == 2
    assert (run_dir / "final" / "COMPLETED.json").exists()
    assert (run_dir / "final" / "reports" / "summary.json").exists()
    assert (run_dir / "final" / "COMPLETED.json").stat().st_mtime_ns >= (
        run_dir / "final" / "manifest.json"
    ).stat().st_mtime_ns
    summary = read_json(run_dir / "reports" / "summary.json")
    assert summary["counts"] == {"expected": 2, "evaluated": 2, "excluded": 0, "failed": 0}
    assert summary["evaluator_versions"]["primary_judge_score"] == "primary-judge-v1"
    rigorous = read_json(run_dir / "evaluations" / "rigorous_report.json")
    assert rigorous["status"] == "COMPLETED"
    assert rigorous["metrics"]["overall"]["exact_match"] == 1.0


def test_full_lifecycle_offline_locomo_marks_mem0_ablation_not_applicable(tmp_path: Path) -> None:
    config = make_locomo_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LoCoMoPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLoCoMoFramework(),
        answerer=FakeAnswerer(["Pixel", "near the kitchen window"]),
    ).run(config)

    result = OfflineLifecycleFinalizer(artifact_store=store, judge=FakeJudge()).finalize("lifecycle-locomo")

    run_dir = tmp_path / "runs" / "lifecycle-locomo"
    assert result.state == "COMPLETED"
    ablation = read_json(run_dir / "evaluations" / "ablation_report.json")
    assert ablation["status"] == "NOT_APPLICABLE"
    assert "Mem0" in ablation["reason"]
    assert (run_dir / "final" / "COMPLETED.json").exists()


def test_finalizer_refuses_incomplete_expected_prediction_set(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    with pytest.raises(InjectedInterrupt, match="after prediction commit"):
        LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=FakeLongMemEvalFramework(),
            answerer=FakeAnswerer(["almonds"]),
        ).run(config, interrupt_at="after-prediction-commit")

    with pytest.raises(StateError, match="expected prediction set is incomplete"):
        OfflineLifecycleFinalizer(artifact_store=store, judge=FakeJudge()).finalize("lifecycle-longmemeval")


def test_terminal_resume_does_not_regenerate_valid_judgments(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(["almonds", "the Rome trip"]),
    ).run(config)
    first_judge = FakeJudge()

    with pytest.raises(InjectedTerminalInterrupt, match="after judge"):
        OfflineLifecycleFinalizer(artifact_store=store, judge=first_judge).finalize(
            "lifecycle-longmemeval",
            interrupt_at="after-judge",
        )

    second_judge = FakeJudge()
    result = OfflineLifecycleFinalizer(artifact_store=store, judge=second_judge).finalize(
        "lifecycle-longmemeval"
    )

    assert result.state == "COMPLETED"
    assert len(first_judge.calls) == 2
    assert second_judge.calls == []


def test_required_evaluator_failure_sets_failed_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(["almonds", "the Rome trip"]),
    ).run(config)

    def fail_required(_evaluations: list[dict[str, Any]], _metadata: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("simulated evaluator failure")

    monkeypatch.setattr(
        "dmf_bench.evaluation.longmemeval.rigorous_report",
        fail_required,
    )

    with pytest.raises(StateError, match="Finalization failed"):
        OfflineLifecycleFinalizer(artifact_store=store, judge=FakeJudge()).finalize("lifecycle-longmemeval")

    status = read_json(tmp_path / "runs" / "lifecycle-longmemeval" / "run-status.json")
    assert status["state"] == "FAILED_EVALUATION"


def test_internal_evaluator_adapters_match_legacy_pure_functions() -> None:
    locomo_items = [
        {
            "generated_answer": "Pixel",
            "ground_truth_answer": "Pixel",
            "category_name": "multi-hop",
            "evidence": ["D1:1"],
            "retrieval": {
                "memories_evaluated": 1,
                "search_results": [{"metadata": {"source_unit_id": "D1:1"}}],
            },
        }
    ]
    longmemeval_items = [
        {
            "generated_answer": "almonds",
            "ground_truth_answer": "almonds",
            "question_type": "single-session-user",
            "answer_session_ids": ["session-a"],
            "retrieval": {
                "memories_evaluated": 1,
                "search_results": [{"metadata": {"source_unit_id": "session-a"}}],
            },
        }
    ]
    metadata = {"top_k": 1, "framework": "dmf"}

    locomo_cutoff, locomo_metrics = legacy_locomo_flat(locomo_items, metadata)
    longmemeval_cutoff, longmemeval_metrics = legacy_longmemeval_flat(longmemeval_items, metadata)

    assert locomo_eval.rigorous_report(locomo_items, metadata)["cutoff_label"] == locomo_cutoff
    assert locomo_eval.rigorous_report(locomo_items, metadata)["metrics"] == locomo_metrics
    assert longmemeval_eval.rigorous_report(longmemeval_items, metadata)["cutoff_label"] == longmemeval_cutoff
    assert longmemeval_eval.rigorous_report(longmemeval_items, metadata)["metrics"] == longmemeval_metrics


def test_full_lifecycle_runner_executes_prediction_to_completed_in_one_call(
    tmp_path: Path,
) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    prediction_runner = LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(["almonds", "the Rome trip"]),
    )

    result = OfflineFullLifecycleRunner(
        prediction_runner=prediction_runner,
        artifact_store=store,
        judge=FakeJudge(),
    ).run(config)

    assert result.state == "COMPLETED"
    assert (tmp_path / "runs" / "lifecycle-longmemeval" / "final" / "COMPLETED.json").exists()
