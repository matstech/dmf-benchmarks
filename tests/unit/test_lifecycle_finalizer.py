import json
import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import AnswererRequest, BenchmarkUnit, FrameworkRunContext, JudgeRequest, RetrievalResult
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import sha256_file
from dmf_bench.benchmarks.locomo import evaluation as locomo_eval
from dmf_bench.benchmarks.longmemeval import evaluation as longmemeval_eval
from dmf_bench.evaluation import OfflineFullLifecycleRunner, OfflineLifecycleFinalizer
from dmf_bench.evaluation.finalizer import InjectedTerminalInterrupt
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.runner import (
    InjectedInterrupt,
    LoCoMoPredictOnlyRunner,
    LongMemEvalPredictOnlyRunner,
)
from dmf_bench.state import StateError, load_lifecycle_checkpoint, plan_resume
from dmf_bench.benchmarks.locomo.rigorous import evaluate_flat as pure_locomo_flat
from dmf_bench.benchmarks.longmemeval.rigorous import evaluate_flat as pure_longmemeval_flat


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


class FakeLongMemEvalFramework:
    name = "dmf"

    def cleanup_unit(
        self,
        _unit: BenchmarkUnit,
        _question: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        del run_context
        return None

    def prepare_unit(
        self,
        _unit: BenchmarkUnit,
        _question: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        del run_context
        return {"barrier": True}

    def retrieve(
        self,
        _unit: BenchmarkUnit,
        question: dict[str, Any],
        _config: dict[str, Any],
        _prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        del run_context
        return RetrievalResult(
            search_results=(
                {"metadata": {"source_unit_ids": list(question["answer_session_ids"])}},
            ),
            cutoff_label="top_1",
            memories_evaluated=1,
        )


class FakeLoCoMoFramework:
    name = "mem0"

    def cleanup_unit(
        self,
        _unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        del run_context
        return None

    def prepare_unit(
        self,
        _unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        del run_context
        return {"barrier": True}

    def retrieve_question(
        self,
        _unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        question: Any,
        _config: dict[str, Any],
        _prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        del run_context
        return RetrievalResult(
            search_results=(
                {"metadata": {"source_unit_ids": list(question.qa_item.get("evidence", []))}},
            ),
            cutoff_label="top_1",
            memories_evaluated=1,
        )


class FakeAnswerer:
    name = "fake-answerer"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)

    def generate(self, _request: AnswererRequest) -> dict[str, Any]:
        answer = self.answers.pop(0) if self.answers else "fallback"
        return {
            "generated_answer": answer,
            "answerer_provider": "fake",
            "answerer_requested_model": "requested-fixture-model",
            "answerer_model": "fixture-model",
            "answerer_finish_reason": "stop",
            "answerer_usage": {"total_tokens": 1},
        }


class FakeJudge:
    name = "fake-judge"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        prediction = request.prediction
        question = str(prediction.get("question", ""))
        expected = str(prediction.get("ground_truth_answer", ""))
        generated = str(prediction.get("generated_answer", ""))
        self.calls.append(question)
        return {
            "judgment": "CORRECT" if expected.lower() in generated.lower() or generated.lower() in expected.lower() else "WRONG",
            "score": 1.0 if expected.lower() in generated.lower() or generated.lower() in expected.lower() else 0.0,
            "reason": "offline fixture judgment",
            "judge_provider": "fake",
            "judge_requested_model": "fixture-judge",
            "judge_model": "fixture-judge",
            "judge_finish_reason": "stop",
            "judge_usage": {"total_tokens": 2},
            "judge_fingerprint": judge_fingerprint(str(request.metadata["benchmark"])),
        }


class BlockingJudge(FakeJudge):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("timed out waiting to release blocking judge")
        return super().judge(request)


class FailingJudge(FakeJudge):
    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        del request
        raise RuntimeError("simulated judge failure")


def make_longmemeval_config(tmp_path: Path) -> dict[str, Any]:
    dataset_path = tmp_path / "longmemeval-mini.json"
    shutil.copy2(FIXTURE_DIR / "longmemeval-mini.json", dataset_path)
    return {
        "schema_version": 2,
        "experiment_id": "lifecycle-longmemeval",
        "benchmark": "longmemeval",
        "framework": "dmf",
        "runtime": {"root": str(tmp_path), "runs_dir": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")},
        "dataset": {"name": "longmemeval", "path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "selection": {"ordered_item_ids": ["lme-001", "lme-002"], "filters": {}, "seed": 7},
        "models": {
            "answerer": {"provider": "fake", "requested_model": "fixture-model", "parameters": {"temperature": 0, "max_tokens": 256}},
            "judge": {"provider": "fake", "requested_model": "fixture-judge", "parameters": {"temperature": 0, "max_tokens": 256}},
        },
        "evaluation": {"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
        "artifact_store": {"type": "local", "uri": str(tmp_path / "runs")},
    }


def make_locomo_config(tmp_path: Path) -> dict[str, Any]:
    dataset_path = tmp_path / "locomo-mini.json"
    shutil.copy2(FIXTURE_DIR / "locomo-mini.json", dataset_path)
    return {
        "schema_version": 2,
        "experiment_id": "lifecycle-locomo",
        "benchmark": "locomo",
        "framework": "mem0",
        "runtime": {"root": str(tmp_path), "runs_dir": str(tmp_path / "runs"), "cache_dir": str(tmp_path / "cache")},
        "dataset": {"name": "locomo", "path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "selection": {"ordered_item_ids": ["conversation-0001"], "filters": {"categories": [1, 2]}, "seed": 7},
        "models": {
            "answerer": {"provider": "fake", "requested_model": "fixture-model", "parameters": {"temperature": 0, "max_tokens": 256}},
            "judge": {"provider": "fake", "requested_model": "fixture-judge", "parameters": {"temperature": 0, "max_tokens": 256}},
        },
        "evaluation": {"required": ["primary_judge_score", "rigorous_report"], "optional": ["ablation_report"]},
        "artifact_store": {"type": "local", "uri": str(tmp_path / "runs")},
    }


def build_full_runner(
    root: Path,
    benchmark: str,
    *,
    answers: list[str],
    judge: FakeJudge | None = None,
) -> tuple[dict[str, Any], LocalArtifactStore, OfflineFullLifecycleRunner]:
    root.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(root / "runs")
    if benchmark == "longmemeval":
        config = make_longmemeval_config(root)
        prediction_runner: Any = LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=FakeLongMemEvalFramework(),
            answerer=FakeAnswerer(answers),
        )
    elif benchmark == "locomo":
        config = make_locomo_config(root)
        prediction_runner = LoCoMoPredictOnlyRunner(
            artifact_store=store,
            framework=FakeLoCoMoFramework(),
            answerer=FakeAnswerer(answers),
        )
    else:
        raise AssertionError(f"unsupported fixture benchmark: {benchmark}")
    return (
        config,
        store,
        OfflineFullLifecycleRunner(
            prediction_runner=prediction_runner,
            artifact_store=store,
            judge=judge or FakeJudge(),
        ),
    )


def scientific_artifact_snapshot(run_dir: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for directory in ("items", "judgments", "evaluations", "reports"):
        for path in sorted((run_dir / directory).rglob("*")):
            if path.is_file():
                relative = path.relative_to(run_dir).as_posix()
                if (
                    relative
                    in {
                        "reports/timing.json",
                        "reports/resources.json",
                        "reports/summary.md",
                    }
                    or relative.endswith("/timing.json")
                    or "/timing/" in relative
                ):
                    continue
                if path.suffix == ".json":
                    payload = read_json(path)
                    normalized = _without_operational_timing(payload)
                    result[relative] = json.dumps(
                        normalized,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                else:
                    result[relative] = path.read_bytes()
    return result


def _without_operational_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_operational_timing(item)
            for key, item in value.items()
            if key not in {"pipeline_timing", "timing"}
        }
    if isinstance(value, list):
        return [_without_operational_timing(item) for item in value]
    return value


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
    assert summary["report_artifacts"]["resources"] == "reports/resources.json"
    resources = read_json(run_dir / "reports" / "resources.json")
    assert resources["attempt"]["attempt_id"]
    assert resources["process"]["cpu_total_seconds"] >= 0
    assert resources["process"]["peak_rss_bytes"] > 0
    assert (run_dir / "final" / "reports" / "resources.json").is_file()
    rigorous = read_json(run_dir / "evaluations" / "rigorous_report.json")
    assert rigorous["status"] == "COMPLETED"
    assert rigorous["metrics"]["overall"]["exact_match"] == 1.0
    prediction = read_json(run_dir / "items" / "lme-001" / "prediction.json")
    assert prediction["answerer_requested_model"] == "requested-fixture-model"
    assert prediction["answerer_finish_reason"] == "stop"
    judgment = read_json(run_dir / "judgments" / "lme-001.json")
    assert judgment["judge_requested_model"] == "fixture-judge"
    assert judgment["judge_finish_reason"] == "stop"
    assert judgment["judge_usage"] == {"total_tokens": 2}


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


def test_finalizer_rejects_v1_predictions_without_compatibility_parser(
    tmp_path: Path,
) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(["almonds", "the Rome trip"]),
    ).run(config)

    def load_v1_predictions(
        run_dir: Path,
        manifest: Any,
    ) -> list[dict[str, Any]]:
        predictions = [
            read_json(run_dir / "items" / item_id / "prediction.json")
            for item_id in manifest.expected_item_ids
        ]
        for prediction in predictions:
            prediction["schema_version"] = 1
        return predictions

    finalizer = OfflineLifecycleFinalizer(
        artifact_store=store,
        judge=FakeJudge(),
        prediction_loaders={"longmemeval": load_v1_predictions},
    )
    with pytest.raises(StateError, match="v1 state is not supported"):
        finalizer.finalize("lifecycle-longmemeval")


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
        "dmf_bench.benchmarks.longmemeval.evaluation.rigorous_report",
        fail_required,
    )

    with pytest.raises(StateError, match="Finalization failed"):
        OfflineLifecycleFinalizer(artifact_store=store, judge=FakeJudge()).finalize("lifecycle-longmemeval")

    status = read_json(tmp_path / "runs" / "lifecycle-longmemeval" / "run-status.json")
    assert status["state"] == "FAILED_EVALUATION"
    checkpoint = load_lifecycle_checkpoint(
        tmp_path
        / "runs"
        / "lifecycle-longmemeval"
        / "phase-checkpoints"
        / "EVALUATING"
        / "checkpoint.json"
    )
    assert checkpoint.status == "FAILED"


def test_internal_evaluator_adapters_match_pure_functions() -> None:
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

    locomo_cutoff, locomo_metrics = pure_locomo_flat(locomo_items, metadata)
    longmemeval_cutoff, longmemeval_metrics = pure_longmemeval_flat(longmemeval_items, metadata)

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


@pytest.mark.parametrize("benchmark", ["longmemeval", "locomo"])
@pytest.mark.parametrize("resume_phase", ["RUNNING", "JUDGING", "EVALUATING", "PUBLISHING"])
def test_resumed_lifecycle_matches_continuous_scientific_artifacts(
    tmp_path: Path,
    benchmark: str,
    resume_phase: str,
) -> None:
    answers = ["almonds", "the Rome trip"] if benchmark == "longmemeval" else [
        "Pixel",
        "near the kitchen window",
    ]
    continuous_root = tmp_path / "continuous"
    continuous_config, continuous_store, continuous_runner = build_full_runner(
        continuous_root,
        benchmark,
        answers=answers,
    )
    continuous = continuous_runner.run(continuous_config)
    assert continuous.state == "COMPLETED"

    resumed_root = tmp_path / "resumed"
    first_config, _first_store, first_runner = build_full_runner(
        resumed_root,
        benchmark,
        answers=(answers[:1] if resume_phase == "RUNNING" else answers),
    )
    if resume_phase == "RUNNING":
        prediction_interrupt = (
            "after-prediction-commit"
            if benchmark == "longmemeval"
            else "after-first-prediction"
        )
        with pytest.raises(InjectedInterrupt):
            first_runner.run(
                first_config,
                prediction_interrupt_at=prediction_interrupt,
            )
    else:
        terminal_interrupt = {
            "JUDGING": "after-first-judgment",
            "EVALUATING": "after-first-evaluator",
            "PUBLISHING": "after-publish-stage",
        }[resume_phase]
        with pytest.raises(InjectedTerminalInterrupt):
            first_runner.run(
                first_config,
                terminal_interrupt_at=terminal_interrupt,
            )

    resume_answers = (
        ["the Rome trip"]
        if benchmark == "longmemeval" and resume_phase == "RUNNING"
        else answers
    )
    resumed_config, resumed_store, resumed_runner = build_full_runner(
        resumed_root,
        benchmark,
        answers=resume_answers,
    )
    resumed = resumed_runner.run(resumed_config, resume=True)

    continuous_run = continuous_store.run_dir(continuous.run_id)
    resumed_run = resumed_store.run_dir(resumed.run_id)
    assert resumed.state == "COMPLETED"
    assert scientific_artifact_snapshot(resumed_run) == scientific_artifact_snapshot(
        continuous_run
    )
    status = read_json(resumed_run / "run-status.json")
    assert status["items"] == {"expected": 2, "committed": 2, "failed": 0}
    expected_units = 2 if benchmark == "longmemeval" else 1
    assert status["units"] == {
        "expected": expected_units,
        "committed": expected_units,
        "failed": 0,
    }
    assert resumed_store.verify_committed(resumed.run_id)["status"] == "verified"


@pytest.mark.parametrize("corruption", ["truncated", "wrong-fingerprint", "missing"])
def test_resume_regenerates_only_invalid_judgment(
    tmp_path: Path,
    corruption: str,
) -> None:
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

    judgment_path = store.run_dir("lifecycle-longmemeval") / "judgments" / "lme-001.json"
    if corruption == "truncated":
        judgment_path.write_text('{"schema_version":2', encoding="utf-8")
    elif corruption == "wrong-fingerprint":
        payload = read_json(judgment_path)
        payload["judge_fingerprint"] = "f" * 64
        write_json_atomic(judgment_path, payload)
    else:
        judgment_path.unlink()

    second_judge = FakeJudge()
    result = OfflineLifecycleFinalizer(
        artifact_store=store,
        judge=second_judge,
    ).finalize("lifecycle-longmemeval")

    assert result.state == "COMPLETED"
    assert len(first_judge.calls) == 2
    assert second_judge.calls == first_judge.calls[:1]


def test_verify_interrupt_reuses_staging_and_receipt(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(["almonds", "the Rome trip"]),
    ).run(config)
    finalizer = OfflineLifecycleFinalizer(artifact_store=store, judge=FakeJudge())

    with pytest.raises(InjectedTerminalInterrupt, match="verification"):
        finalizer.finalize("lifecycle-longmemeval", interrupt_at="after-verify")

    run_dir = store.run_dir("lifecycle-longmemeval")
    staging_dirs_before = sorted(
        path for path in (run_dir / "publish-staging").iterdir() if path.is_dir()
    )
    assert len(staging_dirs_before) == 1
    receipt_path = staging_dirs_before[0] / "verification-receipt.json"
    receipt_digest = sha256_file(receipt_path)

    assert finalizer.finalize("lifecycle-longmemeval").state == "COMPLETED"
    staging_dirs_after = sorted(
        path for path in (run_dir / "publish-staging").iterdir() if path.is_dir()
    )
    assert staging_dirs_after == staging_dirs_before
    assert sha256_file(receipt_path) == receipt_digest


def test_completion_marker_is_authoritative_after_post_commit_interrupt(
    tmp_path: Path,
) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(["almonds", "the Rome trip"]),
    ).run(config)
    first_judge = FakeJudge()
    finalizer = OfflineLifecycleFinalizer(artifact_store=store, judge=first_judge)

    with pytest.raises(InjectedTerminalInterrupt, match="publication commit"):
        finalizer.finalize(
            "lifecycle-longmemeval",
            interrupt_at="after-publish-commit",
        )

    run_dir = store.run_dir("lifecycle-longmemeval")
    assert store.verify_committed("lifecycle-longmemeval")["status"] == "verified"
    second_judge = FakeJudge()
    resumed = OfflineLifecycleFinalizer(
        artifact_store=store,
        judge=second_judge,
    ).finalize("lifecycle-longmemeval")
    assert resumed.state == "COMPLETED"
    assert second_judge.calls == []
    assert read_json(run_dir / "run-status.json")["state"] == "COMPLETED"


@pytest.mark.parametrize(
    ("failing_phase", "expected_state"),
    [
        ("JUDGING", "FAILED_JUDGING"),
        ("REPORTING", "FAILED_REPORTING"),
        ("PUBLISHING", "FAILED_PUBLISHING"),
        ("VERIFYING", "FAILED_VERIFYING"),
    ],
)
def test_terminal_failure_state_matches_actual_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_phase: str,
    expected_state: str,
) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(["almonds", "the Rome trip"]),
    ).run(config)
    judge: FakeJudge = FailingJudge() if failing_phase == "JUDGING" else FakeJudge()
    finalizer = OfflineLifecycleFinalizer(artifact_store=store, judge=judge)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"simulated {failing_phase.lower()} failure")

    if failing_phase == "REPORTING":
        monkeypatch.setattr(finalizer, "_write_reports", fail)
    elif failing_phase == "PUBLISHING":
        monkeypatch.setattr(store, "stage", fail)
    elif failing_phase == "VERIFYING":
        monkeypatch.setattr(store, "verify", fail)

    with pytest.raises(StateError):
        finalizer.finalize("lifecycle-longmemeval")

    run_dir = store.run_dir("lifecycle-longmemeval")
    status = read_json(run_dir / "run-status.json")
    assert status["state"] == expected_state
    assert status["phase"] == failing_phase
    failed_checkpoint = load_lifecycle_checkpoint(
        run_dir / "phase-checkpoints" / failing_phase / "checkpoint.json"
    )
    assert failed_checkpoint.status == "FAILED"
    assert not (run_dir / "final" / "COMPLETED.json").exists()


def test_prediction_failure_sets_failed_running(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    framework = FakeLongMemEvalFramework()

    def fail_prepare(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated prediction failure")

    framework.prepare_unit = fail_prepare  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="prediction failure"):
        LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=framework,
            answerer=FakeAnswerer(["unused"]),
        ).run(config)

    run_dir = store.run_dir("lifecycle-longmemeval")
    status = read_json(run_dir / "run-status.json")
    assert status["state"] == "FAILED_RUNNING"
    assert status["phase"] == "RUNNING"
    checkpoint = load_lifecycle_checkpoint(
        run_dir / "phase-checkpoints" / "RUNNING" / "checkpoint.json"
    )
    assert checkpoint.status == "FAILED"
    assert not (run_dir / "final" / "COMPLETED.json").exists()


def test_full_lifecycle_holds_lock_while_judge_is_blocked(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    config, store, full_runner = build_full_runner(
        tmp_path,
        "longmemeval",
        answers=["almonds", "the Rome trip"],
        judge=BlockingJudge(entered, release),
    )
    errors: list[BaseException] = []

    def run_lifecycle() -> None:
        try:
            full_runner.run(config)
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            errors.append(exc)

    worker = threading.Thread(target=run_lifecycle)
    worker.start()
    assert entered.wait(timeout=5)

    with pytest.raises(StateError, match="already held"):
        OfflineLifecycleFinalizer(artifact_store=store, judge=FakeJudge()).finalize(
            "lifecycle-longmemeval"
        )

    release.set()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert errors == []
    assert plan_resume(store.run_dir("lifecycle-longmemeval")).next_phase == "COMPLETED"


def test_phase_and_attempt_checkpoints_are_digest_protected(tmp_path: Path) -> None:
    config, store, runner = build_full_runner(
        tmp_path,
        "longmemeval",
        answers=["almonds", "the Rome trip"],
    )
    result = runner.run(config)
    run_dir = store.run_dir(result.run_id)

    for phase in ("RUNNING", "JUDGING", "EVALUATING", "REPORTING", "PUBLISHING", "VERIFYING"):
        checkpoint = load_lifecycle_checkpoint(
            run_dir / "phase-checkpoints" / phase / "checkpoint.json"
        )
        assert checkpoint.status == "COMMITTED"
        assert checkpoint.checkpoint_digest
        assert checkpoint.scientific_fingerprint == read_json(
            run_dir / "run-manifest.json"
        )["scientific_fingerprint"]

    attempts = list((run_dir / "attempts").glob("*/attempt.json"))
    assert len(attempts) == 1
    attempt = read_json(attempts[0])
    assert attempt["status"] == "COMPLETED"
    assert len(attempt["attempt_digest"]) == 64
