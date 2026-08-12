import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import AnswererRequest, BenchmarkUnit, FrameworkRunContext, RetrievalResult
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json
from dmf_bench.cli import main
from dmf_bench.contracts import REPORT_SCHEMA_VERSION, sha256_file
from dmf_bench.runner import (
    InjectedInterrupt,
    LongMemEvalPredictOnlyRunner,
)
from dmf_bench.state import StateError, plan_resume


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


class FakeLongMemEvalFramework:
    name = "dmf"

    def __init__(self) -> None:
        self.cleanup_calls: list[str] = []
        self.prepare_calls: list[str] = []
        self.retrieve_calls: list[str] = []

    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        _question: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        del run_context
        self.cleanup_calls.append(unit.unit_id)

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        _question: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        del run_context
        self.prepare_calls.append(unit.unit_id)
        return {
            "qdrant_commit_barrier": True,
            "cleanup_manifest": {
                "framework": self.name,
                "unit_hash": unit.unit_id,
                "collections": [
                    {"name": f"owned-{unit.unit_id}", "role": "primary"}
                ],
                "local_paths": [],
            },
        }

    def retrieve(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        _prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        del run_context
        self.retrieve_calls.append(unit.unit_id)
        return RetrievalResult(
            native_context={
                "question_id": unit.unit_id,
                "answer_session_ids": question["answer_session_ids"],
            },
            native_surface_diagnostics={"result_count": 1},
            cutoff_label="native",
        )


class FailingRetentionLongMemEvalFramework(FakeLongMemEvalFramework):
    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        super().cleanup_unit(
            unit,
            question,
            config,
            run_context=run_context,
        )
        if len(self.cleanup_calls) == 2:
            raise RuntimeError("retention cleanup failed")


class FakeAnswerer:
    name = "fake-answerer"

    def __init__(self, answer: str = "almonds") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def generate(self, request: AnswererRequest) -> dict[str, Any]:
        self.calls.append(
            {
                "system_prompt": request.system_prompt,
                "prompt": request.user_prompt,
                "metadata": request.metadata,
            }
        )
        return {
            "generated_answer": self.answer,
            "answerer_provider": "fake",
            "answerer_model": "fixture-model",
            "answerer_usage": {"total_tokens": 3},
        }


def make_longmemeval_config(tmp_path: Path) -> dict[str, Any]:
    dataset_path = tmp_path / "longmemeval-mini.json"
    shutil.copy2(FIXTURE_DIR / "longmemeval-mini.json", dataset_path)
    framework_config_path = tmp_path / "framework.toml"
    framework_config_path.write_text("[ltm]\nstorage_type = \"qdrant\"\n", encoding="utf-8")
    return {
        "schema_version": 2,
        "experiment_id": "fixture-longmemeval-dmf",
        "benchmark": "longmemeval",
        "framework": "dmf",
        "runtime": {
            "root": str(tmp_path),
            "runs_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
            "metrics_port": 9464,
            "log_level": "INFO",
        },
        "framework_config": {
            "path": str(framework_config_path),
            "sha256": sha256_file(framework_config_path),
            "format": "toml",
            "profile": "fixture",
        },
        "qdrant": {
            "endpoint_env": "QDRANT_URL",
            "retention": "keep",
            "request_timeout_seconds": 10,
        },
        "dataset": {
            "name": "longmemeval",
            "source": "fixture",
            "revision": "fixture-revision",
            "sha256": sha256_file(dataset_path),
            "path": str(dataset_path),
        },
        "selection": {
            "ordered_item_ids": ["lme-001", "lme-002"],
            "filters": {},
            "seed": 7,
        },
        "models": {
            "answerer": {
                "provider": "fake",
                "requested_model": "fixture-model",
                "parameters": {"temperature": 0, "max_tokens": 256},
                "runtime": {"timeout_seconds": 30, "rpm": 100, "max_retries": 0},
            },
            "judge": {
                "provider": "fake",
                "requested_model": "fixture-judge",
                "parameters": {"temperature": 0, "max_tokens": 256},
                "runtime": {"timeout_seconds": 30, "rpm": 100, "max_retries": 0},
            },
        },
        "evaluation": {"required": ["primary_judge_score"], "optional": []},
        "artifact_store": {"type": "local", "uri": str(tmp_path / "runs")},
    }


def test_longmemeval_adapter_materializes_local_reference_and_expected_set(
    tmp_path: Path,
) -> None:
    config = make_longmemeval_config(tmp_path)
    adapter = LongMemEvalAdapter()

    units = adapter.enumerate_units(config)

    assert [unit.unit_id for unit in units] == ["lme-001", "lme-002"]
    assert units[0].item_ids == ("lme-001",)
    assert units[0].metadata["ground_truth_answer"] == "almonds"
    assert adapter.materialize_reference(config).sha256 == config["dataset"]["sha256"]


def test_longmemeval_adapter_filters_and_samples_when_selection_uses_wildcard(
    tmp_path: Path,
) -> None:
    config = make_longmemeval_config(tmp_path)
    config["selection"] = {
        "ordered_item_ids": ["*"],
        "filters": {"question_types": ["single-session-user"]},
        "sample_per_type": 1,
        "seed": 3,
    }

    units = LongMemEvalAdapter().enumerate_units(config)

    assert [unit.unit_id for unit in units] == ["lme-001"]


def test_prompt_surface_builds_v2_prediction(tmp_path: Path) -> None:
    native_config = make_longmemeval_config(tmp_path)
    adapter = LongMemEvalAdapter()
    question = adapter.selected_questions_by_id(native_config)["lme-001"]

    native_input = adapter.build_answerer_input(
        question=question,
        framework_name="dmf",
        retrieval={"native_context": {"memory": "I prefer almonds."}},
    )

    assert "Native context:" in native_input.user_prompt
    assert '"memory": "I prefer almonds."' in native_input.user_prompt

    native_prediction = adapter.build_prediction(
        question=question,
        framework_name="dmf",
        retrieval={
            "native_context": {"memory": "I prefer almonds."},
            "search_results": [
                {"metadata": {"source_unit_id": "session-a"}}
            ],
            "memories_evaluated": 1,
        },
        answerer_input=native_input,
        answerer_output={"generated_answer": "almonds", "answerer_model": "fixture"},
    )
    assert native_prediction["schema_version"] == 2
    assert "protocol" not in native_prediction
    assert native_prediction["ground_truth_answer"] == "almonds"
    assert native_prediction["native"]["native_context"] == {"memory": "I prefer almonds."}
    assert native_prediction["retrieval"]["search_results"][0]["metadata"][
        "source_unit_id"
    ] == "session-a"


def test_predict_only_runner_commits_question_predictions_and_stops_partial(
    tmp_path: Path,
) -> None:
    config = make_longmemeval_config(tmp_path)
    framework = FakeLongMemEvalFramework()
    answerer = FakeAnswerer()
    runner = LongMemEvalPredictOnlyRunner(
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
        framework=framework,
        answerer=answerer,
    )

    result = runner.run(config)
    run_dir = tmp_path / "runs" / "fixture-longmemeval-dmf"

    assert result.state == "PARTIAL"
    assert result.committed_unit_ids == ("lme-001", "lme-002")
    assert result.restarted_unit_ids == ("lme-001", "lme-002")
    assert len(answerer.calls) == 2
    assert read_json(run_dir / "run-status.json")["state"] == "PARTIAL"
    prediction = read_json(run_dir / "items" / "lme-001" / "prediction.json")
    assert prediction["schema_version"] == 2
    assert "protocol" not in prediction
    assert prediction["native"]["native_context"]["question_id"] == "lme-001"
    timing = read_json(run_dir / "items" / "lme-001" / "timing.json")
    assert timing["schema_version"] == REPORT_SCHEMA_VERSION
    checkpoint = read_json(
        run_dir / "checkpoints" / "lme-001" / "checkpoint.json"
    )
    assert checkpoint["status"] == "COMMITTED"
    assert [artifact["path"] for artifact in checkpoint["artifacts"]] == [
        "items/lme-001/prediction.json",
        "items/lme-001/timing.json",
        "items/lme-001/cleanup-manifest.json",
        "items/lme-001/retention.json",
    ]
    assert read_json(run_dir / "items" / "lme-001" / "retention.json")[
        "outcome"
    ] == "kept"
    assert read_json(run_dir / "items" / "lme-001" / "cleanup-manifest.json")[
        "unit_hash"
    ] == "lme-001"


def test_delete_on_success_cleans_each_unit_before_commit(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    config["qdrant"]["retention"] = "delete-on-success"
    framework = FakeLongMemEvalFramework()

    LongMemEvalPredictOnlyRunner(
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
        framework=framework,
        answerer=FakeAnswerer(),
    ).run(config)

    run_dir = tmp_path / "runs" / "fixture-longmemeval-dmf"
    assert framework.cleanup_calls == ["lme-001", "lme-001", "lme-002", "lme-002"]
    assert read_json(run_dir / "items" / "lme-001" / "retention.json")[
        "outcome"
    ] == "deleted"


def test_retention_failure_does_not_commit_unit(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    config["qdrant"]["retention"] = "delete-on-success"

    with pytest.raises(RuntimeError, match="retention cleanup failed"):
        LongMemEvalPredictOnlyRunner(
            artifact_store=LocalArtifactStore(tmp_path / "runs"),
            framework=FailingRetentionLongMemEvalFramework(),
            answerer=FakeAnswerer(),
        ).run(config)

    run_dir = tmp_path / "runs" / "fixture-longmemeval-dmf"
    checkpoint = read_json(run_dir / "checkpoints" / "lme-001" / "checkpoint.json")
    assert checkpoint["status"] == "PREDICTING"
    assert read_json(run_dir / "run-status.json")["state"] == "FAILED_RUNNING"
    assert not (run_dir / "items" / "lme-001" / "retention.json").exists()


def test_resume_skips_committed_questions(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    first_framework = FakeLongMemEvalFramework()
    first_answerer = FakeAnswerer()
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=first_framework,
        answerer=first_answerer,
    ).run(config)

    second_framework = FakeLongMemEvalFramework()
    second_answerer = FakeAnswerer(answer="should-not-be-used")
    result = LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=second_framework,
        answerer=second_answerer,
    ).run(config, resume=True)

    assert result.committed_unit_ids == ("lme-001", "lme-002")
    assert result.restarted_unit_ids == ()
    assert second_answerer.calls == []
    assert second_framework.prepare_calls == []


def test_interrupt_before_prediction_commit_restarts_unit(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    framework = FakeLongMemEvalFramework()

    with pytest.raises(InjectedInterrupt, match="before prediction commit"):
        LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=framework,
            answerer=FakeAnswerer(),
        ).run(config, interrupt_at="before-prediction-commit")

    run_dir = tmp_path / "runs" / "fixture-longmemeval-dmf"
    assert not (run_dir / "items" / "lme-001" / "prediction.json").exists()
    assert plan_resume(run_dir).restart_unit_ids == ("lme-001", "lme-002")

    resumed_answerer = FakeAnswerer()
    resumed = LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=resumed_answerer,
    ).run(config, resume=True)

    assert resumed.committed_unit_ids == ("lme-001", "lme-002")
    assert len(resumed_answerer.calls) == 2


def test_interrupt_after_prediction_commit_does_not_regenerate_committed_unit(
    tmp_path: Path,
) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")

    with pytest.raises(InjectedInterrupt, match="after prediction commit"):
        LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=FakeLongMemEvalFramework(),
            answerer=FakeAnswerer(),
        ).run(config, interrupt_at="after-prediction-commit")

    run_dir = tmp_path / "runs" / "fixture-longmemeval-dmf"
    assert plan_resume(run_dir).committed_unit_ids == ("lme-001",)
    assert plan_resume(run_dir).restart_unit_ids == ("lme-002",)

    resumed_answerer = FakeAnswerer(answer="resumed")
    resumed = LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=resumed_answerer,
    ).run(config, resume=True)

    assert resumed.restarted_unit_ids == ("lme-002",)
    assert len(resumed_answerer.calls) == 1
    assert read_json(run_dir / "items" / "lme-001" / "prediction.json")["generated_answer"] == "almonds"
    assert read_json(run_dir / "items" / "lme-002" / "prediction.json")["generated_answer"] == "resumed"


def test_resume_refuses_seed_dataset_or_model_mismatch(tmp_path: Path) -> None:
    config = make_longmemeval_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LongMemEvalPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLongMemEvalFramework(),
        answerer=FakeAnswerer(),
    ).run(config)

    mismatched = make_longmemeval_config(tmp_path)
    mismatched["selection"]["seed"] = 99

    with pytest.raises(StateError, match="scientific fingerprint mismatch"):
        LongMemEvalPredictOnlyRunner(
            artifact_store=store,
            framework=FakeLongMemEvalFramework(),
            answerer=FakeAnswerer(),
        ).run(mismatched, resume=True)


def test_cli_run_predict_only_plan_is_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_longmemeval_config(tmp_path)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["run", "--config", str(config_path), "--predict-only", "--plan-only"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["predict_only_state"] == "PARTIAL"
    assert output["expected_unit_ids"] == ["lme-001", "lme-002"]
    assert not (tmp_path / "runs").exists()
