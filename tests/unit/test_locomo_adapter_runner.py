import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import (
    AnswererRequest,
    BenchmarkUnit,
    FrameworkRunContext,
    ProgressUpdate,
    RetrievalResult,
)
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json
from dmf_bench.cli import main
from dmf_bench.contracts import REPORT_SCHEMA_VERSION, sha256_file
from dmf_bench.runner import InjectedInterrupt, LoCoMoPredictOnlyRunner
from dmf_bench.state import StateError, plan_resume


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


class FakeLoCoMoFramework:
    name = "dmf"

    def __init__(self) -> None:
        self.cleanup_calls: list[str] = []
        self.prepare_calls: list[str] = []
        self.retrieve_calls: list[str] = []

    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        del run_context
        self.cleanup_calls.append(unit.unit_id)

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        self.prepare_calls.append(unit.unit_id)
        run_context.report_progress(
            ProgressUpdate(
                stage="memory_ingestion",
                label="Ingesting source memory",
                completed=2,
                total=5,
                item_label="fixture records",
            )
        )
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

    def retrieve_question(
        self,
        unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        question: Any,
        config: dict[str, Any],
        _prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        del run_context
        self.retrieve_calls.append(question.question_id)
        return RetrievalResult(
            native_context={
                "conversation_id": unit.unit_id,
                "question_id": question.question_id,
            },
            native_surface_diagnostics={"result_count": 1},
            cutoff_label="native",
        )


class FailingRetentionLoCoMoFramework(FakeLoCoMoFramework):
    def cleanup_unit(
        self,
        unit: BenchmarkUnit,
        conversation: dict[str, Any],
        config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        super().cleanup_unit(
            unit,
            conversation,
            config,
            run_context=run_context,
        )
        if len(self.cleanup_calls) == 2:
            raise RuntimeError("retention cleanup failed")


class FakeAnswerer:
    name = "fake-answerer"

    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = list(answers or ["Pixel", "near the kitchen window"])
        self.calls: list[dict[str, Any]] = []

    def generate(self, request: AnswererRequest) -> dict[str, Any]:
        self.calls.append(
            {
                "system_prompt": request.system_prompt,
                "prompt": request.user_prompt,
                "metadata": request.metadata,
            }
        )
        answer = self.answers.pop(0) if self.answers else "fallback"
        return {
            "generated_answer": answer,
            "answerer_provider": "fake",
            "answerer_model": "fixture-model",
            "answerer_usage": {"total_tokens": 3},
        }


def make_locomo_config(tmp_path: Path) -> dict[str, Any]:
    dataset_path = tmp_path / "locomo-mini.json"
    shutil.copy2(FIXTURE_DIR / "locomo-mini.json", dataset_path)
    framework_config_path = tmp_path / "framework.toml"
    framework_config_path.write_text("[ltm]\nstorage_type = \"qdrant\"\n", encoding="utf-8")
    return {
        "schema_version": 2,
        "experiment_id": "fixture-locomo-dmf",
        "benchmark": "locomo",
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
            "name": "locomo",
            "source": "fixture",
            "revision": "fixture-revision",
            "sha256": sha256_file(dataset_path),
            "path": str(dataset_path),
        },
        "selection": {
            "ordered_item_ids": ["conversation-0001"],
            "filters": {"categories": [1, 2]},
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


def test_locomo_adapter_materializes_units_and_expected_questions(tmp_path: Path) -> None:
    config = make_locomo_config(tmp_path)
    adapter = LoCoMoAdapter()

    units = adapter.enumerate_units(config)

    assert [unit.unit_id for unit in units] == ["conversation-0001"]
    assert units[0].item_ids == ("conv0_q0", "conv0_q1")
    assert units[0].metadata["categories"] == [1, 2]
    assert adapter.expected_question_ids(config) == ("conv0_q0", "conv0_q1")
    assert adapter.materialize_reference(config).sha256 == config["dataset"]["sha256"]


def test_locomo_adapter_filters_categories_and_conversation_ids(tmp_path: Path) -> None:
    config = make_locomo_config(tmp_path)
    config["selection"] = {
        "ordered_item_ids": ["*"],
        "filters": {"conversation_ids": ["conversation-0001"], "categories": [2]},
        "seed": 7,
    }

    units = LoCoMoAdapter().enumerate_units(config)

    assert [unit.unit_id for unit in units] == ["conversation-0001"]
    assert units[0].item_ids == ("conv0_q1",)


def test_locomo_surface_builds_v2_prediction(
    tmp_path: Path,
) -> None:
    native_config = make_locomo_config(tmp_path)
    adapter = LoCoMoAdapter()
    _idx, conversation, questions = adapter.selected_conversations_by_id(native_config)["conversation-0001"]
    question = questions[1]

    native_input = adapter.build_answerer_input(
        conversation=conversation,
        question=question,
        framework_name="dmf",
        retrieval={"native_context": {"memory": "Pixel's bed is near the kitchen window."}},
    )

    assert "Native context:" in native_input.user_prompt
    assert "Pixel's bed is near the kitchen window" in native_input.user_prompt

    native_prediction = adapter.build_prediction(
        conversation=conversation,
        question=question,
        framework_name="dmf",
        retrieval={
            "native_context": {"memory": "Pixel bed"},
            "search_results": [
                {"metadata": {"source_unit_id": "D2:1"}}
            ],
            "memories_evaluated": 1,
        },
        answerer_input=native_input,
        answerer_output={"generated_answer": "near the kitchen window", "answerer_model": "fixture"},
    )
    assert native_prediction["schema_version"] == 2
    assert "protocol" not in native_prediction
    assert native_prediction["question_id"] == "conv0_q1"
    assert native_prediction["ground_truth_answer"] == "near the kitchen window"
    assert native_prediction["native"]["native_context"] == {"memory": "Pixel bed"}
    assert native_prediction["retrieval"]["search_results"][0]["metadata"][
        "source_unit_id"
    ] == "D2:1"


def test_locomo_runner_uses_one_ingestion_and_commits_conversation_atomically(
    tmp_path: Path,
) -> None:
    config = make_locomo_config(tmp_path)
    framework = FakeLoCoMoFramework()
    answerer = FakeAnswerer()

    result = LoCoMoPredictOnlyRunner(
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
        framework=framework,
        answerer=answerer,
    ).run(config)
    run_dir = tmp_path / "runs" / "fixture-locomo-dmf"

    assert result.state == "PARTIAL"
    assert result.committed_unit_ids == ("conversation-0001",)
    assert framework.prepare_calls == ["conversation-0001"]
    assert framework.retrieve_calls == ["conv0_q0", "conv0_q1"]
    assert len(answerer.calls) == 2
    assert read_json(run_dir / "run-status.json")["state"] == "PARTIAL"
    checkpoint = read_json(run_dir / "checkpoints" / "conversation-0001" / "checkpoint.json")
    assert checkpoint["status"] == "COMMITTED"
    assert len(checkpoint["artifacts"]) == 7
    assert read_json(run_dir / "items" / "conversation-0001" / "retention.json")[
        "outcome"
    ] == "kept"
    cleanup_manifest = read_json(
        run_dir / "items" / "conversation-0001" / "cleanup-manifest.json"
    )
    assert cleanup_manifest["collections"][0]["name"] == "owned-conversation-0001"
    assert any(
        artifact["path"].endswith("cleanup-manifest.json")
        for artifact in checkpoint["artifacts"]
    )
    aggregate = read_json(run_dir / "items" / "conversation-0001" / "predictions.json")
    assert aggregate["question_ids"] == ["conv0_q0", "conv0_q1"]
    timing = read_json(
        run_dir / "items" / "conversation-0001" / "timing" / "conv0_q0.json"
    )
    assert timing["schema_version"] == REPORT_SCHEMA_VERSION


def test_locomo_delete_on_success_cleans_before_commit(tmp_path: Path) -> None:
    config = make_locomo_config(tmp_path)
    config["qdrant"]["retention"] = "delete-on-success"
    framework = FakeLoCoMoFramework()

    LoCoMoPredictOnlyRunner(
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
        framework=framework,
        answerer=FakeAnswerer(),
    ).run(config)

    run_dir = tmp_path / "runs" / "fixture-locomo-dmf"
    assert framework.cleanup_calls == ["conversation-0001", "conversation-0001"]
    assert read_json(
        run_dir / "items" / "conversation-0001" / "retention.json"
    )["outcome"] == "deleted"


def test_locomo_retention_failure_does_not_commit_conversation(tmp_path: Path) -> None:
    config = make_locomo_config(tmp_path)
    config["qdrant"]["retention"] = "delete-on-success"

    with pytest.raises(RuntimeError, match="retention cleanup failed"):
        LoCoMoPredictOnlyRunner(
            artifact_store=LocalArtifactStore(tmp_path / "runs"),
            framework=FailingRetentionLoCoMoFramework(),
            answerer=FakeAnswerer(),
        ).run(config)

    run_dir = tmp_path / "runs" / "fixture-locomo-dmf"
    checkpoint = read_json(
        run_dir / "checkpoints" / "conversation-0001" / "checkpoint.json"
    )
    assert checkpoint["status"] == "PREDICTING"
    assert read_json(run_dir / "run-status.json")["state"] == "FAILED_RUNNING"
    assert not (
        run_dir / "items" / "conversation-0001" / "retention.json"
    ).exists()


def test_locomo_resume_skips_committed_conversation(tmp_path: Path) -> None:
    config = make_locomo_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LoCoMoPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLoCoMoFramework(),
        answerer=FakeAnswerer(),
    ).run(config)

    second_framework = FakeLoCoMoFramework()
    second_answerer = FakeAnswerer(["should-not-be-used"])
    result = LoCoMoPredictOnlyRunner(
        artifact_store=store,
        framework=second_framework,
        answerer=second_answerer,
    ).run(config, resume=True)

    assert result.restarted_unit_ids == ()
    assert second_framework.prepare_calls == []
    assert second_answerer.calls == []


def test_locomo_interrupt_after_ingestion_restarts_entire_conversation(tmp_path: Path) -> None:
    config = make_locomo_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")

    with pytest.raises(InjectedInterrupt, match="after ingestion"):
        LoCoMoPredictOnlyRunner(
            artifact_store=store,
            framework=FakeLoCoMoFramework(),
            answerer=FakeAnswerer(),
        ).run(config, interrupt_at="after-ingestion-before-first-prediction")

    run_dir = tmp_path / "runs" / "fixture-locomo-dmf"
    assert plan_resume(run_dir).restart_unit_ids == ("conversation-0001",)
    cleanup_manifest_path = (
        run_dir / "items" / "conversation-0001" / "cleanup-manifest.json"
    )
    assert read_json(cleanup_manifest_path)["unit_hash"] == "conversation-0001"
    activity = read_json(run_dir / "run-status.json")["current_activity"]
    assert activity["stage"] == "memory_ingestion"
    assert activity["completed"] == 2
    assert activity["total"] == 5
    assert activity["memory_system"] == "dmf"
    assert activity["input_record"] == {"number": 1, "total": 1}

    resumed_framework = FakeLoCoMoFramework()
    resumed = LoCoMoPredictOnlyRunner(
        artifact_store=store,
        framework=resumed_framework,
        answerer=FakeAnswerer(),
    ).run(config, resume=True)

    assert resumed.committed_unit_ids == ("conversation-0001",)
    assert resumed_framework.prepare_calls == ["conversation-0001"]


def test_locomo_interrupt_between_predictions_discards_partial_outputs(
    tmp_path: Path,
) -> None:
    config = make_locomo_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")

    with pytest.raises(InjectedInterrupt, match="after first LoCoMo prediction"):
        LoCoMoPredictOnlyRunner(
            artifact_store=store,
            framework=FakeLoCoMoFramework(),
            answerer=FakeAnswerer(["first-partial", "unused"]),
        ).run(config, interrupt_at="after-first-prediction")

    run_dir = tmp_path / "runs" / "fixture-locomo-dmf"
    partial_path = run_dir / "items" / "conversation-0001" / "questions" / "conv0_q0.json"
    assert partial_path.exists()
    assert plan_resume(run_dir).restart_unit_ids == ("conversation-0001",)

    resumed = LoCoMoPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLoCoMoFramework(),
        answerer=FakeAnswerer(["fresh-1", "fresh-2"]),
    ).run(config, resume=True)

    assert resumed.restarted_unit_ids == ("conversation-0001",)
    assert read_json(partial_path)["generated_answer"] == "fresh-1"
    assert read_json(run_dir / "items" / "conversation-0001" / "questions" / "conv0_q1.json")[
        "generated_answer"
    ] == "fresh-2"


def test_locomo_resume_refuses_selection_or_model_mismatch(tmp_path: Path) -> None:
    config = make_locomo_config(tmp_path)
    store = LocalArtifactStore(tmp_path / "runs")
    LoCoMoPredictOnlyRunner(
        artifact_store=store,
        framework=FakeLoCoMoFramework(),
        answerer=FakeAnswerer(),
    ).run(config)

    mismatched = make_locomo_config(tmp_path)
    mismatched["selection"]["filters"]["categories"] = [1]

    with pytest.raises(StateError, match="scientific fingerprint mismatch"):
        LoCoMoPredictOnlyRunner(
            artifact_store=store,
            framework=FakeLoCoMoFramework(),
            answerer=FakeAnswerer(),
        ).run(mismatched, resume=True)


def test_cli_run_predict_only_plan_supports_locomo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_locomo_config(tmp_path)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert main(["run", "--config", str(config_path), "--predict-only", "--plan-only"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["predict_only_state"] == "PARTIAL"
    assert output["expected_unit_ids"] == ["conversation-0001"]
    assert output["expected_question_ids"] == ["conv0_q0", "conv0_q1"]
    assert not (tmp_path / "runs").exists()
