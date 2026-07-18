import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import BenchmarkUnit
from dmf_bench.adapters.longmemeval import LongMemEvalAdapter
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json
from dmf_bench.cli import main
from dmf_bench.contracts import sha256_file
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
    ) -> None:
        self.cleanup_calls.append(unit.unit_id)

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        _question: dict[str, Any],
        _config: dict[str, Any],
    ) -> dict[str, Any]:
        self.prepare_calls.append(unit.unit_id)
        return {"qdrant_commit_barrier": True}

    def retrieve(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        _prepared: dict[str, Any],
    ) -> dict[str, Any]:
        self.retrieve_calls.append(unit.unit_id)
        if config["protocol"] == "native":
            return {
                "native_context": {
                    "question_id": unit.unit_id,
                    "answer_session_ids": question["answer_session_ids"],
                },
                "native_surface_diagnostics": {"result_count": 1},
                "cutoff_label": "native",
            }
        return {
            "search_results": [
                {
                    "memory": "hit",
                    "metadata": {"source_unit_ids": list(question["answer_session_ids"])},
                }
            ],
            "cutoff_label": "top_1",
        }


class FakeAnswerer:
    name = "fake-answerer"

    def __init__(self, answer: str = "almonds") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"prompt": prompt, "metadata": metadata})
        return {
            "generated_answer": self.answer,
            "answerer_provider": "fake",
            "answerer_model": "fixture-model",
            "answerer_usage": {"total_tokens": 3},
        }


def make_longmemeval_config(tmp_path: Path, *, protocol: str = "strict") -> dict[str, Any]:
    dataset_path = tmp_path / "longmemeval-mini.json"
    shutil.copy2(FIXTURE_DIR / "longmemeval-mini.json", dataset_path)
    return {
        "schema_version": 1,
        "experiment_id": f"fixture-longmemeval-dmf-{protocol}",
        "benchmark": "longmemeval",
        "framework": "dmf",
        "protocol": protocol,
        "runtime": {
            "root": str(tmp_path),
            "runs_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
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
                "parameters": {"temperature": 0},
            },
            "judge": {
                "provider": "fake",
                "requested_model": "fixture-judge",
                "parameters": {"temperature": 0},
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


def test_strict_and_native_prompt_surfaces_are_protocol_labeled(tmp_path: Path) -> None:
    strict_config = make_longmemeval_config(tmp_path, protocol="strict")
    native_config = make_longmemeval_config(tmp_path, protocol="native")
    adapter = LongMemEvalAdapter()
    question = adapter.selected_questions_by_id(strict_config)["lme-001"]

    strict_input = adapter.build_answerer_input(
        question=question,
        protocol="strict",
        framework_name="dmf",
        retrieval={"search_results": [{"metadata": {"source_unit_id": "session-a"}}]},
    )
    native_input = adapter.build_answerer_input(
        question=question,
        protocol="native",
        framework_name="dmf",
        retrieval={"native_context": {"memory": "I prefer almonds."}},
    )

    assert "Session Date: 2024/02/01 (Thu) 08:00" in strict_input.user_prompt
    assert '"role": "user"' in strict_input.user_prompt
    assert "Native context:" in native_input.user_prompt
    assert '"memory": "I prefer almonds."' in native_input.user_prompt

    strict_prediction = adapter.build_prediction(
        question=question,
        protocol="strict",
        framework_name="dmf",
        retrieval={"search_results": [{"metadata": {"source_unit_id": "session-a"}}]},
        answerer_input=strict_input,
        answerer_output={"generated_answer": "almonds", "answerer_model": "fixture"},
    )
    assert strict_prediction["protocol_label"] == "strict/longmemeval"
    assert strict_prediction["ground_truth_answer"] == "almonds"

    native_prediction = adapter.build_prediction(
        question=question,
        protocol="native",
        framework_name="dmf",
        retrieval={"native_context": {"memory": "I prefer almonds."}},
        answerer_input=native_input,
        answerer_output={"generated_answer": "almonds", "answerer_model": "fixture"},
    )
    assert native_prediction["protocol_label"] == "native/longmemeval"
    assert native_prediction["native"]["native_context"] == {"memory": "I prefer almonds."}


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
    run_dir = tmp_path / "runs" / "fixture-longmemeval-dmf-strict"

    assert result.state == "PARTIAL"
    assert result.committed_unit_ids == ("lme-001", "lme-002")
    assert result.restarted_unit_ids == ("lme-001", "lme-002")
    assert len(answerer.calls) == 2
    assert read_json(run_dir / "run-status.json")["state"] == "PARTIAL"
    prediction = read_json(run_dir / "items" / "lme-001" / "prediction.json")
    assert prediction["protocol"] == "strict"
    assert prediction["retrieval"]["strict_session_ids"] == ["session-a"]
    assert read_json(run_dir / "checkpoints" / "lme-001" / "checkpoint.json")["status"] == "COMMITTED"


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

    run_dir = tmp_path / "runs" / "fixture-longmemeval-dmf-strict"
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

    run_dir = tmp_path / "runs" / "fixture-longmemeval-dmf-strict"
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
