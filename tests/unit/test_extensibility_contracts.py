from __future__ import annotations

from pathlib import Path
from typing import Any

from dmf_bench.adapters.base import BenchmarkUnit, FrameworkRunContext, JudgeRequest, RetrievalResult
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.contracts import ArtifactRef, RunManifest, UnitCheckpoint, hash_canonical_json, sha256_file
from dmf_bench.evaluation.finalizer import OfflineLifecycleFinalizer
from dmf_bench.evaluation.registry import EvaluationRequirement
from dmf_bench.registry import BenchmarkInfo, FRAMEWORKS, FrameworkInfo, supported_combinations, validate_combination
from dmf_bench.state import UnitState


class ExternalFixtureFramework:
    name = "fixture-memory"

    def cleanup_unit(
        self,
        _unit: BenchmarkUnit,
        _item: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        del run_context
        return None

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        _item: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        del run_context
        return {"prepared_unit_id": unit.unit_id}

    def retrieve(
        self,
        unit: BenchmarkUnit,
        _item: dict[str, Any],
        _config: dict[str, Any],
        prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        del run_context
        return RetrievalResult(
            cutoff_label="fixture",
            native_context={"unit_id": unit.unit_id, "prepared": prepared},
        )


class TinyBenchmarkFixture:
    name = "tinybench"
    atomic_unit = "tiny-unit"
    supported_protocols = ("native",)

    def enumerate_units(self, _config: dict[str, Any]) -> list[BenchmarkUnit]:
        return [BenchmarkUnit(unit_id="tiny-001", item_ids=("tiny-001",), metadata={"question": "2+2?"})]

    def build_prediction(self, unit: BenchmarkUnit) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.name,
            "protocol": "native",
            "framework": ExternalFixtureFramework.name,
            "question_id": unit.unit_id,
            "question": "2+2?",
            "ground_truth_answer": "4",
            "generated_answer": "4",
        }


class FixtureJudge:
    name = "fixture-judge"
    judge_fingerprint = hash_canonical_json(
        {"schema_version": 1, "rubric": "fixture-exact-match-v1"}
    )

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        expected = str(request.prediction.get("ground_truth_answer", ""))
        generated = str(request.prediction.get("generated_answer", ""))
        return {
            "judgment": "CORRECT" if expected == generated else "WRONG",
            "score": 1.0 if expected == generated else 0.0,
            "reason": "fixture exact match",
            "judge_provider": "fixture",
            "judge_requested_model": "fixture-judge",
            "judge_model": "fixture-judge",
            "judge_fingerprint": self.judge_fingerprint,
        }


def tiny_report(evaluations: list[dict[str, Any]], _metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluator": "tiny_report",
        "evaluator_version": "tiny-report-v1",
        "status": "COMPLETED",
        "metrics": {"accuracy": sum(float(item["score"]) for item in evaluations) / len(evaluations)},
    }


tiny_report.evaluator_version = "tiny-report-v1"  # type: ignore[attr-defined]


def create_tiny_committed_run(store: LocalArtifactStore) -> str:
    benchmark = TinyBenchmarkFixture()
    unit = benchmark.enumerate_units({})[0]
    fingerprint_inputs = {
        "schema_version": 1,
        "benchmark": benchmark.name,
        "framework": ExternalFixtureFramework.name,
        "protocol": "native",
        "dataset": {"name": "tinybench", "source": "fixture", "revision": "v1", "sha256": "0" * 64},
        "selection": {"ordered_item_ids": [unit.unit_id], "seed": 1},
        "models": {
            "judge": {
                "provider": "fixture",
                "requested_model": "fixture-judge",
                "parameters": {},
            }
        },
        "judge_contract": {"schema_version": 1, "rubric": "fixture-exact-match-v1"},
        "evaluation": {"required": ["primary_judge_score", "tiny_report"]},
    }
    manifest = RunManifest(
        run_id="tiny-run",
        scientific_fingerprint=hash_canonical_json(fingerprint_inputs),
        fingerprint_inputs=fingerprint_inputs,
        expected_item_ids=(unit.unit_id,),
        atomic_unit=benchmark.atomic_unit,
    )
    run_dir = store.create_run(manifest)
    prediction_path = run_dir / "items" / unit.unit_id / "prediction.json"
    write_json_atomic(prediction_path, benchmark.build_prediction(unit))
    artifact = ArtifactRef(
        path=prediction_path.relative_to(run_dir).as_posix(),
        sha256=sha256_file(prediction_path),
        bytes=prediction_path.stat().st_size,
    )
    checkpoint = UnitCheckpoint(
        run_id=manifest.run_id,
        unit_id=unit.unit_id,
        status=UnitState.COMMITTED.value,
        scientific_fingerprint=manifest.scientific_fingerprint,
        artifacts=(artifact,),
    )
    write_json_atomic(run_dir / "checkpoints" / unit.unit_id / "checkpoint.json", checkpoint.to_dict())
    return manifest.run_id


def load_tiny_predictions(run_dir: Path, manifest: RunManifest) -> list[dict[str, Any]]:
    return [
        {
            **_read_prediction(run_dir / "items" / unit_id / "prediction.json"),
            "protocol_mode": str(manifest.fingerprint_inputs["protocol"]),
        }
        for unit_id in manifest.expected_item_ids
    ]


def _read_prediction(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_explicit_registry_accepts_external_framework_without_default_registry_mutation() -> None:
    benchmarks = {"tinybench": BenchmarkInfo("tinybench", "tiny-unit", ("native",))}
    frameworks = {**FRAMEWORKS, ExternalFixtureFramework.name: FrameworkInfo(ExternalFixtureFramework.name, "fixture")}

    validate_combination(
        "tinybench",
        ExternalFixtureFramework.name,
        "native",
        benchmarks=benchmarks,
        frameworks=frameworks,
        protocols=("native",),
    )

    assert ("tinybench", ExternalFixtureFramework.name, "native") in supported_combinations(
        benchmarks=benchmarks,
        frameworks=frameworks,
        protocols=("native",),
    )
    assert ExternalFixtureFramework.name not in FRAMEWORKS


def test_fixture_benchmark_and_evaluator_extend_finalizer_without_runner_or_store_changes(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    run_id = create_tiny_committed_run(store)

    result = OfflineLifecycleFinalizer(
        artifact_store=store,
        judge=FixtureJudge(),
        prediction_loaders={"tinybench": load_tiny_predictions},
        evaluation_plans={
            ("tinybench", "native", ExternalFixtureFramework.name): (
                EvaluationRequirement("primary_judge_score", required=True),
                EvaluationRequirement("tiny_report", required=True),
            )
        },
        evaluators={"tiny_report": tiny_report},
    ).finalize(run_id)

    final_dir = tmp_path / "runs" / "tiny-run" / "final"
    assert result.state == "COMPLETED"
    assert (final_dir / "COMPLETED.json").exists()
    assert LocalArtifactStore(tmp_path / "runs").verify_committed("tiny-run")["verified_artifact_count"] > 0
