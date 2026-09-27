"""Offline v3 judge identity, checkpoint, and report tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import JudgeRequest
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import RunManifest, UnitCheckpoint, hash_canonical_json
from dmf_bench.evaluation.finalizer import (
    InjectedTerminalInterrupt, OfflineFullLifecycleRunner, OfflineLifecycleFinalizer,
)
from dmf_bench.evaluation.registry import EvaluationRequirement
from dmf_bench.fingerprints import judge_contract_identity
from dmf_bench.state import StateError, artifact_ref_for


class FakeJudge:
    def __init__(self, name: str, rubric_fingerprint: str, *, correct: bool) -> None:
        self.name = name
        self.judge_fingerprint = rubric_fingerprint
        self.correct = correct
        self.calls: list[str] = []

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        self.calls.append(str(request.prediction["question_id"]))
        return {
            "judgment": "CORRECT" if self.correct else "WRONG",
            "score": 1.0 if self.correct else 0.0,
            "reason": "deterministic fixture",
            "judge_provider": "fixture",
            "judge_requested_model": self.name,
            "judge_model": self.name,
            "judge_fingerprint": self.judge_fingerprint,
            "judge_usage": {"total_tokens": 2},
        }


def _run(tmp_path: Path) -> tuple[LocalArtifactStore, str, dict[str, FakeJudge]]:
    store = LocalArtifactStore(tmp_path / "runs")
    contract = judge_contract_identity("longmemeval")
    rubric_fingerprint = hash_canonical_json(contract)
    judges: dict[str, FakeJudge] = {
        "primary": FakeJudge("primary-model", rubric_fingerprint, correct=True),
        "secondary": FakeJudge("secondary-model", rubric_fingerprint, correct=False),
    }
    judge_inputs = []
    for judge_id, judge in judges.items():
        identity = {
            "id": judge_id,
            "model": {"provider": "fixture", "requested_model": judge.name},
            "contract": contract,
        }
        judge_inputs.append({**identity, "fingerprint": hash_canonical_json(identity)})
    inputs = {
        "schema_version": 3,
        "benchmark": "longmemeval",
        "framework": "dmf",
        "expected_question_ids": ["q1", "q2"],
        "judges": judge_inputs,
        "evaluation": {"primary_judge_id": "primary"},
    }
    fingerprint = hash_canonical_json(inputs)
    manifest = RunManifest(
        run_id="multi-judge-fixture", scientific_fingerprint=fingerprint,
        fingerprint_inputs=inputs, expected_item_ids=("unit-1", "unit-2"),
        atomic_unit="longmemeval-question", schema_version=3,
    )
    run_dir = store.create_run(manifest)
    for unit_id, query_id in (("unit-1", "q1"), ("unit-2", "q2")):
        aggregate_path = run_dir / "items" / unit_id / "predictions.json"
        write_json_atomic(aggregate_path, {"schema_version": 3, "question_ids": [query_id], "predictions": [{
            "schema_version": 3, "benchmark": "longmemeval", "framework": "dmf",
            "question_id": query_id, "question": "fixture question",
            "ground_truth_answer": "fixture answer", "generated_answer": "fixture answer",
            "question_type": "single-session-user", "is_abstention": False,
            "answer_session_ids": ["session-a"],
            "answerer_usage": {}, "memory_internal_usage": {}, "pipeline_timing": {},
        }]})
        write_json_atomic(run_dir / "items" / unit_id / "prepared.json", {
            "resources": [], "ingestion_usage": {}, "ingestion_timing": {}, "diagnostics": {},
        })
        write_json_atomic(run_dir / "items" / unit_id / "questions" / f"{query_id}.retrieval.json", {
            "items": [{"memory_id": "memory-1", "content": "fixture memory", "rank": 1,
                       "source_event_ids": ["event-1"], "metadata": {"source_refs": ["session-a"]}}],
            "usage": {}, "timing": {},
        })
        write_json_atomic(run_dir / "items" / unit_id / "questions" / f"{query_id}.context.json", {
            "returned_ids": ["memory-1"], "included_ids": ["memory-1"],
            "excluded_ids": [], "original_tokens": 3, "included_tokens": 3,
        })
        checkpoint = UnitCheckpoint(
            run_id=manifest.run_id, unit_id=unit_id, status="COMMITTED",
            scientific_fingerprint=fingerprint,
            artifacts=(artifact_ref_for(run_dir, aggregate_path),),
        )
        write_json_atomic(run_dir / "checkpoints" / unit_id / "checkpoint.json", checkpoint.to_dict())
    return store, manifest.run_id, judges


def _finalizer(
    store: LocalArtifactStore, judges: dict[str, FakeJudge], *, v3_plan: bool = False,
) -> OfflineLifecycleFinalizer:
    plan = (
        EvaluationRequirement("primary_judge_score", required=True),
        EvaluationRequirement("analysis_rows", required=True),
        EvaluationRequirement("retrieval_report", required=False),
        EvaluationRequirement("judge_agreement", required=False),
    ) if v3_plan else (EvaluationRequirement("primary_judge_score", required=True),)
    return OfflineLifecycleFinalizer(
        artifact_store=store,
        judges=judges,
        primary_judge_id="primary",
        evaluation_plans={
            ("longmemeval", "dmf"): plan,
        },
    )


def test_v3_checkpoints_each_query_judge_and_reports_disagreement(tmp_path: Path) -> None:
    store, run_id, judges = _run(tmp_path)
    result = _finalizer(store, judges).finalize(run_id)
    assert result.state == "COMPLETED"
    run_dir = store.run_dir(run_id)
    assert read_json(run_dir / "run-manifest.json")["schema_version"] == 3
    assert read_json(run_dir / "items" / "unit-1" / "predictions.json")["predictions"][0]["schema_version"] == 3
    assert read_json(run_dir / "judgments" / "primary" / "q1.json")["schema_version"] == 3
    assert read_json(run_dir / "evaluations" / "evaluations.json")[0]["schema_version"] == 3
    assert read_json(run_dir / "evaluations" / "evaluation-summary.json")["schema_version"] == 3
    assert read_json(run_dir / "reports" / "summary.json")["schema_version"] == 3
    assert judges["primary"].calls == ["q1", "q2"]
    assert judges["secondary"].calls == ["q1", "q2"]
    report = read_json(store.run_dir(run_id) / "reports" / "summary.json")["judges"]
    assert report["primary"]["avg_judge_score"] == 1.0
    assert report["secondary"]["secondary"]["avg_judge_score"] == 0.0
    assert report["agreement"]["discordant_query_ids"] == ["q1", "q2"]
    usage = read_json(store.run_dir(run_id) / "reports" / "usage.json")
    assert usage["components"]["judge"]["calls"] == 4
    assert usage["totals"]["evaluation"]["total_tokens"] == 8
    checkpoints = list((store.run_dir(run_id) / "phase-checkpoints" / "JUDGING" / "items").glob("*.json"))
    assert len(checkpoints) == 4


def test_v3_resume_reuses_only_valid_pair_after_fingerprint_tamper(tmp_path: Path) -> None:
    store, run_id, judges = _run(tmp_path)
    with pytest.raises(InjectedTerminalInterrupt, match="after judge"):
        _finalizer(store, judges).finalize(run_id, interrupt_at="after-judge")
    judgment_path = store.run_dir(run_id) / "judgments" / "secondary" / "q1.json"
    tampered = read_json(judgment_path)
    tampered["judge_identity_fingerprint"] = "f" * 64
    write_json_atomic(judgment_path, tampered)
    resumed = {
        "primary": FakeJudge("primary-model", judges["primary"].judge_fingerprint, correct=True),
        "secondary": FakeJudge("secondary-model", judges["secondary"].judge_fingerprint, correct=False),
    }
    assert _finalizer(store, resumed).finalize(run_id).state == "COMPLETED"
    assert resumed["primary"].calls == []
    assert resumed["secondary"].calls == ["q1"]


def test_v3_refuses_judge_rubric_fingerprint_mismatch(tmp_path: Path) -> None:
    store, run_id, judges = _run(tmp_path)
    judges["secondary"].judge_fingerprint = "f" * 64
    with pytest.raises(StateError, match="rubric fingerprint mismatch"):
        _finalizer(store, judges).finalize(run_id)
    assert judges["primary"].calls == []
    assert judges["secondary"].calls == []


def test_v3_analysis_rows_retrieval_and_agreement_artifacts(tmp_path: Path) -> None:
    store, run_id, judges = _run(tmp_path)
    assert _finalizer(store, judges, v3_plan=True).finalize(run_id).state == "COMPLETED"
    run_dir = store.run_dir(run_id)
    retrieval = read_json(run_dir / "evaluations" / "retrieval_report.json")
    assert retrieval["status"] == "COMPLETED"
    assert retrieval["schema_version"] == 3
    assert retrieval["metrics"]["mean_recall_at_k"] == 1.0
    agreement = read_json(run_dir / "evaluations" / "judge_agreement.json")
    assert agreement["status"] == "COMPLETED"
    assert agreement["schema_version"] == 3
    assert agreement["discordant_query_ids"] == ["q1", "q2"]
    rows_path = run_dir / "evaluations" / "analysis_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    assert [row["query_id"] for row in rows] == ["q1", "q2"]
    assert rows[0]["primary_score"] == 1.0
    assert rows[0]["secondary_scores"] == {"secondary": 0.0}
    assert rows[0]["retrieval_metrics"]["recall_at_k"] == 1.0
    assert read_json(run_dir / "evaluations" / "analysis_rows.json")["schema_version"] == 3


def test_full_v3_runner_with_two_deterministic_judges(tmp_path: Path) -> None:
    from test_retrieval_qa_runner import config_v3, runner_for

    config = config_v3(tmp_path, "longmemeval", "dmf")
    config["selection"]["ordered_item_ids"] = ["lme-001"]
    config["models"]["judges"].append({
        **config["models"]["judges"][0],
        "id": "secondary", "requested_model": "second-v1",
    })
    config["evaluation"]["optional"].append("judge_agreement")
    runner, _framework, _answerer = runner_for(tmp_path, "longmemeval", "dmf")
    rubric_fingerprint = hash_canonical_json(judge_contract_identity("longmemeval"))
    judges = {
        "primary": FakeJudge("exact-v1", rubric_fingerprint, correct=True),
        "secondary": FakeJudge("second-v1", rubric_fingerprint, correct=False),
    }
    result = OfflineFullLifecycleRunner(
        prediction_runner=runner,
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
        judges=judges,
        primary_judge_id="primary",
    ).run(config)
    assert result.state == "COMPLETED"
    summary = read_json(tmp_path / "runs" / result.run_id / "reports" / "summary.json")
    assert summary["judges"]["agreement"]["discordant_query_ids"] == ["lme-001"]
    run_dir = tmp_path / "runs" / result.run_id
    assert read_json(run_dir / "run-manifest.json")["schema_version"] == 3
    assert read_json(run_dir / "items" / "lme-001" / "prediction.json")["schema_version"] == 3
    assert read_json(run_dir / "reports" / "usage.json")["schema_version"] == 3
    assert read_json(run_dir / "evaluations" / "analysis_rows.json")["status"] == "COMPLETED"
    assert read_json(run_dir / "evaluations" / "retrieval_report.json")["status"] == "NOT_APPLICABLE"
    assert read_json(run_dir / "evaluations" / "judge_agreement.json")["status"] == "COMPLETED"


def test_full_v3_runner_marks_single_judge_agreement_not_applicable(tmp_path: Path) -> None:
    from test_retrieval_qa_runner import config_v3, runner_for

    config = config_v3(tmp_path, "longmemeval", "mem0")
    config["selection"]["ordered_item_ids"] = ["lme-001"]
    runner, _framework, _answerer = runner_for(tmp_path, "longmemeval", "mem0")
    rubric_fingerprint = hash_canonical_json(judge_contract_identity("longmemeval"))
    judge = FakeJudge("exact-v1", rubric_fingerprint, correct=True)
    result = OfflineFullLifecycleRunner(
        prediction_runner=runner,
        artifact_store=LocalArtifactStore(tmp_path / "runs"),
        judges={"primary": judge}, primary_judge_id="primary",
    ).run(config)
    assert result.state == "COMPLETED"
    run_dir = tmp_path / "runs" / result.run_id
    agreement = read_json(run_dir / "evaluations" / "judge_agreement.json")
    assert agreement["status"] == "NOT_APPLICABLE"
    assert "at least two judges" in agreement["reason"]
