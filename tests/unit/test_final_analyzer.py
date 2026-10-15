"""Synthetic verified-publication tests for the final analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmf_bench.analyzer import _wilcoxon, analyze, load_verified_run, write_analysis
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import RunManifest, hash_canonical_json, sha256_file
from dmf_bench.state import StateError


QUERIES = ("a-1", "a-2", "b-1", "c-1", "d-1")
UNITS = {"a": ("a-1", "a-2"), "b": ("b-1",), "c": ("c-1",), "d": ("d-1",)}


def _published_run(
    store: LocalArtifactStore, run_id: str, framework: str, scores: tuple[float, ...],
    *, profile: str = "locomo-v3", retrieval_na: bool = True,
    primary_model: str = "primary", included_units: tuple[str, ...] = tuple(UNITS),
    bounded: bool = False,
) -> Path:
    contract = {"rubric": "fixture-v1"}
    judges = []
    for judge_id in ("primary", "secondary"):
        requested_model = primary_model if judge_id == "primary" else judge_id
        identity = {"id": judge_id, "model": {"provider": "fixture", "requested_model": requested_model}, "contract": contract}
        judges.append({**identity, "fingerprint": hash_canonical_json(identity)})
    selected_units = {unit_id: UNITS[unit_id] for unit_id in included_units}
    selected_queries = [query_id for query_ids in selected_units.values() for query_id in query_ids]
    inputs = {
        "schema_version": 3, "scientific_profile": profile, "benchmark": "locomo",
        "framework": framework, "dataset": {"revision": "fixture-v1", "sha256": "a" * 64},
        "selection": {"ordered_item_ids": list(selected_units)}, "expected_question_ids": selected_queries,
        "retrieval": {"top_k": 5}, "context_budget": {"max_tokens": 100},
        "answerer": {"provider": "fixture", "requested_model": "answerer"},
        "judges": judges, "evaluation": {"primary_judge_id": "primary"},
    }
    manifest = RunManifest(
        run_id=run_id, scientific_fingerprint=hash_canonical_json(inputs),
        fingerprint_inputs=inputs, expected_item_ids=tuple(selected_units), atomic_unit="conversation",
        schema_version=3,
    )
    run_dir = store.create_run(manifest)
    score_by_query = dict(zip(QUERIES, scores))
    evaluations = []
    rows = []
    retrieval_queries = {}
    for unit_id, query_ids in selected_units.items():
        predictions = []
        for query_id in query_ids:
            score = score_by_query[query_id]
            prediction = {"schema_version": 3, "benchmark": "locomo", "framework": framework, "question_id": query_id,
                          "answerer_usage": {}, "estimated_cost": 0.01}
            predictions.append(prediction)
            judge_results = {}
            for judge in judges:
                judge_id = judge["id"]
                judge_score = score if judge_id == "primary" else 1 - score
                judgment = {
                    **prediction, "scientific_fingerprint": manifest.scientific_fingerprint,
                    "prediction_sha256": hash_canonical_json(prediction), "judge_id": judge_id,
                    "judge_identity_fingerprint": judge["fingerprint"],
                    "judge_fingerprint": hash_canonical_json(contract),
                    "judge_provider": "fixture", "judge_requested_model": judge["model"]["requested_model"],
                    "score": judge_score, "judge_usage": {},
                }
                path = run_dir / "judgments" / judge_id / f"{query_id}.json"
                write_json_atomic(path, judgment)
                judge_results[judge_id] = {"score": judge_score}
            category = 1 if unit_id in {"a", "b"} else 2
            evaluations.append({**prediction, "category": category, "primary_judge_id": "primary", "judges": judge_results})
            metric = {"status": "NOT_APPLICABLE", "reason": "fixture lacks evidence"} if retrieval_na else {"status": "COMPLETED", "recall_at_k": 1.0}
            retrieval_queries[query_id] = metric
            rows.append({
                "schema_version": 3, "run_id": run_id, "scientific_fingerprint": manifest.scientific_fingerprint,
                "benchmark": "locomo", "framework": framework, "query_id": query_id,
                "strata": {"category": category},
                "primary_judge_id": "primary", "primary_score": score,
                "judge_scores": {"primary": score, "secondary": 1 - score},
                "secondary_scores": {"secondary": 1 - score},
                "retrieval_metrics": metric, "returned_memory_count": 0, "included_memory_count": 0,
                "excluded_memory_count": 0, "retrieved_tokens": 0, "included_tokens": 0,
                "answerer_usage": {}, "framework_usage": {"ingestion": None, "retrieval": {}},
                "judge_usage": {"primary": {}, "secondary": {}},
                "timing": {"ingestion": None, "retrieval": None, "answer": None, "judging": {"primary": None, "secondary": None}},
                "estimated_cost": 0.01,
            })
            write_json_atomic(run_dir / "items" / unit_id / "timing" / f"{query_id}.json", {
                "pipeline_timing": {"qa_ms": 10.0},
            })
            if framework == "full-context":
                write_json_atomic(run_dir / "items" / unit_id / "questions" / f"{query_id}.retrieval.json", {
                    "diagnostics": {"baseline_family": "bounded-full-context" if bounded and query_id == "a-1" else "full-context"},
                })
        write_json_atomic(run_dir / "items" / unit_id / "predictions.json", {
            "schema_version": 3, "question_ids": list(query_ids), "predictions": predictions,
        })
    write_json_atomic(run_dir / "evaluations" / "evaluations.json", evaluations)
    write_json_atomic(run_dir / "evaluations" / "retrieval_report.json", {
        "schema_version": 3, "status": "NOT_APPLICABLE" if retrieval_na else "COMPLETED",
        **({"reason": "fixture lacks evidence"} if retrieval_na else {"queries": retrieval_queries}),
    })
    rows_path = run_dir / "evaluations" / "analysis_rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    write_json_atomic(run_dir / "evaluations" / "analysis_rows.json", {
        "schema_version": 3, "status": "COMPLETED", "evaluator_version": "analysis-rows-v3",
        "path": "evaluations/analysis_rows.jsonl", "row_count": len(rows), "sha256": sha256_file(rows_path),
    })
    staged = store.stage(run_id, run_dir)
    store.commit(staged, store.verify(staged))
    return run_dir


def _plan(*run_ids: str) -> dict:
    return {"schema_version": 1, "run_ids": list(run_ids), "comparisons": [{"left": run_ids[0], "right": run_ids[1]}],
            "cells": [{"benchmark": "locomo", "framework": "dmf", "run_id": run_ids[0]},
                      {"benchmark": "locomo", "framework": "mem0", "run_id": run_ids[1]},
                      {"benchmark": "beam", "framework": "amem", "status": "FAILED", "reason": "synthetic failure"}]}


def test_known_paired_statistics_and_output(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    _published_run(store, "left", "dmf", (0, 0, 0, 0, 0))
    _published_run(store, "right", "mem0", (1, 1, 1, 1, 1))
    result = analyze(store, _plan("left", "right"))
    primary = result["comparisons"][0]["primary"]
    assert primary["unit_count"] == 4
    assert primary["query_count"] == 5
    assert primary["mean_delta_right_minus_left"] == 1
    assert primary["bootstrap_95_percentile_ci"] == [1, 1]
    assert primary["wilcoxon"]["p_two_sided"] == 0.125
    assert primary["holm_adjusted_p"] == 0.125
    assert result["comparisons"][0]["secondary_sensitivity"]["secondary"]["mean_delta_right_minus_left"] == -1
    assert result["runs"]["left"]["retrieval"]["status"] == "NOT_APPLICABLE"
    assert result["runs"]["left"]["strata"]["category"]["1"]["unit_count"] == 2
    assert len(result["matrix"]) == 28
    assert next(cell for cell in result["matrix"] if cell["benchmark"] == "beam" and cell["framework"] == "amem")["status"] == "FAILED"
    assert result["frontier_points"][0]["mean_qa_latency_ms"] == 10
    output = tmp_path / "output"
    write_analysis(result, output)
    assert read_json(output / "analysis.json") == result
    assert "VERIFIED" in (output / "matrix.csv").read_text()
    assert "FRONTIER" in (output / "frontier.csv").read_text()


def test_clustered_units_prevent_query_pseudoreplication(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    _published_run(store, "left", "dmf", (0, 0, 0, 0, 0))
    _published_run(store, "right", "mem0", (1, 1, 0, 0, 0))
    primary = analyze(store, _plan("left", "right"))["comparisons"][0]["primary"]
    assert primary["mean_delta_right_minus_left"] == 0.25
    assert primary["wilcoxon"]["nonzero_units"] == 1
    assert _wilcoxon([0, 0])["method"] == "all-zero"


def test_rejects_tampering_missing_item_and_incompatible_controls(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    _published_run(store, "left", "dmf", (0, 0, 0, 0, 0))
    _published_run(store, "right", "mem0", (1, 1, 1, 1, 1), profile="other-profile")
    with pytest.raises(ValueError, match="scientific_profile"):
        analyze(store, _plan("left", "right"))
    path = store.run_dir("left") / "final" / "evaluations" / "analysis_rows.jsonl"
    path.write_text(path.read_text().replace('"primary_score":0', '"primary_score":9', 1))
    with pytest.raises(StateError, match="checksum mismatch"):
        load_verified_run(store, "left")
    path.unlink()
    with pytest.raises(StateError, match="Missing committed artifact"):
        load_verified_run(store, "left")


def test_rejects_judge_mismatch_and_missing_paired_query(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    _published_run(store, "left", "dmf", (0, 0, 0, 0, 0))
    _published_run(store, "right", "mem0", (1, 1, 1, 1, 1))
    right = store.run_dir("right") / "final" / "run-manifest.json"
    data = read_json(right)
    data["fingerprint_inputs"]["judges"][0]["model"]["requested_model"] = "other-judge"
    write_json_atomic(right, data)
    with pytest.raises(StateError, match="artifact size mismatch"):
        analyze(store, _plan("left", "right"))
    left = store.run_dir("left") / "final" / "items" / "a" / "predictions.json"
    left.unlink()
    with pytest.raises(StateError, match="Missing committed artifact"):
        load_verified_run(store, "left")


def test_not_applicable_retrieval_is_distinct_from_zero(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    _published_run(store, "na", "dmf", (0, 0, 0, 0, 0))
    _published_run(store, "available", "mem0", (0, 0, 0, 0, 0), retrieval_na=False)
    result = analyze(store, _plan("na", "available"))
    assert result["runs"]["na"]["retrieval"]["status"] == "NOT_APPLICABLE"
    assert result["runs"]["available"]["retrieval"]["status"] == "AVAILABLE"


def test_full_context_label_uses_retrieval_diagnostics(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    _published_run(store, "bounded", "full-context", (0, 0, 0, 0, 0), bounded=True)
    result = analyze(store, {"schema_version": 1, "run_ids": ["bounded"]})
    assert result["runs"]["bounded"]["profile"] == "bounded-full-context"
    assert result["frontier_points"][0]["frontier_family"] == "full-context"


@pytest.mark.parametrize("change,field", [
    ({"primary_model": "another-primary"}, "judges"),
    ({"included_units": ("a", "b", "c")}, "selection"),
])
def test_rejects_coherently_published_incompatible_runs(tmp_path: Path, change: dict, field: str) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    _published_run(store, "left", "dmf", (0, 0, 0, 0, 0))
    _published_run(store, "right", "mem0", (1, 1, 1, 1, 1), **change)
    with pytest.raises(ValueError, match=field):
        analyze(store, _plan("left", "right"))


def test_cli_analyze_writes_verified_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from dmf_bench.cli import main

    store = LocalArtifactStore(tmp_path / "runs")
    _published_run(store, "left", "dmf", (0, 0, 0, 0, 0))
    _published_run(store, "right", "mem0", (1, 1, 1, 1, 1))
    plan_path = tmp_path / "plan.json"
    write_json_atomic(plan_path, _plan("left", "right"))
    output = tmp_path / "analysis"
    assert main(["analyze", "--runs-dir", str(store.runs_dir), "--plan", str(plan_path), "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["comparisons"] == 1
    assert (output / "comparisons.csv").is_file()


def test_loads_actual_v3_finalizer_publication(tmp_path: Path) -> None:
    from test_multi_judge_finalizer import _finalizer, _run

    store, run_id, judges = _run(tmp_path)
    assert _finalizer(store, judges, v3_plan=True).finalize(run_id).state == "COMPLETED"
    verified = load_verified_run(store, run_id)
    assert [row["query_id"] for row in verified.rows] == ["q1", "q2"]
    assert verified.unit_by_query == {"q1": "unit-1", "q2": "unit-2"}
