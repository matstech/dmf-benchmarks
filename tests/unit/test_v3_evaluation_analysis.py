from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dmf_bench.adapters.base import EvaluationReference
from dmf_bench.evaluation.registry import EvaluationRequirement, evaluation_plan_v3
from dmf_bench.registry import BENCHMARKS, COMPATIBILITY, FRAMEWORKS
from dmf_bench.reporting.analysis import analysis_rows_jsonl, build_analysis_rows


def test_v3_plan_requires_quality_and_reasons_about_applicability() -> None:
    mem0 = evaluation_plan_v3("locomo", "mem0", ("primary",))
    by_name = {item.name: item for item in mem0}
    assert [item.name for item in mem0] == [
        "primary_judge_score", "rigorous_report", "analysis_rows", "retrieval_report",
        "ablation_report", "judge_agreement",
    ]
    assert all(by_name[name].status == "REQUIRED" for name in (
        "primary_judge_score", "rigorous_report", "analysis_rows"))
    assert by_name["retrieval_report"].status == "OPTIONAL"
    for name in ("ablation_report", "judge_agreement"):
        assert by_name[name].status == "NOT_APPLICABLE"
        assert by_name[name].not_applicable_reason

    dmf = evaluation_plan_v3("locomo", "dmf", ("primary", "secondary"))
    by_name = {item.name: item for item in dmf}
    assert by_name["retrieval_report"].status == "OPTIONAL"
    assert by_name["ablation_report"].status == "OPTIONAL"
    assert by_name["judge_agreement"].status == "OPTIONAL"

    no_provenance = {**FRAMEWORKS, "mem0": replace(
        FRAMEWORKS["mem0"], capabilities=FRAMEWORKS["mem0"].capabilities - {"source-provenance"})}
    missing = evaluation_plan_v3("locomo", "mem0", ("primary",), frameworks=no_provenance)
    retrieval = next(item for item in missing if item.name == "retrieval_report")
    assert retrieval.status == "NOT_APPLICABLE"
    assert "source-provenance" in retrieval.not_applicable_reason


def test_v3_plan_uses_registered_pair_and_benchmark_evidence_profile() -> None:
    descriptor = replace(BENCHMARKS["locomo"], evaluator_profile="no-evidence")
    pair = replace(COMPATIBILITY[("locomo", "dmf")], evaluator_profile="no-evidence")
    plan = evaluation_plan_v3(
        "locomo", "dmf", ("primary",), benchmarks={**BENCHMARKS, "locomo": descriptor},
        compatibility={**COMPATIBILITY, ("locomo", "dmf"): pair},
    )
    assert next(item for item in plan if item.name == "retrieval_report").status == "NOT_APPLICABLE"
    with pytest.raises(ValueError, match="profile differs"):
        evaluation_plan_v3("locomo", "dmf", ("primary",),
                           benchmarks={**BENCHMARKS, "locomo": descriptor})
    with pytest.raises(ValueError, match="Unsupported benchmark/framework pair"):
        evaluation_plan_v3("locomo", "dmf", ("primary",), compatibility={})
    with pytest.raises(ValueError, match="judge_ids"):
        evaluation_plan_v3("locomo", "dmf", ("primary", "primary"))


def analysis_fixture() -> dict:
    ids = ("q-2", "q-1")
    return {
        "run_id": "run-1", "scientific_fingerprint": "a" * 64,
        "benchmark": "locomo", "framework": "dmf",
        "expected_query_ids": ids,
        "expected_judge_ids": ("primary", "secondary"),
        "primary_judge_id": "primary",
        "predictions": {
            query_id: {"question_id": query_id, "answerer_usage": {"total_tokens": index + 2}}
            for index, query_id in enumerate(ids)
        },
        "retrievals": {
            query_id: {"items": [{"memory_id": f"{query_id}:m1"}, {"memory_id": f"{query_id}:m2"}],
                       "usage": {"memory_internal": {"calls": 1}},
                       "timing": {"retrieval_seconds": 0.1}}
            for query_id in ids
        },
        "packed_contexts": {
            query_id: {"returned_ids": [f"{query_id}:m1", f"{query_id}:m2"],
                       "included_ids": [f"{query_id}:m1"],
                       "excluded_ids": [f"{query_id}:m2"],
                       "original_tokens": 20, "included_tokens": 10}
            for query_id in ids
        },
        "judgments": {
            query_id: {
                "primary": {"question_id": query_id, "judge_id": "primary", "score": 1.0},
                "secondary": {"question_id": query_id, "judge_id": "secondary", "score": 0.0},
            }
            for query_id in ids
        },
        "references": {
            query_id: EvaluationReference(query_id, "answer", ("e-1",), {"category": index})
            for index, query_id in enumerate(ids)
        },
        "retrieval_metrics": {query_id: {"recall_at_2": 0.5} for query_id in ids},
        "retrieval_requirement": EvaluationRequirement("retrieval_report", required=False),
        "ingestion": {query_id: {"ingestion_timing": {"seconds": 1.0}} for query_id in ids},
    }


def test_analysis_rows_follow_manifest_and_are_deterministic_jsonl() -> None:
    arguments = analysis_fixture()
    rows = build_analysis_rows(**arguments)
    assert [row["query_id"] for row in rows] == ["q-2", "q-1"]
    assert rows[0]["schema_version"] == 3
    assert rows[0]["strata"] == {"category": 0}
    assert rows[0]["primary_score"] == 1.0
    assert rows[0]["secondary_scores"] == {"secondary": 0.0}
    assert rows[0]["retrieval_metrics"] == {"recall_at_2": 0.5}
    assert (rows[0]["returned_memory_count"], rows[0]["included_memory_count"],
            rows[0]["excluded_memory_count"]) == (2, 1, 1)
    assert (rows[0]["retrieved_tokens"], rows[0]["included_tokens"]) == (20, 10)
    assert rows[0]["timing"]["ingestion"] == {"seconds": 1.0}
    assert rows[0]["estimated_cost"] is None
    content = analysis_rows_jsonl(rows)
    assert content == analysis_rows_jsonl(build_analysis_rows(**arguments))
    assert content.endswith(b"\n")
    assert [json.loads(line)["query_id"] for line in content.splitlines()] == ["q-2", "q-1"]
    assert b"ground_truth_answer" not in content


def test_analysis_rows_emit_reason_instead_of_zero_for_unavailable_retrieval() -> None:
    arguments = analysis_fixture()
    arguments["retrieval_metrics"] = None
    arguments["retrieval_requirement"] = EvaluationRequirement(
        "retrieval_report", required=False,
        not_applicable_reason="No official evidence for this benchmark profile.")
    rows = build_analysis_rows(**arguments)
    assert rows[0]["retrieval_metrics"] == {
        "status": "NOT_APPLICABLE", "reason": "No official evidence for this benchmark profile."}


@pytest.mark.parametrize("missing", ["predictions", "retrievals", "packed_contexts", "judgments", "references"])
def test_analysis_rows_fail_closed_on_missing_query(missing: str) -> None:
    arguments = analysis_fixture()
    del arguments[missing]["q-1"]
    with pytest.raises(ValueError, match="exactly the expected query IDs"):
        build_analysis_rows(**arguments)


def test_analysis_rows_reject_missing_judge_packing_mismatch_and_missing_metric() -> None:
    arguments = analysis_fixture()
    del arguments["judgments"]["q-2"]["secondary"]
    with pytest.raises(ValueError, match="every expected judge"):
        build_analysis_rows(**arguments)
    arguments = analysis_fixture()
    arguments["packed_contexts"]["q-2"]["included_ids"] = ["foreign"]
    with pytest.raises(ValueError, match="Packed IDs"):
        build_analysis_rows(**arguments)
    arguments = analysis_fixture()
    arguments["retrieval_metrics"] = None
    with pytest.raises(ValueError, match="Applicable retrieval metrics"):
        build_analysis_rows(**arguments)
