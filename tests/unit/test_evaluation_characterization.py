import pytest

from common.native_reporting import (
    aggregate_native_primary_quality,
    apply_native_primary_judge_score,
)
from common.results_io import ensure_locomo_uniform_schema
from locomo.evaluate_rigorous import evaluate_flat as evaluate_locomo_flat
from longmemeval.evaluate_rigorous import ensure_uniform_schema as ensure_longmemeval_uniform_schema
from longmemeval.evaluate_rigorous import evaluate_flat as evaluate_longmemeval_flat


def test_native_primary_quality_only_uses_native_judge_score() -> None:
    native_item = apply_native_primary_judge_score(
        {"protocol_mode": "native", "category_name": "temporal"},
        judgment="CORRECT",
        score=1.0,
        reason="matches",
        judge_provider="fake",
        judge_model="fake-judge",
    )
    strict_item = apply_native_primary_judge_score(
        {"protocol_mode": "strict", "category_name": "temporal"},
        judgment="WRONG",
        score=0.0,
        reason="misses",
        judge_provider="fake",
        judge_model="fake-judge",
    )

    report = aggregate_native_primary_quality(
        [native_item, strict_item],
        benchmark_name="locomo",
    )

    assert native_item["judge_score"] == 1.0
    assert "judge_score" not in strict_item
    assert report["overall"]["judged_count"] == 1
    assert report["overall"]["missing_count"] == 1
    assert report["overall"]["avg_judge_score"] == 1.0


def test_locomo_rigorous_flat_metrics_known_values() -> None:
    evaluations = [
        {
            "generated_answer": "Pixel",
            "ground_truth_answer": "Pixel",
            "category_name": "multi-hop",
            "evidence": ["D1:1"],
            "retrieval": {
                "memories_evaluated": 2,
                "search_results": [
                    {"metadata": {"source_unit_id": "D1:1"}},
                    {"metadata": {"source_unit_id": "D2:1"}},
                ],
            },
        },
        {
            "generated_answer": "kitchen",
            "ground_truth_answer": "near the kitchen window",
            "category_name": "temporal",
            "evidence": ["D2:1"],
            "retrieval": {
                "memories_evaluated": 1,
                "search_results": [
                    {"metadata": {"source_unit_id": "D1:1"}},
                ],
            },
        },
    ]

    cutoff_label, metrics = evaluate_locomo_flat(evaluations, {"top_k": 2})

    assert cutoff_label == "top_2"
    assert metrics["overall"]["exact_match"] == 0.5
    assert metrics["overall"]["recall_at_k"] == 0.5
    assert metrics["overall"]["precision_at_k"] == 0.25
    assert metrics["overall"]["mrr"] == 0.5
    assert metrics["by_group"]["multi-hop"]["exact_match"] == 1.0
    assert metrics["by_group"]["temporal"]["recall_at_k"] == 0.0


def test_longmemeval_rigorous_flat_metrics_known_values() -> None:
    evaluations = [
        {
            "generated_answer": "almonds",
            "ground_truth_answer": "almonds",
            "question_type": "single-session-user",
            "answer_session_ids": ["session-a"],
            "retrieval": {
                "memories_evaluated": 2,
                "search_results": [
                    {"metadata": {"source_unit_id": "session-a"}},
                    {"metadata": {"source_unit_id": "session-b"}},
                ],
            },
        },
        {
            "generated_answer": "concert",
            "ground_truth_answer": "the Rome trip",
            "question_type": "temporal-reasoning",
            "answer_session_ids": ["session-c"],
            "retrieval": {
                "memories_evaluated": 1,
                "search_results": [
                    {"metadata": {"source_unit_id": "session-d"}},
                ],
            },
        },
    ]

    cutoff_label, metrics = evaluate_longmemeval_flat(evaluations, {"top_k": 2})

    assert cutoff_label == "top_2"
    assert metrics["overall"]["exact_match"] == 0.5
    assert metrics["overall"]["recall_at_k"] == 0.5
    assert metrics["overall"]["precision_at_k"] == 0.25
    assert metrics["overall"]["mrr"] == 0.5
    assert metrics["by_group"]["single-session-user"]["exact_match"] == 1.0
    assert metrics["by_group"]["temporal-reasoning"]["recall_at_k"] == 0.0


def test_incomplete_or_mixed_result_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported LOCOMO evaluation payload"):
        ensure_locomo_uniform_schema([{"question": "missing prediction"}])

    with pytest.raises(ValueError, match="Mixed LongMemEval result schemas"):
        ensure_longmemeval_uniform_schema(
            [
                {"generated_answer": "flat"},
                {"cutoff_results": {"top_1": {"generated_answer": "legacy"}}},
            ]
        )
