from collections.abc import Callable

import pytest

from dmf_bench.reporting.quality import (
    aggregate_native_primary_quality,
    apply_native_primary_judge_score,
)
from dmf_bench.benchmarks.locomo.evaluation import ensure_v2_evaluations
from dmf_bench.benchmarks.locomo.ablation import (
    evaluate_ablation as evaluate_locomo_ablation,
)
from dmf_bench.benchmarks.locomo.rigorous import (
    evaluate_flat as evaluate_locomo_flat,
)
from longmemeval.evaluate_ablation import (
    evaluate_ablation as evaluate_longmemeval_ablation,
)
from longmemeval.evaluate_rigorous import ensure_uniform_schema as ensure_longmemeval_uniform_schema
from longmemeval.evaluate_rigorous import evaluate_flat as evaluate_longmemeval_flat


def test_primary_quality_reports_missing_unscored_items() -> None:
    native_item = apply_native_primary_judge_score(
        {"category_name": "temporal"},
        judgment="CORRECT",
        score=1.0,
        reason="matches",
        judge_provider="fake",
        judge_model="fake-judge",
    )
    unscored_item = {"category_name": "temporal"}

    report = aggregate_native_primary_quality(
        [native_item, unscored_item],
        benchmark_name="locomo",
    )

    assert native_item["judge_score"] == 1.0
    assert "judge_score" not in unscored_item
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


@pytest.mark.parametrize(
    ("evaluate", "reference"),
    [
        (
            evaluate_locomo_ablation,
            {"category_name": "temporal", "evidence": ["D1:1"]},
        ),
        (
            evaluate_longmemeval_ablation,
            {
                "question_type": "single-session-user",
                "answer_session_ids": ["session-a"],
            },
        ),
    ],
)
def test_ablation_distinguishes_available_empty_native_diagnostics(
    evaluate: Callable[..., dict[str, object]],
    reference: dict[str, object],
) -> None:
    item = {
        **reference,
        "retrieval": {
            "search_results": [],
            "recall_diagnostics": {
                "diagnostics_available": True,
                "diagnostic_source": "dmf_structured_native_final_projection",
                "raw_stage_available": False,
                "raw_candidates": [],
                "ranked_candidates_canonical": [],
                "final_candidates_canonical": [],
            },
        },
    }

    result = evaluate([item])

    assert result["stats"]["questions_with_diagnostics"] == 1
    assert result["stats"]["questions_without_diagnostics"] == 0
    assert result["stages"]["final"]["overall"]["recall_at_k"] == 0.0


def test_incomplete_or_mixed_result_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported LOCOMO evaluation payload"):
        ensure_v2_evaluations([{"question": "missing prediction"}])

    with pytest.raises(ValueError, match="Mixed LongMemEval result schemas"):
        ensure_longmemeval_uniform_schema(
            [
                {"generated_answer": "flat"},
                {"cutoff_results": {"top_1": {"generated_answer": "legacy"}}},
            ]
        )
