import hashlib
import json
from pathlib import Path

from dmf_bench.benchmarks.longmemeval.ablation import evaluate_ablation
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.benchmarks.longmemeval.dataset import (
    filter_questions_by_ids,
    normalize_longmemeval_haystack,
    render_longmemeval_session_for_context,
    sample_questions_stratified,
    serialize_longmemeval_session_for_dmf,
    serialize_longmemeval_session_for_mem0,
)
from dmf_bench.benchmarks.longmemeval.prompts import ANSWERER_USER_PROMPT_TEMPLATE
from dmf_bench.benchmarks.longmemeval.rigorous import evaluate_flat
from dmf_bench.fingerprints import judge_fingerprint


def load_longmemeval_fixture() -> list[dict]:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "longmemeval-mini.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_longmemeval_filter_by_ids_preserves_dataset_order() -> None:
    questions = load_longmemeval_fixture()

    selected = filter_questions_by_ids(questions, ["lme-002", "lme-001"])

    assert [question["question_id"] for question in selected] == ["lme-001", "lme-002"]


def test_longmemeval_sampling_is_seeded_and_sorted_by_question_id() -> None:
    questions = load_longmemeval_fixture()

    first = sample_questions_stratified(questions, per_type=1, seed=7)
    second = sample_questions_stratified(questions, per_type=1, seed=7)

    assert [question["question_id"] for question in first] == ["lme-001", "lme-002"]
    assert [question["question_id"] for question in first] == [
        question["question_id"] for question in second
    ]


def test_longmemeval_haystack_is_chronological() -> None:
    question = load_longmemeval_fixture()[0]

    haystack = normalize_longmemeval_haystack(question)

    assert [session["session_id"] for session in haystack] == ["session-a", "session-b"]


def test_longmemeval_framework_serializations_are_stable() -> None:
    haystack = normalize_longmemeval_haystack(load_longmemeval_fixture()[0])

    assert [serialize_longmemeval_session_for_dmf(session) for session in haystack] == [
        ["I prefer almonds as a snack. I will remember that."],
        ["I bought tea. Noted."],
    ]
    assert [serialize_longmemeval_session_for_mem0(session) for session in haystack] == [
        [[
            {"role": "user", "content": "I prefer almonds as a snack."},
            {"role": "assistant", "content": "I will remember that."},
        ]],
        [[
            {"role": "user", "content": "I bought tea."},
            {"role": "assistant", "content": "Noted."},
        ]],
    ]
    assert render_longmemeval_session_for_context(haystack[0]) == (
        "Session session-a (2024/02/01 (Thu) 08:00)\n"
        "User: I prefer almonds as a snack.\n"
        "Assistant: I will remember that."
    )


def test_longmemeval_prompt_and_judge_fingerprint_are_stable() -> None:
    question = load_longmemeval_fixture()[0]
    answerer_input = LongMemEvalAdapter().build_answerer_input(
        question=question,
        framework_name="dmf",
        retrieval={"native_context": {"memory": "I prefer almonds."}},
    )

    assert judge_fingerprint("longmemeval") == (
        "e2eaef074e7972a6ad684b89733897eda6d958a3d4d5a4450dc0b42594c07ad3"
    )
    assert hashlib.sha256(ANSWERER_USER_PROMPT_TEMPLATE.encode()).hexdigest() == (
        "eca46aba4c1023ed177549c216ce73da8e3cb00e9a3445d3228523a3e02af977"
    )
    assert hashlib.sha256(answerer_input.user_prompt.encode()).hexdigest() == (
        "a2605cc04e0ee310d5f05539b7a5064be06372c585fc68bcfd47ffa52c60605a"
    )


def test_longmemeval_v2_metrics_preserve_single_multi_temporal_and_abstention() -> None:
    evaluations = [
        _evaluation(
            generated_answer="almonds",
            ground_truth_answer="almonds",
            question_type="single-session-user",
            answer_session_ids=["session-a"],
            retrieved_session_ids=["session-a", "session-b"],
        ),
        _evaluation(
            generated_answer="the Rome trip",
            ground_truth_answer="the Rome trip",
            question_type="multi-session",
            answer_session_ids=["session-c"],
            retrieved_session_ids=["session-d", "session-c"],
        ),
        _evaluation(
            generated_answer="the Rome trip",
            ground_truth_answer="the Rome trip",
            question_type="temporal-reasoning",
            answer_session_ids=["session-c"],
            retrieved_session_ids=["session-c"],
            diagnostics_available=True,
        ),
        _evaluation(
            generated_answer="I don't know",
            ground_truth_answer="unknown",
            question_type="single-session-assistant",
            answer_session_ids=[],
            retrieved_session_ids=[],
        ),
    ]

    cutoff_label, rigorous = evaluate_flat(evaluations, {"top_k": 2})
    ablation = evaluate_ablation(evaluations)

    assert cutoff_label == "native"
    assert rigorous["overall"] == {
        "exact_match": 1.0,
        "token_f1": 1.0,
        "rouge_l": 1.0,
        "recall_at_k": 0.75,
        "precision_at_k": 0.5,
        "mrr": 0.625,
        "ndcg_at_k": 0.6577324383928644,
    }
    assert rigorous["abstention_count"] == 1
    assert rigorous["by_group"]["multi-session"]["mrr"] == 0.5
    assert rigorous["by_group"]["temporal-reasoning"]["recall_at_k"] == 1.0
    assert ablation["stats"] == {
        "total_questions": 3,
        "questions_with_diagnostics": 1,
        "questions_without_diagnostics": 2,
        "raw_unmappable_candidates_total": 0,
        "avg_candidates_raw": 5 / 3,
        "avg_candidates_ranked": 5 / 3,
        "avg_candidates_final": 5 / 3,
    }
    assert ablation["stages"]["final"]["overall"]["recall_at_k"] == 1.0


def _evaluation(
    *,
    generated_answer: str,
    ground_truth_answer: str,
    question_type: str,
    answer_session_ids: list[str],
    retrieved_session_ids: list[str],
    diagnostics_available: bool = False,
) -> dict:
    search_results = [
        {"metadata": {"source_unit_id": session_id}}
        for session_id in retrieved_session_ids
    ]
    diagnostics = {"diagnostics_available": diagnostics_available}
    if diagnostics_available:
        diagnostics.update(
            {
                "raw_candidates": [],
                "ranked_candidates_canonical": [
                    {"id": "r1", "metadata": {"source_unit_id": answer_session_ids[0]}}
                ],
                "final_candidates_canonical": [
                    {"id": "r1", "metadata": {"source_unit_id": answer_session_ids[0]}}
                ],
            }
        )
    return {
        "generated_answer": generated_answer,
        "ground_truth_answer": ground_truth_answer,
        "question_type": question_type,
        "answer_session_ids": answer_session_ids,
        "cutoff_label": "native",
        "retrieval": {
            "memories_evaluated": len(search_results),
            "search_results": search_results,
            "recall_diagnostics": diagnostics,
        },
    }
