import json
from pathlib import Path

from longmemeval.utils import (
    filter_questions_by_ids,
    normalize_longmemeval_haystack,
    sample_questions_stratified,
)


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
