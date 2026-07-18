import json
from pathlib import Path

from longmemeval.prompts import build_answerer_user_prompt
from longmemeval.utils import (
    build_longmemeval_strict_session_substrate,
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


def test_longmemeval_haystack_is_chronological_before_strict_rendering() -> None:
    question = load_longmemeval_fixture()[0]

    haystack = normalize_longmemeval_haystack(question)
    strict_substrate = build_longmemeval_strict_session_substrate(question)

    assert [session["session_id"] for session in haystack] == ["session-a", "session-b"]
    assert list(strict_substrate) == ["session-a", "session-b"]
    assert strict_substrate["session-a"]["turns"][0] == {
        "role": "user",
        "content": "I prefer almonds as a snack.",
    }


def test_longmemeval_answerer_prompt_snapshot() -> None:
    prompt = build_answerer_user_prompt(
        "### Session 1:\nSession Date: 2024/02/01 (Thu) 08:00\nSession Content:\nUser: I prefer almonds.",
        "What snack does Sam prefer?",
        "2024/02/10 (Sat) 09:00",
    )

    assert prompt == (
        "I will give you several history chats between you and a user.\n"
        "Please answer the question based on the relevant chat history.\n\n\n"
        "History Chats:\n\n"
        "### Session 1:\n"
        "Session Date: 2024/02/01 (Thu) 08:00\n"
        "Session Content:\n"
        "User: I prefer almonds.\n\n"
        "Current Date: 2024/02/10 (Sat) 09:00\n"
        "Question: What snack does Sam prefer?\n"
        "Answer:"
    )
