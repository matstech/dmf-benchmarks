import json
from pathlib import Path

from locomo.prompts import build_answerer_user_prompt
from locomo.qa import enumerate_filtered_qa_items
from locomo.utils import (
    build_locomo_dialog_substrate,
    list_locomo_session_keys,
    serialize_locomo_turn_for_dmf,
    serialize_locomo_turn_for_mem0,
)


def load_locomo_fixture() -> list[dict]:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "locomo-mini.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_locomo_category_filter_preserves_dataset_question_index_order() -> None:
    conversation = load_locomo_fixture()[0]

    selected = enumerate_filtered_qa_items(
        conversation["qa"],
        allowed_categories=[2, 1],
    )

    assert [(index, item["question"]) for index, item in selected] == [
        (0, "What is Alice's cat called?"),
        (1, "Where did Alice move the bed?"),
    ]


def test_locomo_dialog_substrate_uses_canonical_session_order() -> None:
    conversation_data = load_locomo_fixture()[0]["conversation"]

    assert list_locomo_session_keys(conversation_data) == ["session_1", "session_2"]

    substrate = build_locomo_dialog_substrate(conversation_data)

    assert list(substrate) == ["D1", "D2"]
    assert substrate["D1"]["turn_dia_ids"] == ["D1:1", "D1:2"]
    assert "Session Date: 9:00 am on 01 January, 2024" in substrate["D1"]["serialized_dialog"]
    assert "Alice: I adopted a grey cat named Pixel." in substrate["D1"]["serialized_dialog"]


def test_locomo_framework_serialization_keeps_protocol_difference_visible() -> None:
    turn = load_locomo_fixture()[0]["conversation"]["session_2"][0]

    assert serialize_locomo_turn_for_dmf(turn) == (
        "I moved Pixel's bed near the kitchen window. "
        "Image: query cat bed. The image shows a blue pet bed."
    )
    assert serialize_locomo_turn_for_mem0(turn) == (
        "Alice: I moved Pixel's bed near the kitchen window. "
        "Shared image: query cat bed. The image shows a blue pet bed."
    )


def test_locomo_prompt_snapshot_for_temporal_category() -> None:
    prompt = build_answerer_user_prompt(
        "Session Date: 9:00 am on 01 January, 2024\nAlice: Pixel likes salmon.",
        "When did Alice mention Pixel?",
        category=2,
        ground_truth_answer="01 January, 2024",
    )

    assert prompt == (
        "Session Date: 9:00 am on 01 January, 2024\n"
        "Alice: Pixel likes salmon.\n\n"
        "Based on the above context, write an answer in the form of a short phrase for the following question.\n"
        "Answer with exact words from the context whenever possible. Question: When did Alice mention Pixel? "
        "Use the Session Date of the conversation to answer with an approximate date. Short answer:"
    )
