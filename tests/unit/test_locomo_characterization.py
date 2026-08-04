import json
from pathlib import Path

from locomo.qa import enumerate_filtered_qa_items
from locomo.utils import (
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


def test_locomo_framework_serialization_keeps_framework_difference_visible() -> None:
    turn = load_locomo_fixture()[0]["conversation"]["session_2"][0]

    assert serialize_locomo_turn_for_dmf(turn) == (
        "I moved Pixel's bed near the kitchen window. "
        "Image: query cat bed. The image shows a blue pet bed."
    )
    assert serialize_locomo_turn_for_mem0(turn) == (
        "Alice: I moved Pixel's bed near the kitchen window. "
        "Shared image: query cat bed. The image shows a blue pet bed."
    )
