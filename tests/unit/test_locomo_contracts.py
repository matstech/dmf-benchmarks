import json
from pathlib import Path

from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.benchmarks.locomo.dataset import (
    serialize_locomo_turn_for_dmf,
    serialize_locomo_turn_for_mem0,
)
from dmf_bench.benchmarks.locomo.prompts import official_ground_truth_answer
from dmf_bench.benchmarks.locomo.questions import (
    category_name,
    normalize_generated_answer_for_category,
)


def load_locomo_fixture() -> list[dict]:
    fixture_path = Path(__file__).parents[1] / "fixtures" / "locomo-mini.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_locomo_category_filter_preserves_dataset_question_index_order() -> None:
    conversation = load_locomo_fixture()[0]

    selected = LoCoMoAdapter().enumerate_questions(
        conversation_id="conversation-0001",
        conversation_idx=0,
        conversation=conversation,
        allowed_categories=(2, 1),
    )

    assert [(item.question_idx, item.qa_item["question"]) for item in selected] == [
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


def test_locomo_categories_one_through_five_and_adversarial_target_are_stable() -> None:
    assert {category: category_name(category) for category in range(1, 6)} == {
        1: "multi-hop",
        2: "temporal",
        3: "open-domain",
        4: "single-hop",
        5: "adversarial",
    }
    assert official_ground_truth_answer(5, "private answer") == (
        "Not mentioned in the conversation"
    )
    assert normalize_generated_answer_for_category(
        category=5,
        generated_answer="(b)",
        ground_truth_answer="private answer",
    ) == "Not mentioned in the conversation"
