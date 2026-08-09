"""Pure LoCoMo category and answer normalization helpers."""

from __future__ import annotations

from .prompts import normalize_category_5_prediction


def category_name(category: int) -> str:
    return {
        1: "multi-hop",
        2: "temporal",
        3: "open-domain",
        4: "single-hop",
        5: "adversarial",
    }.get(category, f"category-{category}")


def normalize_generated_answer_for_category(
    *,
    category: int,
    generated_answer: str,
    ground_truth_answer: str,
) -> str:
    if category == 5:
        return normalize_category_5_prediction(
            generated_answer,
            ground_truth_answer,
        )
    return generated_answer
