"""LoCoMo answer-target normalization shared by native runners."""

from __future__ import annotations


NO_INFORMATION_ANSWER = "Not mentioned in the conversation"


def official_ground_truth_answer(category: int, answer: str) -> str:
    if category == 5:
        return NO_INFORMATION_ANSWER
    return answer


def normalize_category_5_prediction(prediction: str, answer: str = "") -> str:
    normalized = prediction.strip().lower()
    if normalized in {"b", "(b)"} or normalized.startswith("(b)"):
        return NO_INFORMATION_ANSWER
    if normalized in {"a", "(a)"} or normalized.startswith("(a)"):
        return answer
    return prediction
