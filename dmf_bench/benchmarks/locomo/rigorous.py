"""Pure deterministic rigorous evaluator for LoCoMo v2 outputs."""

from __future__ import annotations

import math
import re
import string
from collections import defaultdict
from typing import Any


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalized_tokens(prediction)
    gold_tokens = normalized_tokens(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = defaultdict(int)
    gold_counts: dict[str, int] = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in gold_tokens:
        gold_counts[token] += 1
    overlap = sum(
        min(count, gold_counts.get(token, 0))
        for token, count in pred_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_index in range(1, len(left) + 1):
        for right_index in range(1, len(right) + 1):
            if left[left_index - 1] == right[right_index - 1]:
                table[left_index][right_index] = table[left_index - 1][right_index - 1] + 1
            else:
                table[left_index][right_index] = max(
                    table[left_index - 1][right_index],
                    table[left_index][right_index - 1],
                )
    return table[-1][-1]


def rouge_l(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalized_tokens(prediction)
    gold_tokens = normalized_tokens(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def retrieved_source_ids(search_results: list[dict[str, Any]], k: int) -> list[str]:
    ids: list[str] = []
    for item in search_results[:k]:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        single_id = metadata.get("source_unit_id")
        if isinstance(single_id, str) and single_id:
            ids.append(single_id)
        multiple_ids = metadata.get("source_unit_ids")
        if isinstance(multiple_ids, list):
            ids.extend(
                value
                for value in multiple_ids
                if isinstance(value, str) and value
            )
    return dedupe_preserve_order(ids)


def dedupe_preserve_order(ids: list[str]) -> list[str]:
    """Keep the first ranked occurrence of each source unit.

    A canonical memory may expose the same source through both
    ``source_unit_id`` and ``source_unit_ids``. Counting those aliases more
    than once inflates binary-gain DCG and can produce NDCG values above one.
    """

    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids) & gold_ids) / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return sum(1 for item in retrieved_ids if item in gold_ids) / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for index, item in enumerate(retrieved_ids, start=1):
        if item in gold_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(retrieved_ids, start=1)
        if item in gold_ids
    )
    ideal_hits = min(len(gold_ids), len(retrieved_ids))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def detected_cutoff_label(item: dict[str, Any], metadata: dict[str, Any]) -> str:
    raw_label = item.get("cutoff_label")
    if isinstance(raw_label, str) and raw_label.strip():
        return raw_label.strip()
    top_k = metadata.get("top_k")
    return f"top_{int(top_k)}" if top_k is not None else "flat"


def evaluate_flat(
    evaluations: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    metric_names = (
        "exact_match",
        "token_f1",
        "rouge_l",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
    )
    overall: dict[str, list[float]] = {name: [] for name in metric_names}
    by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {name: [] for name in metric_names}
    )
    cutoff_label = detected_cutoff_label(evaluations[0], metadata)

    for item in evaluations:
        prediction = str(item.get("generated_answer", item.get("prediction", "")))
        ground_truth = str(item.get("ground_truth_answer", item.get("ground_truth", "")))
        group = str(item.get("category_name", item.get("category", "unknown")))
        retrieval = item.get("retrieval") or {}
        search_results = retrieval.get("search_results") or []
        memories_evaluated = int(
            retrieval.get("memories_evaluated", len(search_results)) or 0
        )
        retrieved_ids = retrieved_source_ids(search_results, memories_evaluated)
        gold_ids = {
            evidence
            for evidence in item.get("evidence", [])
            if isinstance(evidence, str) and evidence
        }
        row = {
            "exact_match": exact_match(prediction, ground_truth),
            "token_f1": token_f1(prediction, ground_truth),
            "rouge_l": rouge_l(prediction, ground_truth),
            "recall_at_k": recall_at_k(retrieved_ids, gold_ids),
            "precision_at_k": precision_at_k(retrieved_ids, gold_ids),
            "mrr": reciprocal_rank(retrieved_ids, gold_ids),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, gold_ids),
        }
        for key, value in row.items():
            overall[key].append(value)
            by_group[group][key].append(value)

    return cutoff_label, {
        "overall": {key: average(values) for key, values in overall.items()},
        "by_group": {
            group: {key: average(values) for key, values in metrics.items()}
            for group, metrics in sorted(by_group.items())
        },
    }
