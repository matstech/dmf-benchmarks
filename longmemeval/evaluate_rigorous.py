# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""
Offline deterministic evaluator for LongMemEval benchmark outputs.

Reads a unified LongMemEval results JSON (flat mono-cutoff schema or legacy
cutoff_results) and prints:

    - exact match
    - token F1
    - ROUGE-L
    - recall@k
    - precision@k
    - MRR
    - nDCG@k
    - retrieval latency summary

All metrics are broken down by question_type.

Supports both the current flat schema (``generated_answer`` at top level)
and the legacy ``cutoff_results`` schema for backward compatibility.

Methodological notes
--------------------
- Gold evidence is session-based: ``answer_session_ids``.
- Retrieved evidence is mapped from ``metadata.source_unit_id`` to session ids,
  then deduplicated preserving rank order.
- Abstention normalisation maps variants like "I don't know", "idk", etc.
  to a canonical token before computing answer metrics.

Usage::

    python3 -m longmemeval.evaluate_rigorous --input <results_json>
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


# ---------------------------------------------------------------------------
# Abstention normalisation
# ---------------------------------------------------------------------------

_ABSTAIN_TOKEN = "__ABSTAIN__"

_ABSTAIN_SYNONYMS = frozenset(
    {
        "i don't know",
        "i do not know",
        "idk",
        "not mentioned",
        "unknown",
        "not specified",
        "n/a",
        "none",
        "no information",
        "cannot be determined",
        "can't be determined",
        "not enough information",
    }
)


def normalize_abstention(text: str) -> str:
    """If *text* is an abstention variant, return ``__ABSTAIN__``; else return
    the original text unchanged."""
    stripped = text.strip().lower()
    stripped_no_punct = stripped.rstrip(string.punctuation).strip()
    if stripped_no_punct in _ABSTAIN_SYNONYMS:
        return _ABSTAIN_TOKEN
    return text


# ---------------------------------------------------------------------------
# Text normalisation & answer metrics
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


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

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, gold_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0

    rows = len(left) + 1
    cols = len(right) + 1
    table = [[0] * cols for _ in range(rows)]

    for i in range(1, rows):
        for j in range(1, cols):
            if left[i - 1] == right[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

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


# ---------------------------------------------------------------------------
# Retrieval helpers — session-level deduplication
# ---------------------------------------------------------------------------


def dedupe_preserve_order(ids: list[str]) -> list[str]:
    """Remove duplicate ids while preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def retrieved_session_ids(
    search_results: list[dict[str, Any]],
    k: int,
) -> list[str]:
    """Extract session ids from the top-*k* search results, deduplicated."""
    raw_ids: list[str] = []

    for item in search_results[:k]:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue

        single_id = metadata.get("source_unit_id")
        if isinstance(single_id, str) and single_id:
            raw_ids.append(single_id)
            continue

        multiple_ids = metadata.get("source_unit_ids")
        if isinstance(multiple_ids, list):
            for value in multiple_ids:
                if isinstance(value, str) and value:
                    raw_ids.append(value)

    return dedupe_preserve_order(raw_ids)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    hits = len(set(retrieved_ids) & gold_ids)
    return hits / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for item in retrieved_ids if item in gold_ids)
    return hits / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for index, item in enumerate(retrieved_ids, start=1):
        if item in gold_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0

    dcg = 0.0
    for index, item in enumerate(retrieved_ids, start=1):
        if item in gold_ids:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(gold_ids), len(retrieved_ids))
    if ideal_hits == 0:
        return 0.0

    idcg = 0.0
    for index in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(index + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_judgment_label(value: Any) -> str:
    return str(value or "").strip().upper()


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


def is_flat_result(item: dict[str, Any]) -> bool:
    """Return whether one evaluation item uses the flat mono-cutoff schema."""
    return (
        "generated_answer" in item
        and not isinstance(item.get("cutoff_results"), dict)
    )


def detected_cutoff_label(
    item: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Infer the cutoff label for a flat-schema evaluation item."""
    raw_label = item.get("cutoff_label")
    if isinstance(raw_label, str) and raw_label.strip():
        return raw_label.strip()

    top_k = metadata.get("top_k")
    if top_k is not None:
        return f"top_{int(top_k)}"

    return "flat"


def ensure_uniform_schema(evaluations: list[dict[str, Any]]) -> str:
    """Ensure all evaluations share one schema. Returns 'flat' or 'legacy'."""
    flat_count = sum(1 for item in evaluations if is_flat_result(item))

    if flat_count == len(evaluations):
        return "flat"
    if flat_count == 0:
        return "legacy"

    raise ValueError(
        "Mixed LongMemEval result schemas detected: "
        f"{flat_count} flat items and {len(evaluations) - flat_count} legacy items. "
        "All evaluations in one bundle must use the same schema."
    )


# ---------------------------------------------------------------------------
# Latency summary
# ---------------------------------------------------------------------------


def latency_summary(
    evaluations: list[dict[str, Any]],
    *,
    field_name: str,
    fallback_field_name: str | None = None,
) -> dict[str, Any]:
    overall_latencies: list[float] = []
    by_group: dict[str, list[float]] = defaultdict(list)

    for item in evaluations:
        retrieval = item.get("retrieval") or {}
        latency = retrieval.get(field_name)
        if latency is None and fallback_field_name is not None:
            latency = retrieval.get(fallback_field_name)
        if latency is None:
            continue

        latency_value = float(latency)
        group = str(item.get("question_type", "unknown"))
        overall_latencies.append(latency_value)
        by_group[group].append(latency_value)

    overall = {
        "count": len(overall_latencies),
        "avg_search_latency_ms": average(overall_latencies),
        "median_search_latency_ms": median(overall_latencies) if overall_latencies else 0.0,
        "min_search_latency_ms": min(overall_latencies) if overall_latencies else 0.0,
        "max_search_latency_ms": max(overall_latencies) if overall_latencies else 0.0,
    }

    grouped = {}
    for group, latencies in sorted(by_group.items()):
        grouped[group] = {
            "count": len(latencies),
            "avg_search_latency_ms": average(latencies),
            "median_search_latency_ms": median(latencies) if latencies else 0.0,
            "min_search_latency_ms": min(latencies) if latencies else 0.0,
            "max_search_latency_ms": max(latencies) if latencies else 0.0,
        }

    return {
        "overall": overall,
        "by_group": grouped,
    }


def judge_summary(evaluations: list[dict[str, Any]]) -> dict[str, Any] | None:
    overall_scores: list[float] = []
    overall_passes = 0
    by_group_scores: dict[str, list[float]] = defaultdict(list)
    by_group_passes: dict[str, int] = defaultdict(int)

    for item in evaluations:
        judgment = _normalize_judgment_label(item.get("judgment"))
        raw_score = item.get("score")
        if judgment not in {"CORRECT", "WRONG", "PASS", "FAIL"} or raw_score is None:
            continue

        score = float(raw_score)
        group = str(item.get("question_type", "unknown"))
        passed = judgment in {"CORRECT", "PASS"}

        overall_scores.append(score)
        if passed:
            overall_passes += 1
        by_group_scores[group].append(score)
        if passed:
            by_group_passes[group] += 1

    if not overall_scores:
        return None

    overall = {
        "judged_count": len(overall_scores),
        "avg_judge_score": average(overall_scores),
        "judge_pass_rate": overall_passes / len(overall_scores),
    }
    grouped = {}
    for group, scores in sorted(by_group_scores.items()):
        grouped[group] = {
            "judged_count": len(scores),
            "avg_judge_score": average(scores),
            "judge_pass_rate": by_group_passes[group] / len(scores),
        }

    return {
        "overall": overall,
        "by_group": grouped,
    }


# ---------------------------------------------------------------------------
# Evaluation — flat schema
# ---------------------------------------------------------------------------


def evaluate_flat(
    evaluations: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Compute all metrics for flat mono-cutoff evaluations."""
    overall: dict[str, list[float]] = {
        "exact_match": [],
        "token_f1": [],
        "rouge_l": [],
        "recall_at_k": [],
        "precision_at_k": [],
        "mrr": [],
        "ndcg_at_k": [],
    }
    by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "exact_match": [],
            "token_f1": [],
            "rouge_l": [],
            "recall_at_k": [],
            "precision_at_k": [],
            "mrr": [],
            "ndcg_at_k": [],
        }
    )

    cutoff_label = detected_cutoff_label(evaluations[0], metadata)
    abstention_count = 0

    for item in evaluations:
        group = str(item.get("question_type", "unknown"))

        # Answer metrics
        raw_prediction = str(item.get("generated_answer", ""))
        raw_ground_truth = str(item.get("ground_truth_answer", ""))

        prediction = normalize_abstention(raw_prediction)
        ground_truth = normalize_abstention(raw_ground_truth)

        if prediction == _ABSTAIN_TOKEN or ground_truth == _ABSTAIN_TOKEN:
            abstention_count += 1

        answer_row = {
            "exact_match": exact_match(prediction, ground_truth),
            "token_f1": token_f1(prediction, ground_truth),
            "rouge_l": rouge_l(prediction, ground_truth),
        }
        for key, value in answer_row.items():
            overall[key].append(value)
            by_group[group][key].append(value)

        # Retrieval metrics (session-level, deduplicated)
        retrieval = item.get("retrieval") or {}
        search_results = retrieval.get("search_results") or []
        memories_evaluated = int(
            retrieval.get("memories_evaluated", len(search_results)) or 0
        )

        gold_ids = {
            sid
            for sid in item.get("answer_session_ids", [])
            if isinstance(sid, str) and sid
        }

        if not search_results:
            for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"):
                overall[key].append(0.0)
                by_group[group][key].append(0.0)
            continue

        deduped_ids = retrieved_session_ids(search_results, memories_evaluated)

        retrieval_row = {
            "recall_at_k": recall_at_k(deduped_ids, gold_ids),
            "precision_at_k": precision_at_k(deduped_ids, gold_ids),
            "mrr": reciprocal_rank(deduped_ids, gold_ids),
            "ndcg_at_k": ndcg_at_k(deduped_ids, gold_ids),
        }
        for key, value in retrieval_row.items():
            overall[key].append(value)
            by_group[group][key].append(value)

    result = {
        "overall": {key: average(values) for key, values in overall.items()},
        "by_group": {
            group: {key: average(values) for key, values in metrics.items()}
            for group, metrics in sorted(by_group.items())
        },
        "abstention_count": abstention_count,
    }

    return cutoff_label, result


# ---------------------------------------------------------------------------
# Evaluation — legacy cutoff_results schema
# ---------------------------------------------------------------------------


def evaluate_legacy(
    evaluations: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Compute all metrics from legacy cutoff_results (picks first cutoff)."""
    # Find the single cutoff label
    cutoff_label: str | None = None
    for item in evaluations:
        cr = item.get("cutoff_results") or {}
        if cr:
            cutoff_label = next(iter(cr.keys()))
            break
    if not cutoff_label:
        raise ValueError("No cutoff_results found in legacy evaluations.")

    overall: dict[str, list[float]] = {
        "exact_match": [],
        "token_f1": [],
        "rouge_l": [],
        "recall_at_k": [],
        "precision_at_k": [],
        "mrr": [],
        "ndcg_at_k": [],
    }
    by_group: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "exact_match": [],
            "token_f1": [],
            "rouge_l": [],
            "recall_at_k": [],
            "precision_at_k": [],
            "mrr": [],
            "ndcg_at_k": [],
        }
    )

    abstention_count = 0

    for item in evaluations:
        cutoff_result = (item.get("cutoff_results") or {}).get(cutoff_label)
        if not isinstance(cutoff_result, dict):
            continue

        group = str(item.get("question_type", "unknown"))

        # Answer metrics
        raw_prediction = str(cutoff_result.get("generated_answer", ""))
        raw_ground_truth = str(item.get("ground_truth_answer", ""))

        prediction = normalize_abstention(raw_prediction)
        ground_truth = normalize_abstention(raw_ground_truth)

        if prediction == _ABSTAIN_TOKEN or ground_truth == _ABSTAIN_TOKEN:
            abstention_count += 1

        answer_row = {
            "exact_match": exact_match(prediction, ground_truth),
            "token_f1": token_f1(prediction, ground_truth),
            "rouge_l": rouge_l(prediction, ground_truth),
        }
        for key, value in answer_row.items():
            overall[key].append(value)
            by_group[group][key].append(value)

        # Retrieval metrics
        search_results = ((item.get("retrieval") or {}).get("search_results")) or []
        memories_evaluated = int(
            cutoff_result.get("memories_evaluated", len(search_results)) or 0
        )

        gold_ids = {
            sid
            for sid in item.get("answer_session_ids", [])
            if isinstance(sid, str) and sid
        }

        if not search_results:
            for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"):
                overall[key].append(0.0)
                by_group[group][key].append(0.0)
            continue

        deduped_ids = retrieved_session_ids(search_results, memories_evaluated)

        retrieval_row = {
            "recall_at_k": recall_at_k(deduped_ids, gold_ids),
            "precision_at_k": precision_at_k(deduped_ids, gold_ids),
            "mrr": reciprocal_rank(deduped_ids, gold_ids),
            "ndcg_at_k": ndcg_at_k(deduped_ids, gold_ids),
        }
        for key, value in retrieval_row.items():
            overall[key].append(value)
            by_group[group][key].append(value)

    result = {
        "overall": {key: average(values) for key, values in overall.items()},
        "by_group": {
            group: {key: average(values) for key, values in metrics.items()}
            for group, metrics in sorted(by_group.items())
        },
        "abstention_count": abstention_count,
    }

    return cutoff_label, result


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "exact_match",
    "token_f1",
    "rouge_l",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
)


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")

    overall = metrics["overall"]
    print("Overall")
    for key in _METRIC_KEYS:
        print(f"  {key}: {overall[key]:.4f}")

    print("By question_type")
    for group, group_metrics in metrics["by_group"].items():
        print(f"  {group}")
        for key in _METRIC_KEYS:
            print(f"    {key}: {group_metrics[key]:.4f}")


def print_latency_summary(summary: dict[str, Any]) -> None:
    print("\n=== retrieval latency ===")

    overall = summary["overall"]
    print("Overall")
    print(f"  count: {overall['count']}")
    print(f"  avg_search_latency_ms: {overall['avg_search_latency_ms']:.2f}")
    print(f"  median_search_latency_ms: {overall['median_search_latency_ms']:.2f}")
    print(f"  min_search_latency_ms: {overall['min_search_latency_ms']:.2f}")
    print(f"  max_search_latency_ms: {overall['max_search_latency_ms']:.2f}")

    print("By question_type")
    for group, group_summary in summary["by_group"].items():
        print(f"  {group}")
        print(f"    count: {group_summary['count']}")
        print(f"    avg_search_latency_ms: {group_summary['avg_search_latency_ms']:.2f}")
        print(f"    median_search_latency_ms: {group_summary['median_search_latency_ms']:.2f}")
        print(f"    min_search_latency_ms: {group_summary['min_search_latency_ms']:.2f}")
        print(f"    max_search_latency_ms: {group_summary['max_search_latency_ms']:.2f}")


def print_latency_summary_with_title(title: str, summary: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")

    overall = summary["overall"]
    print("Overall")
    print(f"  count: {overall['count']}")
    print(f"  avg_search_latency_ms: {overall['avg_search_latency_ms']:.2f}")
    print(f"  median_search_latency_ms: {overall['median_search_latency_ms']:.2f}")
    print(f"  min_search_latency_ms: {overall['min_search_latency_ms']:.2f}")
    print(f"  max_search_latency_ms: {overall['max_search_latency_ms']:.2f}")

    print("By question_type")
    for group, group_summary in summary["by_group"].items():
        print(f"  {group}")
        print(f"    count: {group_summary['count']}")
        print(f"    avg_search_latency_ms: {group_summary['avg_search_latency_ms']:.2f}")
        print(f"    median_search_latency_ms: {group_summary['median_search_latency_ms']:.2f}")
        print(f"    min_search_latency_ms: {group_summary['min_search_latency_ms']:.2f}")
        print(f"    max_search_latency_ms: {group_summary['max_search_latency_ms']:.2f}")


def print_judge_summary(summary: dict[str, Any]) -> None:
    print("\n=== judge metrics ===")

    overall = summary["overall"]
    print("Overall")
    print(f"  judged_count: {overall['judged_count']}")
    print(f"  avg_judge_score: {overall['avg_judge_score']:.4f}")
    print(f"  judge_pass_rate: {overall['judge_pass_rate']:.4f}")

    print("By question_type")
    for group, group_summary in summary["by_group"].items():
        print(f"  {group}")
        print(f"    judged_count: {group_summary['judged_count']}")
        print(f"    avg_judge_score: {group_summary['avg_judge_score']:.4f}")
        print(f"    judge_pass_rate: {group_summary['judge_pass_rate']:.4f}")


def print_token_accounting(data: dict[str, Any]) -> None:
    token_accounting = data.get("token_accounting")
    if not isinstance(token_accounting, dict):
        token_accounting = {}

    answerer = token_accounting.get("answerer")
    if not isinstance(answerer, dict):
        answerer = data.get("answerer_usage")

    if isinstance(answerer, dict):
        answerer_total = int(answerer.get("total_tokens", 0) or 0)
        if answerer_total > 0:
            print("\n=== token accounting (answerer) ===")
            print(f"  prompt_tokens_total: {int(answerer.get('prompt_tokens_total', 0))}")
            print(f"  completion_tokens: {int(answerer.get('completion_tokens', 0))}")
            print(f"  total_tokens: {answerer_total}")

    memory_internal = token_accounting.get("memory_internal")
    if not isinstance(memory_internal, dict):
        return

    memory_total = int(memory_internal.get("total_tokens", 0) or 0)
    if memory_total == 0:
        return

    print("\n=== token accounting (memory internal) ===")
    print(f"  prompt_tokens: {int(memory_internal.get('prompt_tokens', 0))}")
    print(f"  completion_tokens: {int(memory_internal.get('completion_tokens', 0))}")
    print(f"  total_tokens: {memory_total}")


def print_metric_descriptions(data: dict[str, Any]) -> None:
    metric_descriptions = data.get("metric_descriptions")
    if not isinstance(metric_descriptions, dict) or not metric_descriptions:
        return

    ordered_keys = (
        "exact_match",
        "token_f1",
        "rouge_l",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
        "count",
        "avg_search_latency_ms",
        "median_search_latency_ms",
        "min_search_latency_ms",
        "max_search_latency_ms",
        "prompt_tokens_total",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens",
        "calls",
        "judged_count",
        "avg_judge_score",
        "judge_pass_rate",
    )

    print("\n=== metric descriptions ===")
    for key in ordered_keys:
        description = metric_descriptions.get(key)
        if isinstance(description, str) and description.strip():
            print(f"  {key}: {description.strip()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deterministic answer and retrieval metrics for LongMemEval results.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a LongMemEval unified results JSON file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    data = json.loads(input_path.read_text())

    # Benchmark gate
    metadata = data.get("metadata") or {}
    benchmark = metadata.get("benchmark")
    if benchmark != "longmemeval":
        raise SystemExit(
            f"Unsupported benchmark: {benchmark!r}. "
            "This script only supports 'longmemeval'."
        )

    # Evaluations
    evaluations = data.get("evaluations", [])
    if not isinstance(evaluations, list) or not evaluations:
        raise SystemExit("No evaluations found in input JSON.")

    # Schema detection
    schema = ensure_uniform_schema(evaluations)

    # Header
    framework = metadata.get("framework", "unknown")
    print(f"Input: {input_path}")
    print(f"Benchmark: {benchmark}")
    print(f"Framework: {framework}")
    print(f"Questions: {len(evaluations)}")

    # Latency
    pipeline_latency = latency_summary(
        evaluations,
        field_name="retrieval_pipeline_latency_ms",
        fallback_field_name="search_latency_ms",
    )
    print_latency_summary_with_title("retrieval latency (pipeline)", pipeline_latency)

    backend_latency = latency_summary(
        evaluations,
        field_name="backend_search_latency_ms",
    )
    if backend_latency["overall"]["count"] > 0:
        print_latency_summary_with_title("retrieval latency (backend)", backend_latency)

    # Metrics
    if schema == "flat":
        cutoff_label, metrics = evaluate_flat(evaluations, metadata)
    else:
        cutoff_label, metrics = evaluate_legacy(evaluations)

    print(f"Cutoff: {cutoff_label}")
    print_metrics(cutoff_label, metrics)

    # Abstention count
    abstention_count = metrics.get("abstention_count", 0)
    if abstention_count:
        print(f"\nAbstentions (prediction or gold): {abstention_count}/{len(evaluations)}")

    judge_metrics = judge_summary(evaluations)
    if judge_metrics is not None:
        print_judge_summary(judge_metrics)

    # Token accounting
    print_token_accounting(data)

    # Metric descriptions
    print_metric_descriptions(data)


if __name__ == "__main__":
    main()
