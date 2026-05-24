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
Offline deterministic evaluator for LOCOMO benchmark outputs.

This script reads an already-produced LOCOMO results JSON and prints:
- exact match
- token F1
- ROUGE-L
- recall@k
- precision@k
- MRR
- nDCG@k

It supports the current flat LOCOMO schema and the legacy cutoff-nested one.
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

from common.results_io import ensure_locomo_uniform_schema


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
            for value in multiple_ids:
                if isinstance(value, str) and value:
                    ids.append(value)

    return ids


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
        relevance = 1.0 if item in gold_ids else 0.0
        if relevance:
            dcg += relevance / math.log2(index + 1)

    ideal_hits = min(len(gold_ids), len(retrieved_ids))
    if ideal_hits == 0:
        return 0.0

    idcg = 0.0
    for index in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(index + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_judgment_label(value: Any) -> str:
    return str(value or "").strip().upper()


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
        group = str(item.get("category_name", item.get("category", "unknown")))
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
        group = str(item.get("category_name", item.get("category", "unknown")))
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


def is_flat_result(item: dict[str, Any]) -> bool:
    return (
        ("generated_answer" in item or "prediction" in item)
        and not isinstance(item.get("cutoff_results"), dict)
    )


def detected_cutoff_label(
    item: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    raw_label = item.get("cutoff_label")
    if isinstance(raw_label, str) and raw_label.strip():
        return raw_label.strip()

    top_k = metadata.get("top_k")
    if top_k is not None:
        return f"top_{int(top_k)}"

    return "flat"


def evaluate_flat(
    evaluations: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    overall = {
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

    for item in evaluations:
        prediction = str(item.get("generated_answer", item.get("prediction", "")))
        ground_truth = str(item.get("ground_truth_answer", item.get("ground_truth", "")))
        group = str(item.get("category_name", item.get("category", "unknown")))

        search_results = ((item.get("retrieval") or {}).get("search_results")) or []
        memories_evaluated = int(
            (item.get("retrieval") or {}).get("memories_evaluated", len(search_results)) or 0
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


def evaluate_cutoff(evaluations: list[dict[str, Any]], cutoff_label: str) -> dict[str, Any]:
    overall = {
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

    for item in evaluations:
        cutoff_result = (item.get("cutoff_results") or {}).get(cutoff_label)
        if not isinstance(cutoff_result, dict):
            continue

        prediction = str(cutoff_result.get("generated_answer", cutoff_result.get("prediction", "")))
        ground_truth = str(item.get("ground_truth_answer", item.get("ground_truth", "")))
        group = str(item.get("category_name", item.get("category", "unknown")))

        search_results = ((item.get("retrieval") or {}).get("search_results")) or []
        memories_evaluated = int(cutoff_result.get("memories_evaluated", len(search_results)) or 0)
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

    return {
        "overall": {key: average(values) for key, values in overall.items()},
        "by_group": {
            group: {key: average(values) for key, values in metrics.items()}
            for group, metrics in sorted(by_group.items())
        },
    }


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")

    overall = metrics["overall"]
    print("Overall")
    for key in (
        "exact_match",
        "token_f1",
        "rouge_l",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "ndcg_at_k",
    ):
        print(f"  {key}: {overall[key]:.4f}")

    print("By group")
    for group, group_metrics in metrics["by_group"].items():
        print(f"  {group}")
        for key in (
            "exact_match",
            "token_f1",
            "rouge_l",
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "ndcg_at_k",
        ):
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

    print("By group")
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

    print("By group")
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

    print("By group")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute deterministic answer and retrieval metrics for LOCOMO results.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a LOCOMO unified results JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    data = json.loads(input_path.read_text())

    metadata = data.get("metadata") or {}
    benchmark = metadata.get("benchmark")
    if benchmark != "locomo":
        raise SystemExit(f"Unsupported benchmark: {benchmark!r}. This script only supports 'locomo'.")

    evaluations = data.get("evaluations", [])
    if not isinstance(evaluations, list) or not evaluations:
        raise SystemExit("No evaluations found in input JSON.")

    schema = ensure_locomo_uniform_schema(evaluations)

    print(f"Input: {input_path}")
    print(f"Benchmark: {benchmark}")
    print(f"Questions: {len(evaluations)}")
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

    if schema == "flat":
        cutoff_label, metrics = evaluate_flat(evaluations, metadata)
        print(f"Cutoff: {cutoff_label}")
        print_metrics(cutoff_label, metrics)
        judge_metrics = judge_summary(evaluations)
        if judge_metrics is not None:
            print_judge_summary(judge_metrics)
        print_token_accounting(data)
        print_metric_descriptions(data)
        return

    first_cutoff_results = evaluations[0].get("cutoff_results") or {}
    cutoff_labels = list(first_cutoff_results.keys())
    if not cutoff_labels:
        raise SystemExit("No cutoff_results found in input JSON.")

    print(f"Cutoffs: {', '.join(cutoff_labels)}")
    for cutoff_label in cutoff_labels:
        metrics = evaluate_cutoff(evaluations, cutoff_label)
        print_metrics(cutoff_label, metrics)
    print_token_accounting(data)
    print_metric_descriptions(data)


if __name__ == "__main__":
    main()
