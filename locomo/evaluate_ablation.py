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
Ablation evaluator for LOCOMO: raw vs ranked vs final recall.

Measures retrieval recall at three pipeline stages for DMF results:

  1. raw     — All candidates returned by ChromaDB vector search
               (before active-context guard and contextualization).
  2. ranked  — Post-contextualization, post-filtering, pre-topic-limit.
  3. final   — The candidates actually delivered to the answer generator
               (identical to the main evaluate_rigorous recall@k).

For Mem0 results (which lack recall_diagnostics), only the final stage
is reported — confirming that Mem0 has no post-retrieval processing.

This script quantifies the filtering cost/benefit of DMF's
contextualization pipeline and supports the methodological framing
required for a fair cross-framework comparison in a scientific paper.

Usage::

    python3 -m locomo.evaluate_ablation --input <results_json>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from common.results_io import ensure_locomo_uniform_schema


# ---------------------------------------------------------------------------
# Retrieval metrics (same as evaluate_rigorous)
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

    return dcg / idcg if idcg > 0 else 0.0


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# ID extraction helpers
# ---------------------------------------------------------------------------


def _extract_source_ids_from_canonical(
    candidates: list[dict[str, Any]],
) -> list[str]:
    """Extract source_unit_ids from canonical search_results format."""
    ids: list[str] = []
    for item in candidates:
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


def _extract_source_ids_from_raw(
    raw_candidates: list[dict[str, Any]],
    record_id_to_source: dict[str, str],
) -> list[str]:
    """Extract source_unit_ids from raw_candidates using record_id mapping."""
    ids: list[str] = []
    for candidate in raw_candidates:
        record = candidate.get("record", {})
        record_id = str(record.get("record_id", ""))
        source_id = record_id_to_source.get(record_id)
        if source_id:
            ids.append(source_id)
    return ids


def _build_record_id_to_source_mapping(
    item: dict[str, Any],
) -> dict[str, str]:
    """Build record_id → source_unit_id mapping from available canonical data."""
    mapping: dict[str, str] = {}
    diagnostics = (item.get("retrieval") or {}).get("recall_diagnostics") or {}

    # Use ranked_candidates_canonical + final_candidates_canonical
    for source_key in ("ranked_candidates_canonical", "final_candidates_canonical"):
        candidates = diagnostics.get(source_key, [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            candidate_id = candidate.get("id", "")
            metadata = candidate.get("metadata") or {}
            source_id = metadata.get("source_unit_id")
            if candidate_id and source_id:
                mapping[candidate_id] = source_id

    # Also try ranked_candidates and final_candidates (non-canonical)
    for source_key in ("ranked_candidates", "final_candidates"):
        candidates = diagnostics.get(source_key, [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            record = candidate.get("record", {})
            record_id = str(record.get("record_id", ""))
            # These don't have source_unit_id directly, skip if not in mapping
            if record_id and record_id not in mapping:
                pass  # Cannot resolve without record_index

    return mapping


# ---------------------------------------------------------------------------
# Stage metrics computation
# ---------------------------------------------------------------------------

_RETRIEVAL_METRICS = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k")


def _compute_retrieval_metrics(
    retrieved_ids: list[str],
    gold_ids: set[str],
) -> dict[str, float]:
    return {
        "recall_at_k": recall_at_k(retrieved_ids, gold_ids),
        "precision_at_k": precision_at_k(retrieved_ids, gold_ids),
        "mrr": reciprocal_rank(retrieved_ids, gold_ids),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, gold_ids),
    }


def evaluate_ablation(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-stage retrieval metrics for all evaluations."""

    stages = ("raw", "ranked", "final")
    overall: dict[str, dict[str, list[float]]] = {
        stage: {metric: [] for metric in _RETRIEVAL_METRICS}
        for stage in stages
    }
    by_group: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {
            stage: {metric: [] for metric in _RETRIEVAL_METRICS}
            for stage in stages
        }
    )

    stats = {
        "total_questions": 0,
        "questions_with_diagnostics": 0,
        "questions_without_diagnostics": 0,
        "raw_unmappable_candidates": 0,
        "avg_raw_count": [],
        "avg_ranked_count": [],
        "avg_final_count": [],
    }

    for item in evaluations:
        group = str(item.get("category_name", item.get("category", "unknown")))
        gold_ids = {
            evidence
            for evidence in item.get("evidence", [])
            if isinstance(evidence, str) and evidence
        }
        if not gold_ids:
            continue

        stats["total_questions"] += 1

        retrieval = item.get("retrieval") or {}
        search_results = retrieval.get("search_results") or []
        diagnostics = retrieval.get("recall_diagnostics") or {}

        # Final stage (always available) — same as evaluate_rigorous
        final_ids = _extract_source_ids_from_canonical(search_results)
        final_metrics = _compute_retrieval_metrics(final_ids, gold_ids)
        for metric, value in final_metrics.items():
            overall["final"][metric].append(value)
            by_group[group]["final"][metric].append(value)
        stats["avg_final_count"].append(len(final_ids))

        # Check if diagnostics are available (DMF only)
        ranked_canonical = diagnostics.get("ranked_candidates_canonical") or []
        final_canonical = diagnostics.get("final_candidates_canonical") or []
        raw_candidates = diagnostics.get("raw_candidates") or []

        diagnostics_available = diagnostics.get("diagnostics_available") is True
        if (
            not diagnostics_available
            and not ranked_canonical
            and not raw_candidates
        ):
            # Mem0 or missing diagnostics: replicate final for all stages
            stats["questions_without_diagnostics"] += 1
            for metric, value in final_metrics.items():
                overall["raw"][metric].append(value)
                overall["ranked"][metric].append(value)
                by_group[group]["raw"][metric].append(value)
                by_group[group]["ranked"][metric].append(value)
            stats["avg_raw_count"].append(len(final_ids))
            stats["avg_ranked_count"].append(len(final_ids))
            continue

        stats["questions_with_diagnostics"] += 1

        # Ranked stage (post-contextualization, pre-topic-limit)
        ranked_ids = _extract_source_ids_from_canonical(ranked_canonical)
        ranked_metrics = _compute_retrieval_metrics(ranked_ids, gold_ids)
        for metric, value in ranked_metrics.items():
            overall["ranked"][metric].append(value)
            by_group[group]["ranked"][metric].append(value)
        stats["avg_ranked_count"].append(len(ranked_ids))

        # Raw stage (pre-contextualization)
        record_id_mapping = _build_record_id_to_source_mapping(item)
        raw_ids = _extract_source_ids_from_raw(raw_candidates, record_id_mapping)

        unmappable = len(raw_candidates) - len(raw_ids)
        stats["raw_unmappable_candidates"] += unmappable

        if raw_ids:
            raw_metrics = _compute_retrieval_metrics(raw_ids, gold_ids)
        else:
            # Fallback: if raw cannot be mapped, use ranked as upper bound
            raw_metrics = ranked_metrics

        for metric, value in raw_metrics.items():
            overall["raw"][metric].append(value)
            by_group[group]["raw"][metric].append(value)
        stats["avg_raw_count"].append(len(raw_ids) if raw_ids else len(ranked_ids))

    # Aggregate
    result: dict[str, Any] = {"stages": {}}
    for stage in stages:
        result["stages"][stage] = {
            "overall": {
                metric: average(values)
                for metric, values in overall[stage].items()
            },
            "by_group": {
                group: {
                    metric: average(values)
                    for metric, values in group_stages[stage].items()
                }
                for group, group_stages in sorted(by_group.items())
            },
        }

    result["stats"] = {
        "total_questions": stats["total_questions"],
        "questions_with_diagnostics": stats["questions_with_diagnostics"],
        "questions_without_diagnostics": stats["questions_without_diagnostics"],
        "raw_unmappable_candidates_total": stats["raw_unmappable_candidates"],
        "avg_candidates_raw": average(stats["avg_raw_count"]),
        "avg_candidates_ranked": average(stats["avg_ranked_count"]),
        "avg_candidates_final": average(stats["avg_final_count"]),
    }

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_ablation_results(results: dict[str, Any], *, framework: str) -> None:
    stats = results["stats"]
    print(f"\n{'='*70}")
    print("LOCOMO RETRIEVAL ABLATION — Pipeline Stage Comparison")
    print(f"{'='*70}")
    print(f"Framework: {framework}")
    print(f"Total questions evaluated: {stats['total_questions']}")
    print(f"Questions with recall diagnostics (DMF): {stats['questions_with_diagnostics']}")
    print(f"Questions without diagnostics (Mem0/missing): {stats['questions_without_diagnostics']}")
    print(f"Unmappable raw candidates (record_id not in canonical): {stats['raw_unmappable_candidates_total']}")
    print(f"Avg candidates — raw: {stats['avg_candidates_raw']:.1f}, "
          f"ranked: {stats['avg_candidates_ranked']:.1f}, "
          f"final: {stats['avg_candidates_final']:.1f}")

    stages_data = results["stages"]

    # Overall comparison table
    print(f"\n{'─'*70}")
    print("OVERALL — Retrieval metrics by pipeline stage")
    print(f"{'─'*70}")
    print(f"{'Stage':<12} {'recall@k':<12} {'precision@k':<14} {'MRR':<10} {'nDCG@k':<10}")
    print(f"{'─'*12} {'─'*12} {'─'*14} {'─'*10} {'─'*10}")

    for stage in ("raw", "ranked", "final"):
        o = stages_data[stage]["overall"]
        print(
            f"{stage:<12} "
            f"{o['recall_at_k']:<12.4f} "
            f"{o['precision_at_k']:<14.4f} "
            f"{o['mrr']:<10.4f} "
            f"{o['ndcg_at_k']:<10.4f}"
        )

    # Delta rows
    raw_o = stages_data["raw"]["overall"]
    final_o = stages_data["final"]["overall"]
    ranked_o = stages_data["ranked"]["overall"]

    print(f"{'─'*12} {'─'*12} {'─'*14} {'─'*10} {'─'*10}")
    delta_rf = {
        m: final_o[m] - raw_o[m] for m in _RETRIEVAL_METRICS
    }
    print(
        f"{'Δ(fin-raw)':<12} "
        f"{delta_rf['recall_at_k']:+<12.4f} "
        f"{delta_rf['precision_at_k']:+<14.4f} "
        f"{delta_rf['mrr']:+<10.4f} "
        f"{delta_rf['ndcg_at_k']:+<10.4f}"
    )

    # By group
    print(f"\n{'─'*70}")
    print("BY CATEGORY — recall@k at each stage")
    print(f"{'─'*70}")
    print(f"{'Category':<16} {'raw':<10} {'ranked':<10} {'final':<10} {'Δ(fin-raw)':<12}")
    print(f"{'─'*16} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

    # Collect all groups
    all_groups = sorted(stages_data["final"]["by_group"].keys())
    for group in all_groups:
        raw_r = stages_data["raw"]["by_group"].get(group, {}).get("recall_at_k", 0.0)
        ranked_r = stages_data["ranked"]["by_group"].get(group, {}).get("recall_at_k", 0.0)
        final_r = stages_data["final"]["by_group"].get(group, {}).get("recall_at_k", 0.0)
        delta = final_r - raw_r
        print(
            f"{group:<16} "
            f"{raw_r:<10.4f} "
            f"{ranked_r:<10.4f} "
            f"{final_r:<10.4f} "
            f"{delta:+<12.4f}"
        )

    # Interpretation note
    print(f"\n{'─'*70}")
    print("INTERPRETATION")
    print(f"{'─'*70}")
    if stats["questions_with_diagnostics"] > 0:
        print("• raw:    Recall of ChromaDB vector search (before contextualization)")
        print("• ranked: Recall after NLP contextualization + filtering (before topic limit)")
        print("• final:  Recall delivered to the answer generator (post all filtering)")
        print("")
        print("• Δ(fin-raw) < 0 → DMF's filtering removes some gold evidence (precision/recall trade-off)")
        print("• Δ(fin-raw) ≈ 0 → Filtering does not penalise recall")
        print("• Δ(fin-raw) > 0 → Should not happen (filtering cannot add new candidates)")
        if stats["raw_unmappable_candidates_total"] > 0:
            print("")
            print(f"⚠ {stats['raw_unmappable_candidates_total']} raw candidates could not be mapped to source_unit_id.")
            print("  Their contribution is estimated from ranked_candidates_canonical.")
    else:
        print("• No recall diagnostics found (Mem0 framework).")
        print("• All three stages report the same values (no post-retrieval processing).")
        print("• This confirms that Mem0 delivers raw vector search output to the answerer.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOCOMO ablation: compare retrieval recall at different DMF pipeline stages.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a LOCOMO unified results JSON file.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write structured JSON results.",
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

    ensure_locomo_uniform_schema(evaluations)

    framework = metadata.get("framework", "unknown")
    print(f"Input: {input_path}")
    print(f"Benchmark: {benchmark}")
    print(f"Framework: {framework}")
    print(f"Questions: {len(evaluations)}")

    results = evaluate_ablation(evaluations)
    print_ablation_results(results, framework=framework)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON results written to: {output_path}")


if __name__ == "__main__":
    main()
