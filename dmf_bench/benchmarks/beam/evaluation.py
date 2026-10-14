"""BEAM report grouped by published ability and context scale."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .spec import EVALUATOR_VERSION, PROTOCOL_LABEL


def rigorous_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: defaultdict(list) for name in ("scale", "ability", "category", "difficulty")
    }
    rows = []
    for item in evaluations:
        if item.get("protocol_label") != PROTOCOL_LABEL:
            raise ValueError("BEAM report requires the retrieval-controlled protocol label.")
        timing = item.get("retrieval", {}).get("timing", {})
        row = {
            "question_id": item["question_id"], "scale": item["scale"],
            "ability": item["ability"], "category": item["category"],
            "difficulty": item["difficulty"], "context_chars": item["context_chars"],
            "score": float(item["score"]),
            "answerer_tokens": item.get("answerer_usage", {}).get("total_tokens", 0),
            "judge_tokens": item.get("judge_usage", {}).get("total_tokens", 0),
            "retrieval_ms": timing.get("retrieval_pipeline_ms"),
            "bounded_context": item.get("retrieval", {}).get("diagnostics", {}).get("baseline_family") == "bounded-full-context",
        }
        rows.append(row)
        for name in groups:
            groups[name][str(row[name])].append(row)

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        times = [float(row["retrieval_ms"]) for row in values if row["retrieval_ms"] is not None]
        return {"count": len(values),
                "mean_score": sum(row["score"] for row in values) / len(values),
                "mean_answerer_tokens": sum(row["answerer_tokens"] for row in values) / len(values),
                "mean_judge_tokens": sum(row["judge_tokens"] for row in values) / len(values),
                "mean_retrieval_ms": sum(times) / len(times) if times else None,
                "bounded_count": sum(row["bounded_context"] for row in values)}

    return {
        "schema_version": 3, "benchmark": "beam", "evaluator": "rigorous_report",
        "evaluator_version": EVALUATOR_VERSION, "status": "COMPLETED",
        "protocol_label": PROTOCOL_LABEL, "framework": metadata.get("framework"),
        "count": len(rows),
        "by": {name: {key: summarize(values) for key, values in sorted(group.items())}
               for name, group in groups.items()},
        "rows": rows,
    }
