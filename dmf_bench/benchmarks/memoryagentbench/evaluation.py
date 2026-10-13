"""Task-specific MemoryAgentBench scoring for the pinned upstream release."""

from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any, Sequence

from .spec import EVALUATOR_VERSION


def normalize_answer(value: str) -> str:
    text = value.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_output(value: str) -> str:
    match = re.search(r"(?:Answer:)(.*)(?:\n|$)", value, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"^(.*)(?:\n|$)", value)
    if match is None:
        return ""
    return re.sub(r"^Answer:", "", match.group(1).strip(), flags=re.IGNORECASE).strip()


def _answers(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("Ground truth answers must be a string or list.")
    result = []
    for item in value:
        if isinstance(item, list):
            result.extend(str(nested) for nested in item)
        else:
            result.append(str(item))
    if not result:
        raise ValueError("Ground truth answers are empty.")
    return result


def deterministic_score(task: str, prediction: str, expected: Any, *, movie_candidates: Sequence[str] = ()) -> dict[str, Any]:
    if task == "recsys":
        if not isinstance(expected, dict) or not isinstance(expected.get("names"), list):
            raise ValueError("Recsys reference requires official movie names.")
        predicted = parse_recommendations(prediction, movie_candidates)
        gold = expected["names"]
        score = sum(name in predicted[:5] for name in gold) / len(gold)
        return {"metric": "Recall@5", "score": score, "parsed_output": predicted[:20]}
    answers = _answers(expected)
    candidates = [parse_output(prediction)] if task in {"icl", "eventqa"} else [prediction, parse_output(prediction)]
    exact = max(normalize_answer(candidate) == normalize_answer(answer)
                for candidate in candidates for answer in answers)
    substring = max(normalize_answer(answer) in normalize_answer(candidate)
                    for candidate in candidates for answer in answers)
    metric = "exact_match" if task in {"icl", "detective_qa"} else "substring_exact_match"
    result = {
        "metric": metric,
        "score": float(exact if metric == "exact_match" else substring),
        "exact_match": bool(exact),
        "substring_exact_match": bool(substring),
        "parsed_output": parse_output(prediction),
    }
    if task == "eventqa":
        result["eventqa_recall"] = int(all(answer.lower() in prediction.lower() for answer in answers))
    return result


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, start=1):
        current = [row]
        for column, b in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (a != b)))
        previous = current
    return previous[-1]


def parse_recommendations(text: str, movie_candidates: Sequence[str]) -> list[str]:
    if "1." in text:
        recommendations = text.split("1.", maxsplit=1)[1]
    else:
        recommendations = text.replace(",", "\n")
    unique_movies = sorted(set(movie_candidates))
    exact = {movie.lower(): movie for movie in unique_movies}
    result = []
    for line in recommendations.split("\n"):
        item = re.sub(r"\([^()]*\)", "", line)
        item = re.sub(r"^(?:\d+[\.\)、]?\s*[-—–]?\s*)?", "", item)
        item = " ".join(item.split())
        if item.lower() in exact:
            result.append(exact[item.lower()])
        elif unique_movies:
            result.append(min(unique_movies, key=lambda movie: _edit_distance(item.lower(), movie.lower())))
    return result


def rigorous_report(evaluations: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "competence": defaultdict(list),
        "task": defaultdict(list),
        "source": defaultdict(list),
        "answer_format": defaultdict(list),
        "context_bucket": defaultdict(list),
    }
    rows = []
    for item in evaluations:
        if item.get("protocol_label") != "MemoryAgentBench retrieval-controlled adaptation":
            raise ValueError("MemoryAgentBench evaluation requires the retrieval-controlled protocol label.")
        score = float(item["score"])
        context_chars = int(item["context_chars"])
        bucket = "<=32k" if context_chars <= 32768 else "<=256k" if context_chars <= 262144 else ">256k"
        retrieval_timing = item.get("retrieval", {}).get("timing", {})
        row = {
            "question_id": item["question_id"],
            "official_qa_pair_id": item["official_qa_pair_id"],
            "competence": item["competence"],
            "task": item["task"],
            "source": item["source"],
            "official_metric": item["official_metric"],
            "score": score,
            "context_chars": context_chars,
            "answer_format": item["answer_format"],
            "answerer_tokens": item.get("answerer_usage", {}).get("total_tokens", 0),
            "judge_tokens": item.get("judge_usage", {}).get("total_tokens", 0),
            "retrieval_ms": retrieval_timing.get("retrieval_pipeline_ms"),
            "retrieved_tokens": item.get("packed_context", {}).get("original_tokens", 0),
            "included_tokens": item.get("packed_context", {}).get("included_tokens", 0),
        }
        rows.append(row)
        for dimension in ("competence", "task", "source", "answer_format"):
            groups[dimension][str(item[dimension])].append(row)
        groups["context_bucket"][bucket].append(row)

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        timed = [float(row["retrieval_ms"]) for row in values if row["retrieval_ms"] is not None]
        return {
            "count": len(values),
            "mean_score": sum(row["score"] for row in values) / len(values),
            "mean_score_percent": 100 * sum(row["score"] for row in values) / len(values),
            "mean_answerer_tokens": sum(row["answerer_tokens"] for row in values) / len(values),
            "mean_judge_tokens": sum(row["judge_tokens"] for row in values) / len(values),
            "mean_retrieval_ms": sum(timed) / len(timed) if timed else None,
        }
    return {
        "schema_version": 3,
        "benchmark": "memoryagentbench",
        "evaluator": "rigorous_report",
        "evaluator_version": EVALUATOR_VERSION,
        "status": "COMPLETED",
        "protocol_label": "MemoryAgentBench retrieval-controlled adaptation",
        "framework": metadata.get("framework"),
        "count": len(rows),
        "by": {
            dimension: {key: summarize(values)
                        for key, values in sorted(groups[dimension].items())}
            for dimension in groups
        },
        "rows": rows,
    }
