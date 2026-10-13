"""Pinned MemoryAgentBench task taxonomy and protocol labels."""

from __future__ import annotations


ADAPTER_VERSION = "memoryagentbench-v1"
EVALUATOR_VERSION = "memoryagentbench-evaluator-v1"
DATASET_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
CODE_REVISION = "538026089d1a8a8eff05121d0db89b388f360eba"
PROFILES = frozenset({
    "memoryagentbench-ar-pilot-v1",
    "memoryagentbench-core-v1",
    "memoryagentbench-full-v1",
})


def task_for_source(source: str) -> tuple[str, str]:
    if source.startswith("eventqa_"):
        return "eventqa", "substring_exact_match"
    if source.startswith("ruler_qa"):
        return "ruler_qa", "substring_exact_match"
    if source == "longmemeval_s*":
        return "longmemeval", "llm_judge"
    if source.startswith("icl_"):
        return "icl", "exact_match"
    if source == "recsys_redial_full":
        return "recsys", "Recall@5"
    if source == "detective_qa":
        return "detective_qa", "exact_match"
    if source.startswith("factconsolidation_"):
        return "factconsolidation", "substring_exact_match"
    if source == "infbench_sum_eng_shots2":
        raise ValueError("InfBench summarization requires the official three-call judge and is not supported in this retrieval-controlled profile.")
    raise ValueError(f"Unknown MemoryAgentBench source: {source!r}")


def format_for_task(task: str) -> str:
    return {
        "eventqa": "event-continuation",
        "ruler_qa": "short-answer",
        "longmemeval": "short-answer",
        "icl": "numeric-label",
        "recsys": "ranked-movie-list",
        "detective_qa": "multiple-choice",
        "factconsolidation": "short-answer",
    }[task]
