"""Explicit benchmark/framework evaluation requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dmf_bench.registry import (
    BENCHMARKS,
    COMPATIBILITY,
    FRAMEWORKS,
    BenchmarkDescriptor,
    CompatibilityRecord,
    FrameworkDescriptor,
    validate_combination,
)


@dataclass(frozen=True)
class EvaluationRequirement:
    name: str
    required: bool
    not_applicable_reason: str | None = None

    @property
    def status(self) -> str:
        return "NOT_APPLICABLE" if self.not_applicable_reason else "REQUIRED" if self.required else "OPTIONAL"


def evaluation_plan_for(
    *,
    benchmark: str,
    framework: str,
    plans: dict[tuple[str, str], tuple[EvaluationRequirement, ...]] | None = None,
) -> tuple[EvaluationRequirement, ...]:
    """Return an explicit evaluator table for one benchmark/framework pair."""
    if plans is not None:
        key = (benchmark, framework)
        if key not in plans:
            raise ValueError(f"Unsupported evaluation plan: {benchmark!r}/{framework!r}.")
        return plans[key]

    if benchmark not in {"locomo", "longmemeval"}:
        raise ValueError(f"Unsupported benchmark for evaluation: {benchmark!r}.")
    if framework not in {"dmf", "mem0"}:
        raise ValueError(f"Unsupported framework for evaluation: {framework!r}.")

    ablation_not_applicable = None
    if framework != "dmf":
        ablation_not_applicable = "Ablation report requires DMF recall diagnostics; Mem0 has no post-retrieval stages."

    return (
        EvaluationRequirement("primary_judge_score", required=True),
        EvaluationRequirement("rigorous_report", required=True),
        EvaluationRequirement(
            "ablation_report",
            required=False,
            not_applicable_reason=ablation_not_applicable,
        ),
    )


_EVIDENCE_PROFILES = frozenset({"locomo-v3", "longmemeval-v3"})


def evaluation_plan_v3(
    benchmark: str,
    framework: str,
    judge_ids: tuple[str, ...],
    *,
    benchmarks: Mapping[str, BenchmarkDescriptor] | None = None,
    frameworks: Mapping[str, FrameworkDescriptor] | None = None,
    compatibility: Mapping[tuple[str, str], CompatibilityRecord] | None = None,
) -> tuple[EvaluationRequirement, ...]:
    """Resolve v3 evaluator applicability from registered pair and capabilities."""
    selected_benchmarks = BENCHMARKS if benchmarks is None else benchmarks
    selected_frameworks = FRAMEWORKS if frameworks is None else frameworks
    selected_compatibility = COMPATIBILITY if compatibility is None else compatibility
    pair = validate_combination(
        benchmark, framework, benchmarks=selected_benchmarks,
        frameworks=selected_frameworks, compatibility=selected_compatibility,
    )
    descriptor = selected_benchmarks[benchmark]
    memory = selected_frameworks[framework]
    if descriptor.lifecycle != "retrieval-qa-v1":
        raise ValueError("V3 evaluation requires a retrieval-qa-v1 benchmark.")
    if pair.evaluator_profile != descriptor.evaluator_profile:
        raise ValueError("Compatibility evaluator profile differs from benchmark descriptor.")
    if not isinstance(judge_ids, tuple) or not judge_ids or any(
        not isinstance(judge_id, str) or not judge_id.strip() for judge_id in judge_ids
    ) or len(set(judge_ids)) != len(judge_ids):
        raise ValueError("judge_ids must be a non-empty tuple of unique IDs.")

    evidence_available = descriptor.evaluator_profile in _EVIDENCE_PROFILES
    provenance_available = "source-provenance" in memory.capabilities
    if not evidence_available:
        retrieval_reason = "Benchmark descriptor has no official retrieval evidence."
    elif not provenance_available:
        retrieval_reason = "Framework descriptor has no source-provenance capability."
    else:
        retrieval_reason = None
    ablation_reason = None if framework == "dmf" else "DMF recall stages are unavailable for this framework."
    agreement_reason = None if len(judge_ids) > 1 else "Judge agreement requires at least two judges."
    return (
        EvaluationRequirement("primary_judge_score", required=True),
        EvaluationRequirement("rigorous_report", required=True),
        EvaluationRequirement("analysis_rows", required=True),
        EvaluationRequirement("retrieval_report", required=False,
                              not_applicable_reason=retrieval_reason),
        EvaluationRequirement("ablation_report", required=False,
                              not_applicable_reason=ablation_reason),
        EvaluationRequirement("judge_agreement", required=False,
                              not_applicable_reason=agreement_reason),
    )
