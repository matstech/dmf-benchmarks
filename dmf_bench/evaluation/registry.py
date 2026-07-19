"""Explicit benchmark/protocol evaluation requirements."""

from __future__ import annotations

from dataclasses import dataclass


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
    protocol: str,
    framework: str,
    plans: dict[tuple[str, str, str], tuple[EvaluationRequirement, ...]] | None = None,
) -> tuple[EvaluationRequirement, ...]:
    """Return an explicit evaluator table for one benchmark/protocol/framework."""
    if plans is not None:
        key = (benchmark, protocol, framework)
        if key not in plans:
            raise ValueError(f"Unsupported evaluation plan: {benchmark!r}/{protocol!r}/{framework!r}.")
        return plans[key]

    if benchmark not in {"locomo", "longmemeval"}:
        raise ValueError(f"Unsupported benchmark for evaluation: {benchmark!r}.")
    if protocol not in {"strict", "native"}:
        raise ValueError(f"Unsupported protocol for evaluation: {protocol!r}.")
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
