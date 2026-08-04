"""Explicit registry for benchmark and framework names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    unit_type: str


@dataclass(frozen=True)
class FrameworkInfo:
    name: str
    storage_backend: str


BENCHMARKS: dict[str, BenchmarkInfo] = {
    "locomo": BenchmarkInfo(
        name="locomo",
        unit_type="locomo-conversation",
    ),
    "longmemeval": BenchmarkInfo(
        name="longmemeval",
        unit_type="longmemeval-question",
    ),
}

FRAMEWORKS: dict[str, FrameworkInfo] = {
    "dmf": FrameworkInfo(name="dmf", storage_backend="qdrant-server"),
    "mem0": FrameworkInfo(name="mem0", storage_backend="qdrant-server"),
}


def supported_combinations(
    *,
    benchmarks: dict[str, BenchmarkInfo] | None = None,
    frameworks: dict[str, FrameworkInfo] | None = None,
) -> list[tuple[str, str]]:
    """Return supported benchmark/framework pairs in deterministic order."""
    selected_benchmarks = benchmarks or BENCHMARKS
    selected_frameworks = frameworks or FRAMEWORKS
    return [
        (benchmark, framework)
        for benchmark in sorted(selected_benchmarks)
        for framework in sorted(selected_frameworks)
    ]


def validate_combination(
    benchmark: str,
    framework: str,
    *,
    benchmarks: dict[str, BenchmarkInfo] | None = None,
    frameworks: dict[str, FrameworkInfo] | None = None,
) -> None:
    selected_benchmarks = benchmarks or BENCHMARKS
    selected_frameworks = frameworks or FRAMEWORKS
    if benchmark not in selected_benchmarks:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}. Supported: {', '.join(sorted(selected_benchmarks))}."
        )
    if framework not in selected_frameworks:
        raise ValueError(
            f"Unsupported framework {framework!r}. Supported: {', '.join(sorted(selected_frameworks))}."
        )
