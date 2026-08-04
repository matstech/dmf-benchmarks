"""Explicit registry for benchmark, framework, and protocol names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    unit_type: str
    protocols: tuple[str, ...]


@dataclass(frozen=True)
class FrameworkInfo:
    name: str
    storage_backend: str


BENCHMARKS: dict[str, BenchmarkInfo] = {
    "locomo": BenchmarkInfo(
        name="locomo",
        unit_type="locomo-conversation",
        protocols=("native",),
    ),
    "longmemeval": BenchmarkInfo(
        name="longmemeval",
        unit_type="longmemeval-question",
        protocols=("native",),
    ),
}

FRAMEWORKS: dict[str, FrameworkInfo] = {
    "dmf": FrameworkInfo(name="dmf", storage_backend="qdrant-server"),
    "mem0": FrameworkInfo(name="mem0", storage_backend="qdrant-server"),
}

PROTOCOLS: tuple[str, ...] = ("native",)


def supported_combinations(
    *,
    benchmarks: dict[str, BenchmarkInfo] | None = None,
    frameworks: dict[str, FrameworkInfo] | None = None,
    protocols: tuple[str, ...] | None = None,
) -> list[tuple[str, str, str]]:
    """Return supported benchmark/framework/protocol triples in deterministic order."""
    selected_benchmarks = benchmarks or BENCHMARKS
    selected_frameworks = frameworks or FRAMEWORKS
    selected_protocols = protocols or PROTOCOLS
    return [
        (benchmark, framework, protocol)
        for benchmark in sorted(selected_benchmarks)
        for framework in sorted(selected_frameworks)
        for protocol in selected_protocols
        if protocol in selected_benchmarks[benchmark].protocols
    ]


def validate_combination(
    benchmark: str,
    framework: str,
    protocol: str,
    *,
    benchmarks: dict[str, BenchmarkInfo] | None = None,
    frameworks: dict[str, FrameworkInfo] | None = None,
    protocols: tuple[str, ...] | None = None,
) -> None:
    selected_benchmarks = benchmarks or BENCHMARKS
    selected_frameworks = frameworks or FRAMEWORKS
    selected_protocols = protocols or PROTOCOLS
    if benchmark not in selected_benchmarks:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}. Supported: {', '.join(sorted(selected_benchmarks))}."
        )
    if framework not in selected_frameworks:
        raise ValueError(
            f"Unsupported framework {framework!r}. Supported: {', '.join(sorted(selected_frameworks))}."
        )
    if protocol not in selected_protocols:
        raise ValueError(
            f"Unsupported protocol {protocol!r}. Supported: {', '.join(selected_protocols)}."
        )
    if protocol not in selected_benchmarks[benchmark].protocols:
        raise ValueError(
            f"Protocol {protocol!r} is not supported for benchmark {benchmark!r}."
        )
