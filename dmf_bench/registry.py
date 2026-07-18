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
        protocols=("strict", "native"),
    ),
    "longmemeval": BenchmarkInfo(
        name="longmemeval",
        unit_type="longmemeval-question",
        protocols=("strict", "native"),
    ),
}

FRAMEWORKS: dict[str, FrameworkInfo] = {
    "dmf": FrameworkInfo(name="dmf", storage_backend="qdrant-server"),
    "mem0": FrameworkInfo(name="mem0", storage_backend="qdrant-server"),
}

PROTOCOLS: tuple[str, ...] = ("strict", "native")


def supported_combinations() -> list[tuple[str, str, str]]:
    """Return supported benchmark/framework/protocol triples in deterministic order."""
    return [
        (benchmark, framework, protocol)
        for benchmark in sorted(BENCHMARKS)
        for framework in sorted(FRAMEWORKS)
        for protocol in PROTOCOLS
        if protocol in BENCHMARKS[benchmark].protocols
    ]


def validate_combination(benchmark: str, framework: str, protocol: str) -> None:
    if benchmark not in BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}. Supported: {', '.join(sorted(BENCHMARKS))}."
        )
    if framework not in FRAMEWORKS:
        raise ValueError(
            f"Unsupported framework {framework!r}. Supported: {', '.join(sorted(FRAMEWORKS))}."
        )
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"Unsupported protocol {protocol!r}. Supported: {', '.join(PROTOCOLS)}."
        )
    if protocol not in BENCHMARKS[benchmark].protocols:
        raise ValueError(
            f"Protocol {protocol!r} is not supported for benchmark {benchmark!r}."
        )
