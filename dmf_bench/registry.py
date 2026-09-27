"""Explicit descriptors and benchmark/framework compatibility records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


ComponentFactory = Callable[[dict[str, Any]], Any]


def _locomo_factory(config: dict[str, Any]) -> Any:
    from .benchmarks.locomo.adapter import LoCoMoAdapter

    return LoCoMoAdapter()


def _longmemeval_factory(config: dict[str, Any]) -> Any:
    from .benchmarks.longmemeval.adapter import LongMemEvalAdapter

    return LongMemEvalAdapter()


def _dmf_factory(config: dict[str, Any]) -> Any:
    from .adapters.dmf import dmf_framework_factories

    return dmf_framework_factories()["dmf"](config)


def _mem0_factory(config: dict[str, Any]) -> Any:
    from .adapters.mem0 import mem0_framework_factories

    return mem0_framework_factories()["mem0"](config)


@dataclass(frozen=True)
class BenchmarkDescriptor:
    name: str
    adapter_version: str
    lifecycle: str
    atomic_unit: str
    factory: ComponentFactory
    evaluator_profile: str

    @property
    def unit_type(self) -> str:
        """Preserve the existing CLI field name during the v3 cutover."""
        return self.atomic_unit


@dataclass(frozen=True)
class FrameworkDescriptor:
    name: str
    adapter_version: str
    factory: ComponentFactory
    config_formats: frozenset[str]
    storage_kinds: frozenset[str]
    capabilities: frozenset[str]
    distribution: str

    @property
    def storage_backend(self) -> str:
        """Preserve the existing CLI field name during the v3 cutover."""
        return sorted(self.storage_kinds)[0]


@dataclass(frozen=True)
class CompatibilityRecord:
    benchmark: str
    framework: str
    status: str
    evaluator_profile: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"certified", "experimental"}:
            raise ValueError(f"Unsupported compatibility status: {self.status!r}.")
        if not self.benchmark or not self.framework or not self.evaluator_profile:
            raise ValueError("Compatibility record identifiers cannot be empty.")


BENCHMARKS: dict[str, BenchmarkDescriptor] = {
    "locomo": BenchmarkDescriptor(
        "locomo", "locomo-v3", "retrieval-qa-v1", "locomo-conversation",
        _locomo_factory, "locomo-v3",
    ),
    "longmemeval": BenchmarkDescriptor(
        "longmemeval", "longmemeval-v3", "retrieval-qa-v1",
        "longmemeval-question", _longmemeval_factory, "longmemeval-v3",
    ),
}

FRAMEWORKS: dict[str, FrameworkDescriptor] = {
    "dmf": FrameworkDescriptor(
        "dmf", "dmf-v3", _dmf_factory, frozenset({"toml"}),
        frozenset({"qdrant-server"}),
        frozenset({"native-score", "usage", "timestamps", "source-provenance", "cleanup-manifest", "deterministic-resource-naming"}),
        "dmf-memory==0.3.0",
    ),
    "mem0": FrameworkDescriptor(
        "mem0", "mem0-v3", _mem0_factory, frozenset({"yaml"}),
        frozenset({"qdrant-server"}),
        frozenset({"native-score", "usage", "source-provenance", "cleanup-manifest", "deterministic-resource-naming"}),
        "mem0ai@8db3430d20f8b76cb7f80fb30df048321863392f",
    ),
}

COMPATIBILITY: dict[tuple[str, str], CompatibilityRecord] = {
    ("locomo", "dmf"): CompatibilityRecord(
        "locomo", "dmf", "certified", "locomo-v3",
        "Local adapter, Qdrant, resume, and deterministic container fixtures passed.",
    ),
    ("locomo", "mem0"): CompatibilityRecord(
        "locomo", "mem0", "certified", "locomo-v3",
        "Local adapter, Qdrant, resume, and deterministic container fixtures passed.",
    ),
    ("longmemeval", "dmf"): CompatibilityRecord(
        "longmemeval", "dmf", "certified", "longmemeval-v3",
        "Local adapter, Qdrant, resume, and deterministic container fixtures passed.",
    ),
    ("longmemeval", "mem0"): CompatibilityRecord(
        "longmemeval", "mem0", "certified", "longmemeval-v3",
        "Local adapter, Qdrant, resume, and deterministic container fixtures passed.",
    ),
}


def supported_combinations(
    *,
    benchmarks: Mapping[str, BenchmarkDescriptor] | None = None,
    frameworks: Mapping[str, FrameworkDescriptor] | None = None,
    compatibility: Mapping[tuple[str, str], CompatibilityRecord] | None = None,
) -> list[tuple[str, str]]:
    """Return only explicitly registered pairs in deterministic order."""
    selected_benchmarks = BENCHMARKS if benchmarks is None else benchmarks
    selected_frameworks = FRAMEWORKS if frameworks is None else frameworks
    selected_compatibility = COMPATIBILITY if compatibility is None else compatibility
    return sorted(
        pair for pair, record in selected_compatibility.items()
        if pair == (record.benchmark, record.framework)
        and record.status in {"certified", "experimental"}
        and pair[0] in selected_benchmarks
        and pair[1] in selected_frameworks
    )


def validate_combination(
    benchmark: str,
    framework: str,
    *,
    benchmarks: Mapping[str, BenchmarkDescriptor] | None = None,
    frameworks: Mapping[str, FrameworkDescriptor] | None = None,
    compatibility: Mapping[tuple[str, str], CompatibilityRecord] | None = None,
) -> CompatibilityRecord:
    selected_benchmarks = BENCHMARKS if benchmarks is None else benchmarks
    selected_frameworks = FRAMEWORKS if frameworks is None else frameworks
    selected_compatibility = COMPATIBILITY if compatibility is None else compatibility
    if benchmark not in selected_benchmarks:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}. Supported: {', '.join(sorted(selected_benchmarks))}."
        )
    if framework not in selected_frameworks:
        raise ValueError(
            f"Unsupported framework {framework!r}. Supported: {', '.join(sorted(selected_frameworks))}."
        )
    pair = (benchmark, framework)
    if pair not in supported_combinations(
        benchmarks=selected_benchmarks,
        frameworks=selected_frameworks,
        compatibility=selected_compatibility,
    ):
        raise ValueError(f"Unsupported benchmark/framework pair: {benchmark}/{framework}.")
    return selected_compatibility[pair]
