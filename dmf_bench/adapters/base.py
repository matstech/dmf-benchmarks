"""Small Protocol contracts for benchmark, framework, answerer, and judge adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


class ResumeCapability(str, Enum):
    RESTART_UNIT = "restart-unit"
    SNAPSHOT_UNIT = "snapshot-unit"
    RESUME_STEP = "resume-step"


class FrameworkCapability(str, Enum):
    NATIVE_SURFACE = "native-surface"
    USAGE = "usage"
    QDRANT_SERVER = "qdrant-server"
    CLEANUP_MANIFEST = "cleanup-manifest"


@dataclass(frozen=True)
class BenchmarkUnit:
    unit_id: str
    item_ids: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ProgressUpdate:
    """Framework-neutral progress for the activity inside one atomic unit."""

    stage: str
    label: str
    completed: int
    total: int
    item_label: str

    def __post_init__(self) -> None:
        if not self.stage or not self.label or not self.item_label:
            raise ValueError("Progress stage, label, and item label cannot be empty.")
        if self.completed < 0 or self.total < 0:
            raise ValueError("Progress counts cannot be negative.")
        if self.completed > self.total:
            raise ValueError("Progress completed count cannot exceed total count.")


ProgressReporter = Callable[[ProgressUpdate], None]


@dataclass(frozen=True)
class FrameworkRunContext:
    """Operational run identity required for isolated framework resources."""

    run_id: str
    scientific_fingerprint: str
    run_dir: Path
    progress_reporter: ProgressReporter | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def report_progress(self, update: ProgressUpdate) -> None:
        """Publish optional live progress without coupling an adapter to storage."""
        if self.progress_reporter is not None:
            self.progress_reporter(update)


@dataclass(frozen=True)
class AnswererRequest:
    """Complete prompt envelope passed to an answerer transport."""

    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeRequest:
    """Prediction and benchmark metadata required by a scientific judge."""

    prediction: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """Framework-neutral retrieval payload consumed by benchmark adapters."""

    cutoff_label: str
    search_results: tuple[dict[str, Any], ...] = ()
    native_context: Any = ""
    native_surface_diagnostics: dict[str, Any] = field(default_factory=dict)
    recall_diagnostics: dict[str, Any] = field(default_factory=dict)
    memory_internal_usage: dict[str, Any] = field(default_factory=dict)
    memories_evaluated: int = 0
    timing: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cutoff_label:
            raise ValueError("RetrievalResult.cutoff_label cannot be empty.")
        if self.memories_evaluated < 0:
            raise ValueError("RetrievalResult.memories_evaluated cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cutoff_label": self.cutoff_label,
            "search_results": [dict(item) for item in self.search_results],
            "native_context": self.native_context,
            "native_surface_diagnostics": dict(self.native_surface_diagnostics),
            "recall_diagnostics": dict(self.recall_diagnostics),
            "memory_internal_usage": dict(self.memory_internal_usage),
            "memories_evaluated": self.memories_evaluated,
            "timing": dict(self.timing),
        }


@runtime_checkable
class BenchmarkAdapter(Protocol):
    name: str

    def enumerate_units(self, config: dict[str, Any]) -> list[BenchmarkUnit]:
        """Return the ordered benchmark units for a resolved config."""


@runtime_checkable
class FrameworkAdapter(Protocol):
    name: str
    resume_capability: ResumeCapability
    capabilities: frozenset[FrameworkCapability]

    def resources_for_unit(self, run_hash: str, unit_id: str) -> Any:
        """Return all framework resources owned by one run/unit."""

    def validate_runtime(self) -> None:
        """Fail if the required backend/runtime is not available."""


@runtime_checkable
class AnswererAdapter(Protocol):
    name: str

    def generate(self, request: AnswererRequest) -> dict[str, Any]:
        """Return one deterministic answer payload for the runner boundary."""


@runtime_checkable
class JudgeAdapter(Protocol):
    name: str

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        """Return one judge payload for the runner boundary."""


@dataclass(frozen=True)
class LocalFileResource:
    path: Path
    role: str
