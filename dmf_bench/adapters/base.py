"""Small Protocol contracts for benchmark, framework, answerer, and judge adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


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


@runtime_checkable
class BenchmarkAdapter(Protocol):
    name: str
    supported_protocols: tuple[str, ...]

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

    def generate(self, prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return one deterministic answer payload for the runner boundary."""


@runtime_checkable
class JudgeAdapter(Protocol):
    name: str

    def judge(self, question: str, expected: str, generated: str) -> dict[str, Any]:
        """Return one judge payload for the runner boundary."""


@dataclass(frozen=True)
class LocalFileResource:
    path: Path
    role: str
