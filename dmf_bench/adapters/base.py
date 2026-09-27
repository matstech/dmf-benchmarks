"""Small Protocol contracts for benchmark, framework, answerer, and judge adapters."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


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


_EVALUATION_ONLY_KEYS = frozenset(
    {
        "answer", "expectedanswer", "groundtruth", "groundtruthanswer",
        "goldanswer", "evidence", "evidencerefs", "evidenceids",
        "judge", "judgeid", "judgemetadata", "rubric", "strata",
        "evaluationreference", "evaluationmetadata", "evaluatoronly",
        "correctanswer", "targetanswer", "label", "labels",
    }
)
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$")


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _json_value(value: Any, name: str) -> Any:
    """Check and detach JSON data; reject non-string keys and non-finite numbers."""
    active: set[int] = set()

    def check(item: Any) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{name} contains a non-finite number.")
            return
        if isinstance(item, Mapping):
            if id(item) in active:
                raise ValueError(f"{name} contains a cycle.")
            active.add(id(item))
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{name} contains a non-string JSON key.")
            for child in item.values():
                check(child)
            active.remove(id(item))
            return
        if isinstance(item, (list, tuple)):
            if id(item) in active:
                raise ValueError(f"{name} contains a cycle.")
            active.add(id(item))
            for child in item:
                check(child)
            active.remove(id(item))
            return
        raise ValueError(f"{name} must contain only JSON values.")

    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _safe_metadata(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    clean = _json_value(value, name)

    def check_keys(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if re.sub(r"[^a-z0-9]", "", key.lower()) in _EVALUATION_ONLY_KEYS:
                    raise ValueError(f"{name} contains an evaluator-only field: {key}.")
                check_keys(child)
        elif isinstance(item, list):
            for child in item:
                check_keys(child)

    check_keys(clean)
    return clean


def _timestamp(value: str | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _UTC_TIMESTAMP.fullmatch(value):
        raise ValueError(f"{name} must be a UTC RFC 3339 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid UTC RFC 3339 timestamp.") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must be UTC.")


def _unique_ids(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple.")
    for value in values:
        _nonempty(value, name)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate IDs.")


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    session_id: str
    sequence: int
    role: str
    content: str
    occurred_at: str | None
    source_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.event_id, "event_id")
        _nonempty(self.session_id, "session_id")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer.")
        if self.role not in {"user", "assistant", "system", "observation"}:
            raise ValueError("Unsupported memory event role.")
        _nonempty(self.content, "content")
        _timestamp(self.occurred_at, "occurred_at")
        _unique_ids(self.source_refs, "source_refs")
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata, "event metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "session_id": self.session_id,
                "sequence": self.sequence, "role": self.role, "content": self.content,
                "occurred_at": self.occurred_at, "source_refs": list(self.source_refs),
                "metadata": _safe_metadata(self.metadata, "event metadata")}


@dataclass(frozen=True)
class MemoryQuery:
    query_id: str
    text: str
    as_of: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.query_id, "query_id")
        _nonempty(self.text, "text")
        _timestamp(self.as_of, "as_of")
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata, "query metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {"query_id": self.query_id, "text": self.text, "as_of": self.as_of,
                "metadata": _safe_metadata(self.metadata, "query metadata")}


@dataclass(frozen=True)
class EvaluationReference:
    query_id: str
    expected_answer: Any
    evidence_refs: tuple[str, ...]
    strata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty(self.query_id, "query_id")
        object.__setattr__(self, "expected_answer", _json_value(self.expected_answer, "expected_answer"))
        _unique_ids(self.evidence_refs, "evidence_refs")
        if not isinstance(self.strata, Mapping):
            raise ValueError("strata must be a mapping.")
        object.__setattr__(self, "strata", _json_value(self.strata, "strata"))

    def to_dict(self) -> dict[str, Any]:
        return {"query_id": self.query_id, "expected_answer": _json_value(self.expected_answer, "expected_answer"),
                "evidence_refs": list(self.evidence_refs), "strata": _json_value(self.strata, "strata")}


@dataclass(frozen=True)
class BenchmarkCase:
    unit: BenchmarkUnit
    events: tuple[MemoryEvent, ...]
    queries: tuple[MemoryQuery, ...]
    references: Mapping[str, EvaluationReference]

    def __post_init__(self) -> None:
        if not isinstance(self.unit, BenchmarkUnit):
            raise ValueError("unit must be a BenchmarkUnit.")
        _nonempty(self.unit.unit_id, "unit_id")
        _unique_ids(self.unit.item_ids, "item_ids")
        if not self.unit.item_ids or not isinstance(self.events, tuple) or not isinstance(self.queries, tuple):
            raise ValueError("case requires item IDs and tuple events/queries.")
        if any(not isinstance(event, MemoryEvent) for event in self.events):
            raise ValueError("events must contain MemoryEvent values.")
        if any(not isinstance(query, MemoryQuery) for query in self.queries):
            raise ValueError("queries must contain MemoryQuery values.")
        _unique_ids(tuple(event.event_id for event in self.events), "event IDs")
        query_ids = tuple(query.query_id for query in self.queries)
        _unique_ids(query_ids, "query IDs")
        if query_ids != self.unit.item_ids:
            raise ValueError("query IDs must equal ordered unit item IDs.")
        sequences: dict[str, int] = {}
        for event in self.events:
            previous = sequences.get(event.session_id)
            if previous is not None and event.sequence <= previous:
                raise ValueError("event sequences must increase within each session.")
            sequences[event.session_id] = event.sequence
        if not isinstance(self.references, Mapping) or set(self.references) != set(query_ids):
            raise ValueError("references must contain exactly one entry per query.")
        for query_id, reference in self.references.items():
            if not isinstance(reference, EvaluationReference) or reference.query_id != query_id:
                raise ValueError("reference query ID mismatch.")
        object.__setattr__(self, "unit", BenchmarkUnit(
            self.unit.unit_id,
            self.unit.item_ids,
            _safe_metadata(self.unit.metadata, "unit metadata"),
        ))
        object.__setattr__(self, "references", dict(self.references))

    def to_dict(self) -> dict[str, Any]:
        return {"unit": {"unit_id": self.unit.unit_id, "item_ids": list(self.unit.item_ids),
                         "metadata": _safe_metadata(self.unit.metadata, "unit metadata")},
                "events": [event.to_dict() for event in self.events],
                "queries": [query.to_dict() for query in self.queries],
                "references": {query_id: reference.to_dict() for query_id, reference in self.references.items()}}


@dataclass(frozen=True)
class RetrievedMemory:
    memory_id: str
    content: str
    rank: int
    native_score: float | None = None
    native_score_kind: str | None = None
    occurred_at: str | None = None
    source_event_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.memory_id, "memory_id")
        _nonempty(self.content, "content")
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer.")
        if self.native_score is not None:
            if isinstance(self.native_score, bool) or not isinstance(self.native_score, (int, float)) or not math.isfinite(self.native_score):
                raise ValueError("native_score must be finite.")
            _nonempty(self.native_score_kind, "native_score_kind")
        elif self.native_score_kind is not None:
            raise ValueError("native_score_kind requires native_score.")
        _timestamp(self.occurred_at, "occurred_at")
        _unique_ids(self.source_event_ids, "source_event_ids")
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata, "memory metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "content": self.content, "rank": self.rank,
                "native_score": self.native_score, "native_score_kind": self.native_score_kind,
                "occurred_at": self.occurred_at, "source_event_ids": list(self.source_event_ids),
                "metadata": _safe_metadata(self.metadata, "memory metadata")}


@dataclass(frozen=True)
class CanonicalRetrievalResult:
    """Validated v3 retrieval data; legacy RetrievalResult remains for v2 callers."""

    items: tuple[RetrievedMemory, ...]
    raw_payload: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or any(not isinstance(item, RetrievedMemory) for item in self.items):
            raise ValueError("items must be a tuple of RetrievedMemory values.")
        _unique_ids(tuple(item.memory_id for item in self.items), "memory IDs")
        if tuple(item.rank for item in self.items) != tuple(range(1, len(self.items) + 1)):
            raise ValueError("retrieved memory ranks must be contiguous and ordered from one.")
        object.__setattr__(self, "raw_payload", _json_value(self.raw_payload, "raw_payload"))
        for name in ("diagnostics", "usage", "timing"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping.")
            object.__setattr__(self, name, _json_value(value, name))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 3,
                "items": [item.to_dict() for item in self.items],
                "raw_payload": _json_value(self.raw_payload, "raw_payload"),
                "diagnostics": _json_value(self.diagnostics, "diagnostics"),
                "usage": _json_value(self.usage, "usage"),
                "timing": _json_value(self.timing, "timing")}


@dataclass(frozen=True)
class OwnedResource:
    resource_id: str
    kind: str
    role: str
    locator: str

    def __post_init__(self) -> None:
        for name in ("resource_id", "kind", "role", "locator"):
            _nonempty(getattr(self, name), name)

    def to_dict(self) -> dict[str, str]:
        return {"resource_id": self.resource_id, "kind": self.kind,
                "role": self.role, "locator": self.locator}


@dataclass(frozen=True)
class PreparedMemoryUnit:
    handle: Any
    resources: tuple[OwnedResource, ...]
    ingestion_usage: Mapping[str, Any]
    ingestion_timing: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.resources, tuple) or any(not isinstance(item, OwnedResource) for item in self.resources):
            raise ValueError("resources must be a tuple of OwnedResource values.")
        _unique_ids(tuple(resource.resource_id for resource in self.resources), "resource IDs")
        for name in ("ingestion_usage", "ingestion_timing", "diagnostics"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping.")
            object.__setattr__(self, name, _json_value(value, name))

    def to_dict(self) -> dict[str, Any]:
        """Serialize persisted preparation evidence without the process-local handle."""
        return {"resources": [resource.to_dict() for resource in self.resources],
                "ingestion_usage": _json_value(self.ingestion_usage, "ingestion_usage"),
                "ingestion_timing": _json_value(self.ingestion_timing, "ingestion_timing"),
                "diagnostics": _json_value(self.diagnostics, "diagnostics")}


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
