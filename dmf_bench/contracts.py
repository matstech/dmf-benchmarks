"""Serializable contracts shared by local state and artifact storage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .atomic_io import canonical_json_bytes


EXPERIMENT_CONFIG_SCHEMA_VERSION = 2
RUN_MANIFEST_SCHEMA_VERSION = 2
ATTEMPT_SCHEMA_VERSION = 2
LIFECYCLE_CHECKPOINT_SCHEMA_VERSION = 2
UNIT_CHECKPOINT_SCHEMA_VERSION = 2
EVALUATION_REF_SCHEMA_VERSION = 2
RUN_STATUS_SCHEMA_VERSION = 2
COMPLETION_MARKER_SCHEMA_VERSION = 2
PREDICTION_SCHEMA_VERSION = 2
JUDGMENT_SCHEMA_VERSION = 2
EVALUATION_SCHEMA_VERSION = 2
REPORT_SCHEMA_VERSION = 2
PUBLICATION_MANIFEST_SCHEMA_VERSION = 2
STRUCTURED_LOG_SCHEMA_VERSION = 2
SCIENTIFIC_FINGERPRINT_SCHEMA_VERSION = 2
PROVENANCE_SCHEMA_VERSION = 2


RunPhase = Literal[
    "PREFLIGHT",
    "RUNNING",
    "JUDGING",
    "EVALUATING",
    "REPORTING",
    "PUBLISHING",
    "VERIFYING",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_canonical_json(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def scientific_fingerprint(payload: dict[str, Any]) -> str:
    return hash_canonical_json(payload)


def assert_sha256(value: str, *, field_name: str = "sha256") -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest.")


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("artifact path cannot be empty.")
        assert_sha256(self.sha256, field_name="artifact sha256")
        if self.bytes < 0:
            raise ValueError("artifact bytes cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            bytes=int(data["bytes"]),
        )


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    scientific_fingerprint: str
    fingerprint_inputs: dict[str, Any]
    expected_item_ids: tuple[str, ...]
    atomic_unit: str
    resume_policy: str = "restart-unit"
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: int = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"RunManifest schema_version must be {RUN_MANIFEST_SCHEMA_VERSION}; "
                f"v{self.schema_version} state is not supported."
            )
        if not self.run_id:
            raise ValueError("run_id cannot be empty.")
        assert_sha256(self.scientific_fingerprint, field_name="scientific_fingerprint")
        if not self.expected_item_ids:
            raise ValueError("expected_item_ids cannot be empty.")
        if self.resume_policy not in {"restart-unit", "snapshot-unit", "resume-step"}:
            raise ValueError(f"Unsupported resume_policy: {self.resume_policy!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "scientific_fingerprint": self.scientific_fingerprint,
            "fingerprint_inputs": self.fingerprint_inputs,
            "expected_item_ids": list(self.expected_item_ids),
            "atomic_unit": self.atomic_unit,
            "resume_policy": self.resume_policy,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        return cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]),
            scientific_fingerprint=str(data["scientific_fingerprint"]),
            fingerprint_inputs=dict(data["fingerprint_inputs"]),
            expected_item_ids=tuple(str(item) for item in data["expected_item_ids"]),
            atomic_unit=str(data["atomic_unit"]),
            resume_policy=str(data.get("resume_policy", "restart-unit")),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    run_id: str
    started_at: str
    entry_phase: str = "RUNNING"
    resume: bool = False
    resumed_from_attempt_id: str | None = None
    status: str = "IN_PROGRESS"
    completed_at: str | None = None
    schema_version: int = ATTEMPT_SCHEMA_VERSION
    attempt_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != ATTEMPT_SCHEMA_VERSION:
            raise ValueError(
                f"Attempt schema_version must be {ATTEMPT_SCHEMA_VERSION}; "
                f"v{self.schema_version} state is not supported."
            )
        if not self.attempt_id or not self.run_id or not self.started_at:
            raise ValueError("attempt_id, run_id and started_at cannot be empty.")
        if self.entry_phase not in {
            "RUNNING",
            "JUDGING",
            "EVALUATING",
            "REPORTING",
            "PUBLISHING",
            "VERIFYING",
            "COMPLETED",
        }:
            raise ValueError(f"Unsupported attempt entry_phase: {self.entry_phase!r}.")
        if self.status not in {"IN_PROGRESS", "PARTIAL", "COMPLETED", "FAILED", "INTERRUPTED"}:
            raise ValueError(f"Unsupported attempt status: {self.status!r}.")
        if self.status in {"PARTIAL", "COMPLETED", "FAILED", "INTERRUPTED"} and not self.completed_at:
            raise ValueError("Terminal attempt status requires completed_at.")
        if self.attempt_digest:
            assert_sha256(self.attempt_digest, field_name="attempt_digest")

    def without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "entry_phase": self.entry_phase,
            "resume": self.resume,
            "resumed_from_attempt_id": self.resumed_from_attempt_id,
            "status": self.status,
            "completed_at": self.completed_at,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.without_digest()
        data["attempt_digest"] = self.attempt_digest or hash_canonical_json(data)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attempt":
        attempt = cls(
            schema_version=int(data["schema_version"]),
            attempt_id=str(data["attempt_id"]),
            run_id=str(data["run_id"]),
            started_at=str(data["started_at"]),
            entry_phase=str(data.get("entry_phase", "RUNNING")),
            resume=bool(data.get("resume", False)),
            resumed_from_attempt_id=(
                str(data["resumed_from_attempt_id"])
                if data.get("resumed_from_attempt_id") is not None
                else None
            ),
            status=str(data.get("status", "IN_PROGRESS")),
            completed_at=(
                str(data["completed_at"])
                if data.get("completed_at") is not None
                else None
            ),
            attempt_digest=str(data["attempt_digest"]),
        )
        if attempt.attempt_digest != hash_canonical_json(attempt.without_digest()):
            raise ValueError("Attempt digest mismatch.")
        return attempt


@dataclass(frozen=True)
class LifecycleCheckpoint:
    """Digest-protected checkpoint for one terminal lifecycle phase."""

    run_id: str
    attempt_id: str
    phase: str
    status: str
    scientific_fingerprint: str
    input_fingerprint: str
    expected_item_ids: tuple[str, ...]
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    predecessor_digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = LIFECYCLE_CHECKPOINT_SCHEMA_VERSION
    checkpoint_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != LIFECYCLE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "LifecycleCheckpoint schema_version must be "
                f"{LIFECYCLE_CHECKPOINT_SCHEMA_VERSION}; "
                f"v{self.schema_version} state is not supported."
            )
        if not self.run_id or not self.attempt_id:
            raise ValueError("run_id and attempt_id cannot be empty.")
        if self.phase not in {
            "RUNNING",
            "JUDGING",
            "EVALUATING",
            "REPORTING",
            "PUBLISHING",
            "VERIFYING",
        }:
            raise ValueError(f"Unsupported lifecycle phase: {self.phase!r}.")
        if self.status not in {"IN_PROGRESS", "COMMITTED", "FAILED"}:
            raise ValueError(f"Unsupported lifecycle checkpoint status: {self.status!r}.")
        assert_sha256(self.scientific_fingerprint, field_name="scientific_fingerprint")
        assert_sha256(self.input_fingerprint, field_name="input_fingerprint")
        if not self.expected_item_ids:
            raise ValueError("LifecycleCheckpoint expected_item_ids cannot be empty.")
        if self.predecessor_digest is not None:
            assert_sha256(self.predecessor_digest, field_name="predecessor_digest")
        if self.status == "COMMITTED" and not self.artifacts:
            raise ValueError("Committed lifecycle checkpoint requires artifacts.")
        if self.checkpoint_digest:
            assert_sha256(self.checkpoint_digest, field_name="checkpoint_digest")

    def without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "phase": self.phase,
            "status": self.status,
            "scientific_fingerprint": self.scientific_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "expected_item_ids": list(self.expected_item_ids),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "predecessor_digest": self.predecessor_digest,
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.without_digest()
        data["checkpoint_digest"] = self.checkpoint_digest or hash_canonical_json(data)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LifecycleCheckpoint":
        checkpoint = cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]),
            attempt_id=str(data["attempt_id"]),
            phase=str(data["phase"]),
            status=str(data["status"]),
            scientific_fingerprint=str(data["scientific_fingerprint"]),
            input_fingerprint=str(data["input_fingerprint"]),
            expected_item_ids=tuple(str(item) for item in data["expected_item_ids"]),
            artifacts=tuple(
                ArtifactRef.from_dict(artifact)
                for artifact in data.get("artifacts", [])
            ),
            predecessor_digest=(
                str(data["predecessor_digest"])
                if data.get("predecessor_digest") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
            checkpoint_digest=str(data["checkpoint_digest"]),
        )
        if checkpoint.checkpoint_digest != hash_canonical_json(checkpoint.without_digest()):
            raise ValueError("Lifecycle checkpoint digest mismatch.")
        return checkpoint


@dataclass(frozen=True)
class UnitCheckpoint:
    run_id: str
    unit_id: str
    status: str
    scientific_fingerprint: str
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    schema_version: int = UNIT_CHECKPOINT_SCHEMA_VERSION
    checkpoint_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != UNIT_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"UnitCheckpoint schema_version must be {UNIT_CHECKPOINT_SCHEMA_VERSION}; "
                f"v{self.schema_version} state is not supported."
            )
        if not self.run_id or not self.unit_id:
            raise ValueError("run_id and unit_id cannot be empty.")
        assert_sha256(self.scientific_fingerprint, field_name="scientific_fingerprint")
        if self.status not in {
            "PENDING",
            "PREPARING",
            "INGESTING",
            "PREDICTING",
            "PREDICTED",
            "JUDGING",
            "JUDGED",
            "COMMITTED",
            "FAILED",
        }:
            raise ValueError(f"Unsupported unit status: {self.status!r}.")
        if self.checkpoint_digest:
            assert_sha256(self.checkpoint_digest, field_name="checkpoint_digest")

    def without_digest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "unit_id": self.unit_id,
            "status": self.status,
            "scientific_fingerprint": self.scientific_fingerprint,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.without_digest()
        data["checkpoint_digest"] = self.checkpoint_digest or hash_canonical_json(data)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnitCheckpoint":
        artifacts = tuple(
            ArtifactRef.from_dict(artifact)
            for artifact in data.get("artifacts", [])
        )
        checkpoint = cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]),
            unit_id=str(data["unit_id"]),
            status=str(data["status"]),
            scientific_fingerprint=str(data["scientific_fingerprint"]),
            artifacts=artifacts,
            checkpoint_digest=str(data["checkpoint_digest"]),
        )
        expected_digest = hash_canonical_json(checkpoint.without_digest())
        if checkpoint.checkpoint_digest != expected_digest:
            raise ValueError("Unit checkpoint digest mismatch.")
        return checkpoint


@dataclass(frozen=True)
class EvaluationRef:
    evaluation_id: str
    status: str
    artifact: ArtifactRef
    schema_version: int = EVALUATION_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_REF_SCHEMA_VERSION:
            raise ValueError(
                f"EvaluationRef schema_version must be {EVALUATION_REF_SCHEMA_VERSION}; "
                f"v{self.schema_version} state is not supported."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "status": self.status,
            "artifact": self.artifact.to_dict(),
        }


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    state: str
    phase: str
    expected: int
    committed: int
    failed: int = 0
    expected_units: int | None = None
    committed_units: int | None = None
    failed_units: int = 0
    schema_version: int = RUN_STATUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_STATUS_SCHEMA_VERSION:
            raise ValueError(
                f"RunStatus schema_version must be {RUN_STATUS_SCHEMA_VERSION}; "
                f"v{self.schema_version} state is not supported."
            )

    def to_dict(self) -> dict[str, Any]:
        expected_units = self.expected if self.expected_units is None else self.expected_units
        committed_units = self.committed if self.committed_units is None else self.committed_units
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "state": self.state,
            "phase": self.phase,
            "items": {
                "expected": self.expected,
                "committed": self.committed,
                "failed": self.failed,
            },
            "units": {
                "expected": expected_units,
                "committed": committed_units,
                "failed": self.failed_units,
            },
        }


@dataclass(frozen=True)
class CompletionMarker:
    run_id: str
    scientific_fingerprint: str
    final_manifest_sha256: str
    verified_artifact_count: int
    schema_version: int = COMPLETION_MARKER_SCHEMA_VERSION
    state: str = "COMPLETED"

    def __post_init__(self) -> None:
        if self.schema_version != COMPLETION_MARKER_SCHEMA_VERSION:
            raise ValueError(
                "CompletionMarker schema_version must be "
                f"{COMPLETION_MARKER_SCHEMA_VERSION}; "
                f"v{self.schema_version} state is not supported."
            )
        if self.state != "COMPLETED":
            raise ValueError("CompletionMarker state must be COMPLETED.")
        assert_sha256(self.scientific_fingerprint, field_name="scientific_fingerprint")
        assert_sha256(self.final_manifest_sha256, field_name="final_manifest_sha256")
        if self.verified_artifact_count < 1:
            raise ValueError("verified_artifact_count must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "state": self.state,
            "scientific_fingerprint": self.scientific_fingerprint,
            "final_manifest_sha256": self.final_manifest_sha256,
            "verification": {
                "status": "verified",
                "verified_artifact_count": self.verified_artifact_count,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompletionMarker":
        verification = data.get("verification")
        if not isinstance(verification, dict) or verification.get("status") != "verified":
            raise ValueError("Completion marker verification must be verified.")
        return cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]),
            state=str(data["state"]),
            scientific_fingerprint=str(data["scientific_fingerprint"]),
            final_manifest_sha256=str(data["final_manifest_sha256"]),
            verified_artifact_count=int(verification["verified_artifact_count"]),
        )
