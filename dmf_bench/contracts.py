"""Serializable contracts shared by local state and artifact storage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .atomic_io import canonical_json_bytes


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
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("RunManifest schema_version must be 1.")
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
        )


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    run_id: str
    started_at: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class UnitCheckpoint:
    run_id: str
    unit_id: str
    status: str
    scientific_fingerprint: str
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    schema_version: int = 1
    checkpoint_digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("UnitCheckpoint schema_version must be 1.")
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
    schema_version: int = 1

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
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
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
        }


@dataclass(frozen=True)
class CompletionMarker:
    run_id: str
    scientific_fingerprint: str
    final_manifest_sha256: str
    verified_artifact_count: int
    schema_version: int = 1
    state: str = "COMPLETED"

    def __post_init__(self) -> None:
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
