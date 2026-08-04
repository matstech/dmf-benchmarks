"""Local artifact store with staged verify and final commit marker."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import (
    ArtifactRef,
    CompletionMarker,
    PUBLICATION_MANIFEST_SCHEMA_VERSION,
    RunManifest,
    hash_canonical_json,
    sha256_file,
)
from dmf_bench.state import StateError, load_manifest


@dataclass(frozen=True)
class StagedPublication:
    run_id: str
    staging_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class VerificationReceipt:
    run_id: str
    manifest_sha256: str
    artifact_count: int
    receipt_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "artifact_count": self.artifact_count,
            "status": "verified",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, receipt_path: Path) -> "VerificationReceipt":
        if data.get("status") != "verified":
            raise ValueError("Verification receipt status must be verified.")
        return cls(
            run_id=str(data["run_id"]),
            manifest_sha256=str(data["manifest_sha256"]),
            artifact_count=int(data["artifact_count"]),
            receipt_path=receipt_path,
        )


class LocalArtifactStore:
    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / safe_run_id(run_id)

    def create_run(self, manifest: RunManifest) -> Path:
        run_dir = self.run_dir(manifest.run_id)
        if run_dir.exists():
            raise StateError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        for relative in ("logs", "attempts", "checkpoints", "items", "evaluations", "reports", "metrics"):
            (run_dir / relative).mkdir()
        write_json_atomic(run_dir / "run-manifest.json", manifest.to_dict())
        return run_dir

    def inspect(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        manifest = load_manifest(run_dir)
        completion_path = run_dir / "final" / "COMPLETED.json"
        final_manifest_path = run_dir / "final" / "manifest.json"
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest": manifest.to_dict(),
            "completed": completion_path.exists(),
            "completion_path": str(completion_path) if completion_path.exists() else None,
            "final_manifest_path": str(final_manifest_path) if final_manifest_path.exists() else None,
        }

    def inspect_committed(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        base = self.inspect(run_id)
        final_dir = run_dir / "final"
        completion_path = final_dir / "COMPLETED.json"
        final_manifest_path = final_dir / "manifest.json"
        completion = read_json(completion_path) if completion_path.exists() else None
        final_manifest = read_json(final_manifest_path) if final_manifest_path.exists() else None
        artifacts = []
        if isinstance(final_manifest, dict):
            artifacts = list(final_manifest.get("artifacts") or [])
        return {
            **base,
            "backend": "local-only",
            "shared_volume_root": str(self.runs_dir),
            "api_mode": "read-only",
            "writable_api": False,
            "retry_policy": local_only_retry_policy(),
            "completion": completion,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }

    def verify_committed(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        manifest = load_manifest(run_dir)
        final_dir = run_dir / "final"
        completion_path = final_dir / "COMPLETED.json"
        final_manifest_path = final_dir / "manifest.json"

        if not completion_path.is_file():
            raise StateError(f"Missing completion marker: {completion_path}")
        if not final_manifest_path.is_file():
            raise StateError(f"Missing final manifest: {final_manifest_path}")

        completion = read_json(completion_path)
        if not isinstance(completion, dict):
            raise StateError("Completion marker must be a JSON object.")
        try:
            CompletionMarker.from_dict(completion)
        except (KeyError, TypeError, ValueError) as exc:
            raise StateError(f"Invalid completion marker: {exc}") from exc
        final_manifest = read_json(final_manifest_path)
        if not isinstance(final_manifest, dict):
            raise StateError("Final manifest must be a JSON object.")
        if final_manifest.get("schema_version") != PUBLICATION_MANIFEST_SCHEMA_VERSION:
            raise StateError(
                "Final publication manifest must declare "
                f"schema_version={PUBLICATION_MANIFEST_SCHEMA_VERSION}."
            )

        if completion.get("run_id") != manifest.run_id:
            raise StateError("Completion marker run_id mismatch.")
        if completion.get("scientific_fingerprint") != manifest.scientific_fingerprint:
            raise StateError("Completion marker scientific_fingerprint mismatch.")
        if completion.get("final_manifest_sha256") != sha256_file(final_manifest_path):
            raise StateError("Completion marker final_manifest_sha256 mismatch.")
        if final_manifest.get("run_id") != manifest.run_id:
            raise StateError("Final manifest run_id mismatch.")
        if final_manifest.get("scientific_fingerprint") != manifest.scientific_fingerprint:
            raise StateError("Final manifest scientific_fingerprint mismatch.")

        artifacts = final_manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise StateError("Final manifest must list at least one artifact.")

        verified: list[dict[str, Any]] = []
        for raw_artifact in artifacts:
            artifact = ArtifactRef.from_dict(raw_artifact)
            path = resolve_committed_artifact_path(final_dir, artifact.path)
            if not path.is_file():
                raise StateError(f"Missing committed artifact: {artifact.path}")
            observed_bytes = path.stat().st_size
            observed_sha256 = sha256_file(path)
            if observed_bytes != artifact.bytes:
                raise StateError(f"Committed artifact size mismatch: {artifact.path}")
            if observed_sha256 != artifact.sha256:
                raise StateError(f"Committed artifact checksum mismatch: {artifact.path}")
            verified.append(artifact.to_dict())

        expected_count = int(
            ((completion.get("verification") or {}).get("verified_artifact_count") or 0)
        )
        if expected_count != len(verified):
            raise StateError("Completion marker verified_artifact_count mismatch.")

        return {
            "run_id": manifest.run_id,
            "backend": "local-only",
            "shared_volume_root": str(self.runs_dir),
            "retry_policy": local_only_retry_policy(),
            "status": "verified",
            "completion_path": str(completion_path),
            "final_manifest_path": str(final_manifest_path),
            "final_manifest_sha256": completion["final_manifest_sha256"],
            "verified_artifact_count": len(verified),
            "artifacts": verified,
        }

    def committed_artifact_path(self, run_id: str, artifact_path: str) -> Path:
        self.verify_committed(run_id)
        final_dir = self.run_dir(run_id) / "final"
        manifest = read_json(final_dir / "manifest.json")
        if not isinstance(manifest, dict):
            raise StateError("Final manifest must be a JSON object.")
        allowed = {str(item.get("path")) for item in manifest.get("artifacts", []) if isinstance(item, dict)}
        if artifact_path not in allowed:
            raise StateError(f"Artifact is not listed in final manifest: {artifact_path}")
        return resolve_committed_artifact_path(final_dir, artifact_path)

    def stage(
        self,
        run_id: str,
        source_dir: str | Path,
        *,
        publication_id: str | None = None,
    ) -> StagedPublication:
        run_dir = self.run_dir(run_id)
        if (run_dir / "final" / "COMPLETED.json").exists():
            self.verify_committed(run_id)
            raise StateError(f"Run is already committed: {run_id}")
        source_path = Path(source_dir)
        if not source_path.is_dir():
            raise StateError(f"Publication source is not a directory: {source_path}")

        resolved_publication_id = safe_publication_id(publication_id or uuid.uuid4().hex)
        staging_root = run_dir / "publish-staging"
        staging_dir = staging_root / resolved_publication_id
        manifest_path = staging_dir / "manifest.json"
        staged = StagedPublication(
            run_id=run_id,
            staging_dir=staging_dir,
            manifest_path=manifest_path,
        )
        if staging_dir.exists():
            self._validate_staged(staged)
            return staged

        staging_root.mkdir(parents=True, exist_ok=True)
        temporary_dir = staging_root / f".{resolved_publication_id}.tmp-{uuid.uuid4().hex}"
        temporary_dir.mkdir()
        payload_dir = temporary_dir / "payload"
        payload_dir.mkdir()
        try:
            copy_tree_contents(source_path, payload_dir)

            manifest = load_manifest(run_dir)
            artifacts = build_artifact_refs(payload_dir)
            manifest_payload = {
                "schema_version": PUBLICATION_MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "scientific_fingerprint": manifest.scientific_fingerprint,
                "artifacts": [artifact.to_dict() for artifact in artifacts],
            }
            write_json_atomic(temporary_dir / "manifest.json", manifest_payload)
            os.replace(temporary_dir, staging_dir)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
        self._validate_staged(staged)
        return staged

    def load_staged(self, run_id: str, staging_dir: str | Path) -> StagedPublication:
        run_dir = self.run_dir(run_id).resolve()
        resolved = Path(staging_dir).resolve()
        staging_root = (run_dir / "publish-staging").resolve()
        if staging_root not in resolved.parents or resolved.parent != staging_root:
            raise StateError(f"Unsafe publication staging directory: {staging_dir}")
        staged = StagedPublication(
            run_id=run_id,
            staging_dir=resolved,
            manifest_path=resolved / "manifest.json",
        )
        self._validate_staged(staged)
        return staged

    def verify(self, staged: StagedPublication) -> VerificationReceipt:
        manifest = self._validate_staged(staged)
        artifacts = list(manifest["artifacts"])

        receipt = VerificationReceipt(
            run_id=staged.run_id,
            manifest_sha256=sha256_file(staged.manifest_path),
            artifact_count=len(artifacts),
            receipt_path=staged.staging_dir / "verification-receipt.json",
        )
        if receipt.receipt_path.exists():
            payload = read_json(receipt.receipt_path)
            if not isinstance(payload, dict):
                raise StateError("Verification receipt must be a JSON object.")
            try:
                existing = VerificationReceipt.from_dict(
                    payload,
                    receipt_path=receipt.receipt_path,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise StateError(f"Invalid verification receipt: {exc}") from exc
            if existing != receipt:
                raise StateError("Verification receipt does not match staged publication.")
            return existing
        write_json_atomic(receipt.receipt_path, receipt.to_dict())
        return receipt

    def commit(self, staged: StagedPublication, receipt: VerificationReceipt) -> CompletionMarker:
        run_dir = self.run_dir(staged.run_id)
        final_dir = run_dir / "final"
        completion_path = final_dir / "COMPLETED.json"
        if completion_path.exists():
            self.verify_committed(staged.run_id)
            payload = read_json(completion_path)
            if not isinstance(payload, dict):
                raise StateError("Completion marker must be a JSON object.")
            try:
                return CompletionMarker.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise StateError(f"Invalid completion marker: {exc}") from exc
        manifest = load_manifest(run_dir)
        staged_manifest = self._validate_staged(staged)
        if receipt.run_id != staged.run_id:
            raise StateError("Verification receipt run_id mismatch.")
        if receipt.receipt_path != staged.staging_dir / "verification-receipt.json":
            raise StateError("Verification receipt path mismatch.")
        if receipt.manifest_sha256 != sha256_file(staged.manifest_path):
            raise StateError("Verification receipt manifest digest mismatch.")
        if receipt.artifact_count != len(staged_manifest["artifacts"]):
            raise StateError("Verification receipt artifact count mismatch.")

        final_dir.mkdir(exist_ok=True)
        payload_dir = staged.staging_dir / "payload"
        copy_tree_contents(payload_dir, final_dir)
        final_manifest_path = final_dir / "manifest.json"
        shutil.copy2(staged.manifest_path, final_manifest_path)

        for raw_artifact in staged_manifest["artifacts"]:
            artifact = ArtifactRef.from_dict(raw_artifact)
            path = resolve_committed_artifact_path(final_dir, artifact.path)
            if not path.is_file():
                raise StateError(f"Missing final artifact before commit: {artifact.path}")
            if path.stat().st_size != artifact.bytes or sha256_file(path) != artifact.sha256:
                raise StateError(f"Final artifact verification failed before commit: {artifact.path}")

        marker = CompletionMarker(
            run_id=staged.run_id,
            scientific_fingerprint=manifest.scientific_fingerprint,
            final_manifest_sha256=sha256_file(final_manifest_path),
            verified_artifact_count=receipt.artifact_count,
        )
        write_json_atomic(completion_path, marker.to_dict())
        self.verify_committed(staged.run_id)
        return marker

    def _validate_staged(self, staged: StagedPublication) -> dict[str, Any]:
        if staged.run_id != safe_run_id(staged.run_id):
            raise StateError(f"Unsafe run_id: {staged.run_id!r}")
        run_dir = self.run_dir(staged.run_id).resolve()
        staging_dir = staged.staging_dir.resolve()
        staging_root = (run_dir / "publish-staging").resolve()
        if staging_dir.parent != staging_root:
            raise StateError(f"Unsafe publication staging directory: {staging_dir}")
        if staged.manifest_path.resolve() != staging_dir / "manifest.json":
            raise StateError("Publication manifest path mismatch.")
        manifest = read_json(staged.manifest_path)
        if not isinstance(manifest, dict):
            raise StateError("Publication manifest must be a JSON object.")
        run_manifest = load_manifest(run_dir)
        if manifest.get("schema_version") != PUBLICATION_MANIFEST_SCHEMA_VERSION:
            raise StateError(
                "Publication manifest must declare "
                f"schema_version={PUBLICATION_MANIFEST_SCHEMA_VERSION}."
            )
        if manifest.get("run_id") != staged.run_id:
            raise StateError("Publication manifest run_id mismatch.")
        if manifest.get("scientific_fingerprint") != run_manifest.scientific_fingerprint:
            raise StateError("Publication manifest scientific fingerprint mismatch.")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise StateError("Publication manifest must list at least one artifact.")
        payload_dir = staging_dir / "payload"
        seen: set[str] = set()
        for raw_artifact in artifacts:
            if not isinstance(raw_artifact, dict):
                raise StateError("Publication manifest artifact must be a JSON object.")
            try:
                artifact = ArtifactRef.from_dict(raw_artifact)
            except (KeyError, TypeError, ValueError) as exc:
                raise StateError(f"Invalid publication artifact: {exc}") from exc
            if artifact.path in seen:
                raise StateError(f"Duplicate staged artifact: {artifact.path}")
            seen.add(artifact.path)
            path = resolve_committed_artifact_path(payload_dir, artifact.path)
            if not path.is_file():
                raise StateError(f"Missing staged artifact: {artifact.path}")
            if path.stat().st_size != artifact.bytes:
                raise StateError(f"Staged artifact size mismatch: {artifact.path}")
            if sha256_file(path) != artifact.sha256:
                raise StateError(f"Staged artifact checksum mismatch: {artifact.path}")
        return manifest


def build_artifact_refs(root: Path) -> tuple[ArtifactRef, ...]:
    refs: list[ArtifactRef] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        refs.append(
            ArtifactRef(
                path=relative,
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
            )
        )
    return tuple(refs)


def copy_tree_contents(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] in {"publish-staging", "final"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def safe_run_id(run_id: str) -> str:
    normalized = str(run_id).strip()
    path = Path(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or "/" in normalized
        or "\\" in normalized
    ):
        raise StateError(f"Unsafe run_id: {run_id!r}")
    return normalized


def safe_publication_id(publication_id: str) -> str:
    normalized = safe_run_id(publication_id)
    if normalized.startswith("."):
        raise StateError(f"Unsafe publication_id: {publication_id!r}")
    return normalized


def resolve_committed_artifact_path(final_dir: Path, artifact_path: str) -> Path:
    relative = Path(artifact_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise StateError(f"Unsafe artifact path: {artifact_path!r}")
    return final_dir / relative


def local_only_retry_policy() -> dict[str, str]:
    return {
        "status": "NOT_APPLICABLE",
        "reason": "local-only filesystem publication uses atomic rename and explicit verify; no remote retry loop or SDK is involved.",
    }
