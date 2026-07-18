"""Local artifact store with staged verify and final commit marker."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import (
    ArtifactRef,
    CompletionMarker,
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


class LocalArtifactStore:
    def __init__(self, runs_dir: str | Path) -> None:
        self.runs_dir = Path(runs_dir)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

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
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest": manifest.to_dict(),
            "completed": completion_path.exists(),
            "completion_path": str(completion_path) if completion_path.exists() else None,
        }

    def stage(self, run_id: str, source_dir: str | Path) -> StagedPublication:
        run_dir = self.run_dir(run_id)
        if (run_dir / "final" / "COMPLETED.json").exists():
            raise StateError(f"Run is already committed: {run_id}")
        source_path = Path(source_dir)
        if not source_path.is_dir():
            raise StateError(f"Publication source is not a directory: {source_path}")

        staging_dir = run_dir / "publish-staging" / uuid.uuid4().hex
        staging_dir.mkdir(parents=True)
        payload_dir = staging_dir / "payload"
        payload_dir.mkdir()
        copy_tree_contents(source_path, payload_dir)

        artifacts = build_artifact_refs(payload_dir)
        manifest_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "artifacts": [artifact.to_dict() for artifact in artifacts],
        }
        manifest_path = staging_dir / "manifest.json"
        write_json_atomic(manifest_path, manifest_payload)
        return StagedPublication(run_id=run_id, staging_dir=staging_dir, manifest_path=manifest_path)

    def verify(self, staged: StagedPublication) -> VerificationReceipt:
        manifest = read_json(staged.manifest_path)
        if not isinstance(manifest, dict):
            raise StateError("Publication manifest must be a JSON object.")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise StateError("Publication manifest must list at least one artifact.")

        payload_dir = staged.staging_dir / "payload"
        for raw_artifact in artifacts:
            artifact = ArtifactRef.from_dict(raw_artifact)
            path = payload_dir / artifact.path
            if not path.is_file():
                raise StateError(f"Missing staged artifact: {artifact.path}")
            if path.stat().st_size != artifact.bytes:
                raise StateError(f"Staged artifact size mismatch: {artifact.path}")
            if sha256_file(path) != artifact.sha256:
                raise StateError(f"Staged artifact checksum mismatch: {artifact.path}")

        receipt = VerificationReceipt(
            run_id=staged.run_id,
            manifest_sha256=sha256_file(staged.manifest_path),
            artifact_count=len(artifacts),
            receipt_path=staged.staging_dir / "verification-receipt.json",
        )
        write_json_atomic(receipt.receipt_path, receipt.to_dict())
        return receipt

    def commit(self, staged: StagedPublication, receipt: VerificationReceipt) -> CompletionMarker:
        run_dir = self.run_dir(staged.run_id)
        final_dir = run_dir / "final"
        completion_path = final_dir / "COMPLETED.json"
        if completion_path.exists():
            raise StateError(f"Run is already committed: {staged.run_id}")
        manifest = load_manifest(run_dir)
        if receipt.run_id != staged.run_id:
            raise StateError("Verification receipt run_id mismatch.")

        final_dir.mkdir(exist_ok=True)
        payload_dir = staged.staging_dir / "payload"
        copy_tree_contents(payload_dir, final_dir)
        final_manifest_path = final_dir / "manifest.json"
        shutil.copy2(staged.manifest_path, final_manifest_path)

        marker = CompletionMarker(
            run_id=staged.run_id,
            scientific_fingerprint=manifest.scientific_fingerprint,
            final_manifest_sha256=sha256_file(final_manifest_path),
            verified_artifact_count=receipt.artifact_count,
        )
        write_json_atomic(completion_path, marker.to_dict())
        return marker


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
