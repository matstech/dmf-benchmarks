"""OCI publication helpers for complete benchmark run snapshots."""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.contracts import sha256_file
from dmf_bench.state import StateError


OFFICIAL_RUN_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_TYPE = "application/vnd.dmf-bench.run.v1+tar"
DEFAULT_LAYER_TYPE = "application/vnd.dmf-bench.run.layer.v1.tar+gzip"


class OciPublishError(RuntimeError):
    """Raised when a run cannot be packaged or pushed as an OCI artifact."""


def publish_run_oci(
    *,
    run_id: str,
    runs_dir: str | Path,
    ref: str,
    subject: str | None = None,
    output_dir: str | Path | None = None,
    oras_bin: str = "oras",
    dry_run: bool = False,
) -> dict[str, Any]:
    store = LocalArtifactStore(runs_dir)
    verification = store.verify_committed(run_id)
    run_dir = store.run_dir(run_id)
    if not run_dir.is_dir():
        raise StateError(f"Run directory not found: {run_dir}")

    official = write_official_run_metadata(
        run_dir,
        run_id=run_id,
        ref=ref,
        subject=subject,
        verification=verification,
    )
    sha256sums_path = write_sha256sums(run_dir)

    if output_dir is not None:
        archive_dir = Path(output_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{run_id}.run.tar.gz"
        cleanup_archive = False
        result = _publish_with_archive(
            run_dir=run_dir,
            archive_path=archive_path,
            ref=ref,
            subject=subject,
            oras_bin=oras_bin,
            dry_run=dry_run,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="dmf-bench-oci-") as temp_dir:
            archive_path = Path(temp_dir) / f"{run_id}.run.tar.gz"
            result = _publish_with_archive(
                run_dir=run_dir,
                archive_path=archive_path,
                ref=ref,
                subject=subject,
                oras_bin=oras_bin,
                dry_run=dry_run,
            )
            cleanup_archive = True

    payload = {
        "schema_version": OFFICIAL_RUN_SCHEMA_VERSION,
        "kind": "dmf-bench-oci-publication",
        "run_id": run_id,
        "ref": ref,
        "subject": subject or "",
        "artifact_type": DEFAULT_ARTIFACT_TYPE,
        "layer_type": DEFAULT_LAYER_TYPE,
        "dry_run": dry_run,
        "pushed": result["pushed"],
        "archive_path": "" if cleanup_archive else str(archive_path.resolve()),
        "archive_sha256": result["archive_sha256"],
        "archive_bytes": result["archive_bytes"],
        "official_run_path": str(official["path"]),
        "sha256sums_path": str(sha256sums_path.resolve()),
        "oras": result["oras"],
    }
    return payload


def _publish_with_archive(
    *,
    run_dir: Path,
    archive_path: Path,
    ref: str,
    subject: str | None,
    oras_bin: str,
    dry_run: bool,
) -> dict[str, Any]:
    create_run_archive(run_dir, archive_path)
    archive_sha256 = sha256_file(archive_path)
    archive_bytes = archive_path.stat().st_size
    command = oras_push_command(
        oras_bin=oras_bin,
        ref=ref,
        archive_path=archive_path,
        subject=subject,
    )
    oras_result: dict[str, Any] = {
        "command": command,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    pushed = False
    if not dry_run:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
        oras_result.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        if completed.returncode != 0:
            raise OciPublishError(
                f"oras push failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        pushed = True
    return {
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "pushed": pushed,
        "oras": oras_result,
    }


def write_official_run_metadata(
    run_dir: Path,
    *,
    run_id: str,
    ref: str,
    subject: str | None,
    verification: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": OFFICIAL_RUN_SCHEMA_VERSION,
        "kind": "dmf-bench-official-run",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "oci": {
            "ref": ref,
            "subject": subject or "",
            "artifact_type": DEFAULT_ARTIFACT_TYPE,
            "layer_type": DEFAULT_LAYER_TYPE,
        },
        "image": {
            "digest": os.getenv("DMF_BENCH_IMAGE_DIGEST", "unknown"),
            "ref": os.getenv("DMF_BENCH_IMAGE", ""),
        },
        "harness": {
            "commit": os.getenv("DMF_BENCH_HARNESS_COMMIT", "unknown"),
        },
        "run": {
            "directory": str(run_dir.resolve()),
            "completion_path": verification.get("completion_path", ""),
            "final_manifest_path": verification.get("final_manifest_path", ""),
            "final_manifest_sha256": verification.get("final_manifest_sha256", ""),
            "verified_artifact_count": verification.get("verified_artifact_count", 0),
        },
    }
    path = run_dir / "OFFICIAL-RUN.json"
    write_json_atomic(path, payload)
    return {"path": path.resolve(), "payload": payload}


def write_sha256sums(run_dir: Path) -> Path:
    target = run_dir / "SHA256SUMS"
    lines: list[str] = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        if relative == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def create_run_archive(run_dir: Path, archive_path: Path) -> Path:
    if not run_dir.is_dir():
        raise OciPublishError(f"Run directory not found: {run_dir}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    root_name = run_dir.name
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            archive.add(path, arcname=str(Path(root_name) / path.relative_to(run_dir)))
    return archive_path


def oras_push_command(
    *,
    oras_bin: str,
    ref: str,
    archive_path: Path,
    subject: str | None,
) -> list[str]:
    command = [
        oras_bin,
        "push",
        "--artifact-type",
        DEFAULT_ARTIFACT_TYPE,
    ]
    if subject:
        command.extend(["--subject", subject])
    command.extend(
        [
            ref,
            f"{archive_path.resolve()}:{DEFAULT_LAYER_TYPE}",
        ]
    )
    return command
