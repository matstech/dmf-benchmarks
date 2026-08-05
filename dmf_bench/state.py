"""Run state transitions, locking, and pure resume planning."""

from __future__ import annotations

import errno
import fcntl
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from .atomic_io import read_json, write_json_atomic
from .contracts import (
    ArtifactRef,
    Attempt,
    LifecycleCheckpoint,
    RunManifest,
    RUN_STATUS_SCHEMA_VERSION,
    UnitCheckpoint,
    sha256_file,
)


class StateError(RuntimeError):
    """Raised when persisted state is invalid or a transition is rejected."""


class RunState(str, Enum):
    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    RUNNING = "RUNNING"
    JUDGING = "JUDGING"
    EVALUATING = "EVALUATING"
    REPORTING = "REPORTING"
    PUBLISHING = "PUBLISHING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    INTERRUPTING = "INTERRUPTING"
    INTERRUPTED = "INTERRUPTED"
    RESUMING = "RESUMING"
    FAILED_PREFLIGHT = "FAILED_PREFLIGHT"
    FAILED_RUNNING = "FAILED_RUNNING"
    FAILED_JUDGING = "FAILED_JUDGING"
    FAILED_EVALUATION = "FAILED_EVALUATION"
    FAILED_REPORTING = "FAILED_REPORTING"
    FAILED_PUBLISHING = "FAILED_PUBLISHING"
    FAILED_VERIFYING = "FAILED_VERIFYING"


class UnitState(str, Enum):
    PENDING = "PENDING"
    PREPARING = "PREPARING"
    INGESTING = "INGESTING"
    PREDICTING = "PREDICTING"
    PREDICTED = "PREDICTED"
    JUDGING = "JUDGING"
    JUDGED = "JUDGED"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"


RUN_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.PREFLIGHT},
    RunState.PREFLIGHT: {
        RunState.RUNNING,
        RunState.JUDGING,
        RunState.EVALUATING,
        RunState.FAILED_PREFLIGHT,
        RunState.INTERRUPTING,
    },
    RunState.RUNNING: {
        RunState.JUDGING,
        RunState.PARTIAL,
        RunState.FAILED_RUNNING,
        RunState.INTERRUPTING,
    },
    RunState.JUDGING: {
        RunState.EVALUATING,
        RunState.PARTIAL,
        RunState.FAILED_JUDGING,
        RunState.INTERRUPTING,
    },
    RunState.EVALUATING: {
        RunState.REPORTING,
        RunState.PARTIAL,
        RunState.FAILED_EVALUATION,
        RunState.INTERRUPTING,
    },
    RunState.REPORTING: {RunState.PUBLISHING, RunState.FAILED_REPORTING, RunState.INTERRUPTING},
    RunState.PUBLISHING: {RunState.VERIFYING, RunState.FAILED_PUBLISHING, RunState.INTERRUPTING},
    RunState.VERIFYING: {RunState.COMPLETED, RunState.FAILED_VERIFYING, RunState.INTERRUPTING},
    RunState.INTERRUPTING: {RunState.INTERRUPTED},
    RunState.INTERRUPTED: {RunState.RESUMING},
    RunState.FAILED_PREFLIGHT: {RunState.RESUMING},
    RunState.FAILED_RUNNING: {RunState.RESUMING},
    RunState.FAILED_JUDGING: {RunState.RESUMING},
    RunState.FAILED_EVALUATION: {RunState.RESUMING},
    RunState.FAILED_REPORTING: {RunState.RESUMING},
    RunState.FAILED_PUBLISHING: {RunState.RESUMING},
    RunState.FAILED_VERIFYING: {RunState.RESUMING},
    RunState.RESUMING: {
        RunState.PREFLIGHT,
        RunState.RUNNING,
        RunState.JUDGING,
        RunState.EVALUATING,
        RunState.REPORTING,
        RunState.PUBLISHING,
        RunState.VERIFYING,
    },
    RunState.PARTIAL: {RunState.RESUMING},
    RunState.COMPLETED: set(),
}


UNIT_TRANSITIONS: dict[UnitState, set[UnitState]] = {
    UnitState.PENDING: {UnitState.PREPARING},
    UnitState.PREPARING: {UnitState.INGESTING, UnitState.FAILED},
    UnitState.INGESTING: {UnitState.PREDICTING, UnitState.FAILED},
    UnitState.PREDICTING: {UnitState.PREDICTED, UnitState.FAILED},
    UnitState.PREDICTED: {UnitState.JUDGING, UnitState.COMMITTED, UnitState.FAILED},
    UnitState.JUDGING: {UnitState.JUDGED, UnitState.FAILED},
    UnitState.JUDGED: {UnitState.COMMITTED, UnitState.FAILED},
    UnitState.COMMITTED: set(),
    UnitState.FAILED: {UnitState.PREPARING},
}


def transition_run_state(current: RunState, next_state: RunState) -> RunState:
    if next_state not in RUN_TRANSITIONS[current]:
        raise StateError(f"Invalid run transition: {current.value} -> {next_state.value}.")
    return next_state


def transition_unit_state(current: UnitState, next_state: UnitState) -> UnitState:
    if next_state not in UNIT_TRANSITIONS[current]:
        raise StateError(f"Invalid unit transition: {current.value} -> {next_state.value}.")
    return next_state


class RunLock:
    """Exclusive non-distributed lock for one local run directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: Any = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            self._file = None
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise StateError(f"Run lock is already held: {self.path}") from exc
            raise

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@dataclass(frozen=True)
class ResumePlan:
    run_id: str
    next_phase: str
    committed_unit_ids: tuple[str, ...]
    restart_unit_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "next_phase": self.next_phase,
            "committed_unit_ids": list(self.committed_unit_ids),
            "restart_unit_ids": list(self.restart_unit_ids),
            "reason": self.reason,
        }


def load_manifest(run_dir: str | Path) -> RunManifest:
    data = read_json(Path(run_dir) / "run-manifest.json")
    if not isinstance(data, dict):
        raise StateError("run-manifest.json must contain a JSON object.")
    try:
        return RunManifest.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError(f"Invalid run manifest: {exc}") from exc


def load_run_status(
    run_dir: str | Path,
    *,
    required: bool = True,
) -> dict[str, Any] | None:
    path = Path(run_dir) / "run-status.json"
    if not path.is_file():
        if required:
            raise StateError(f"Missing run status: {path}")
        return None
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise StateError("run-status.json must contain a JSON object.")
    if payload.get("schema_version") != RUN_STATUS_SCHEMA_VERSION:
        observed_version = payload.get("schema_version")
        raise StateError(
            "run-status.json must declare "
            f"schema_version={RUN_STATUS_SCHEMA_VERSION}; "
            f"v{observed_version} state is not supported."
        )
    state = str(payload.get("state", ""))
    if state not in {candidate.value for candidate in RunState}:
        raise StateError(f"run-status.json contains an unsupported state: {state!r}.")
    return payload


def validate_run_state_version(run_dir: str | Path) -> RunManifest:
    manifest = load_manifest(run_dir)
    load_run_status(run_dir, required=False)
    return manifest


def load_unit_checkpoint(path: str | Path) -> UnitCheckpoint:
    data = read_json(path)
    if not isinstance(data, dict):
        raise StateError(f"Unit checkpoint must contain a JSON object: {path}")
    try:
        return UnitCheckpoint.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError(f"Invalid unit checkpoint {path}: {exc}") from exc


def new_attempt(
    *,
    run_id: str,
    entry_phase: str,
    resume: bool,
    resumed_from_attempt_id: str | None = None,
) -> Attempt:
    return Attempt(
        attempt_id=uuid4().hex,
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        entry_phase=entry_phase,
        resume=resume,
        resumed_from_attempt_id=resumed_from_attempt_id,
    )


def write_attempt(run_dir: str | Path, attempt: Attempt) -> Path:
    run_path = Path(run_dir)
    manifest = load_manifest(run_path)
    if attempt.run_id != manifest.run_id:
        raise StateError("Attempt run_id does not match run manifest.")
    path = run_path / "attempts" / attempt.attempt_id / "attempt.json"
    write_json_atomic(path, attempt.to_dict())
    return path


def finish_attempt(
    run_dir: str | Path,
    attempt: Attempt,
    *,
    status: str,
) -> Attempt:
    finished = replace(
        attempt,
        status=status,
        completed_at=datetime.now(timezone.utc).isoformat(),
        attempt_digest="",
    )
    write_attempt(run_dir, finished)
    return finished


def latest_attempt_id(run_dir: str | Path) -> str | None:
    attempts_root = Path(run_dir) / "attempts"
    candidates: list[tuple[str, str]] = []
    if not attempts_root.is_dir():
        return None
    for path in attempts_root.glob("*/attempt.json"):
        try:
            payload = read_json(path)
            if not isinstance(payload, dict):
                continue
            attempt = Attempt.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append((attempt.started_at, attempt.attempt_id))
    return max(candidates)[1] if candidates else None


def lifecycle_checkpoint_path(
    run_dir: str | Path,
    phase: str,
    *,
    item_id: str | None = None,
) -> Path:
    root = Path(run_dir) / "phase-checkpoints" / phase.upper()
    if item_id is None:
        return root / "checkpoint.json"
    safe_item = item_id.replace("/", "_").replace("\\", "_")
    return root / "items" / f"{safe_item}.json"


def write_lifecycle_checkpoint(
    run_dir: str | Path,
    checkpoint: LifecycleCheckpoint,
    *,
    item_id: str | None = None,
) -> Path:
    run_path = Path(run_dir)
    manifest = load_manifest(run_path)
    if checkpoint.run_id != manifest.run_id:
        raise StateError("Lifecycle checkpoint run_id mismatch.")
    if checkpoint.scientific_fingerprint != manifest.scientific_fingerprint:
        raise StateError("Lifecycle checkpoint scientific fingerprint mismatch.")
    path = lifecycle_checkpoint_path(run_path, checkpoint.phase, item_id=item_id)
    write_json_atomic(path, checkpoint.to_dict())
    return path


def load_lifecycle_checkpoint(path: str | Path) -> LifecycleCheckpoint:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise StateError(f"Lifecycle checkpoint must contain a JSON object: {path}")
    try:
        return LifecycleCheckpoint.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError(f"Invalid lifecycle checkpoint {path}: {exc}") from exc


def artifact_ref_for(run_dir: str | Path, path: str | Path) -> ArtifactRef:
    run_path = Path(run_dir).resolve()
    artifact_path = Path(path).resolve()
    if run_path not in artifact_path.parents:
        raise StateError(f"Artifact path is outside run directory: {artifact_path}")
    if not artifact_path.is_file():
        raise StateError(f"Artifact path is not a file: {artifact_path}")
    return ArtifactRef(
        path=artifact_path.relative_to(run_path).as_posix(),
        sha256=sha256_file(artifact_path),
        bytes=artifact_path.stat().st_size,
    )


def validate_artifact_refs(
    run_dir: str | Path,
    artifacts: tuple[ArtifactRef, ...],
) -> None:
    if not artifacts:
        raise StateError("Committed checkpoint does not reference any artifacts.")
    run_path = Path(run_dir).resolve()
    seen: set[str] = set()
    for artifact in artifacts:
        relative = Path(artifact.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise StateError(f"Unsafe checkpoint artifact path: {artifact.path}")
        if artifact.path in seen:
            raise StateError(f"Duplicate checkpoint artifact path: {artifact.path}")
        seen.add(artifact.path)
        path = run_path / relative
        if not path.is_file():
            raise StateError(f"Missing checkpoint artifact: {artifact.path}")
        if path.stat().st_size != artifact.bytes:
            raise StateError(f"Checkpoint artifact size mismatch: {artifact.path}")
        if sha256_file(path) != artifact.sha256:
            raise StateError(f"Checkpoint artifact checksum mismatch: {artifact.path}")


def expected_terminal_item_ids(manifest: RunManifest) -> tuple[str, ...]:
    raw = manifest.fingerprint_inputs.get("expected_question_ids")
    if isinstance(raw, list) and raw:
        item_ids = tuple(str(item) for item in raw)
        if len(set(item_ids)) != len(item_ids):
            raise StateError("Manifest expected_question_ids contains duplicates.")
        return item_ids
    return manifest.expected_item_ids


def _valid_phase_checkpoint(
    run_path: Path,
    manifest: RunManifest,
    *,
    phase: str,
    expected_item_ids: tuple[str, ...],
    predecessor_digest: str | None,
) -> LifecycleCheckpoint | None:
    path = lifecycle_checkpoint_path(run_path, phase)
    if not path.exists():
        return None
    checkpoint = load_lifecycle_checkpoint(path)
    if checkpoint.run_id != manifest.run_id:
        raise StateError(f"{phase} checkpoint run_id mismatch.")
    if checkpoint.phase != phase or checkpoint.status != "COMMITTED":
        return None
    if checkpoint.scientific_fingerprint != manifest.scientific_fingerprint:
        raise StateError(f"{phase} checkpoint fingerprint mismatch.")
    if checkpoint.expected_item_ids != expected_item_ids:
        raise StateError(f"{phase} checkpoint expected item set mismatch.")
    if checkpoint.predecessor_digest != predecessor_digest:
        return None
    try:
        validate_artifact_refs(run_path, checkpoint.artifacts)
    except StateError:
        return None
    return checkpoint


def plan_resume(run_dir: str | Path) -> ResumePlan:
    run_path = Path(run_dir)
    manifest = validate_run_state_version(run_path)
    completion_marker = run_path / "final" / "COMPLETED.json"
    if completion_marker.exists():
        from dmf_bench.artifacts.local import LocalArtifactStore

        LocalArtifactStore(run_path.parent).verify_committed(manifest.run_id)
        return ResumePlan(
            run_id=manifest.run_id,
            next_phase="COMPLETED",
            committed_unit_ids=manifest.expected_item_ids,
            restart_unit_ids=(),
            reason="run already completed",
        )

    committed: list[str] = []
    restart: list[str] = []
    for unit_id in manifest.expected_item_ids:
        checkpoint_path = run_path / "checkpoints" / unit_id / "checkpoint.json"
        if not checkpoint_path.exists():
            restart.append(unit_id)
            continue
        checkpoint = load_unit_checkpoint(checkpoint_path)
        if checkpoint.run_id != manifest.run_id:
            raise StateError(f"Checkpoint run_id mismatch for unit {unit_id}.")
        if checkpoint.unit_id != unit_id:
            raise StateError(f"Checkpoint unit_id mismatch for unit {unit_id}.")
        if checkpoint.scientific_fingerprint != manifest.scientific_fingerprint:
            raise StateError(f"Checkpoint fingerprint mismatch for unit {unit_id}.")
        if checkpoint.status == UnitState.COMMITTED.value:
            try:
                validate_artifact_refs(run_path, checkpoint.artifacts)
            except StateError:
                restart.append(unit_id)
            else:
                committed.append(unit_id)
        else:
            restart.append(unit_id)

    if restart:
        return ResumePlan(
            run_id=manifest.run_id,
            next_phase="RUNNING",
            committed_unit_ids=tuple(committed),
            restart_unit_ids=tuple(restart),
            reason="prediction units remain incomplete",
        )

    terminal_ids = expected_terminal_item_ids(manifest)
    predecessor: str | None = None
    for phase, next_phase, reason in (
        ("JUDGING", "JUDGING", "judgments are incomplete or invalid"),
        ("EVALUATING", "EVALUATING", "evaluation artifacts are incomplete or invalid"),
        ("REPORTING", "REPORTING", "reports are incomplete or invalid"),
        ("PUBLISHING", "PUBLISHING", "publication staging is incomplete or invalid"),
        ("VERIFYING", "VERIFYING", "publication verification or completion is incomplete"),
    ):
        checkpoint = _valid_phase_checkpoint(
            run_path,
            manifest,
            phase=phase,
            expected_item_ids=terminal_ids,
            predecessor_digest=predecessor,
        )
        if checkpoint is None:
            return ResumePlan(
                run_id=manifest.run_id,
                next_phase=next_phase,
                committed_unit_ids=tuple(committed),
                restart_unit_ids=(),
                reason=reason,
            )
        predecessor = checkpoint.checkpoint_digest

    return ResumePlan(
        run_id=manifest.run_id,
        next_phase="VERIFYING",
        committed_unit_ids=tuple(committed),
        restart_unit_ids=(),
        reason="verified staging exists but completion marker is missing",
    )
