"""Run state transitions, locking, and pure resume planning."""

from __future__ import annotations

import errno
import fcntl
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .atomic_io import read_json
from .contracts import RunManifest, UnitCheckpoint


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


def load_unit_checkpoint(path: str | Path) -> UnitCheckpoint:
    data = read_json(path)
    if not isinstance(data, dict):
        raise StateError(f"Unit checkpoint must contain a JSON object: {path}")
    try:
        return UnitCheckpoint.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise StateError(f"Invalid unit checkpoint {path}: {exc}") from exc


def plan_resume(run_dir: str | Path) -> ResumePlan:
    run_path = Path(run_dir)
    manifest = load_manifest(run_path)
    completion_marker = run_path / "final" / "COMPLETED.json"
    if completion_marker.exists():
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

    return ResumePlan(
        run_id=manifest.run_id,
        next_phase="JUDGING",
        committed_unit_ids=tuple(committed),
        restart_unit_ids=(),
        reason="all prediction units are committed",
    )
