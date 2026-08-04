import json
import subprocess
import sys
from pathlib import Path

import pytest

import dmf_bench.atomic_io as atomic_io
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import AtomicWriteError, canonical_json_bytes, read_json, write_json_atomic
from dmf_bench.cli import main
from dmf_bench.contracts import (
    ArtifactRef,
    RunManifest,
    UnitCheckpoint,
    hash_canonical_json,
    scientific_fingerprint,
    sha256_file,
)
from dmf_bench.state import (
    RunLock,
    RunState,
    StateError,
    UnitState,
    load_unit_checkpoint,
    plan_resume,
    transition_run_state,
    transition_unit_state,
)


FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def make_manifest(run_id: str = "run-001", expected: tuple[str, ...] = ("unit-1",)) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        scientific_fingerprint=FINGERPRINT_A,
        fingerprint_inputs={"benchmark": "fixture", "seed": 7},
        expected_item_ids=expected,
        atomic_unit="longmemeval-question",
    )


def write_checkpoint(
    run_dir: Path,
    *,
    run_id: str = "run-001",
    unit_id: str = "unit-1",
    status: str = "COMMITTED",
    fingerprint: str = FINGERPRINT_A,
) -> None:
    artifact_path = run_dir / "items" / unit_id / "result.json"
    write_json_atomic(artifact_path, {"unit_id": unit_id})
    artifacts = (
        ArtifactRef(
            path=artifact_path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(artifact_path),
            bytes=artifact_path.stat().st_size,
        ),
    ) if status == "COMMITTED" else ()
    checkpoint = UnitCheckpoint(
        run_id=run_id,
        unit_id=unit_id,
        status=status,
        scientific_fingerprint=fingerprint,
        artifacts=artifacts,
    )
    write_json_atomic(
        run_dir / "checkpoints" / unit_id / "checkpoint.json",
        checkpoint.to_dict(),
    )


def test_canonical_json_is_stable_and_rejects_nan() -> None:
    left = {"b": [2, 1], "a": {"x": "é"}}
    right = {"a": {"x": "é"}, "b": [2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_bytes(left) == b'{"a":{"x":"\xc3\xa9"},"b":[2,1]}'

    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_scientific_fingerprint_changes_only_when_payload_changes() -> None:
    payload = {"benchmark": "locomo", "model": "fixed", "seed": 42}

    assert scientific_fingerprint(payload) == hash_canonical_json(dict(reversed(payload.items())))
    assert scientific_fingerprint(payload) != scientific_fingerprint({**payload, "seed": 43})


def test_run_and_unit_state_transitions_are_explicit() -> None:
    assert transition_run_state(RunState.CREATED, RunState.PREFLIGHT) == RunState.PREFLIGHT
    assert transition_run_state(RunState.RUNNING, RunState.JUDGING) == RunState.JUDGING
    assert transition_unit_state(UnitState.PENDING, UnitState.PREPARING) == UnitState.PREPARING
    assert transition_unit_state(UnitState.PREDICTED, UnitState.COMMITTED) == UnitState.COMMITTED

    with pytest.raises(StateError, match="Invalid run transition"):
        transition_run_state(RunState.CREATED, RunState.COMPLETED)
    with pytest.raises(StateError, match="Invalid unit transition"):
        transition_unit_state(UnitState.COMMITTED, UnitState.PREPARING)


def test_truncated_json_and_semantic_incomplete_checkpoint_are_rejected(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"schema_version": 1', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON file"):
        read_json(truncated)

    incomplete = tmp_path / "checkpoint.json"
    write_json_atomic(incomplete, {"schema_version": 1, "run_id": "run-001"})

    with pytest.raises(StateError, match="Invalid unit checkpoint"):
        load_unit_checkpoint(incomplete)


def test_atomic_write_failure_keeps_previous_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "checkpoint.json"
    write_json_atomic(path, {"version": 1, "value": "old"})

    def fail_replace(_source: str | Path, _target: str | Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)

    with pytest.raises(AtomicWriteError):
        write_json_atomic(path, {"version": 1, "value": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "value": "old"}


def test_run_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    lock_path = tmp_path / "run.lock"

    with RunLock(lock_path):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from dmf_bench.state import RunLock, StateError\n"
                    "import sys\n"
                    f"lock = RunLock({str(lock_path)!r})\n"
                    "try:\n"
                    "    lock.acquire()\n"
                    "except StateError:\n"
                    "    sys.exit(7)\n"
                    "else:\n"
                    "    lock.release()\n"
                    "    sys.exit(0)\n"
                ),
            ],
            check=False,
        )

    assert result.returncode == 7


def test_local_artifact_store_writes_completion_marker_last(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    run_dir = store.create_run(make_manifest())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_json_atomic(source_dir / "results.json", {"ok": True})

    staged = store.stage("run-001", source_dir)

    assert not (run_dir / "final" / "COMPLETED.json").exists()
    assert staged.manifest_path.exists()

    receipt = store.verify(staged)

    assert receipt.artifact_count == 1
    assert not (run_dir / "final" / "COMPLETED.json").exists()

    marker = store.commit(staged, receipt)

    assert marker.state == "COMPLETED"
    assert (run_dir / "final" / "manifest.json").exists()
    assert (run_dir / "final" / "results.json").exists()
    assert (run_dir / "final" / "COMPLETED.json").exists()


def test_local_artifact_stage_verify_and_commit_are_idempotent(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    store.create_run(make_manifest())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_json_atomic(source_dir / "results.json", {"ok": True})

    first_stage = store.stage("run-001", source_dir, publication_id="report-digest")
    second_stage = store.stage("run-001", source_dir, publication_id="report-digest")
    first_receipt = store.verify(first_stage)
    second_receipt = store.verify(second_stage)
    first_marker = store.commit(first_stage, first_receipt)
    second_marker = store.commit(second_stage, second_receipt)

    assert first_stage == second_stage
    assert first_receipt == second_receipt
    assert first_marker == second_marker
    assert store.verify_committed("run-001")["status"] == "verified"


def test_local_artifact_store_refuses_committed_overwrite(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    store.create_run(make_manifest())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    write_json_atomic(source_dir / "results.json", {"ok": True})

    staged = store.stage("run-001", source_dir)
    receipt = store.verify(staged)
    store.commit(staged, receipt)

    with pytest.raises(StateError, match="already committed"):
        store.stage("run-001", source_dir)


def test_resume_planner_skips_only_committed_valid_checkpoints(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    run_dir = store.create_run(make_manifest(expected=("unit-1", "unit-2", "unit-3")))
    write_checkpoint(run_dir, unit_id="unit-1", status="COMMITTED")
    write_checkpoint(run_dir, unit_id="unit-2", status="PREDICTED")

    plan = plan_resume(run_dir)

    assert plan.next_phase == "RUNNING"
    assert plan.committed_unit_ids == ("unit-1",)
    assert plan.restart_unit_ids == ("unit-2", "unit-3")


def test_resume_planner_fails_closed_on_fingerprint_mismatch(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    run_dir = store.create_run(make_manifest())
    write_checkpoint(run_dir, fingerprint=FINGERPRINT_B)

    with pytest.raises(StateError, match="fingerprint mismatch"):
        plan_resume(run_dir)


def test_resume_planner_moves_to_judging_when_predictions_committed(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "runs")
    run_dir = store.create_run(make_manifest())
    write_checkpoint(run_dir, status="COMMITTED")

    plan = plan_resume(run_dir)

    assert plan.next_phase == "JUDGING"
    assert plan.committed_unit_ids == ("unit-1",)
    assert plan.restart_unit_ids == ()


def test_cli_inspect_and_resume_plan_only_are_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_dir = tmp_path / "runs"
    store = LocalArtifactStore(runs_dir)
    run_dir = store.create_run(make_manifest())
    before_paths = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))

    assert main(["inspect", "--runs-dir", str(runs_dir), "--run-id", "run-001"]) == 0
    inspect_output = json.loads(capsys.readouterr().out)

    assert inspect_output["run_id"] == "run-001"
    assert inspect_output["completed"] is False

    assert main(["resume", "--plan-only", "--runs-dir", str(runs_dir), "--run-id", "run-001"]) == 0
    plan_output = json.loads(capsys.readouterr().out)

    assert plan_output["next_phase"] == "RUNNING"
    assert plan_output["restart_unit_ids"] == ["unit-1"]
    assert sorted(path.relative_to(run_dir) for path in run_dir.rglob("*")) == before_paths
