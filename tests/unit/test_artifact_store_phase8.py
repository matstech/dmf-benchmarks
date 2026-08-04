import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dmf_bench.api import create_app
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.cli import main
from dmf_bench.contracts import RunManifest
from dmf_bench.state import StateError


FINGERPRINT = "c" * 64


def make_manifest(run_id: str = "run-001") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        scientific_fingerprint=FINGERPRINT,
        fingerprint_inputs={"benchmark": "fixture", "artifact_backend": "local-only"},
        expected_item_ids=("unit-1",),
        atomic_unit="fixture-unit",
    )


def commit_fixture_run(store: LocalArtifactStore, run_id: str = "run-001") -> Path:
    run_dir = store.create_run(make_manifest(run_id))
    write_json_atomic(run_dir / "evaluations" / "evaluations.json", [{"ok": True}])
    write_json_atomic(run_dir / "reports" / "summary.json", {"expected": 1, "evaluated": 1})
    (run_dir / "reports" / "summary.md").write_text("# report\n", encoding="utf-8")

    staged = store.stage(run_id, run_dir)
    receipt = store.verify(staged)
    store.commit(staged, receipt)
    return run_dir


def test_local_only_committed_verify_and_inspect_on_shared_volume(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "shared-runs")
    commit_fixture_run(store)

    verified = store.verify_committed("run-001")
    inspected = store.inspect_committed("run-001")

    assert verified["status"] == "verified"
    assert verified["backend"] == "local-only"
    assert verified["shared_volume_root"] == str(tmp_path / "shared-runs")
    assert verified["verified_artifact_count"] >= 3
    assert verified["retry_policy"]["status"] == "NOT_APPLICABLE"
    assert inspected["api_mode"] == "read-only"
    assert inspected["writable_api"] is False
    assert inspected["artifact_count"] == verified["verified_artifact_count"]


def test_local_only_verify_detects_checksum_mismatch(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "shared-runs")
    run_dir = commit_fixture_run(store)
    target = run_dir / "final" / "reports" / "summary.json"
    target.write_text(target.read_text(encoding="utf-8").replace("1", "2", 1), encoding="utf-8")

    with pytest.raises(StateError, match="checksum mismatch"):
        store.verify_committed("run-001")


def test_local_only_verify_detects_missing_completion_marker(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "shared-runs")
    run_dir = commit_fixture_run(store)
    (run_dir / "final" / "COMPLETED.json").unlink()

    with pytest.raises(StateError, match="Missing completion marker"):
        store.verify_committed("run-001")


def test_local_only_restore_copy_remains_verifiable_by_digest(tmp_path: Path) -> None:
    source_store = LocalArtifactStore(tmp_path / "source-runs")
    source_run_dir = commit_fixture_run(source_store)
    restored_root = tmp_path / "restored-runs"
    shutil.copytree(source_run_dir, restored_root / "run-001")

    restored = LocalArtifactStore(restored_root).verify_committed("run-001")

    assert restored["status"] == "verified"
    assert restored["final_manifest_sha256"] == source_store.verify_committed("run-001")[
        "final_manifest_sha256"
    ]


def test_artifact_api_lists_verifies_and_downloads_json(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "shared-runs")
    commit_fixture_run(store)
    client = TestClient(create_app(tmp_path / "shared-runs"))

    landing = client.get("/")
    health = client.get("/health")
    services = client.get("/services")
    run = client.get("/runs/run-001")
    artifacts = client.get("/runs/run-001/artifacts")
    verified = client.get("/runs/run-001/verify")
    download = client.get("/runs/run-001/artifacts/reports/summary.json")

    assert landing.status_code == 200
    assert landing.headers["content-type"].startswith("text/html")
    assert "DMF Benchmarks" in landing.text
    assert health.json() == {
        "status": "ok",
        "mode": "read-only",
        "writable": False,
    }
    assert services.json()["services"]["grafana"] == {
        "label": "Grafana",
        "port": 3000,
        "path": "/d/dmf-benchmarks/dmf-benchmarks",
    }
    assert run.json()["completed"] is True
    assert artifacts.json()["artifacts"]
    assert verified.json()["status"] == "verified"
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/json")
    assert download.json() == {"evaluated": 1, "expected": 1}


def test_artifact_api_service_catalog_uses_published_ports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAFANA_PORT", "4300")
    monkeypatch.setenv("PROMETHEUS_PORT", "49090")
    monkeypatch.setenv("QDRANT_HTTP_PORT", "46333")
    client = TestClient(create_app(tmp_path / "shared-runs"))

    services = client.get("/services").json()["services"]

    assert services["grafana"]["port"] == 4300
    assert services["prometheus"]["port"] == 49090
    assert services["qdrant"]["port"] == 46333


def test_artifact_api_refuses_writes_and_does_not_mutate_shared_volume(tmp_path: Path) -> None:
    runs_dir = tmp_path / "shared-runs"
    store = LocalArtifactStore(runs_dir)
    run_dir = store.create_run(make_manifest())
    client = TestClient(create_app(runs_dir))
    before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))

    response = client.post(
        "/runs/run-001/artifacts",
        json={
            "artifacts": [
                {"path": "evaluations/evaluations.json", "payload": [{"ok": True}]},
                {"path": "reports/summary.json", "payload": {"evaluated": 1, "expected": 1}},
            ]
        },
    )

    assert response.status_code == 405
    assert sorted(path.relative_to(run_dir) for path in run_dir.rglob("*")) == before


def test_artifact_api_rejects_unlisted_or_unsafe_artifact_path(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "shared-runs")
    commit_fixture_run(store)
    client = TestClient(create_app(tmp_path / "shared-runs"))

    assert client.get("/runs/run-001/artifacts/../run-manifest.json").status_code == 404
    assert client.get("/runs/run-001/artifacts/not-listed.json").status_code == 404


def test_cli_does_not_expose_artifact_api_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["api", "--help"])

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_cli_inspect_and_verify_committed_artifacts_are_read_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_dir = tmp_path / "shared-runs"
    run_dir = commit_fixture_run(LocalArtifactStore(runs_dir))
    before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))

    assert main(["inspect", "--runs-dir", str(runs_dir), "--run-id", "run-001"]) == 0
    inspect_output = json.loads(capsys.readouterr().out)
    assert inspect_output["backend"] == "local-only"

    assert main(["verify", "--runs-dir", str(runs_dir), "--run-id", "run-001"]) == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output["status"] == "verified"
    assert sorted(path.relative_to(run_dir) for path in run_dir.rglob("*")) == before


def test_phase8_does_not_add_s3_backend_file() -> None:
    assert not Path("dmf_bench/artifacts/s3.py").exists()


def test_committed_artifact_access_rejects_unsafe_run_id(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "shared-runs")

    with pytest.raises(StateError, match="Unsafe run_id"):
        store.run_dir("../outside")
