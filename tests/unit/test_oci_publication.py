import json
import tarfile
from pathlib import Path

from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.cli import _copy_dataset_materialization_manifest, main
from dmf_bench.contracts import RunManifest, sha256_file
from dmf_bench.oci import publish_run_oci


def commit_fixture_run(store: LocalArtifactStore, run_id: str = "run-001") -> Path:
    run_dir = store.create_run(
        RunManifest(
            run_id=run_id,
            scientific_fingerprint="d" * 64,
            fingerprint_inputs={"benchmark": "fixture"},
            expected_item_ids=("unit-1",),
            atomic_unit="fixture-unit",
        )
    )
    write_json_atomic(run_dir / "resolved-config.json", {"run_id": run_id})
    write_json_atomic(run_dir / "reports" / "usage.json", {"tokens": 10})
    write_json_atomic(run_dir / "reports" / "timing.json", {"seconds": 1.5})
    write_json_atomic(run_dir / "items" / "unit-1" / "prediction.json", {"ok": True})
    staged = store.stage(run_id, run_dir)
    receipt = store.verify(staged)
    store.commit(staged, receipt)
    return run_dir


def test_publish_run_oci_dry_run_packages_complete_run_directory(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = commit_fixture_run(LocalArtifactStore(runs_dir))

    result = publish_run_oci(
        run_id="run-001",
        runs_dir=runs_dir,
        ref="ghcr.io/acme/dmf-benchmarks-runs:run-001",
        subject="ghcr.io/acme/dmf-benchmarks@sha256:" + "a" * 64,
        output_dir=tmp_path / "bundles",
        dry_run=True,
    )

    archive_path = Path(result["archive_path"])
    assert result["pushed"] is False
    assert archive_path.is_file()
    assert result["archive_sha256"] == sha256_file(archive_path)
    assert (run_dir / "OFFICIAL-RUN.json").is_file()
    assert (run_dir / "SHA256SUMS").is_file()

    official = json.loads((run_dir / "OFFICIAL-RUN.json").read_text(encoding="utf-8"))
    assert official["kind"] == "dmf-bench-official-run"
    assert official["oci"]["ref"] == "ghcr.io/acme/dmf-benchmarks-runs:run-001"
    assert official["oci"]["subject"].startswith("ghcr.io/acme/dmf-benchmarks@sha256:")

    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert "run-001/run-manifest.json" in names
    assert "run-001/resolved-config.json" in names
    assert "run-001/final/COMPLETED.json" in names
    assert "run-001/final/reports/usage.json" in names
    assert "run-001/OFFICIAL-RUN.json" in names
    assert "run-001/SHA256SUMS" in names


def test_cli_publish_run_oci_dry_run_prints_publication_payload(
    tmp_path: Path,
    capsys,
) -> None:
    runs_dir = tmp_path / "runs"
    commit_fixture_run(LocalArtifactStore(runs_dir))

    assert main(
        [
            "publish-run-oci",
            "--run-id",
            "run-001",
            "--runs-dir",
            str(runs_dir),
            "--ref",
            "ghcr.io/acme/dmf-benchmarks-runs:run-001",
            "--output-dir",
            str(tmp_path / "bundles"),
            "--dry-run",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "dmf-bench-oci-publication"
    assert payload["dry_run"] is True
    assert payload["oras"]["command"][:3] == ["oras", "push", "--artifact-type"]


def test_dataset_materialization_manifest_is_copied_into_run_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "cache" / "datasets" / "manifest.json"
    write_json_atomic(manifest, {"kind": "dataset-materialization", "schema_version": 1})
    run_dir = tmp_path / "runs" / "run-001"
    run_dir.mkdir(parents=True)

    copied = _copy_dataset_materialization_manifest(
        {"dataset": {"materialization_manifest": str(manifest)}},
        run_dir,
    )

    assert copied == run_dir / "datasets" / "materialization-manifest.json"
    assert json.loads(copied.read_text(encoding="utf-8"))["kind"] == "dataset-materialization"
