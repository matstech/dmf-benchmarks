import json
import shutil
from pathlib import Path

import pytest

from dmf_bench.cli import main
from dmf_bench.config import resolve_config
from dmf_bench.contracts import RunManifest, hash_canonical_json, sha256_file
from dmf_bench.datasets import load_dataset_registry, materialize_dataset, registry_as_dict
from dmf_bench.presets import BUILTIN_PRESETS, resolve_preset
from dmf_bench.provenance import build_run_provenance


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


def write_registry(tmp_path: Path, *, source_path: Path | None = None, sha256: str | None = None) -> Path:
    dataset_source = source_path or (FIXTURE_DIR / "locomo-mini.json")
    registry = {
        "schema_version": 1,
        "datasets": [
            {
                "schema_version": 1,
                "dataset_id": "fixture-locomo",
                "benchmark": "locomo",
                "source_url": dataset_source.resolve().as_uri(),
                "revision": "fixture-revision",
                "sha256": sha256 or sha256_file(dataset_source),
                "filename": "locomo-mini.json",
                "expected_schema": "locomo-mini-json-array-v1",
            }
        ],
    }
    registry_path = tmp_path / "dataset-registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return registry_path


def write_config(tmp_path: Path, *, dataset_path: Path, sha256: str | None = None) -> Path:
    selected_dataset_path = dataset_path
    if not dataset_path.resolve().is_relative_to(tmp_path.resolve()):
        selected_dataset_path = tmp_path / dataset_path.name
        shutil.copy2(dataset_path, selected_dataset_path)
    config = json.loads((FIXTURE_DIR / "experiment-valid.json").read_text(encoding="utf-8"))
    config["runtime"] = {
        "root": str(tmp_path),
        "runs_dir": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"),
        "metrics_port": 9464,
        "log_level": "INFO",
    }
    framework_config_path = tmp_path / "framework.toml"
    framework_config_path.write_text("[ltm]\nstorage_type = \"qdrant\"\n", encoding="utf-8")
    config["framework_config"] = {
        "path": str(framework_config_path.resolve()),
        "sha256": sha256_file(framework_config_path),
        "format": "toml",
        "profile": "fixture",
    }
    config["dataset"] = {
        "name": "locomo",
        "source": "fixture",
        "revision": "fixture-revision",
        "sha256": sha256 or sha256_file(selected_dataset_path),
        "path": str(selected_dataset_path.resolve()),
        "registry_id": "fixture-locomo",
        "expected_schema": "locomo-mini-json-array-v1",
    }
    config["artifact_store"] = {"type": "local", "uri": str(tmp_path / "runs")}
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_dataset_registry_parses_offline_and_declares_unpinned_official_limits() -> None:
    registry = load_dataset_registry()
    payload = registry_as_dict(registry)

    assert "locomo-official-unpinned" in registry
    assert "longmemeval-s-official-unpinned" in registry
    assert registry["locomo-official-unpinned"].pinned is False
    assert payload["schema_version"] == 1


def test_materialize_dataset_from_local_source_verifies_hash_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    registry_path = write_registry(tmp_path)

    materialized = materialize_dataset(
        dataset_id="fixture-locomo",
        output_dir=tmp_path / "datasets",
        registry_path=registry_path,
    )

    assert materialized.read_bytes() == (FIXTURE_DIR / "locomo-mini.json").read_bytes()
    assert sha256_file(materialized) == sha256_file(FIXTURE_DIR / "locomo-mini.json")
    assert not list((tmp_path / "datasets").glob("*.tmp"))


def test_materialize_dataset_refuses_download_hash_mismatch_without_publish(
    tmp_path: Path,
) -> None:
    registry_path = write_registry(tmp_path, sha256="f" * 64)

    with pytest.raises(ValueError, match="Downloaded dataset SHA-256 mismatch"):
        materialize_dataset(
            dataset_id="fixture-locomo",
            output_dir=tmp_path / "datasets",
            registry_path=registry_path,
        )

    assert not (tmp_path / "datasets" / "locomo-mini.json").exists()
    assert not list((tmp_path / "datasets").glob("*.tmp"))


def test_materialized_dataset_mutation_blocks_validate(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo-mini.json"
    dataset_path.write_bytes((FIXTURE_DIR / "locomo-mini.json").read_bytes())
    config_path = write_config(tmp_path, dataset_path=dataset_path)
    dataset_path.write_text(dataset_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        resolve_config(config_path)


def test_cli_materialize_dataset_prints_config_fragment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = write_registry(tmp_path)

    assert main(
        [
            "materialize-dataset",
            "--dataset-id",
            "fixture-locomo",
            "--output-dir",
            str(tmp_path / "datasets"),
            "--registry",
            str(registry_path),
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"]["name"] == "locomo"
    assert payload["dataset"]["sha256"] == sha256_file(FIXTURE_DIR / "locomo-mini.json")


def test_preset_fingerprint_and_resolution_are_stable(tmp_path: Path) -> None:
    preset = BUILTIN_PRESETS["paper/locomo-dmf-native-v1"]
    framework_config_path = tmp_path / "framework.toml"
    framework_config_path.write_text("[ltm]\nstorage_type = \"qdrant\"\n", encoding="utf-8")
    first = resolve_preset(
        "paper/locomo-dmf-native-v1",
        dataset_path=tmp_path / "locomo10.json",
        framework_config_path=framework_config_path,
        runtime_root=tmp_path,
    )
    second = resolve_preset(
        "paper/locomo-dmf-native-v1",
        dataset_path=tmp_path / "locomo10.json",
        framework_config_path=framework_config_path,
        runtime_root=tmp_path,
    )

    assert first == second
    assert first["preset"]["profile"] == "paper"
    assert first["preset"]["fingerprint"] == preset.fingerprint()
    assert first["preset"]["fingerprint"] != BUILTIN_PRESETS["default/locomo-dmf-native"].fingerprint()


def test_manifest_provenance_excludes_secrets_and_separates_operational_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "QDRANT_URL",
        "http://user:must-not-leak@qdrant:6333/private?token=must-not-leak",
    )
    dataset_path = FIXTURE_DIR / "locomo-mini.json"
    config_path = write_config(tmp_path, dataset_path=dataset_path)
    resolved = resolve_config(config_path)
    resolved.data["models"]["answerer"]["api_key"] = "must-not-leak"
    fingerprint_inputs = {
        "schema_version": 1,
        "benchmark": resolved.data["benchmark"],
        "framework": resolved.data["framework"],
        "protocol": resolved.data["protocol"],
        "dataset": resolved.data["dataset"],
        "selection": resolved.data["selection"],
    }
    manifest = RunManifest(
        run_id="run-001",
        scientific_fingerprint=hash_canonical_json(fingerprint_inputs),
        fingerprint_inputs=fingerprint_inputs,
        expected_item_ids=("conversation-0001",),
        atomic_unit="locomo-conversation",
        provenance=build_run_provenance(
            resolved.data,
            fingerprint_inputs=fingerprint_inputs,
            source_path=config_path,
        ),
    ).to_dict()

    serialized = json.dumps(manifest, sort_keys=True)
    assert "must-not-leak" not in serialized
    assert manifest["provenance"]["operational"]["qdrant"]["url"] == (
        "http://qdrant:6333"
    )
    assert manifest["provenance"]["scientific"]["dataset"]["sha256"] == sha256_file(dataset_path)
    assert manifest["provenance"]["scientific"]["framework_config"]["sha256"] == resolved.data["framework_config"]["sha256"]
    assert "platform" in manifest["provenance"]["operational"]
    assert "remote_provider_bitwise_reproducibility" in manifest["provenance"]["limits"]
