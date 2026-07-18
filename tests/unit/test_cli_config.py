import json
import subprocess
import sys
from pathlib import Path

import pytest

from dmf_bench.cli import main
from dmf_bench.config import resolve_config
from dmf_bench.contracts import sha256_file
from dmf_bench.registry import supported_combinations


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


def write_valid_config(tmp_path: Path, *, dataset_path: Path | None = None) -> Path:
    selected_dataset_path = dataset_path or (FIXTURE_DIR / "locomo-mini.json")
    config = json.loads((FIXTURE_DIR / "experiment-valid.json").read_text(encoding="utf-8"))
    config["runtime"] = {
        "root": str(tmp_path),
        "runs_dir": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"),
    }
    config["dataset"] = {
        "name": "locomo",
        "source": "fixture",
        "revision": "fixture-revision",
        "sha256": sha256_file(selected_dataset_path),
        "path": str(selected_dataset_path.resolve()),
    }
    config["artifact_store"] = {"type": "local", "uri": str(tmp_path / "runs")}
    target = tmp_path / "experiment-valid.json"
    target.write_text(json.dumps(config), encoding="utf-8")
    return target


def test_supported_combinations_are_explicit_and_deterministic() -> None:
    assert supported_combinations() == [
        ("locomo", "dmf", "strict"),
        ("locomo", "dmf", "native"),
        ("locomo", "mem0", "strict"),
        ("locomo", "mem0", "native"),
        ("longmemeval", "dmf", "strict"),
        ("longmemeval", "dmf", "native"),
        ("longmemeval", "mem0", "strict"),
        ("longmemeval", "mem0", "native"),
    ]


def test_validate_resolves_absolute_paths_and_redacts_secrets(tmp_path: Path) -> None:
    resolved = resolve_config(write_valid_config(tmp_path))

    redacted = resolved.redacted()

    assert resolved.source_path.is_absolute()
    assert redacted["models"]["answerer"]["api_key"] == "<redacted>"
    assert "must-not-leak" not in json.dumps(redacted)


def test_validate_rejects_invalid_config_before_runtime_io() -> None:
    with pytest.raises(ValueError, match="Unsupported protocol"):
        resolve_config(FIXTURE_DIR / "experiment-invalid.json")


def test_validate_rejects_mutated_materialized_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "locomo-mini.json"
    dataset_path.write_text((FIXTURE_DIR / "locomo-mini.json").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    config_path = write_valid_config(tmp_path, dataset_path=dataset_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"]["sha256"] = sha256_file(FIXTURE_DIR / "locomo-mini.json")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        resolve_config(config_path)


def test_cli_list_prints_supported_surface(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0

    output = capsys.readouterr().out

    assert "benchmarks:" in output
    assert "locomo/dmf/strict" in output
    assert "longmemeval/mem0/native" in output


def test_cli_validate_prints_redacted_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", "--config", str(write_valid_config(tmp_path))]) == 0

    output = capsys.readouterr().out

    assert '"source_path"' in output
    assert '"api_key": "<redacted>"' in output
    assert "must-not-leak" not in output


def test_module_entrypoint_matches_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dmf_bench", "list"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "supported combinations:" in result.stdout
