import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dmf_bench.cli import main
from dmf_bench.config import resolve_config
from dmf_bench.atomic_io import read_json
from dmf_bench.contracts import sha256_file
from dmf_bench.registry import supported_combinations
from dmf_bench.secrets import load_runtime_secret_files


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


def write_valid_config(tmp_path: Path, *, dataset_path: Path | None = None) -> Path:
    selected_dataset_path = dataset_path or (tmp_path / "locomo-mini.json")
    if dataset_path is None:
        selected_dataset_path.write_bytes((FIXTURE_DIR / "locomo-mini.json").read_bytes())
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
        "sha256": sha256_file(selected_dataset_path),
        "path": str(selected_dataset_path.resolve()),
    }
    config["artifact_store"] = {"type": "local", "uri": str(tmp_path / "runs")}
    target = tmp_path / "experiment-valid.json"
    target.write_text(json.dumps(config), encoding="utf-8")
    return target


def test_supported_combinations_are_explicit_and_deterministic() -> None:
    assert supported_combinations() == [
        ("locomo", "dmf"),
        ("locomo", "mem0"),
        ("longmemeval", "dmf"),
        ("longmemeval", "mem0"),
    ]


def test_validate_resolves_absolute_paths_and_redacts_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://user:must-not-leak@example.test/v1?api_key=must-not-leak#secret",
    )
    resolved = resolve_config(write_valid_config(tmp_path))

    redacted = resolved.redacted()

    assert resolved.source_path.is_absolute()
    assert redacted["runtime"]["environment"]["secrets_present"]["OPENAI_API_KEY"] is True
    assert redacted["runtime"]["environment"]["endpoints"]["OPENAI_BASE_URL"] == (
        "https://example.test"
    )
    assert redacted["models"]["answerer"]["parameters"]["max_tokens"] == 4096
    assert "must-not-leak" not in json.dumps(redacted)


def test_validate_rejects_inline_secrets(tmp_path: Path) -> None:
    config_path = write_valid_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["models"]["answerer"]["api_key"] = "must-not-leak"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Inline secret field"):
        resolve_config(config_path)


def test_runtime_secret_file_populates_provider_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_path = tmp_path / "openai-key"
    secret_path.write_text("file-secret\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(secret_path))

    load_runtime_secret_files()

    assert os.environ["OPENAI_API_KEY"] == "file-secret"


def test_validate_rejects_mutated_framework_config(tmp_path: Path) -> None:
    config_path = write_valid_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["framework_config"]["path"]).write_text("mutated = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="framework_config SHA-256 mismatch"):
        resolve_config(config_path)


def test_validate_resolves_bundle_relative_resource_paths(tmp_path: Path) -> None:
    config_path = write_valid_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["framework_config"]["path"] = "framework.toml"
    config["dataset"]["path"] = "locomo-mini.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    resolved = resolve_config(config_path)

    assert resolved.data["framework_config"]["path"] == str(
        (tmp_path / "framework.toml").resolve()
    )
    assert resolved.data["dataset"]["path"] == str(
        (tmp_path / "locomo-mini.json").resolve()
    )


def test_validate_rejects_paths_outside_runtime_root(tmp_path: Path) -> None:
    config_path = write_valid_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"]["path"] = str(FIXTURE_DIR / "locomo-mini.json")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset.path must be contained"):
        resolve_config(config_path)


def test_resolved_config_persists_redacted_canonical_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv(
        "QDRANT_URL",
        "http://user:must-not-leak@qdrant:6333/ignored?token=must-not-leak",
    )
    resolved = resolve_config(write_valid_config(tmp_path))

    target = resolved.persist_redacted(tmp_path / "run")
    persisted = read_json(target)

    assert target.name == "resolved-config.json"
    assert persisted["runtime"]["environment"]["secrets_present"]["OPENAI_API_KEY"] is True
    assert persisted["runtime"]["environment"]["endpoints"]["QDRANT_URL"] == (
        "http://qdrant:6333"
    )
    assert "must-not-leak" not in target.read_text(encoding="utf-8")


def test_validate_rejects_invalid_config_before_runtime_io() -> None:
    with pytest.raises(ValueError, match="root must be absolute"):
        resolve_config(FIXTURE_DIR / "experiment-invalid.json")


def test_validate_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    config_path = write_valid_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["unexpected_field"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"unsupported fields: \['unexpected_field'\]",
    ):
        resolve_config(config_path)


def test_validate_rejects_v1_config_before_preflight(tmp_path: Path) -> None:
    config_path = write_valid_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="schema_version=2; v1 configs are not supported",
    ):
        resolve_config(config_path)


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
    assert "protocols:" not in output
    assert "longmemeval/mem0" in output
    assert output.count("  locomo/") == 2
    assert output.count("  longmemeval/") == 2


def test_cli_validate_prints_redacted_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", "--config", str(write_valid_config(tmp_path))]) == 0

    output = capsys.readouterr().out

    assert '"source_path"' in output
    assert '"max_tokens": 4096' in output
    assert '"api_key"' not in output


def test_module_entrypoint_matches_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dmf_bench", "list"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "supported combinations:" in result.stdout


@pytest.mark.parametrize("removed_command", ["report", "publish"])
def test_runtime_cli_does_not_advertise_placeholder_commands(
    removed_command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([removed_command])

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
