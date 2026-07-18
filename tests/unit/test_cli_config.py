import json
import subprocess
import sys
from pathlib import Path

import pytest

from dmf_bench.cli import main
from dmf_bench.config import resolve_config
from dmf_bench.registry import supported_combinations


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


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


def test_validate_resolves_absolute_paths_and_redacts_secrets() -> None:
    resolved = resolve_config(FIXTURE_DIR / "experiment-valid.json")

    redacted = resolved.redacted()

    assert resolved.source_path.is_absolute()
    assert redacted["models"]["answerer"]["api_key"] == "<redacted>"
    assert "must-not-leak" not in json.dumps(redacted)


def test_validate_rejects_invalid_config_before_runtime_io() -> None:
    with pytest.raises(ValueError, match="Unsupported protocol"):
        resolve_config(FIXTURE_DIR / "experiment-invalid.json")


def test_cli_list_prints_supported_surface(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0

    output = capsys.readouterr().out

    assert "benchmarks:" in output
    assert "locomo/dmf/strict" in output
    assert "longmemeval/mem0/native" in output


def test_cli_validate_prints_redacted_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--config", str(FIXTURE_DIR / "experiment-valid.json")]) == 0

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
