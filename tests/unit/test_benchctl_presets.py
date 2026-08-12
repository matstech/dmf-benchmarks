from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dmf_benchctl.cli import main
from dmf_benchctl.presets import pin_operator_config


ROOT = Path(__file__).resolve().parents[2]


def test_config_list_works_outside_source_checkout(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["config", "list"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in payload["presets"]] == [
        "locomo-dmf",
        "locomo-mem0",
        "longmemeval-dmf",
        "longmemeval-mem0",
    ]


def test_installed_layout_uses_packaged_compose_and_operator_workspace(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(
        [
            "--image",
            "ghcr.io/example/dmf-benchmarks:tag",
            "--dry-run",
            "stack",
            "status",
        ]
    ) == 0

    command = capsys.readouterr().out.strip()
    assert "docker compose" in command
    assert str(ROOT / "deploy" / "compose.yaml") in command
    assert command.endswith("ps")


def test_config_export_creates_portable_editable_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "my-experiment"

    assert main(
        [
            "--project-dir",
            str(ROOT),
            "--work-dir",
            str(tmp_path),
            "config",
            "export",
            "locomo-mem0",
            str(output_dir),
        ]
    ) == 0

    experiment_path = output_dir / "experiment.json"
    framework_path = output_dir / "mem0-settings.yaml"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    assert framework_path.is_file()
    assert experiment["framework"] == "mem0"
    assert experiment["framework_config"]["path"] == "mem0-settings.yaml"
    assert experiment["framework_config"]["sha256"] == hashlib.sha256(
        framework_path.read_bytes()
    ).hexdigest()


def test_pin_operator_config_refreshes_edited_framework_checksum(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "my-experiment"
    assert main(
        [
            "--project-dir",
            str(ROOT),
            "config",
            "export",
            "locomo-mem0",
            str(output_dir),
        ]
    ) == 0
    framework_path = output_dir / "mem0-settings.yaml"
    framework_path.write_text(
        framework_path.read_text(encoding="utf-8") + "\n# operator edit\n",
        encoding="utf-8",
    )

    assert pin_operator_config(
        output_dir / "experiment.json",
        runtime_dir=ROOT,
    ) is True

    experiment = json.loads(
        (output_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert experiment["framework_config"]["sha256"] == hashlib.sha256(
        framework_path.read_bytes()
    ).hexdigest()
