from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from dmf_benchctl.cli import main


ROOT = Path(__file__).resolve().parents[2]


class RecordingRunner:
    def __init__(self, return_codes: Sequence[int] = ()) -> None:
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []
        self.return_codes = list(return_codes)

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> int:
        self.calls.append((list(command), cwd, dict(env)))
        return self.return_codes.pop(0) if self.return_codes else 0


def test_run_starts_core_services_and_delegates_to_dmf_bench() -> None:
    runner = RecordingRunner()

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "--image",
            "ghcr.io/example/dmf-benchmarks:20260809T071720Z-RC",
            "--pull",
            "always",
            "run",
            "--config",
            "smoke/config/experiment-locomo-dmf.json",
            "--run-id",
            "official-locomo-dmf",
        ],
        runner=runner,
    )

    assert result == 0
    assert len(runner.calls) == 2
    up, run = (call[0] for call in runner.calls)
    assert up[-7:] == [
        "up",
        "-d",
        "--wait",
        "--pull",
        "always",
        "qdrant",
        "artifact-api",
    ]
    assert str(ROOT / "smoke" / "compose.yaml") in up
    assert run[-8:] == [
        "--entrypoint",
        "dmf-bench",
        "benchmark",
        "run",
        "--config",
        "/bench/smoke/config/experiment-locomo-dmf.json",
        "--run-id",
        "official-locomo-dmf",
    ]
    assert "--no-deps" in run
    assert runner.calls[0][1] == ROOT
    assert runner.calls[0][2]["DMF_BENCH_IMAGE"].endswith(
        ":20260809T071720Z-RC"
    )


def test_run_can_start_observability_and_allow_model_downloads() -> None:
    runner = RecordingRunner()

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "run",
            "--config",
            "/bench/config/experiment-locomo-dmf.json",
            "--run-id",
            "run-with-observability",
            "--observability",
            "--allow-downloads",
        ],
        runner=runner,
    )

    assert result == 0
    assert runner.calls[0][0][-4:] == [
        "qdrant",
        "artifact-api",
        "prometheus",
        "grafana",
    ]
    assert runner.calls[1][2]["HF_HUB_OFFLINE"] == "0"


def test_external_config_is_mounted_read_only() -> None:
    runner = RecordingRunner()
    config = ROOT / "tests" / "fixtures" / "experiment-valid.json"

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "validate",
            "--config",
            str(config),
        ],
        runner=runner,
    )

    assert result == 0
    command = runner.calls[0][0]
    assert command[-3:] == [
        "validate",
        "--config",
        "/bench/operator/experiment-valid.json",
    ]
    assert "--volume" in command
    assert f"{config}:/bench/operator/experiment-valid.json:ro" in command


def test_read_only_run_commands_use_shared_volume_without_starting_services() -> None:
    for operation in ("verify", "status", "inspect"):
        runner = RecordingRunner()
        result = main(
            [
                "--project-dir",
                str(ROOT),
                operation,
                "--run-id",
                "completed-run",
            ],
            runner=runner,
        )

        assert result == 0
        assert len(runner.calls) == 1
        command = runner.calls[0][0]
        assert command[-3:] == [operation, "--run-id", "completed-run"]
        assert command[command.index("--entrypoint") + 1] == "dmf-bench"


def test_stack_failure_prevents_runtime_execution() -> None:
    runner = RecordingRunner(return_codes=[17])

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "run",
            "--config",
            "/bench/config/experiment-locomo-dmf.json",
            "--run-id",
            "not-started",
        ],
        runner=runner,
    )

    assert result == 17
    assert len(runner.calls) == 1


def test_runtime_forwards_advanced_dmf_bench_arguments() -> None:
    runner = RecordingRunner()

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "runtime",
            "--",
            "list-datasets",
        ],
        runner=runner,
    )

    assert result == 0
    assert runner.calls[0][0][-2:] == ["benchmark", "list-datasets"]


def test_stack_down_preserves_named_volumes() -> None:
    runner = RecordingRunner()

    result = main(
        ["--project-dir", str(ROOT), "stack", "down"],
        runner=runner,
    )

    assert result == 0
    command = runner.calls[0][0]
    assert command[-2:] == ["down", "--remove-orphans"]
    assert "--volumes" not in command
    assert "-v" not in command


def test_publish_run_packages_with_dmf_bench_then_pushes_with_oras(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()
    output_dir = tmp_path / "official-bundles"

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "publish-run",
            "--run-id",
            "official-run",
            "--ref",
            "ghcr.io/example/dmf-benchmarks-runs:official-run",
            "--subject",
            "ghcr.io/example/dmf-benchmarks@sha256:abc",
            "--output-dir",
            str(output_dir),
        ],
        runner=runner,
    )

    assert result == 0
    assert output_dir.is_dir()
    package_command = runner.calls[0][0]
    assert package_command[package_command.index("--entrypoint") + 1] == "dmf-bench"
    assert "publish-run-oci" in package_command
    assert "--dry-run" in package_command
    assert f"{output_dir}:/bench/export" in package_command
    assert runner.calls[1][0] == [
        "oras",
        "push",
        "--artifact-type",
        "application/vnd.dmf-bench.run.v1+tar",
        "--subject",
        "ghcr.io/example/dmf-benchmarks@sha256:abc",
        "ghcr.io/example/dmf-benchmarks-runs:official-run",
        (
            f"{output_dir / 'official-run.run.tar.gz'}:"
            "application/vnd.dmf-bench.run.layer.v1.tar+gzip"
        ),
    ]
