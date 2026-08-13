from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from types import SimpleNamespace
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from dmf_benchctl.cli import _operator_status, _print_operator_status, main
from dmf_benchctl.preparation import (
    CommandOutput,
    build_preparation_state,
    load_experiment_metadata,
)
from dmf_benchctl.orchestrators.docker_compose import DockerComposeOrchestrator


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


class RecordingOutputRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> CommandOutput:
        del cwd, env
        self.calls.append(list(command))
        if command[:3] == ["docker", "image", "inspect"]:
            return CommandOutput(
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "RepoDigests": [
                                "ghcr.io/example/dmf-benchmarks@sha256:abc123"
                            ],
                            "Architecture": "arm64",
                            "Os": "linux",
                            "Config": {
                                "Labels": {
                                    "org.opencontainers.image.revision": "deadbeef"
                                }
                            },
                        }
                    ]
                ),
                stderr="",
            )
        if command[:3] == ["docker", "run", "--rm"]:
            return CommandOutput(
                returncode=0,
                stdout=json.dumps(
                    {
                        "schema_version": 1,
                        "datasets": [
                            {
                                "dataset_id": "locomo-official-v1",
                                "benchmark": "locomo",
                                "source_url": "https://example.test/locomo.json",
                                "revision": "locomo-commit",
                                "sha256": "1" * 64,
                                "filename": "locomo.json",
                                "expected_schema": "locomo-v1",
                                "pinned": True,
                            },
                            {
                                "dataset_id": "longmemeval-s-official-v1",
                                "benchmark": "longmemeval",
                                "source_url": "https://example.test/longmemeval.json",
                                "revision": "longmemeval-commit",
                                "sha256": "2" * 64,
                                "filename": "longmemeval.json",
                                "expected_schema": "longmemeval-v1",
                                "pinned": True,
                            },
                        ],
                    }
                ),
                stderr="",
            )
        return CommandOutput(
            returncode=0,
            stdout=json.dumps({"OSType": "linux", "Architecture": "aarch64"}),
            stderr="",
        )


def _write_exported_experiment(
    path: Path,
    *,
    experiment_id: str,
    framework: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "benchmark": "locomo",
                "framework": framework,
                "dataset": {
                    "name": "locomo",
                    "registry_id": "locomo-official-v1",
                    "sampling": {
                        "fraction": 0.05,
                        "unit": "conversation",
                        "seed": 7,
                        "rounding": "ceil",
                    },
                },
                "models": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_exported_generic_filenames_generate_distinct_matrix_run_ids(
    tmp_path: Path,
) -> None:
    mem0 = _write_exported_experiment(
        tmp_path / "locomo-mem0" / "experiment.json",
        experiment_id="custom-locomo-mem0",
        framework="mem0",
    )
    dmf = _write_exported_experiment(
        tmp_path / "locomo-dmf" / "experiment.json",
        experiment_id="custom-locomo-dmf",
        framework="dmf",
    )

    metadata = load_experiment_metadata([mem0, dmf])
    state = build_preparation_state(
        project_dir=tmp_path,
        image={"resolved_ref": "example@sha256:abc"},
        experiments=metadata,
        batch_id="20260813T053827Z",
        observability=True,
        allow_downloads=True,
    )

    assert [item["run_id"] for item in state["experiments"]] == [
        "20260813T053827Z-locomo-mem0",
        "20260813T053827Z-locomo-dmf",
    ]
def test_preparation_rejects_duplicate_benchmark_framework_configs(
    tmp_path: Path,
) -> None:
    first = _write_exported_experiment(
        tmp_path / "first" / "experiment.json",
        experiment_id="first-locomo-mem0",
        framework="mem0",
    )
    second = _write_exported_experiment(
        tmp_path / "second" / "experiment.json",
        experiment_id="second-locomo-mem0",
        framework="mem0",
    )

    with pytest.raises(
        ValueError,
        match="one config per benchmark/framework; duplicate locomo/mem0",
    ):
        load_experiment_metadata([first, second])


def test_quiet_runtime_redirects_stdout_and_stderr_to_local_log(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured.update(kwargs)
        kwargs["stdout"].write("container stdout\ncontainer stderr\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "dmf_benchctl.orchestrators.docker_compose.subprocess.run",
        fake_run,
    )
    log_path = tmp_path / "logs" / "run.log"
    orchestrator = DockerComposeOrchestrator(
        project_dir=ROOT,
        compose_files=(ROOT / "deploy" / "compose.yaml",),
    )

    result = orchestrator.run_runtime(
        ["run", "--config", "/bench/config/experiment.json"],
        quiet=True,
        output_path=log_path,
    )

    assert result == 0
    assert log_path.read_text(encoding="utf-8") == (
        "container stdout\ncontainer stderr\n"
    )
    assert "--use-aliases" in captured["command"]
    assert captured["stderr"] == -2


def test_runtime_mounts_provider_key_as_ephemeral_file_without_environment_leak(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    observed: dict[str, object] = {}

    def runner(command, *, cwd, env):
        del cwd
        serialized = " ".join(command)
        observed["command"] = list(command)
        observed["environment"] = dict(env)
        mount = next(
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--volume" and "/run/secrets/openai_api_key" in command[index + 1]
        )
        host_path = Path(mount.split(":", 1)[0])
        observed["host_path"] = host_path
        observed["secret_during_call"] = host_path.read_text(encoding="utf-8")
        assert "sensitive-key" not in serialized
        return 0

    orchestrator = DockerComposeOrchestrator(
        project_dir=project,
        compose_files=(ROOT / "deploy" / "compose.yaml",),
        environment={"OPENAI_API_KEY": "sensitive-key", "PATH": "/bin"},
        runner=runner,
    )

    assert orchestrator.run_runtime(["run", "--config", "/bench/config.json"]) == 0

    command = observed["command"]
    environment = observed["environment"]
    assert isinstance(command, list)
    assert isinstance(environment, dict)
    assert "OPENAI_API_KEY" not in environment
    assert "OPENAI_API_KEY_FILE=/run/secrets/openai_api_key" in command
    assert observed["secret_during_call"] == "sensitive-key\n"
    assert not Path(str(observed["host_path"])).exists()


def test_operator_status_renames_internal_units_and_renders_progress(
    capsys,
) -> None:
    status = _operator_status(
        {
            "schema_version": 2,
            "run_id": "sample-run",
            "state": "RUNNING",
            "phase": "RUNNING",
            "items": {"committed": 42, "expected": 103, "failed": 0},
            "units": {"committed": 4, "expected": 10, "failed": 0},
        }
    )

    assert status["progress"] == {"percentage": 40.8}
    assert status["evaluation_items"] == {
        "completed": 42,
        "total": 103,
        "failed": 0,
    }
    assert status["input_records"] == {
        "completed": 4,
        "total": 10,
        "failed": 0,
    }
    assert "items" not in status
    assert "units" not in status

    _print_operator_status(status)
    output = capsys.readouterr().out
    assert "[############------------------]  40.8%" in output
    assert "42 of 103 completed" in output
    assert "4 of 10 processed" in output
    assert "questions or scored records" in output
    assert "source conversations or records" in output


def test_run_detaches_worker_and_prints_inspection_commands(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project = tmp_path / "project"
    (project / "deploy").mkdir(parents=True)
    (project / "config").mkdir()
    (project / "deploy" / "compose.yaml").write_text(
        "services: {}\n",
        encoding="utf-8",
    )
    config = project / "config" / "experiment.json"
    config.write_text("{}\n", encoding="utf-8")
    popen_calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        pid = 4321

    def fake_popen(command, **kwargs):
        popen_calls.append((list(command), dict(kwargs)))
        return FakeProcess()

    monkeypatch.setattr("dmf_benchctl.cli.subprocess.Popen", fake_popen)

    result = main(
        [
            "--project-dir",
            str(project),
            "--image",
            "ghcr.io/example/dmf-benchmarks@sha256:abc123",
            "run",
            "--config",
            str(config),
            "--run-id",
            "detached-run",
        ]
    )

    assert result == 0
    assert len(popen_calls) == 1
    command, kwargs = popen_calls[0]
    assert command[:4] == [
        sys.executable,
        "-m",
        "dmf_benchctl",
        "--foreground-worker",
    ]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] == subprocess.DEVNULL
    job = json.loads(
        (project / ".dmf-bench/jobs/detached-run.json").read_text(encoding="utf-8")
    )
    assert job["pid"] == 4321
    assert job["run_ids"] == ["detached-run"]
    output = json.loads(capsys.readouterr().out)
    assert output["detached"] is True
    assert output["worker_log"] == ".dmf-bench/logs/detached-run.worker.log"
    assert "status --run-id detached-run" in output["status_commands"][0]


def test_prepare_locks_image_validates_smoke_suite_and_writes_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = RecordingRunner()
    output_runner = RecordingOutputRunner()
    state_path = tmp_path / "prepared.json"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "--image",
            "ghcr.io/example/dmf-benchmarks:rc",
            "prepare",
            "--suite",
            "smoke",
            "--state-file",
            str(state_path),
            "--batch-id",
            "20260811T120000Z",
            "--observability",
            "--allow-downloads",
        ],
        runner=runner,
        output_runner=output_runner,
    )

    assert result == 0
    assert runner.calls[0][0] == ["docker", "version"]
    assert runner.calls[1][0] == ["docker", "compose", "version"]
    assert runner.calls[2][0] == [
        "docker",
        "pull",
        "ghcr.io/example/dmf-benchmarks:rc",
    ]
    assert output_runner.calls == [
        ["docker", "image", "inspect", "ghcr.io/example/dmf-benchmarks:rc"],
        ["docker", "info", "--format", "{{json .}}"],
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "dmf-bench",
            "ghcr.io/example/dmf-benchmarks@sha256:abc123",
            "list-datasets",
        ],
    ]
    assert runner.calls[3][0][-2:] == ["config", "--quiet"]
    assert runner.calls[4][0][-4:] == [
        "qdrant",
        "artifact-api",
        "prometheus",
        "grafana",
    ]
    validation_calls = runner.calls[5:]
    assert len(validation_calls) == 4
    assert all("--materialize-datasets" in call[0] for call in validation_calls)
    assert all("--quiet" in call[0] for call in validation_calls)
    assert all(
        call[2]["DMF_BENCH_IMAGE"]
        == "ghcr.io/example/dmf-benchmarks@sha256:abc123"
        for call in runner.calls[3:]
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["schema_version"] == 2
    assert state["batch_id"] == "20260811T120000Z"
    assert state["image"] == {
        "digest": "sha256:abc123",
        "platform": "linux/arm64",
        "requested_ref": "ghcr.io/example/dmf-benchmarks:rc",
        "resolved_ref": "ghcr.io/example/dmf-benchmarks@sha256:abc123",
        "revision": "deadbeef",
    }
    assert [item["run_id"] for item in state["experiments"]] == [
        "20260811T120000Z-locomo-dmf",
        "20260811T120000Z-locomo-mem0",
        "20260811T120000Z-longmemeval-dmf",
        "20260811T120000Z-longmemeval-mem0",
    ]
    assert all(len(item["config_sha256"]) == 64 for item in state["experiments"])
    assert all(item["dataset_lock"]["sha256"] in {"1" * 64, "2" * 64} for item in state["experiments"])


def test_run_prepared_executes_the_locked_batch_sequentially(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = RecordingRunner(
        return_codes=[0, 4, 0, 3, 0, 3, 0, 3, 0]
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    state_path = tmp_path / "prepared.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "prepared_at": "2026-08-11T12:00:00Z",
                "batch_id": "20260811T120000Z",
                "project_dir": str(ROOT),
                "image": {
                    "resolved_ref": "ghcr.io/example/dmf-benchmarks@sha256:abc123",
                    "digest": "sha256:abc123",
                    "revision": "deadbeef",
                },
                "observability": True,
                "allow_downloads": True,
                "experiments": [
                    {
                        "host_path": str(ROOT / path),
                        "run_id": f"batch-{Path(path).stem.removeprefix('experiment-')}",
                        "requested_models": [
                            {
                                "role": "answerer",
                                "provider": "openai",
                                "model": "gpt-4.1-mini",
                            }
                        ],
                    }
                    for path in (
                        "smoke/config/experiment-locomo-dmf.json",
                        "smoke/config/experiment-locomo-mem0.json",
                        "smoke/config/experiment-longmemeval-dmf.json",
                        "smoke/config/experiment-longmemeval-mem0.json",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "run",
            "--prepared",
            str(state_path),
        ],
        runner=runner,
    )

    assert result == 0
    assert len(runner.calls) == 9
    assert runner.calls[0][0][-4:] == [
        "qdrant",
        "artifact-api",
        "prometheus",
        "grafana",
    ]
    execution_calls = runner.calls[2::2]
    assert [call[0][-1] for call in execution_calls] == [
        "batch-locomo-dmf",
        "batch-locomo-mem0",
        "batch-longmemeval-dmf",
        "batch-longmemeval-mem0",
    ]
    assert execution_calls[0][0][execution_calls[0][0].index("benchmark") + 1] == "resume"
    assert all(
        call[0][call[0].index("benchmark") + 1] == "run"
        for call in execution_calls[1:]
    )
    for _, _, environment in runner.calls:
        assert environment["DMF_BENCH_IMAGE"].endswith("@sha256:abc123")
        assert environment["DMF_BENCH_IMAGE_DIGEST"] == "sha256:abc123"
        assert environment["DMF_BENCH_HARNESS_COMMIT"] == "deadbeef"
        assert environment["HF_HUB_OFFLINE"] == "0"
    assert all("--quiet" in call[0] for call in runner.calls[1:])


def test_run_prepared_rejects_config_changed_after_v2_preparation(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    (project / "deploy").mkdir(parents=True)
    (project / "deploy" / "compose.yaml").write_text(
        "services: {}\n",
        encoding="utf-8",
    )
    config = project / "experiment.json"
    original = b'{"benchmark":"locomo"}\n'
    config.write_bytes(original)
    state_path = project / "prepared.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "project_dir": str(project),
                "batch_id": "locked-batch",
                "image": {
                    "resolved_ref": "ghcr.io/example/image@sha256:abc",
                    "digest": "sha256:abc",
                    "revision": "deadbeef",
                },
                "experiments": [
                    {
                        "host_path": str(config),
                        "config_sha256": hashlib.sha256(original).hexdigest(),
                        "dataset_lock": {"dataset_id": "locomo-official-v1"},
                        "run_id": "locked-run",
                        "requested_models": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config.write_text('{"benchmark":"longmemeval"}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--project-dir",
                str(project),
                "run",
                "--prepared",
                str(state_path),
            ],
            runner=RecordingRunner(),
        )

    assert exc_info.value.code == 2


def test_resume_prepared_uses_locked_image_and_quiet_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = RecordingRunner()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    run_id = "batch-locomo-dmf"
    state_path = tmp_path / "prepared.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_dir": str(ROOT),
                "image": {
                    "resolved_ref": "ghcr.io/example/dmf-benchmarks@sha256:abc123",
                    "digest": "sha256:abc123",
                    "revision": "deadbeef",
                },
                "allow_downloads": True,
                "experiments": [
                    {
                        "host_path": str(
                            ROOT / "smoke/config/experiment-locomo-dmf.json"
                        ),
                        "run_id": run_id,
                        "requested_models": [
                            {
                                "role": "answerer",
                                "provider": "openai",
                                "model": "gpt-4.1-mini",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = main(
        [
            "--project-dir",
            str(ROOT),
            "resume",
            "--prepared",
            str(state_path),
            "--run-id",
            run_id,
        ],
        runner=runner,
    )

    assert result == 0
    assert len(runner.calls) == 2
    runtime_command = runner.calls[1][0]
    assert runtime_command[runtime_command.index("benchmark") :] == [
        "benchmark",
        "resume",
        "--run-id",
        run_id,
    ]
    assert "--quiet" in runtime_command
    assert "OPENAI_API_KEY_FILE=/run/secrets/openai_api_key" in runtime_command
    assert runner.calls[1][2]["DMF_BENCH_IMAGE"].endswith("@sha256:abc123")
    assert runner.calls[1][2]["HF_HUB_OFFLINE"] == "0"


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
    assert run[-9:] == [
        "--entrypoint",
        "dmf-bench",
        "--quiet",
        "benchmark",
        "run",
        "--config",
        "/bench/smoke/config/experiment-locomo-dmf.json",
        "--run-id",
        "official-locomo-dmf",
    ]
    assert "--no-deps" in run
    assert "--quiet" in run
    assert runner.calls[0][1] == ROOT
    assert runner.calls[0][2]["DMF_BENCH_IMAGE"].endswith(
        ":20260809T071720Z-RC"
    )


def test_run_can_start_observability_and_allow_model_downloads(monkeypatch) -> None:
    runner = RecordingRunner()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

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
            "--follow",
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
    assert "--quiet" not in runner.calls[1][0]
    base_url_option = runner.calls[1][0].index("OPENAI_BASE_URL")
    assert runner.calls[1][0][base_url_option - 1] == "--env"


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
    assert f"{config.parent}:/bench/operator:ro" in command


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
