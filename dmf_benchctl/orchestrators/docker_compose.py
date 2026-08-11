"""Docker Compose implementation of the benchmark orchestrator contract."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .base import CommandRunner


OPTIONAL_RUNTIME_ENVIRONMENT = (
    "OPENAI_BASE_URL",
    "OPENROUTER_BASE_URL",
    "OLLAMA_BASE_URL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> int:
    """Run a command with inherited standard streams."""

    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=False,
    ).returncode


@dataclass(slots=True)
class DockerComposeOrchestrator:
    """Translate control-plane operations into Docker Compose commands."""

    project_dir: Path
    compose_files: tuple[Path, ...]
    environment: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    runner: CommandRunner = run_command
    dry_run: bool = False

    def _base_command(self, *, job_profile: bool = False) -> list[str]:
        command = ["docker", "compose"]
        for compose_file in self.compose_files:
            command.extend(("-f", str(compose_file)))
        if job_profile:
            command.extend(("--profile", "job"))
        return command

    def _execute(
        self,
        command: Sequence[str],
        *,
        output_path: Path | None = None,
    ) -> int:
        if self.dry_run:
            print(shlex.join(command))
            return 0
        if output_path is not None and self.runner is run_command:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as stream:
                return subprocess.run(
                    list(command),
                    cwd=self.project_dir,
                    env=dict(self.environment),
                    check=False,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).returncode
        return self.runner(command, cwd=self.project_dir, env=self.environment)

    def run_external(self, command: Sequence[str]) -> int:
        """Run a non-Compose helper that is still owned by the control plane."""

        return self._execute(command)

    def pull_image(self, image_ref: str) -> int:
        """Pull one explicit image reference before resolving its registry digest."""

        return self._execute(("docker", "pull", image_ref))

    def validate_config(self) -> int:
        return self._execute((*self._base_command(), "config", "--quiet"))

    def build_images(self, services: Sequence[str]) -> int:
        return self._execute((*self._base_command(), "build", *services))

    def stack_up(self, services: Sequence[str], *, pull: str = "missing") -> int:
        return self._execute(
            (
                *self._base_command(),
                "up",
                "-d",
                "--wait",
                "--pull",
                pull,
                *services,
            )
        )

    def stack_down(self) -> int:
        return self._execute(
            (*self._base_command(), "down", "--remove-orphans")
        )

    def stack_stop(self, services: Sequence[str]) -> int:
        return self._execute((*self._base_command(), "stop", *services))

    def stack_status(self) -> int:
        return self._execute((*self._base_command(), "ps"))

    def run_runtime(
        self,
        arguments: Sequence[str],
        *,
        volumes: Sequence[str] = (),
        quiet: bool = False,
        output_path: Path | None = None,
    ) -> int:
        command = [
            *self._base_command(job_profile=True),
            "run",
            "--rm",
            "--no-deps",
            "--use-aliases",
            "--entrypoint",
            "dmf-bench",
        ]
        if quiet:
            command.append("--quiet")
        for name in OPTIONAL_RUNTIME_ENVIRONMENT:
            if str(self.environment.get(name, "")).strip():
                command.extend(("--env", name))
        for volume in volumes:
            command.extend(("--volume", volume))
        command.extend(("benchmark", *arguments))
        return self._execute(command, output_path=output_path if quiet else None)
