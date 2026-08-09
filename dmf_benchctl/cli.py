"""User-facing control plane for Docker-native benchmark operations."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .orchestrators import DockerComposeOrchestrator
from .orchestrators.base import CommandRunner
from .orchestrators.docker_compose import run_command


CORE_SERVICES = ("qdrant", "artifact-api")
OBSERVABILITY_SERVICES = ("prometheus", "grafana")
IMAGE_SERVICES = ("artifact-api", "benchmark")
RUN_ARTIFACT_TYPE = "application/vnd.dmf-bench.run.v1+tar"
RUN_LAYER_TYPE = "application/vnd.dmf-bench.run.layer.v1.tar+gzip"


def _project_dir(value: str | None) -> Path:
    if value:
        candidate = Path(value).expanduser().resolve()
        if not (candidate / "deploy" / "compose.yaml").is_file():
            raise ValueError(
                f"project directory does not contain deploy/compose.yaml: {candidate}"
            )
        return candidate

    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "deploy" / "compose.yaml").is_file():
            return candidate.resolve()
    raise ValueError(
        "cannot locate deploy/compose.yaml; run from the repository or use --project-dir"
    )


def _resolve_file(project_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def _deduplicate(paths: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


def _runtime_path(
    project_dir: Path,
    value: str,
) -> tuple[str, tuple[str, ...], tuple[Path, ...]]:
    """Map a host config path to its stable path in the benchmark container."""

    if value == "/bench" or value.startswith("/bench/"):
        return value, (), ()

    host_path = _resolve_file(project_dir, value)
    if not host_path.is_file():
        raise ValueError(f"configuration file does not exist: {host_path}")

    mappings = (
        (project_dir / "config", Path("/bench/config"), None),
        (
            project_dir / "smoke",
            Path("/bench/smoke"),
            project_dir / "smoke" / "compose.yaml",
        ),
        (project_dir / "datasets", Path("/bench/datasets"), None),
    )
    for host_root, container_root, override in mappings:
        try:
            relative = host_path.relative_to(host_root.resolve())
        except ValueError:
            continue
        overrides = (override.resolve(),) if override is not None else ()
        return str(container_root / relative), (), overrides

    container_path = Path("/bench/operator") / host_path.name
    volume = f"{host_path}:{container_path}:ro"
    return str(container_path), (volume,), ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmf-benchctl",
        description=(
            "Host-side control plane for Docker-native DMF benchmark operations."
        ),
    )
    parser.add_argument(
        "--project-dir",
        help="Repository directory containing deploy/compose.yaml (auto-detected by default).",
    )
    parser.add_argument(
        "--compose-override",
        action="append",
        default=[],
        help="Additional Compose file; may be repeated.",
    )
    parser.add_argument(
        "--image",
        help="Benchmark image reference; overrides DMF_BENCH_IMAGE.",
    )
    parser.add_argument(
        "--pull",
        choices=("always", "missing", "never"),
        default="missing",
        help="Image pull policy used while starting services.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print orchestrator commands without executing them.",
    )

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", help="Validate the orchestrator configuration.")

    image_parser = commands.add_parser("image", help="Manage benchmark images.")
    image_commands = image_parser.add_subparsers(dest="image_command", required=True)
    image_commands.add_parser("build", help="Build the benchmark and Artifact API images.")

    stack_parser = commands.add_parser("stack", help="Manage local benchmark services.")
    stack_commands = stack_parser.add_subparsers(dest="stack_command", required=True)
    stack_up = stack_commands.add_parser("up", help="Start local services.")
    stack_up.add_argument(
        "--observability",
        action="store_true",
        help="Also start Prometheus and Grafana.",
    )
    stack_stop = stack_commands.add_parser("stop", help="Stop local services without removing them.")
    stack_stop.add_argument(
        "--observability",
        action="store_true",
        help="Also stop Prometheus and Grafana.",
    )
    stack_commands.add_parser("down", help="Stop and remove services, preserving volumes.")
    stack_commands.add_parser("status", help="Show local service state.")

    run_parser = commands.add_parser("run", help="Start dependencies and run one experiment.")
    run_parser.add_argument("--config", required=True, help="Host or /bench experiment config path.")
    run_parser.add_argument("--run-id", required=True, help="Unique run identifier.")
    run_parser.add_argument("--predict-only", action="store_true")
    run_parser.add_argument("--plan-only", action="store_true")
    run_parser.add_argument(
        "--observability",
        action="store_true",
        help="Start Prometheus and Grafana with the core services.",
    )
    run_parser.add_argument(
        "--no-stack-up",
        action="store_true",
        help="Use an already running stack.",
    )
    run_parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Set HF_HUB_OFFLINE=0 for this operation.",
    )

    validate_parser = commands.add_parser("validate", help="Validate one experiment config in the image.")
    validate_parser.add_argument("--config", required=True, help="Host or /bench experiment config path.")
    validate_parser.add_argument("--materialize-datasets", action="store_true")
    validate_parser.add_argument("--allow-downloads", action="store_true")

    for name in ("verify", "status", "inspect"):
        command_parser = commands.add_parser(name, help=f"Run dmf-bench {name} in the image.")
        command_parser.add_argument("--run-id", required=True)

    resume_parser = commands.add_parser("resume", help="Resume an existing run in the image.")
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--plan-only", action="store_true")
    resume_parser.add_argument(
        "--no-stack-up",
        action="store_true",
        help="Use an already running stack.",
    )

    publish_parser = commands.add_parser(
        "publish-run",
        help="Package a completed run and optionally publish it as an OCI artifact.",
    )
    publish_parser.add_argument("--run-id", required=True)
    publish_parser.add_argument("--ref", required=True, help="OCI destination reference.")
    publish_parser.add_argument("--subject", help="Producing image digest reference.")
    publish_parser.add_argument(
        "--output-dir",
        default="bundles",
        help="Host directory where the run archive is retained.",
    )
    publish_parser.add_argument("--oras-bin", default="oras")
    publish_parser.add_argument(
        "--dry-run",
        dest="publication_dry_run",
        action="store_true",
        help="Build and verify the archive without pushing it.",
    )

    runtime_parser = commands.add_parser(
        "runtime",
        help="Run an advanced dmf-bench command inside the benchmark image.",
    )
    runtime_parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Arguments passed verbatim to dmf-bench.",
    )
    return parser


def _orchestrator(
    args: argparse.Namespace,
    *,
    project_dir: Path,
    extra_compose_files: Sequence[Path] = (),
    runner: CommandRunner,
) -> DockerComposeOrchestrator:
    compose_files = [project_dir / "deploy" / "compose.yaml"]
    compose_files.extend(
        _resolve_file(project_dir, value) for value in args.compose_override
    )
    compose_files.extend(extra_compose_files)
    for compose_file in compose_files:
        if not compose_file.is_file():
            raise ValueError(f"Compose file does not exist: {compose_file}")

    environment = dict(os.environ)
    if args.image:
        environment["DMF_BENCH_IMAGE"] = args.image
    if getattr(args, "allow_downloads", False):
        environment["HF_HUB_OFFLINE"] = "0"

    return DockerComposeOrchestrator(
        project_dir=project_dir,
        compose_files=_deduplicate(compose_files),
        environment=environment,
        runner=runner,
        dry_run=bool(args.dry_run),
    )


def _services(observability: bool) -> tuple[str, ...]:
    return CORE_SERVICES + (OBSERVABILITY_SERVICES if observability else ())


def main(
    argv: list[str] | None = None,
    *,
    runner: CommandRunner = run_command,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        project_dir = _project_dir(args.project_dir)
        config_path: str | None = None
        volumes: tuple[str, ...] = ()
        automatic_overrides: tuple[Path, ...] = ()
        if hasattr(args, "config"):
            config_path, volumes, automatic_overrides = _runtime_path(
                project_dir,
                args.config,
            )
        orchestrator = _orchestrator(
            args,
            project_dir=project_dir,
            extra_compose_files=automatic_overrides,
            runner=runner,
        )
    except ValueError as exc:
        parser.exit(2, f"dmf-benchctl: error: {exc}\n")

    try:
        if args.command == "config":
            return orchestrator.validate_config()
        if args.command == "image":
            return orchestrator.build_images(IMAGE_SERVICES)
        if args.command == "stack":
            if args.stack_command == "up":
                return orchestrator.stack_up(
                    _services(args.observability),
                    pull=args.pull,
                )
            if args.stack_command == "stop":
                return orchestrator.stack_stop(_services(args.observability))
            if args.stack_command == "down":
                return orchestrator.stack_down()
            return orchestrator.stack_status()
        if args.command == "run":
            if not args.no_stack_up:
                result = orchestrator.stack_up(
                    _services(args.observability),
                    pull=args.pull,
                )
                if result != 0:
                    return result
            runtime_args = ["run", "--config", str(config_path), "--run-id", args.run_id]
            if args.predict_only:
                runtime_args.append("--predict-only")
            if args.plan_only:
                runtime_args.append("--plan-only")
            return orchestrator.run_runtime(runtime_args, volumes=volumes)
        if args.command == "validate":
            runtime_args = ["validate", "--config", str(config_path)]
            if args.materialize_datasets:
                runtime_args.append("--materialize-datasets")
            return orchestrator.run_runtime(runtime_args, volumes=volumes)
        if args.command in {"verify", "status", "inspect"}:
            return orchestrator.run_runtime(
                [args.command, "--run-id", args.run_id]
            )
        if args.command == "resume":
            if not args.no_stack_up:
                result = orchestrator.stack_up(CORE_SERVICES, pull=args.pull)
                if result != 0:
                    return result
            runtime_args = ["resume", "--run-id", args.run_id]
            if args.plan_only:
                runtime_args.append("--plan-only")
            return orchestrator.run_runtime(runtime_args)
        if args.command == "publish-run":
            output_dir = _resolve_file(project_dir, args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            runtime_args = [
                "publish-run-oci",
                "--run-id",
                args.run_id,
                "--ref",
                args.ref,
                "--output-dir",
                "/bench/export",
                "--dry-run",
            ]
            if args.subject:
                runtime_args.extend(("--subject", args.subject))
            result = orchestrator.run_runtime(
                runtime_args,
                volumes=(f"{output_dir}:/bench/export",),
            )
            if result != 0 or args.publication_dry_run:
                return result
            archive = output_dir / f"{args.run_id}.run.tar.gz"
            oras_command = [
                args.oras_bin,
                "push",
                "--artifact-type",
                RUN_ARTIFACT_TYPE,
            ]
            if args.subject:
                oras_command.extend(("--subject", args.subject))
            oras_command.extend(
                (args.ref, f"{archive.resolve()}:{RUN_LAYER_TYPE}")
            )
            return orchestrator.run_external(oras_command)
        runtime_args = list(args.arguments)
        if runtime_args[:1] == ["--"]:
            runtime_args = runtime_args[1:]
        if not runtime_args:
            parser.error("runtime requires a dmf-bench command")
        return orchestrator.run_runtime(runtime_args)
    except FileNotFoundError as exc:
        print(f"dmf-benchctl: error: executable not found: {exc.filename}", file=sys.stderr)
        return 127
    except OSError as exc:
        print(f"dmf-benchctl: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
