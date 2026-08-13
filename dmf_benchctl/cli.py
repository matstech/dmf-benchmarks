"""User-facing control plane for Docker-native benchmark operations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .orchestrators import DockerComposeOrchestrator
from .orchestrators.base import CommandRunner
from .orchestrators.docker_compose import run_command
from .presets import export_preset, pin_operator_config, preset_catalog
from .preparation import (
    DEFAULT_PREPARATION_PATH,
    SMOKE_CONFIG_PATHS,
    OutputRunner,
    build_preparation_state,
    default_batch_id,
    inspect_dataset_registry,
    inspect_image,
    lock_experiment_datasets,
    load_experiment_metadata,
    load_preparation_state,
    require_provider_credentials,
    run_output_command,
    verify_preparation_inputs,
    write_preparation_state,
)
from .resources import resolve_runtime_layout


CORE_SERVICES = ("qdrant", "artifact-api")
OBSERVABILITY_SERVICES = ("prometheus", "grafana")
IMAGE_SERVICES = ("artifact-api", "benchmark")
RUN_ARTIFACT_TYPE = "application/vnd.dmf-bench.run.v1+tar"
RUN_LAYER_TYPE = "application/vnd.dmf-bench.run.layer.v1.tar+gzip"
STATUS_BAR_WIDTH = 30
PHASE_LABELS = {
    "PREFLIGHT": "Preparing runtime",
    "RUNNING": "Processing benchmark data",
    "JUDGING": "Judging generated answers",
    "EVALUATING": "Calculating evaluation metrics",
    "REPORTING": "Writing reports",
    "PUBLISHING": "Publishing artifacts",
    "VERIFYING": "Verifying published artifacts",
    "RESUMING": "Resuming interrupted work",
    "COMPLETED": "Completed",
    "PARTIAL": "Partially completed",
    "INTERRUPTING": "Stopping at a safe boundary",
    "INTERRUPTED": "Interrupted",
}


def _resolve_file(project_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def _deduplicate(paths: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


def _credential_environment(project_dir: Path) -> dict[str, str]:
    """Return host credentials plus non-exported values declared in local .env."""

    environment = dict(os.environ)
    dotenv = project_dir / ".env"
    if not dotenv.is_file():
        return environment
    try:
        lines = dotenv.read_text(encoding="utf-8").splitlines()
    except OSError:
        return environment
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in environment:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        environment[key] = value
    return environment


def _operation_log_path(project_dir: Path, operation_id: str) -> Path:
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in operation_id
    ).strip(".")
    if not safe_name:
        safe_name = "operation"
    return project_dir / ".dmf-bench" / "logs" / f"{safe_name}.log"


def _last_log_line(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return next((line.strip() for line in reversed(lines) if line.strip()), None)


def _parse_json_output(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read runtime output {path}: {exc}") from exc
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        detail = _last_log_line(path) or "runtime returned no JSON status"
        raise ValueError(detail)
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"runtime returned invalid JSON status: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("runtime status must contain a JSON object")
    return payload


def _status_count(payload: object, key: str) -> int:
    if not isinstance(payload, dict):
        return 0
    value = payload.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _operator_status(payload: dict[str, object]) -> dict[str, object]:
    items = payload.get("items")
    records = payload.get("units")
    completed_items = _status_count(items, "committed")
    total_items = _status_count(items, "expected")
    failed_items = _status_count(items, "failed")
    completed_records = _status_count(records, "committed")
    total_records = _status_count(records, "expected")
    failed_records = _status_count(records, "failed")
    percentage = (
        min(100.0, max(0.0, completed_items / total_items * 100.0))
        if total_items
        else 0.0
    )
    state = str(payload.get("state", "UNKNOWN"))
    phase = str(payload.get("phase", state))
    phase_label = None
    if state.startswith("FAILED_"):
        phase_label = f"Failed during {state.removeprefix('FAILED_').lower()}"
    if phase_label is None:
        phase_label = PHASE_LABELS.get(phase)
    if phase_label is None:
        phase_label = phase.replace("_", " ").title()
    current_activity_payload = payload.get("current_activity")
    current_activity: dict[str, object] | None = None
    if isinstance(current_activity_payload, dict):
        activity_completed = _status_count(current_activity_payload, "completed")
        activity_total = _status_count(current_activity_payload, "total")
        activity_percentage = (
            min(100.0, max(0.0, activity_completed / activity_total * 100.0))
            if activity_total
            else 0.0
        )
        input_record_payload = current_activity_payload.get("input_record")
        input_record = (
            {
                "number": _status_count(input_record_payload, "number"),
                "total": _status_count(input_record_payload, "total"),
            }
            if isinstance(input_record_payload, dict)
            else {"number": 0, "total": total_records}
        )
        current_activity = {
            "stage": str(current_activity_payload.get("stage", "unknown")),
            "label": str(current_activity_payload.get("label", "Current activity")),
            "memory_system": str(
                current_activity_payload.get("memory_system", "unknown")
            ),
            "completed": activity_completed,
            "total": activity_total,
            "percentage": round(activity_percentage, 1),
            "item_label": str(current_activity_payload.get("item_label", "items")),
            "input_record": input_record,
            "updated_at": str(current_activity_payload.get("updated_at", "")),
        }
    return {
        "schema_version": 1,
        "run_id": str(payload.get("run_id", "")),
        "state": state,
        "phase": phase,
        "phase_label": phase_label,
        "progress": {
            "percentage": round(percentage, 1),
        },
        "evaluation_items": {
            "completed": completed_items,
            "total": total_items,
            "failed": failed_items,
        },
        "input_records": {
            "completed": completed_records,
            "total": total_records,
            "failed": failed_records,
        },
        "current_activity": current_activity,
        "ready_for_verification": state == "COMPLETED",
    }


def _progress_bar(percentage: float) -> str:
    completed = round(STATUS_BAR_WIDTH * percentage / 100.0)
    completed = min(STATUS_BAR_WIDTH, max(0, completed))
    return f"[{'#' * completed}{'-' * (STATUS_BAR_WIDTH - completed)}]"


def _print_operator_status(status: dict[str, object]) -> None:
    progress = status["progress"]
    evaluation_items = status["evaluation_items"]
    records = status["input_records"]
    current_activity = status.get("current_activity")
    assert isinstance(progress, dict)
    assert isinstance(evaluation_items, dict)
    assert isinstance(records, dict)
    percentage = float(progress["percentage"])
    print(f"Run:              {status['run_id']}")
    print(f"State:            {status['state']}")
    print(f"Current phase:    {status['phase_label']} ({status['phase']})")
    print(f"Overall progress: {_progress_bar(percentage)} {percentage:5.1f}%")
    print(
        "Evaluation items:"
        f" {evaluation_items['completed']} of {evaluation_items['total']} completed"
        f"; {evaluation_items['failed']} failed (questions or scored records)"
    )
    print(
        "Input records:   "
        f"{records['completed']} of {records['total']} processed"
        f"; {records['failed']} failed (source conversations or records)"
    )
    if isinstance(current_activity, dict):
        activity_percentage = float(current_activity["percentage"])
        input_record = current_activity["input_record"]
        assert isinstance(input_record, dict)
        print(
            "Current activity:"
            f" {current_activity['label']} ({current_activity['stage']})"
        )
        print(
            "Partial progress:"
            f" {_progress_bar(activity_percentage)} {activity_percentage:5.1f}%"
            f" — {current_activity['completed']} of {current_activity['total']}"
            f" {current_activity['item_label']}"
        )
        print(
            "Current record:  "
            f" {input_record['number']} of {input_record['total']}"
        )
    if bool(status["ready_for_verification"]):
        print("Next action:      run verify for this RUN_ID")
    else:
        print("Next action:      run status again later")


def _write_local_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _active_job_pid(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
        os.kill(pid, 0)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return pid


def _launch_background_worker(
    *,
    project_dir: Path,
    original_argv: Sequence[str],
    job_id: str,
    run_ids: Sequence[str],
    image_ref: str | None,
) -> int:
    job_path = project_dir / ".dmf-bench" / "jobs" / f"{job_id}.json"
    active_pid = _active_job_pid(job_path)
    if active_pid is not None:
        raise ValueError(f"job {job_id!r} is already running with pid {active_pid}")

    worker_log = _operation_log_path(project_dir, f"{job_id}.worker")
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "dmf_benchctl",
        "--foreground-worker",
        *original_argv,
    ]
    with worker_log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=project_dir,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    relative_worker_log = str(worker_log.relative_to(project_dir))
    relative_job_path = str(job_path.relative_to(project_dir))
    payload: dict[str, object] = {
        "schema_version": 1,
        "state": "STARTED",
        "detached": True,
        "job_id": job_id,
        "pid": process.pid,
        "run_ids": list(run_ids),
        "worker_log": relative_worker_log,
        "image": image_ref,
        "command": ["dmf-benchctl", *original_argv],
    }
    _write_local_json(job_path, payload)
    inspection_prefix = "dmf-benchctl"
    if image_ref:
        inspection_prefix = f"{inspection_prefix} --image {image_ref}"
    print(
        json.dumps(
            {
                **payload,
                "job_file": relative_job_path,
                "status_commands": [
                    f"{inspection_prefix} status --run-id {run_id}"
                    for run_id in run_ids
                ],
                "verify_commands": [
                    f"{inspection_prefix} verify --run-id {run_id}"
                    for run_id in run_ids
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _runtime_path(
    runtime_dir: Path,
    workspace_dir: Path,
    value: str,
) -> tuple[str, tuple[str, ...], tuple[Path, ...]]:
    """Map a host config path to its stable path in the benchmark container."""

    if value == "/bench" or value.startswith("/bench/"):
        return value, (), ()

    host_path = _resolve_file(workspace_dir, value)
    if not host_path.is_file():
        raise ValueError(f"configuration file does not exist: {host_path}")

    mappings = (
        (runtime_dir / "config", Path("/bench/config"), None),
        (
            runtime_dir / "smoke",
            Path("/bench/smoke"),
            runtime_dir / "smoke" / "compose.yaml",
        ),
        (runtime_dir / "datasets", Path("/bench/datasets"), None),
    )
    for host_root, container_root, override in mappings:
        try:
            relative = host_path.relative_to(host_root.resolve())
        except ValueError:
            continue
        overrides = (override.resolve(),) if override is not None else ()
        return str(container_root / relative), (), overrides

    container_path = Path("/bench/operator") / host_path.name
    volume = f"{host_path.parent}:/bench/operator:ro"
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
        help="Source checkout containing runtime assets; intended for development overrides.",
    )
    parser.add_argument(
        "--work-dir",
        help="Writable operator workspace; defaults to the checkout or current directory.",
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
    parser.add_argument(
        "--foreground-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    commands = parser.add_subparsers(dest="command", required=True)
    config_parser = commands.add_parser(
        "config",
        help="Inspect, export, or validate controller configuration.",
    )
    config_commands = config_parser.add_subparsers(dest="config_command")
    config_commands.add_parser("list", help="List built-in experiment presets.")
    config_export = config_commands.add_parser(
        "export",
        help="Export an editable copy of a built-in preset.",
    )
    config_export.add_argument("preset", help="Preset name shown by config list.")
    config_export.add_argument("output_dir", help="Directory receiving experiment.json and settings.")
    config_export.add_argument(
        "--force",
        action="store_true",
        help="Replace generated files if they already exist.",
    )

    prepare_parser = commands.add_parser(
        "prepare",
        help="Prepare and lock a reproducible batch without provider calls.",
    )
    prepare_source = prepare_parser.add_mutually_exclusive_group(required=True)
    prepare_source.add_argument(
        "--config",
        action="append",
        help="Experiment config to prepare; repeat for a batch.",
    )
    prepare_source.add_argument(
        "--suite",
        choices=("smoke",),
        help="Prepare a built-in configuration suite.",
    )
    prepare_parser.add_argument(
        "--state-file",
        default=str(DEFAULT_PREPARATION_PATH),
        help="Local preparation context written atomically.",
    )
    prepare_parser.add_argument("--batch-id", help="Stable batch identifier; UTC time by default.")
    prepare_parser.add_argument("--observability", action="store_true")
    prepare_parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow dataset and model downloads during preparation and later runs.",
    )

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

    run_parser = commands.add_parser(
        "run",
        help="Launch one experiment or a prepared batch as a detached job.",
    )
    run_parser.add_argument("--config", help="Host or /bench experiment config path.")
    run_parser.add_argument("--run-id", help="Unique run identifier.")
    run_parser.add_argument(
        "--prepared",
        nargs="?",
        const=str(DEFAULT_PREPARATION_PATH),
        help="Run every experiment from a prepared context.",
    )
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
    run_parser.add_argument(
        "--follow",
        action="store_true",
        help="Run in the foreground and stream benchmark container logs.",
    )

    validate_parser = commands.add_parser("validate", help="Validate one experiment config in the image.")
    validate_parser.add_argument("--config", required=True, help="Host or /bench experiment config path.")
    validate_parser.add_argument("--materialize-datasets", action="store_true")
    validate_parser.add_argument("--allow-downloads", action="store_true")

    for name in ("verify", "inspect", "evaluate"):
        command_parser = commands.add_parser(name, help=f"Run dmf-bench {name} in the image.")
        command_parser.add_argument("--run-id", required=True)

    status_parser = commands.add_parser(
        "status",
        help="Show operator-friendly run progress.",
    )
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the stable operator status schema as JSON.",
    )

    resume_parser = commands.add_parser("resume", help="Resume an existing run in the image.")
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--plan-only", action="store_true")
    resume_parser.add_argument(
        "--no-stack-up",
        action="store_true",
        help="Use an already running stack.",
    )
    resume_parser.add_argument(
        "--prepared",
        nargs="?",
        const=str(DEFAULT_PREPARATION_PATH),
        help="Use image provenance and policies from a prepared context.",
    )
    resume_parser.add_argument(
        "--follow",
        action="store_true",
        help="Stream benchmark container logs until completion.",
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
    runtime_dir: Path,
    workspace_dir: Path,
    extra_compose_files: Sequence[Path] = (),
    environment_overrides: dict[str, str] | None = None,
    runner: CommandRunner,
) -> DockerComposeOrchestrator:
    compose_files = [runtime_dir / "deploy" / "compose.yaml"]
    compose_files.extend(
        _resolve_file(workspace_dir, value) for value in args.compose_override
    )
    compose_files.extend(extra_compose_files)
    for compose_file in compose_files:
        if not compose_file.is_file():
            raise ValueError(f"Compose file does not exist: {compose_file}")

    environment = _credential_environment(workspace_dir)
    if args.image:
        environment["DMF_BENCH_IMAGE"] = args.image
    if getattr(args, "allow_downloads", False):
        environment["HF_HUB_OFFLINE"] = "0"
    if environment_overrides:
        environment.update(environment_overrides)

    return DockerComposeOrchestrator(
        project_dir=workspace_dir,
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
    output_runner: OutputRunner = run_output_command,
) -> int:
    parser = build_parser()
    original_argv = list(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(original_argv)

    try:
        layout = resolve_runtime_layout(args.project_dir, args.work_dir)
        runtime_dir = layout.runtime_dir
        project_dir = layout.workspace_dir
        project_dir.mkdir(parents=True, exist_ok=True)
        prepared_state: dict[str, object] | None = None
        prepared_path: Path | None = None
        if args.command in {"run", "resume"} and args.prepared:
            if args.command == "run" and (args.config or args.run_id):
                raise ValueError("--prepared cannot be combined with --config or --run-id")
            prepared_path = _resolve_file(project_dir, args.prepared)
            prepared_state = load_preparation_state(prepared_path)
            prepared_project = Path(str(prepared_state["project_dir"])).resolve()
            if prepared_project != project_dir:
                raise ValueError(
                    "prepared context belongs to another project directory: "
                    f"{prepared_project}"
                )
            verify_preparation_inputs(prepared_state)
            prepared_image = str(prepared_state["image"]["resolved_ref"])  # type: ignore[index]
            if args.image and args.image != prepared_image:
                raise ValueError("--image conflicts with the image locked by --prepared")
            args.image = prepared_image
            if bool(prepared_state.get("allow_downloads")):
                args.allow_downloads = True

        config_path: str | None = None
        volumes: tuple[str, ...] = ()
        automatic_overrides: tuple[Path, ...] = ()
        if isinstance(getattr(args, "config", None), str):
            if not args.config.startswith("/bench/"):
                pin_operator_config(
                    _resolve_file(project_dir, args.config),
                    runtime_dir=runtime_dir,
                )
            config_path, volumes, automatic_overrides = _runtime_path(
                runtime_dir,
                project_dir,
                args.config,
            )

        prepared_entries: list[dict[str, object]] = []
        if prepared_state is not None:
            prepared_entries = list(prepared_state["experiments"])  # type: ignore[arg-type]
            override_paths: list[Path] = []
            for entry in prepared_entries:
                _, _, entry_overrides = _runtime_path(
                    runtime_dir,
                    project_dir,
                    str(entry["host_path"]),
                )
                override_paths.extend(entry_overrides)
            automatic_overrides = _deduplicate(override_paths)

        environment_overrides: dict[str, str] = {}
        if prepared_state is not None:
            prepared_image_data = prepared_state["image"]
            assert isinstance(prepared_image_data, dict)
            environment_overrides = {
                "DMF_BENCH_IMAGE_DIGEST": str(prepared_image_data["digest"]),
                "DMF_BENCH_HARNESS_COMMIT": str(prepared_image_data["revision"]),
            }
        orchestrator = _orchestrator(
            args,
            runtime_dir=runtime_dir,
            workspace_dir=project_dir,
            extra_compose_files=automatic_overrides,
            environment_overrides=environment_overrides,
            runner=runner,
        )
    except ValueError as exc:
        parser.exit(2, f"dmf-benchctl: error: {exc}\n")

    try:
        if (
            args.command == "run"
            and not args.follow
            and not args.plan_only
            and not args.dry_run
            and not args.foreground_worker
            and runner is run_command
        ):
            if prepared_state is not None:
                require_provider_credentials(
                    prepared_entries,
                    _credential_environment(project_dir),
                )
                run_ids = [str(entry["run_id"]) for entry in prepared_entries]
                job_id = str(prepared_state["batch_id"])
                image_ref = str(prepared_state["image"]["resolved_ref"])  # type: ignore[index]
            else:
                if not args.config or not args.run_id:
                    raise ValueError("run requires --config and --run-id, or --prepared")
                run_ids = [args.run_id]
                job_id = args.run_id
                image_ref = args.image or os.getenv("DMF_BENCH_IMAGE")
            return _launch_background_worker(
                project_dir=project_dir,
                original_argv=original_argv,
                job_id=job_id,
                run_ids=run_ids,
                image_ref=image_ref,
            )
        if args.command == "config" and args.config_command == "list":
            print(json.dumps({"presets": preset_catalog()}, indent=2, sort_keys=True))
            return 0
        if args.command == "config" and args.config_command == "export":
            experiment_path, framework_path = export_preset(
                runtime_dir=runtime_dir,
                preset_name=args.preset,
                output_dir=_resolve_file(project_dir, args.output_dir),
                force=bool(args.force),
            )
            print(
                json.dumps(
                    {
                        "preset": args.preset,
                        "experiment": str(experiment_path),
                        "framework_config": str(framework_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "config":
            return orchestrator.validate_config()
        if args.command == "prepare":
            if args.dry_run:
                raise ValueError("prepare cannot create a locked context with --dry-run")
            local_environment = _credential_environment(project_dir)
            requested_image = args.image or local_environment.get(
                "DMF_BENCH_IMAGE", ""
            ).strip()
            if not requested_image:
                raise ValueError("prepare requires --image or DMF_BENCH_IMAGE")
            config_values = (
                list(SMOKE_CONFIG_PATHS)
                if args.suite == "smoke"
                else list(args.config or ())
            )
            host_configs = [
                (runtime_dir / value).resolve()
                if args.suite == "smoke"
                else _resolve_file(project_dir, value)
                for value in config_values
            ]
            missing = [str(path) for path in host_configs if not path.is_file()]
            if missing:
                raise ValueError(f"configuration file does not exist: {missing[0]}")
            for host_config in host_configs:
                pin_operator_config(host_config, runtime_dir=runtime_dir)
            experiments = load_experiment_metadata(host_configs)
            require_provider_credentials(experiments, local_environment)

            if orchestrator.run_external(("docker", "version")) != 0:
                return 1
            if orchestrator.run_external(("docker", "compose", "version")) != 0:
                return 1
            if orchestrator.pull_image(requested_image) != 0:
                return 1
            image = inspect_image(
                requested_image,
                project_dir=project_dir,
                environment=orchestrator.environment,
                output_runner=output_runner,
            )
            image_registry = inspect_dataset_registry(
                image["resolved_ref"],
                project_dir=project_dir,
                environment=orchestrator.environment,
                output_runner=output_runner,
            )
            experiments = lock_experiment_datasets(experiments, image_registry)
            prepare_overrides: list[Path] = []
            for host_config in host_configs:
                _, _, config_overrides = _runtime_path(
                    runtime_dir, project_dir, str(host_config)
                )
                prepare_overrides.extend(config_overrides)
            pinned_orchestrator = _orchestrator(
                args,
                runtime_dir=runtime_dir,
                workspace_dir=project_dir,
                extra_compose_files=_deduplicate(prepare_overrides),
                environment_overrides={
                    "DMF_BENCH_IMAGE": image["resolved_ref"],
                    "DMF_BENCH_IMAGE_DIGEST": image["digest"],
                    "DMF_BENCH_HARNESS_COMMIT": image["revision"],
                },
                runner=runner,
            )
            if pinned_orchestrator.validate_config() != 0:
                return 1
            if pinned_orchestrator.stack_up(
                _services(args.observability), pull="missing"
            ) != 0:
                return 1
            for host_config in host_configs:
                runtime_config, config_volumes, _ = _runtime_path(
                    runtime_dir, project_dir, str(host_config)
                )
                result = pinned_orchestrator.run_runtime(
                    ["validate", "--config", runtime_config, "--materialize-datasets"],
                    volumes=config_volumes,
                    quiet=True,
                    output_path=_operation_log_path(
                        project_dir,
                        f"prepare-{host_config.stem}",
                    ),
                )
                if result != 0:
                    return result
            state = build_preparation_state(
                project_dir=project_dir,
                image=image,
                experiments=experiments,
                batch_id=args.batch_id or default_batch_id(),
                observability=bool(args.observability),
                allow_downloads=bool(args.allow_downloads),
            )
            state_path = _resolve_file(project_dir, args.state_file)
            write_preparation_state(state_path, state)
            print(json.dumps({"prepared": str(state_path), **state}, indent=2, sort_keys=True))
            return 0
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
            if prepared_state is not None:
                require_provider_credentials(
                    prepared_entries,
                    _credential_environment(project_dir),
                )
                if not args.no_stack_up:
                    result = orchestrator.stack_up(
                        _services(
                            bool(args.observability)
                            or bool(prepared_state.get("observability"))
                        ),
                        pull=args.pull,
                    )
                    if result != 0:
                        return result
                for entry in prepared_entries:
                    runtime_config, entry_volumes, _ = _runtime_path(
                        runtime_dir,
                        project_dir,
                        str(entry["host_path"]),
                    )
                    run_id = str(entry["run_id"])
                    status_result = orchestrator.run_runtime(
                        ["status", "--run-id", run_id],
                        quiet=True,
                        output_path=_operation_log_path(
                            project_dir,
                            f"{run_id}.status",
                        ),
                    )
                    if status_result in {0, 4, 5}:
                        runtime_args = ["resume", "--run-id", run_id]
                        operation = "resuming"
                        log_path = _operation_log_path(
                            project_dir,
                            f"{run_id}.resume",
                        )
                    elif status_result == 3:
                        runtime_args = [
                            "run",
                            "--config",
                            runtime_config,
                            "--run-id",
                            run_id,
                        ]
                        operation = "running"
                        log_path = _operation_log_path(project_dir, run_id)
                        if args.predict_only:
                            runtime_args.append("--predict-only")
                    else:
                        print(
                            f"dmf-benchctl: cannot inspect {run_id}; "
                            f"status exited with code {status_result}",
                            file=sys.stderr,
                        )
                        return status_result
                    if args.plan_only:
                        runtime_args.append("--plan-only")
                    print(
                        f"dmf-benchctl: {operation} {run_id} "
                        f"(log: {log_path.relative_to(project_dir)}) ...",
                        flush=True,
                    )
                    result = orchestrator.run_runtime(
                        runtime_args,
                        volumes=entry_volumes,
                        quiet=not bool(args.follow),
                        output_path=log_path,
                    )
                    if result != 0:
                        print(
                            f"dmf-benchctl: {run_id} failed with exit code {result}",
                            file=sys.stderr,
                        )
                        detail = _last_log_line(log_path)
                        if detail:
                            print(f"dmf-benchctl: {detail}", file=sys.stderr)
                        return result
                    print(f"dmf-benchctl: completed {run_id}", flush=True)
                return 0
            if not args.config or not args.run_id:
                raise ValueError("run requires --config and --run-id, or --prepared")
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
            log_path = _operation_log_path(project_dir, f"{args.run_id}.resume")
            print(
                f"dmf-benchctl: running {args.run_id} "
                f"(log: {log_path.relative_to(project_dir)}) ...",
                flush=True,
            )
            result = orchestrator.run_runtime(
                runtime_args,
                volumes=volumes,
                quiet=not bool(args.follow),
                output_path=log_path,
            )
            if result != 0:
                print(
                    f"dmf-benchctl: {args.run_id} failed with exit code {result}",
                    file=sys.stderr,
                )
                detail = _last_log_line(log_path)
                if detail:
                    print(f"dmf-benchctl: {detail}", file=sys.stderr)
                return result
            print(f"dmf-benchctl: completed {args.run_id}", flush=True)
            return 0
        if args.command == "validate":
            runtime_args = ["validate", "--config", str(config_path)]
            if args.materialize_datasets:
                runtime_args.append("--materialize-datasets")
            return orchestrator.run_runtime(runtime_args, volumes=volumes)
        if args.command in {"verify", "inspect", "evaluate"}:
            return orchestrator.run_runtime(
                [args.command, "--run-id", args.run_id]
            )
        if args.command == "status":
            if runner is not run_command:
                return orchestrator.run_runtime(["status", "--run-id", args.run_id])
            status_output = _operation_log_path(
                project_dir,
                f"{args.run_id}.status-command",
            )
            result = orchestrator.run_runtime(
                ["status", "--run-id", args.run_id],
                quiet=True,
                output_path=status_output,
            )
            try:
                status = _operator_status(_parse_json_output(status_output))
            except ValueError as exc:
                print(f"dmf-benchctl: status error: {exc}", file=sys.stderr)
                return result or 3
            if args.json:
                print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
            else:
                _print_operator_status(status)
            return result
        if args.command == "resume":
            if prepared_state is not None:
                require_provider_credentials(
                    prepared_entries,
                    _credential_environment(project_dir),
                )
                prepared_run_ids = {
                    str(entry["run_id"]) for entry in prepared_entries
                }
                if args.run_id not in prepared_run_ids:
                    raise ValueError(
                        f"run_id is not part of the prepared context: {args.run_id}"
                    )
            if not args.no_stack_up:
                result = orchestrator.stack_up(CORE_SERVICES, pull=args.pull)
                if result != 0:
                    return result
            runtime_args = ["resume", "--run-id", args.run_id]
            if args.plan_only:
                runtime_args.append("--plan-only")
            log_path = _operation_log_path(project_dir, args.run_id)
            print(
                f"dmf-benchctl: resuming {args.run_id} "
                f"(log: {log_path.relative_to(project_dir)}) ...",
                flush=True,
            )
            result = orchestrator.run_runtime(
                runtime_args,
                quiet=not bool(args.follow),
                output_path=log_path,
            )
            if result != 0:
                print(
                    f"dmf-benchctl: {args.run_id} failed with exit code {result}",
                    file=sys.stderr,
                )
                detail = _last_log_line(log_path)
                if detail:
                    print(f"dmf-benchctl: {detail}", file=sys.stderr)
                return result
            print(f"dmf-benchctl: completed {args.run_id}", flush=True)
            return 0
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
    except ValueError as exc:
        print(f"dmf-benchctl: error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"dmf-benchctl: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
