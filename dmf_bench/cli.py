"""Command line interface for the DMF benchmark harness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, TextIO

from .benchmarks.locomo.adapter import LoCoMoAdapter
from .benchmarks.longmemeval.adapter import LongMemEvalAdapter
from .artifacts import LocalArtifactStore
from .atomic_io import read_json
from .config import ResolvedConfig, resolve_config, validate_config
from .datasets import load_dataset_registry, materialize_dataset_record, registry_as_dict
from .derived_evaluation import derive_locomo_evaluation
from .execution import CancellationController, RunInterrupted
from .logging_config import JsonEventLogger, redact
from .metrics import BenchmarkMetrics, MetricsServer, start_metrics_endpoint, stop_metrics_endpoint
from .oci import OciPublishError, publish_run_oci
from .registry import BENCHMARKS, FRAMEWORKS, supported_combinations
from .runtime import RuntimeApplication, assemble_application
from .secrets import load_runtime_secret_files
from .state import (
    StateError,
    load_run_status,
    plan_resume,
    validate_run_state_version,
)


EXIT_SUCCESS = 0
EXIT_CONFIG = 2
EXIT_RUNTIME = 3
EXIT_SCIENTIFIC_FAILURE = 4
EXIT_INCOMPLETE = 5

ApplicationBuilder = Callable[..., RuntimeApplication]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmf-bench",
        description="DMF benchmark harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List supported benchmarks and frameworks.")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an experiment config and print the resolved non-secret config.",
    )
    validate_parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    validate_parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), help="Override benchmark.")
    validate_parser.add_argument("--framework", choices=sorted(FRAMEWORKS), help="Override framework.")
    validate_parser.add_argument(
        "--materialize-datasets",
        action="store_true",
        help="Materialize source-based dataset entries before validating.",
    )
    validate_parser.add_argument(
        "--dataset-registry",
        help="Optional dataset registry JSON. Defaults to the built-in registry.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a complete benchmark lifecycle.",
    )
    run_parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    run_parser.add_argument("--run-id", help="Override run id.")
    run_parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Run only prediction; this is a PARTIAL run.",
    )
    run_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the lifecycle plan without mutating state.",
    )
    run_parser.add_argument(
        "--dataset-registry",
        help="Optional dataset registry JSON. Defaults to the built-in registry.",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a committed local artifact run without mutating it.",
    )
    verify_parser.add_argument("--run-id", required=True, help="Run id to verify.")
    verify_parser.add_argument(
        "--runs-dir",
        default=os.getenv("DMF_BENCH_RUNS_DIR", "/bench/runs"),
        help="Shared volume directory containing run directories.",
    )

    oci_parser = subparsers.add_parser(
        "publish-run-oci",
        help="Package a committed run directory and publish it as a GHCR/OCI artifact.",
    )
    oci_parser.add_argument("--run-id", required=True, help="Committed run id to publish.")
    oci_parser.add_argument(
        "--runs-dir",
        default=os.getenv("DMF_BENCH_RUNS_DIR", "/bench/runs"),
        help="Directory containing run directories.",
    )
    oci_parser.add_argument("--ref", required=True, help="OCI destination ref, e.g. ghcr.io/org/runs:RUN_ID.")
    oci_parser.add_argument(
        "--subject",
        help="Optional OCI subject image digest, e.g. ghcr.io/org/dmf-benchmarks@sha256:...",
    )
    oci_parser.add_argument(
        "--output-dir",
        help="Optional directory where the generated tar.gz bundle is kept.",
    )
    oci_parser.add_argument(
        "--oras-bin",
        default="oras",
        help="ORAS executable used for push.",
    )
    oci_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and verify the bundle without pushing to the registry.",
    )

    dataset_parser = subparsers.add_parser(
        "materialize-dataset",
        help="Explicitly materialize a pinned dataset from the versioned registry.",
    )
    dataset_parser.add_argument("--dataset-id", required=True, help="Dataset registry id.")
    dataset_parser.add_argument("--output-dir", required=True, help="Directory where the dataset file is published.")
    dataset_parser.add_argument(
        "--registry",
        help="Optional dataset registry JSON. Defaults to the built-in registry.",
    )
    dataset_parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Allow unpinned registry entries and record the observed source SHA-256 in manifest.json.",
    )
    dataset_parser.add_argument("--sample-fraction", type=float, help="Optional sample fraction in (0, 1].")
    dataset_parser.add_argument("--sample-seed", type=int, default=42, help="Sampling seed.")
    dataset_parser.add_argument(
        "--sample-rounding",
        choices=("ceil", "floor", "round"),
        default="ceil",
        help="How fractional sample counts are rounded.",
    )
    dataset_parser.add_argument("--sample-unit", help="Sampling unit, e.g. qa or record.")
    dataset_parser.add_argument("--sample-stratify-by", help="Optional sampling stratum field.")

    subparsers.add_parser("list-datasets", help="List dataset registry entries.")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Derive corrected deterministic reports from a committed run.",
    )
    evaluate_parser.add_argument("--run-id", required=True, help="Committed source run id.")
    evaluate_parser.add_argument(
        "--runs-dir",
        default=os.getenv("DMF_BENCH_RUNS_DIR", "/bench/runs"),
        help="Directory containing committed runs and derived reports.",
    )

    future_commands = {
        "resume": "Resume an existing run from its persisted resolved config.",
        "status": "Inspect persisted run status without mutating it.",
        "health": "Check the local runner volume without mutating it.",
        "inspect": "Inspect manifest and local artifact state.",
        "report": "Run report maintenance flow (not implemented yet).",
        "publish": "Run publish maintenance flow (not implemented yet).",
    }
    for name, help_text in future_commands.items():
        command_parser = subparsers.add_parser(name, help=help_text)
        if name in {"inspect", "resume", "status"}:
            command_parser.add_argument("--run-id", required=True, help="Run id to inspect.")
            command_parser.add_argument(
                "--runs-dir",
                default=os.getenv("DMF_BENCH_RUNS_DIR", "/bench/runs"),
                help="Directory containing run directories.",
            )
        if name == "resume":
            command_parser.add_argument(
                "--plan-only",
                action="store_true",
                help="Print the resume plan without mutating state.",
            )
        if name == "health":
            command_parser.add_argument(
                "--runs-dir",
                default=os.getenv("DMF_BENCH_RUNS_DIR", "/bench/runs"),
                help="Shared benchmark run volume to check.",
            )

    return parser


def main(
    argv: list[str] | None = None,
    *,
    application_builder: ApplicationBuilder | None = None,
) -> int:
    try:
        load_runtime_secret_files()
    except ValueError as exc:
        raise SystemExit(f"dmf-bench: error: {exc}") from exc
    parser = build_parser()
    args = parser.parse_args(argv)

    internal_application = os.getenv("DMF_BENCH_INTERNAL_APPLICATION", "")
    if application_builder is None and internal_application:
        if internal_application != "container-fixture":
            parser.exit(
                2,
                "dmf-bench: error: unsupported internal application profile\n",
            )
        from .container_fixture import fixture_application_builder

        application_builder = fixture_application_builder

    if args.command == "list":
        print_list(sys.stdout)
        return 0

    if args.command == "validate":
        try:
            resolved = resolve_config(
                args.config,
                benchmark=args.benchmark,
                framework=args.framework,
                materialize_datasets=bool(args.materialize_datasets),
                dataset_registry_path=args.dataset_registry,
            )
        except ValueError as exc:
            parser.exit(2, f"dmf-bench validate: error: {exc}\n")
        print(
            json.dumps(resolved.redacted(), ensure_ascii=False, indent=2, sort_keys=True),
            file=sys.stdout,
        )
        return 0

    if args.command == "inspect":
        try:
            store = LocalArtifactStore(args.runs_dir)
            result = store.inspect(args.run_id)
            if result["completed"] or result["final_manifest_path"] is not None:
                result = store.inspect_committed(args.run_id)
        except (StateError, ValueError) as exc:
            parser.exit(3, f"dmf-bench inspect: error: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
        return 0

    if args.command == "verify":
        try:
            result = LocalArtifactStore(args.runs_dir).verify_committed(args.run_id)
        except (StateError, ValueError) as exc:
            parser.exit(3, f"dmf-bench verify: error: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
        return 0

    if args.command == "publish-run-oci":
        try:
            result = publish_run_oci(
                run_id=args.run_id,
                runs_dir=args.runs_dir,
                ref=args.ref,
                subject=args.subject,
                output_dir=args.output_dir,
                oras_bin=args.oras_bin,
                dry_run=bool(args.dry_run),
            )
        except (OciPublishError, OSError, StateError, ValueError) as exc:
            parser.exit(3, f"dmf-bench publish-run-oci: error: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
        return 0

    if args.command == "list-datasets":
        result = registry_as_dict(load_dataset_registry())
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
        return 0

    if args.command == "materialize-dataset":
        try:
            sampling = None
            if args.sample_fraction is not None:
                sampling = {
                    "fraction": args.sample_fraction,
                    "seed": args.sample_seed,
                    "rounding": args.sample_rounding,
                }
                if args.sample_unit:
                    sampling["unit"] = args.sample_unit
                if args.sample_stratify_by:
                    sampling["stratify_by"] = args.sample_stratify_by
            materialized = materialize_dataset_record(
                dataset_id=args.dataset_id,
                output_dir=args.output_dir,
                registry_path=args.registry,
                sampling=sampling,
                allow_unpinned=bool(args.allow_unpinned),
            )
            result = {
                "schema_version": 1,
                "dataset": materialized.dataset_config,
                "manifest_path": str(materialized.manifest_path.resolve()),
            }
        except (OSError, ValueError) as exc:
            parser.exit(3, f"dmf-bench materialize-dataset: error: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
        return 0

    if args.command == "evaluate":
        try:
            result = derive_locomo_evaluation(
                run_id=args.run_id,
                runs_dir=args.runs_dir,
            )
        except (OSError, StateError, ValueError) as exc:
            parser.exit(3, f"dmf-bench evaluate: error: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
        return 0

    if args.command == "run":
        try:
            resolved = resolve_config(
                args.config,
                materialize_datasets=not bool(args.plan_only),
                dataset_registry_path=args.dataset_registry,
            )
            if resolved.data["benchmark"] == "longmemeval":
                units = LongMemEvalAdapter().enumerate_units(resolved.data)
                expected_question_ids = [unit.unit_id for unit in units]
            elif resolved.data["benchmark"] == "locomo":
                adapter = LoCoMoAdapter()
                units = adapter.enumerate_units(resolved.data)
                expected_question_ids = list(adapter.expected_question_ids(resolved.data))
            else:
                raise ValueError(
                    "run --predict-only currently supports benchmark='longmemeval' or 'locomo'."
                )
        except (FileNotFoundError, ValueError) as exc:
            parser.exit(2, f"dmf-bench run: error: {exc}\n")
        if args.plan_only:
            result = {
                "run_id": args.run_id or resolved.data.get("experiment_id"),
                "state": "PLANNED",
                "benchmark": resolved.data["benchmark"],
                "framework": resolved.data["framework"],
                "mode": "predict-only" if args.predict_only else "full",
                "expected_final_state": "PARTIAL" if args.predict_only else "COMPLETED",
                "resume": False,
                "expected_unit_ids": [unit.unit_id for unit in units],
                "expected_question_ids": expected_question_ids,
            }
            if args.predict_only:
                result["predict_only_state"] = "PARTIAL"
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
            return 0
        return _execute(
            resolved,
            run_id=args.run_id,
            resume=False,
            predict_only=bool(args.predict_only),
            application_builder=application_builder,
        )

    if args.command == "resume":
        try:
            run_dir = LocalArtifactStore(args.runs_dir).run_dir(args.run_id)
            if args.plan_only:
                result = plan_resume(run_dir).to_dict()
                print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
                return EXIT_SUCCESS
            resolved = _load_resume_config(run_dir, runs_dir=Path(args.runs_dir))
        except (OSError, StateError, ValueError) as exc:
            return _error(f"dmf-bench resume: error: {exc}", EXIT_RUNTIME)
        return _execute(
            resolved,
            run_id=args.run_id,
            resume=True,
            predict_only=False,
            application_builder=application_builder,
        )

    if args.command == "status":
        return _status(args.runs_dir, args.run_id)

    if args.command == "health":
        return _health(args.runs_dir)

    parser.exit(
        2,
        f"dmf-bench: error: command {args.command!r} is not implemented yet\n",
    )
    return 2


def _execute(
    resolved: ResolvedConfig,
    *,
    run_id: str | None,
    resume: bool,
    predict_only: bool,
    application_builder: ApplicationBuilder | None,
) -> int:
    config = resolved.data
    runtime = config.get("runtime") or {}
    resolved_run_id = str(run_id or config.get("experiment_id") or "")
    labels = {
        "run_id": resolved_run_id or None,
        "benchmark": config.get("benchmark"),
        "framework": config.get("framework"),
    }
    events = JsonEventLogger(level=str(runtime.get("log_level", "INFO")))
    metrics = BenchmarkMetrics()
    metrics_server: MetricsServer | None = None
    active_run_dir: Path | None = None
    active_attempt_id: str | None = None
    controller = CancellationController()

    def on_run_ready(run_dir: Path, attempt: Any) -> None:
        nonlocal active_run_dir, active_attempt_id
        active_run_dir = run_dir
        active_attempt_id = str(getattr(attempt, "attempt_id", "")) or None
        resolved.persist_redacted(run_dir)
        _copy_dataset_materialization_manifest(config, run_dir)
        events.bind_file(run_dir / "logs" / "events.jsonl")
        events.event(
            "run.started",
            "Benchmark lifecycle started.",
            attempt_id=active_attempt_id,
            phase="RUNNING",
            **labels,
        )

    try:
        events.event(
            "run.preflight.started",
            "Benchmark preflight started.",
            phase="PREFLIGHT",
            **labels,
        )
        port = _metrics_port(runtime)
        metrics_server = start_metrics_endpoint(metrics=metrics, port=port)
        builder = application_builder or assemble_application
        application = builder(config, metrics=metrics, events=events)
        if resume:
            active_run_dir = application.artifact_store.run_dir(resolved_run_id)
            events.bind_file(active_run_dir / "logs" / "events.jsonl")
            metrics.restore_run_status(active_run_dir)
        events.event(
            "run.preflight.completed",
            "Benchmark preflight completed.",
            phase="PREFLIGHT",
            outcome="success",
            **labels,
        )
        with controller.installed():
            result = application.run(
                config,
                run_id=run_id,
                resume=resume,
                predict_only=predict_only,
                cancel_check=controller.check,
                on_run_ready=on_run_ready,
            )
        if active_run_dir is not None:
            metrics.restore_run_status(active_run_dir)
            metrics.write_snapshot(active_run_dir)
        events.event(
            "run.completed",
            "Benchmark lifecycle finished.",
            attempt_id=active_attempt_id,
            phase=str(getattr(result, "state", "COMPLETED")),
            outcome=str(getattr(result, "state", "COMPLETED")).lower(),
            **labels,
        )
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stdout)
        return EXIT_SUCCESS
    except RunInterrupted as exc:
        _snapshot_if_available(metrics, active_run_dir)
        events.event(
            "run.interrupt.requested",
            "Benchmark interruption requested.",
            level="WARNING",
            attempt_id=active_attempt_id,
            phase="INTERRUPTING",
            outcome="interrupted",
            error_type=type(exc).__name__,
            **labels,
        )
        events.event(
            "run.interrupted",
            "Benchmark stopped at a safe lifecycle boundary.",
            level="WARNING",
            attempt_id=active_attempt_id,
            phase="INTERRUPTED",
            outcome="interrupted",
            error_type=type(exc).__name__,
            **labels,
        )
        return exc.exit_code
    except Exception as exc:
        _snapshot_if_available(metrics, active_run_dir)
        error_detail = str(redact(str(exc))).strip() or "No error detail available."
        events.event(
            "run.failed",
            f"Benchmark lifecycle failed: {error_detail}",
            level="ERROR",
            attempt_id=active_attempt_id,
            phase=_persisted_phase(active_run_dir, default="PREFLIGHT"),
            outcome="failed",
            error_type=type(exc).__name__,
            **labels,
        )
        code = (
            EXIT_SCIENTIFIC_FAILURE
            if _persisted_state(active_run_dir).startswith("FAILED_")
            else EXIT_RUNTIME
        )
        print(
            f"dmf-bench: error: {type(exc).__name__}: {error_detail}",
            file=sys.stderr,
        )
        return code
    finally:
        if metrics_server is not None:
            stop_metrics_endpoint(metrics_server)
        events.close()


def _copy_dataset_materialization_manifest(config: dict[str, Any], run_dir: Path) -> Path | None:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return None
    manifest = dataset.get("materialization_manifest")
    if not isinstance(manifest, str) or not manifest.strip():
        return None
    source = Path(manifest)
    if not source.is_file():
        raise ValueError(f"Dataset materialization manifest not found: {source}")
    target = run_dir / "datasets" / "materialization-manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _load_resume_config(run_dir: Path, *, runs_dir: Path) -> ResolvedConfig:
    validate_run_state_version(run_dir)
    config_path = run_dir / "resolved-config.json"
    payload = read_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Persisted resolved config must be a JSON object: {config_path}")
    data = json.loads(json.dumps(payload))
    data.pop("source_path", None)
    runtime = data.get("runtime")
    artifact_store = data.get("artifact_store")
    if not isinstance(runtime, dict) or not isinstance(artifact_store, dict):
        raise ValueError("Persisted resolved config is missing runtime or artifact_store.")
    runtime.pop("environment", None)
    authoritative_runs_dir = runs_dir.resolve()
    runtime["runs_dir"] = str(authoritative_runs_dir)
    artifact_store["uri"] = str(authoritative_runs_dir)
    validate_config(data, source_path=config_path.resolve())
    data["source_path"] = str(config_path.resolve())
    return ResolvedConfig(source_path=config_path.resolve(), data=data)


def _status(runs_dir: str | Path, run_id: str) -> int:
    store = LocalArtifactStore(runs_dir)
    run_dir = store.run_dir(run_id)
    try:
        validate_run_state_version(run_dir)
        payload = load_run_status(run_dir)
        assert payload is not None
        state = str(payload.get("state", ""))
        if state == "COMPLETED":
            store.verify_committed(run_id)
            payload = {**payload, "verified": True}
    except (OSError, StateError, ValueError) as exc:
        return _error(f"dmf-bench status: error: {exc}", EXIT_RUNTIME)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
    if state.startswith("FAILED_"):
        return EXIT_SCIENTIFIC_FAILURE
    if state in {"PARTIAL", "INTERRUPTING", "INTERRUPTED"}:
        return EXIT_INCOMPLETE
    return EXIT_SUCCESS


def _health(runs_dir: str | Path) -> int:
    path = Path(runs_dir)
    readable = path.is_dir() and os.access(path, os.R_OK | os.X_OK)
    writable = path.is_dir() and os.access(path, os.W_OK | os.X_OK)
    healthy = readable and writable
    payload = {
        "schema_version": 1,
        "status": "healthy" if healthy else "unhealthy",
        "runs_dir": str(path),
        "read_only_check": True,
        "readable": readable,
        "writable": writable,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stdout)
    return EXIT_SUCCESS if healthy else EXIT_RUNTIME


def _metrics_port(runtime: dict[str, Any]) -> int:
    raw = os.getenv("DMF_BENCH_METRICS_PORT", str(runtime.get("metrics_port", 9464)))
    try:
        port = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("DMF_BENCH_METRICS_PORT must be an integer.") from exc
    if not 0 <= port <= 65535:
        raise ValueError("DMF_BENCH_METRICS_PORT must be between 0 and 65535.")
    return port


def _snapshot_if_available(metrics: BenchmarkMetrics, run_dir: Path | None) -> None:
    if run_dir is None or not (run_dir / "run-status.json").is_file():
        return
    try:
        metrics.restore_run_status(run_dir)
        metrics.write_snapshot(run_dir)
    except (OSError, StateError, ValueError):
        return


def _persisted_state(run_dir: Path | None) -> str:
    if run_dir is None:
        return ""
    try:
        payload = read_json(run_dir / "run-status.json")
    except (OSError, ValueError):
        return ""
    return str(payload.get("state", "")) if isinstance(payload, dict) else ""


def _persisted_phase(run_dir: Path | None, *, default: str) -> str:
    if run_dir is None:
        return default
    try:
        payload = read_json(run_dir / "run-status.json")
    except (OSError, ValueError):
        return default
    return str(payload.get("phase", default)) if isinstance(payload, dict) else default


def _error(message: str, code: int) -> int:
    print(message, file=sys.stderr)
    return code


def print_list(stream: TextIO) -> None:
    print("benchmarks:", file=stream)
    for benchmark, info in sorted(BENCHMARKS.items()):
        print(f"  {benchmark}: unit={info.unit_type}", file=stream)

    print("frameworks:", file=stream)
    for framework, info in sorted(FRAMEWORKS.items()):
        print(f"  {framework}: storage_backend={info.storage_backend}", file=stream)

    print("supported combinations:", file=stream)
    for benchmark, framework in supported_combinations():
        print(f"  {benchmark}/{framework}", file=stream)
