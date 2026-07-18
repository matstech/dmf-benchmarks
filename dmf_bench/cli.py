"""Command line interface for the DMF benchmark harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import TextIO

from .adapters.locomo import LoCoMoAdapter
from .adapters.longmemeval import LongMemEvalAdapter
from .artifacts import LocalArtifactStore
from .config import resolve_config
from .registry import BENCHMARKS, FRAMEWORKS, PROTOCOLS, supported_combinations
from .state import StateError, plan_resume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmf-bench",
        description="DMF benchmark harness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List supported benchmarks, frameworks, and protocols.")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an experiment config and print the resolved non-secret config.",
    )
    validate_parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    validate_parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), help="Override benchmark.")
    validate_parser.add_argument("--framework", choices=sorted(FRAMEWORKS), help="Override framework.")
    validate_parser.add_argument("--protocol", choices=PROTOCOLS, help="Override protocol.")

    run_parser = subparsers.add_parser(
        "run",
        help="Run a benchmark lifecycle; currently exposes predict-only planning.",
    )
    run_parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    run_parser.add_argument("--run-id", help="Override run id.")
    run_parser.add_argument("--resume", action="store_true", help="Resume an existing run.")
    run_parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Run only prediction; this is a PARTIAL run.",
    )
    run_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the prediction plan without mutating state.",
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

    future_commands = {
        "materialize-dataset": "Materialize a pinned dataset (not implemented yet).",
        "resume": "Plan resume for an existing run; currently supports --plan-only.",
        "status": "Inspect run status (not implemented yet).",
        "health": "Check runner health (not implemented yet).",
        "inspect": "Inspect manifest and local artifact state.",
        "evaluate": "Run evaluation maintenance flow (not implemented yet).",
        "report": "Run report maintenance flow (not implemented yet).",
        "publish": "Run publish maintenance flow (not implemented yet).",
    }
    for name, help_text in future_commands.items():
        command_parser = subparsers.add_parser(name, help=help_text)
        if name in {"inspect", "resume"}:
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        print_list(sys.stdout)
        return 0

    if args.command == "validate":
        try:
            resolved = resolve_config(
                args.config,
                benchmark=args.benchmark,
                framework=args.framework,
                protocol=args.protocol,
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
            result = LocalArtifactStore(args.runs_dir).inspect_committed(args.run_id)
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

    if args.command == "run":
        if not args.predict_only:
            parser.exit(2, "dmf-bench run: error: this phase requires --predict-only\n")
        try:
            resolved = resolve_config(args.config)
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
                "protocol": resolved.data["protocol"],
                "predict_only_state": "PARTIAL",
                "resume": bool(args.resume),
                "expected_unit_ids": [unit.unit_id for unit in units],
                "expected_question_ids": expected_question_ids,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
            return 0
        parser.exit(
            2,
            "dmf-bench run: error: CLI execution requires injected "
            "framework and answerer adapters; use --plan-only here\n",
        )

    if args.command == "resume":
        if not args.plan_only:
            parser.exit(
                2,
                "dmf-bench resume: error: resume currently supports only --plan-only\n",
            )
        try:
            run_dir = LocalArtifactStore(args.runs_dir).run_dir(args.run_id)
            result = plan_resume(run_dir).to_dict()
        except (StateError, ValueError) as exc:
            parser.exit(3, f"dmf-bench resume: error: {exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stdout)
        return 0

    parser.exit(
        2,
        f"dmf-bench: error: command {args.command!r} is not implemented yet\n",
    )
    return 2


def print_list(stream: TextIO) -> None:
    print("benchmarks:", file=stream)
    for benchmark, info in sorted(BENCHMARKS.items()):
        print(f"  {benchmark}: unit={info.unit_type}", file=stream)

    print("frameworks:", file=stream)
    for framework, info in sorted(FRAMEWORKS.items()):
        print(f"  {framework}: storage_backend={info.storage_backend}", file=stream)

    print("protocols:", file=stream)
    for protocol in PROTOCOLS:
        print(f"  {protocol}", file=stream)

    print("supported combinations:", file=stream)
    for benchmark, framework, protocol in supported_combinations():
        print(f"  {benchmark}/{framework}/{protocol}", file=stream)
