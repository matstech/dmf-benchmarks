"""Command line interface for the DMF benchmark harness."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from .config import resolve_config
from .registry import BENCHMARKS, FRAMEWORKS, PROTOCOLS, supported_combinations


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

    future_commands = {
        "materialize-dataset": "Materialize a pinned dataset (not implemented in Phase 2).",
        "run": "Run a complete benchmark lifecycle (not implemented in Phase 2).",
        "resume": "Resume an existing run (not implemented in Phase 2).",
        "status": "Inspect run status (not implemented in Phase 2).",
        "health": "Check runner health (not implemented in Phase 2).",
        "inspect": "Inspect manifest and artifacts (not implemented in Phase 2).",
        "evaluate": "Run evaluation maintenance flow (not implemented in Phase 2).",
        "report": "Run report maintenance flow (not implemented in Phase 2).",
        "publish": "Run publish maintenance flow (not implemented in Phase 2).",
    }
    for name, help_text in future_commands.items():
        subparsers.add_parser(name, help=help_text)

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

    parser.exit(
        2,
        f"dmf-bench: error: command {args.command!r} is not implemented in Phase 2\n",
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
