"""Deterministic 2-benchmark x 2-framework certification harness.

This module is reachable only through an explicit ``python -m`` invocation in
the Phase 21 container gate. Production experiment configs continue to use the
normal ``dmf-bench`` entrypoint and cannot select fixture providers.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from dmf_bench.cli import main as cli_main
from dmf_bench.container_fixture import fixture_application_builder
from dmf_bench.contracts import REPORT_SCHEMA_VERSION, sha256_file
from dmf_bench.registry import supported_combinations


LOCOMO_DATASET = Path("/bench/fixtures/locomo-mini.json")
LONGMEMEVAL_DATASET = Path("/bench/fixtures/longmemeval-mini.json")
CONFIG_DIR = Path("/bench/config")


def build_matrix_configs(
    base_config: dict[str, Any],
    *,
    run_prefix: str,
    config_dir: Path = CONFIG_DIR,
    locomo_dataset: Path = LOCOMO_DATASET,
    longmemeval_dataset: Path = LONGMEMEVAL_DATASET,
) -> list[dict[str, Any]]:
    """Resolve all supported combinations against deterministic fixtures."""
    configs: list[dict[str, Any]] = []
    for benchmark, framework in supported_combinations():
        config = deepcopy(base_config)
        run_id = f"{run_prefix}-{benchmark}-{framework}"
        framework_suffix = "toml" if framework == "dmf" else "yaml"
        framework_path = (
            config_dir / f"{benchmark}_{framework}_qdrant_settings.{framework_suffix}"
        )
        dataset_path = (
            locomo_dataset if benchmark == "locomo" else longmemeval_dataset
        )

        config.update(
            {
                "experiment_id": run_id,
                "benchmark": benchmark,
                "framework": framework,
            }
        )
        config["framework_config"] = {
            "path": str(framework_path),
            "sha256": sha256_file(framework_path),
            "format": framework_suffix,
            "profile": "docker-qdrant-fixture-v1",
        }
        config["dataset"] = {
            "name": benchmark,
            "source": "fixture",
            "revision": "mini",
            "sha256": sha256_file(dataset_path),
            "path": str(dataset_path),
        }
        config["selection"] = (
            {
                "ordered_item_ids": ["conversation-0001"],
                "filters": {"categories": [1, 2]},
                "seed": 7,
            }
            if benchmark == "locomo"
            else {
                "ordered_item_ids": ["lme-001", "lme-002"],
                "filters": {},
                "seed": 7,
            }
        )
        config["evaluation"] = {
            "required": ["primary_judge_score", "rigorous_report"],
            "optional": ["ablation_report"],
        }
        configs.append(config)
    return configs


def run_matrix(
    *,
    base_config_path: Path,
    run_prefix: str,
    cli: Callable[..., int] = cli_main,
) -> int:
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    configs = build_matrix_configs(base_config, run_prefix=run_prefix)
    completed: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="dmf-bench-matrix-", dir="/tmp") as raw_dir:
        directory = Path(raw_dir)
        for config in configs:
            run_id = str(config["experiment_id"])
            config_path = directory / f"{run_id}.json"
            config_path.write_text(
                json.dumps(config, sort_keys=True),
                encoding="utf-8",
            )
            exit_code = cli(
                ["run", "--config", str(config_path), "--run-id", run_id],
                application_builder=fixture_application_builder,
            )
            if exit_code != 0:
                return int(exit_code)
            completed.append(
                {
                    "run_id": run_id,
                    "benchmark": str(config["benchmark"]),
                    "framework": str(config["framework"]),
                }
            )

    print(
        json.dumps(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "state": "COMPLETED",
                "completed_count": len(completed),
                "runs": completed,
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic container certification matrix."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_matrix(
        base_config_path=args.config,
        run_prefix=str(args.run_prefix),
    )


if __name__ == "__main__":
    raise SystemExit(main())
