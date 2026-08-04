from __future__ import annotations

import json
from pathlib import Path

from dmf_bench.container_matrix import build_matrix_configs
from dmf_bench.contracts import sha256_file
from dmf_bench.registry import supported_combinations


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"
CONFIG_DIR = Path(__file__).parents[2] / "config"


def test_container_matrix_resolves_all_combinations_with_pinned_inputs() -> None:
    base = json.loads(
        (FIXTURE_DIR / "docker-locomo-execution.json").read_text(encoding="utf-8")
    )

    configs = build_matrix_configs(
        base,
        run_prefix="matrix-fixture",
        config_dir=CONFIG_DIR,
        locomo_dataset=FIXTURE_DIR / "locomo-mini.json",
        longmemeval_dataset=FIXTURE_DIR / "longmemeval-mini.json",
    )

    assert [
        (config["benchmark"], config["framework"])
        for config in configs
    ] == supported_combinations()
    assert len({config["experiment_id"] for config in configs}) == 4
    for config in configs:
        framework_path = Path(config["framework_config"]["path"])
        dataset_path = Path(config["dataset"]["path"])
        assert config["runtime"]["execution_profile"] == (
            "docker-qdrant-fixture-v1"
        )
        assert config["models"]["answerer"]["provider"] == "fixture"
        assert config["models"]["judge"]["provider"] == "fixture"
        assert config["framework_config"]["sha256"] == sha256_file(framework_path)
        assert config["dataset"]["sha256"] == sha256_file(dataset_path)
