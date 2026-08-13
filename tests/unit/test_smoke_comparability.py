import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG_DIR = ROOT / "smoke" / "config"


def _load_configs() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(SMOKE_CONFIG_DIR.glob("experiment-*.json"))
    ]


def test_smoke_pairs_use_identical_scientific_inputs_except_memory_system() -> None:
    configs = _load_configs()
    assert len(configs) == 4

    for benchmark in ("locomo", "longmemeval"):
        pair = [config for config in configs if config["benchmark"] == benchmark]
        assert {config["framework"] for config in pair} == {"dmf", "mem0"}
        first, second = pair
        for field in ("dataset", "selection", "models", "evaluation"):
            assert first[field] == second[field]
        assert all(
            config["qdrant"]["retention"] == "delete-on-success"
            for config in pair
        )


def test_smoke_sampling_bounds_complete_ingestion_and_answering_records() -> None:
    by_benchmark = {config["benchmark"]: config for config in _load_configs()}

    locomo = by_benchmark["locomo"]
    assert locomo["dataset"]["sampling"] == {
        "fraction": 0.005,
        "unit": "conversation",
        "seed": 7,
        "rounding": "ceil",
    }
    assert locomo["selection"]["ordered_item_ids"] == ["*"]

    longmemeval = by_benchmark["longmemeval"]
    assert longmemeval["dataset"]["sampling"] == {
        "fraction": 0.005,
        "unit": "record",
        "seed": 7,
        "rounding": "ceil",
    }
    assert longmemeval["selection"]["ordered_item_ids"] == ["*"]


def test_smoke_judges_use_bounded_response_recovery() -> None:
    for config in _load_configs():
        assert config["models"]["judge"]["runtime"]["response_max_retries"] == 1
