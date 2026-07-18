import os
from pathlib import Path

import pytest

from dmf_bench.adapters.dmf import DmfQdrantFrameworkAdapter
from dmf_bench.adapters.locomo import LoCoMoAdapter
from dmf_bench.adapters.qdrant_lifecycle import QdrantLifecycleManager
from dmf_bench.contracts import sha256_file


@pytest.mark.integration
def test_locomo_conversation_schedules_qdrant_resources() -> None:
    qdrant_url = os.getenv("DMF_BENCH_TEST_QDRANT_URL")
    if not qdrant_url:
        pytest.skip("Set DMF_BENCH_TEST_QDRANT_URL to run Qdrant integration.")

    dataset_path = Path(__file__).parents[1] / "fixtures" / "locomo-mini.json"
    config = {
        "benchmark": "locomo",
        "framework": "dmf",
        "protocol": "strict",
        "dataset": {
            "name": "locomo",
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "selection": {
            "ordered_item_ids": ["conversation-0001"],
            "filters": {"categories": [1, 2]},
            "seed": 7,
        },
        "models": {
            "answerer": {
                "provider": "fake",
                "requested_model": "fixture-model",
                "parameters": {"temperature": 0},
            }
        },
    }
    unit = LoCoMoAdapter().enumerate_units(config)[0]
    resource = DmfQdrantFrameworkAdapter(vector_size=8).resources_for_unit(
        "phase6run",
        unit.unit_id,
    )
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url, api_key=os.getenv("DMF_BENCH_TEST_QDRANT_API_KEY"))
    manager = QdrantLifecycleManager(client)

    manager.assert_absent(resource)
    try:
        manager.create_collections(resource)
        manager.verify_counts(resource, minimum_count_by_role={})
    finally:
        manager.delete_and_wait(resource)

    manager.assert_absent(resource)
