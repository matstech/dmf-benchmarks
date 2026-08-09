import os
import uuid

import pytest

from dmf_bench.adapters.qdrant_lifecycle import (
    CollectionRole,
    QDRANT_COLLECTION_NAMESPACE,
    QdrantLifecycleManager,
    build_cleanup_manifest,
)


@pytest.mark.integration
def test_qdrant_server_lifecycle_roundtrip() -> None:
    qdrant_url = os.getenv("DMF_BENCH_TEST_QDRANT_URL")
    if not qdrant_url:
        pytest.skip("Set DMF_BENCH_TEST_QDRANT_URL to run Qdrant lifecycle integration.")

    from qdrant_client import QdrantClient, models

    client = QdrantClient(url=qdrant_url, api_key=os.getenv("DMF_BENCH_TEST_QDRANT_API_KEY"))
    manager = QdrantLifecycleManager(client)
    unit_id = f"integration-{uuid.uuid4().hex}"
    manifest = build_cleanup_manifest(
        run_hash=uuid.uuid4().hex,
        framework="dmf",
        unit_id=unit_id,
        roles=(CollectionRole.PRIMARY,),
        vector_size=4,
    )
    assert manifest.collections[0].name.startswith(
        f"{QDRANT_COLLECTION_NAMESPACE}_"
    )

    try:
        manager.check_ready()
        manager.assert_absent(manifest)
        manager.create_collections(manifest)
        client.upsert(
            collection_name=manifest.collections[0].name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"test": "dmf-bench"},
                )
            ],
        )
        manager.verify_counts(
            manifest,
            minimum_count_by_role={CollectionRole.PRIMARY: 1},
        )
    finally:
        manager.delete_and_wait(manifest)
