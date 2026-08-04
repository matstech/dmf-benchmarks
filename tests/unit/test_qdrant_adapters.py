from pathlib import Path

import pytest

from dmf_bench.adapters.base import FrameworkCapability, ResumeCapability
from dmf_bench.adapters.dmf import DmfQdrantFrameworkAdapter
from dmf_bench.adapters.mem0 import Mem0QdrantFrameworkAdapter
from dmf_bench.adapters.qdrant_lifecycle import (
    CollectionRole,
    QdrantLifecycleError,
    QdrantLifecycleManager,
    build_cleanup_manifest,
    qdrant_collection_name,
)


class FakeCount:
    def __init__(self, count: int) -> None:
        self.count = count


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, int] = {}
        self.created: list[str] = []
        self.deleted: list[str] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.collections[collection_name] = 0
        self.created.append(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)
        self.deleted.append(collection_name)

    def count(self, collection_name: str, exact: bool = True) -> FakeCount:
        return FakeCount(self.collections[collection_name])


def test_collection_names_are_stable_bounded_and_role_specific() -> None:
    first = qdrant_collection_name(
        run_hash="a" * 64,
        framework="DMF",
        unit_id="conversation/with/a/very/long/id",
        role=CollectionRole.PRIMARY,
    )
    second = qdrant_collection_name(
        run_hash="a" * 64,
        framework="DMF",
        unit_id="conversation/with/a/very/long/id",
        role=CollectionRole.PRIMARY,
    )
    cards = qdrant_collection_name(
        run_hash="a" * 64,
        framework="DMF",
        unit_id="conversation/with/a/very/long/id",
        role=CollectionRole.CARDS,
    )

    assert first == second
    assert first != cards
    assert len(first) <= 63
    assert first.startswith("bench_aaaaaaaaaaaaaaaa_dmf_")


def test_dmf_adapter_declares_qdrant_resources_and_restart_policy() -> None:
    adapter = DmfQdrantFrameworkAdapter(vector_size=768)

    manifest = adapter.resources_for_unit("a" * 64, "unit-1")

    assert adapter.resume_capability is ResumeCapability.RESTART_UNIT
    assert FrameworkCapability.QDRANT_SERVER in adapter.capabilities
    assert [collection.role for collection in manifest.collections] == [
        CollectionRole.PRIMARY,
        CollectionRole.CARDS,
    ]
    assert all(collection.vector_size == 768 for collection in manifest.collections)


def test_mem0_adapter_declares_collections_and_sqlite_cleanup(tmp_path: Path) -> None:
    adapter = Mem0QdrantFrameworkAdapter(vector_size=768, sqlite_root=tmp_path)

    manifest = adapter.resources_for_unit("b" * 64, "question/1")

    assert [collection.role for collection in manifest.collections] == [
        CollectionRole.PRIMARY,
        CollectionRole.ENTITIES,
    ]
    history_path = str(tmp_path / "question_1.sqlite")
    assert manifest.local_paths == (
        history_path,
        f"{history_path}-wal",
        f"{history_path}-shm",
    )


def test_qdrant_lifecycle_create_verify_and_cleanup() -> None:
    client = FakeQdrantClient()
    manager = QdrantLifecycleManager(client)
    manifest = build_cleanup_manifest(
        run_hash="c" * 64,
        framework="dmf",
        unit_id="unit-1",
        roles=(CollectionRole.PRIMARY, CollectionRole.CARDS),
        vector_size=4,
    )

    manager.check_ready()
    manager.assert_absent(manifest)
    manager.create_collections(manifest)

    assert client.created == [collection.name for collection in manifest.collections]

    client.collections[manifest.collections[0].name] = 2
    client.collections[manifest.collections[1].name] = 1
    manager.verify_counts(
        manifest,
        minimum_count_by_role={
            CollectionRole.PRIMARY: 1,
            CollectionRole.CARDS: 1,
        },
    )
    manager.delete_and_wait(manifest)

    assert client.collections == {}
    assert client.deleted == [collection.name for collection in manifest.collections]


def test_qdrant_lifecycle_fails_on_existing_or_missing_counts() -> None:
    client = FakeQdrantClient()
    manager = QdrantLifecycleManager(client)
    manifest = build_cleanup_manifest(
        run_hash="d" * 64,
        framework="mem0",
        unit_id="unit-1",
        roles=(CollectionRole.PRIMARY,),
        vector_size=4,
    )

    client.collections[manifest.collections[0].name] = 0

    with pytest.raises(QdrantLifecycleError, match="already exist"):
        manager.assert_absent(manifest)
    with pytest.raises(QdrantLifecycleError, match="expected at least 1"):
        manager.verify_counts(
            manifest,
            minimum_count_by_role={CollectionRole.PRIMARY: 1},
        )


def test_framework_runtime_validation_uses_server_capable_dependencies(tmp_path: Path) -> None:
    DmfQdrantFrameworkAdapter(vector_size=4).validate_runtime()
    Mem0QdrantFrameworkAdapter(vector_size=4, sqlite_root=tmp_path).validate_runtime()
