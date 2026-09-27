from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import BenchmarkUnit, FrameworkRunContext, MemoryEvent, MemoryQuery, OwnedResource
from dmf_bench.adapters.mem0 import (
    MEM0_V3_PAIRING_POLICY,
    Mem0EngineBundle,
    Mem0QdrantFrameworkAdapter,
    Mem0RuntimeError,
)
from dmf_bench.adapters.qdrant_lifecycle import CollectionRole, QdrantLifecycleError
from dmf_bench.frameworks.mem0_config import load_mem0_config
from dmf_bench.frameworks.mem0_runtime import empty_memory_internal_usage


class FakeCount:
    def __init__(self, count: int) -> None:
        self.count = count


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: dict[str, int] = {}
        self.deleted: list[str] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.collections[collection_name] = 0

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)
        self.collections.pop(collection_name)

    def count(self, collection_name: str, exact: bool = True) -> FakeCount:
        assert exact
        return FakeCount(self.collections[collection_name])


class FakeBackend:
    def __init__(self, client: FakeQdrant, primary: str) -> None:
        self.client = client
        self.primary = primary
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.usage = empty_memory_internal_usage(available=True, framework="mem0")

    def add(self, messages: list[dict[str, str]], *, user_id: str,
            timestamp: int | None, metadata: dict[str, Any]) -> int:
        self.add_calls.append({"messages": messages, "user_id": user_id,
                               "timestamp": timestamp, "metadata": metadata})
        self.rows.append({"id": f"m-{len(self.rows)+1}",
                          "memory": " ".join(message["content"] for message in messages),
                          "score": 0.8, "created_at": timestamp,
                          "metadata": metadata})
        self.client.collections[self.primary] += 1
        self.usage["calls"] += 1
        return 1

    def search_raw(self, query: str, *, user_id: str, top_k: int) -> dict[str, Any]:
        self.search_calls.append({"query": query, "user_id": user_id, "top_k": top_k})
        self.usage["calls"] += 1
        return {"results": self.rows[:top_k]}

    def get_usage(self) -> dict[str, Any]:
        return dict(self.usage)


@dataclass
class FakeBuilder:
    backend: FakeBackend | None = None

    def build(self, *, mem0_config: Any, cleanup_manifest: Any,
              qdrant_client: FakeQdrant, history_path: Path) -> Mem0EngineBundle:
        del mem0_config
        for collection in cleanup_manifest.collections:
            qdrant_client.create_collection(collection.name, object())
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(history_path) as connection:
            connection.execute("CREATE TABLE history (id TEXT)")
        primary = next(item.name for item in cleanup_manifest.collections
                       if item.role is CollectionRole.PRIMARY)
        self.backend = FakeBackend(qdrant_client, primary)
        return Mem0EngineBundle(memory=object(), backend=self.backend)  # type: ignore[arg-type]


def setup(tmp_path: Path) -> tuple[Mem0QdrantFrameworkAdapter, FakeQdrant, FakeBuilder,
                                   BenchmarkUnit, FrameworkRunContext, dict[str, Any]]:
    client = FakeQdrant()
    builder = FakeBuilder()
    config_path = Path(__file__).parents[2] / "config" / "locomo_mem0_qdrant_settings.yaml"
    adapter = Mem0QdrantFrameworkAdapter(
        vector_size=768, qdrant_client=client,
        mem0_config=load_mem0_config(config_path), engine_builder=builder,
    )
    unit = BenchmarkUnit("unit-1", ("q-1",), {})
    context = FrameworkRunContext("run-1", "a" * 64, tmp_path / "runs" / "run-1")
    config = {"runtime": {"cache_dir": str(tmp_path / "cache")}}
    return adapter, client, builder, unit, context, config


def events() -> tuple[MemoryEvent, ...]:
    return (
        MemoryEvent("e-1", "s-1", 0, "user", "My cat is Pixel", "2026-01-02T03:04:05Z",
                    source_refs=("source-1",)),
        MemoryEvent("e-2", "s-1", 1, "assistant", "I remember Pixel", "2026-01-02T03:04:05Z"),
        MemoryEvent("e-3", "s-2", 0, "user", "I moved", None),
    )


def test_mem0_v3_ingests_normalized_events_pairs_and_retrieves_without_reference(tmp_path: Path) -> None:
    adapter, client, builder, unit, context, config = setup(tmp_path)
    declared = adapter.resources_for_unit_v3(unit, config, context)
    assert [resource.role for resource in declared] == ["primary", "entities", "history", "wal", "shm"]
    assert all(not resource.locator.startswith(str(tmp_path)) for resource in declared)
    prepared = adapter.prepare_unit_v3(unit, events(), config, context)
    backend = builder.backend
    assert backend is not None
    assert prepared.resources == declared
    assert prepared.diagnostics["pairing_policy"] == MEM0_V3_PAIRING_POLICY
    assert prepared.diagnostics["ingested_batches"] == 2
    assert [len(call["messages"]) for call in backend.add_calls] == [2, 1]
    assert backend.add_calls[0]["metadata"]["source_event_ids"] == ["e-1", "e-2"]
    assert backend.add_calls[0]["metadata"]["source_refs"] == ["source-1"]
    assert backend.add_calls[0]["timestamp"] == 1767323045
    assert adapter.verify_prepared_v3(unit, prepared, config, context)["verified"] is True

    result = adapter.retrieve_v3(unit, MemoryQuery("q-1", "Where is Pixel?", None),
                                 prepared, config, context)
    assert backend.search_calls[0]["query"] == "Where is Pixel?"
    assert [item.rank for item in result.items] == [1, 2]
    assert result.items[0].source_event_ids == ("e-1", "e-2")
    assert result.items[0].native_score == 0.8
    assert result.items[0].native_score_kind == "similarity"
    assert result.items[0].occurred_at == "2026-01-02T03:04:05Z"
    config["retrieval"] = {"max_results": 1}
    limited = adapter.retrieve_v3(unit, MemoryQuery("q-1", "Where is Pixel?", None),
                                  prepared, config, context)
    assert len(limited.items) == 1
    assert backend.search_calls[-1]["top_k"] == 1
    assert "handle" not in prepared.to_dict()
    json.dumps(prepared.to_dict())
    json.dumps(result.to_dict())

    client.collections["foreign"] = 3
    evidence = adapter.cleanup_unit_v3(unit, declared, config, context)
    assert evidence["verified"] is True
    assert client.collections == {"foreign": 3}
    assert set(client.deleted) == {resource.locator for resource in declared[:2]}
    assert not (tmp_path / "cache" / declared[2].locator).exists()


def test_mem0_v3_rejects_foreign_cleanup_and_invalid_prepared_state(tmp_path: Path) -> None:
    adapter, client, _builder, unit, context, config = setup(tmp_path)
    resources = adapter.resources_for_unit_v3(unit, config, context)
    client.collections["foreign"] = 4
    foreign = (OwnedResource("foreign", "qdrant-collection", "primary", "foreign"),) + resources[1:]
    with pytest.raises(Mem0RuntimeError, match="owned unit manifest"):
        adapter.cleanup_unit_v3(unit, foreign, config, context)
    assert client.collections == {"foreign": 4}
    prepared = adapter.prepare_unit_v3(unit, events(), config, context)
    other_context = FrameworkRunContext("another-run", "a" * 64, context.run_dir)
    with pytest.raises(Mem0RuntimeError, match="another unit or run"):
        adapter.verify_prepared_v3(unit, prepared, config, other_context)
    with pytest.raises(Mem0RuntimeError, match="not part"):
        adapter.retrieve_v3(unit, MemoryQuery("foreign-query", "Hi", None), prepared, config, context)


def test_mem0_v3_barrier_detects_missing_points_or_history(tmp_path: Path) -> None:
    adapter, client, _builder, unit, context, config = setup(tmp_path)
    prepared = adapter.prepare_unit_v3(unit, events(), config, context)
    primary = prepared.resources[0].locator
    client.collections[primary] -= 1
    with pytest.raises(QdrantLifecycleError, match="count mismatch"):
        adapter.verify_prepared_v3(unit, prepared, config, context)
    client.collections[primary] += 1
    history = tmp_path / "cache" / prepared.resources[2].locator
    history.unlink()
    with pytest.raises(Mem0RuntimeError, match="was not created"):
        adapter.verify_prepared_v3(unit, prepared, config, context)


def test_mem0_v3_rejects_unordered_events_before_creating_resources(tmp_path: Path) -> None:
    adapter, client, _builder, unit, context, config = setup(tmp_path)
    with pytest.raises(Mem0RuntimeError, match="ordered"):
        adapter.prepare_unit_v3(unit, events()[:2][::-1], config, context)
    assert client.collections == {}


def test_mem0_v3_maps_observation_role_without_altering_content(tmp_path: Path) -> None:
    adapter, _client, builder, unit, context, config = setup(tmp_path)
    observation = MemoryEvent("e-1", "s-1", 0, "observation", "Exact observation", None)
    adapter.prepare_unit_v3(unit, (observation,), config, context)
    backend = builder.backend
    assert backend is not None
    assert backend.add_calls[0]["messages"] == [{"role": "user", "content": "Exact observation"}]


def test_mem0_v3_uses_separate_qdrant_namespace(tmp_path: Path) -> None:
    adapter, _client, _builder, unit, context, config = setup(tmp_path)
    old_resources = adapter.resources_for_unit_v3(unit, config, context)
    config["schema_version"] = 3
    new_resources = adapter.resources_for_unit_v3(unit, config, context)
    assert old_resources[0].locator.startswith("dmf_bench_v2_")
    assert new_resources[0].locator.startswith("dmf_bench_v3_")
