"""DMF v3 normalized-event and owned-resource checks without providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dmf_bench.adapters.base import BenchmarkUnit, FrameworkRunContext, MemoryEvent, MemoryQuery
from dmf_bench.adapters.dmf import DmfEngineBundle, DmfQdrantFrameworkAdapter, DmfRuntimeError
from dmf_bench.frameworks.dmf_context import DmfNativeContextSurface


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
        del self.collections[collection_name]

    def count(self, collection_name: str, exact: bool = True) -> Any:
        assert exact
        return SimpleNamespace(count=self.collections[collection_name])


class FakePipeline:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def analyze_interaction_with_vector(self, *, text: str, is_system: bool, provenance: Any) -> tuple[Any, list[float]]:
        self.seen.append((text, provenance.role))
        return SimpleNamespace(raw_metadata={}), [0.1]


class FakeMemory:
    def __init__(self) -> None:
        self.entries: list[Any] = []

    @property
    def size(self) -> int:
        return len(self.entries)

    def add_interaction(self, text: str, report: Any, vector: Any) -> Any:
        entry = SimpleNamespace(record_id=f"record-{len(self.entries) + 1}", timestamp=0.0)
        self.entries.append(entry)
        return entry


@dataclass
class FakeBuilder:
    bundle: DmfEngineBundle | None = None

    def build(self, *, dmf_config: Any, cleanup_manifest: Any, qdrant_client: Any, cards_path: Path) -> DmfEngineBundle:
        cards_path.parent.mkdir(parents=True, exist_ok=True)
        cards_path.write_text("owned card")
        self.bundle = DmfEngineBundle(
            pipeline=FakePipeline(),
            scoring=SimpleNamespace(calculate_score=lambda report, text: None),
            memory_engine=FakeMemory(),
            embedding_engine=object(),
            memory_api=object(),
        )
        return self.bundle


def _setup(tmp_path: Path) -> tuple[DmfQdrantFrameworkAdapter, FakeQdrant, FakeBuilder, BenchmarkUnit, dict, FrameworkRunContext]:
    client = FakeQdrant()
    builder = FakeBuilder()
    unit = BenchmarkUnit("unit-1", ("query-1",), {})
    config = {"schema_version": 3, "runtime": {"cache_dir": str(tmp_path / "cache")}}
    context = FrameworkRunContext("run-1", "a" * 64, tmp_path / "run-1")

    def surface(*, memory: Any, query_text: str, record_index: dict) -> DmfNativeContextSurface:
        assert query_text == "What happened?"
        assert memory is builder.bundle.memory_api
        return DmfNativeContextSurface(
            native_context="unused", query_vector=None, surface_marker="fixture",
            recalled_section_present=False, active_section_present=True,
            raw_retrieval_outputs={"search_results": [
                {"id": "record-1", "memory": "Event one", "score": 0.8},
            ]}, result_count=1, context_metrics={},
        )

    adapter = DmfQdrantFrameworkAdapter(
        vector_size=1, qdrant_client=client, dmf_config=object(),
        engine_builder=builder, native_surface_builder=surface,
    )
    return adapter, client, builder, unit, config, context


def test_v3_ingests_only_events_verifies_barrier_and_retrieves(tmp_path: Path) -> None:
    adapter, client, builder, unit, config, context = _setup(tmp_path)
    assert all(
        resource.locator.startswith("dmf_bench_v3_")
        for resource in adapter.resources_for_unit_v3(unit, config, context)
        if resource.kind == "qdrant-collection"
    )
    events = (MemoryEvent("event-1", "session-1", 0, "user", "Event one",
                          "2024-01-01T09:00:00Z", ("source-1",)),)
    prepared = adapter.prepare_unit_v3(unit, events, config, context)
    assert builder.bundle.pipeline.seen == [("Event one", "user")]
    assert prepared.to_dict()["resources"] == [item.to_dict() for item in prepared.resources]
    assert adapter.verify_prepared_v3(unit, prepared, config, context)["verified"] is True
    result = adapter.retrieve_v3(
        unit, MemoryQuery("query-1", "What happened?", None), prepared, config, context,
    )
    assert result.items[0].source_event_ids == ("event-1",)
    assert result.items[0].native_score_kind == "dmf-native"
    assert result.items[0].content == "Event one"
    assert adapter.cleanup_unit_v3(unit, prepared.resources, config, context)["verified"] is True
    assert client.collections == {}


def test_manifest_namespace_defaults_to_v2(tmp_path: Path) -> None:
    adapter, _client, _builder, unit, config, context = _setup(tmp_path)
    del config["schema_version"]
    manifest = adapter._manifest_for_context(unit, config, context)
    assert manifest.namespace == "dmf_bench_v2"
    assert all(item.name.startswith("dmf_bench_v2_") for item in manifest.collections)


def test_v3_rejects_foreign_manifest_and_failed_barrier(tmp_path: Path) -> None:
    adapter, client, _builder, unit, config, context = _setup(tmp_path)
    resources = adapter.resources_for_unit_v3(unit, config, context)
    foreign = adapter.resources_for_unit_v3(BenchmarkUnit("other", ("query-1",), {}), config, context)
    with pytest.raises(DmfRuntimeError, match="outside this run/unit"):
        adapter.cleanup_unit_v3(unit, foreign, config, context)
    prepared = adapter.prepare_unit_v3(
        unit, (MemoryEvent("event-1", "session-1", 0, "user", "Event one", None),),
        config, context,
    )
    primary = next(item.locator for item in resources if item.role == "primary")
    client.collections[primary] = 2
    with pytest.raises(DmfRuntimeError, match="commit barrier"):
        adapter.verify_prepared_v3(unit, prepared, config, context)


def test_factory_prefers_v3_storage_and_keeps_v2_qdrant_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path(__file__).resolve().parents[2] / "config" / "locomo_dmf_qdrant_settings.toml"
    client = FakeQdrant()
    calls: list[tuple[str, str | None, float]] = []

    def client_factory(endpoint: str, api_key: str | None, timeout: float) -> FakeQdrant:
        calls.append((endpoint, api_key, timeout))
        return client

    monkeypatch.setattr(DmfQdrantFrameworkAdapter, "validate_runtime", lambda self: None)
    monkeypatch.setenv("DMF_V3_QDRANT_URL", "http://v3-qdrant:6333")
    monkeypatch.setenv("QDRANT_URL", "http://v2-qdrant:6333")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    config = {
        "framework_config": {"path": str(config_path)},
        "runtime": {"cache_dir": str(tmp_path / "cache")},
        "storage": {
            "kind": "qdrant-server", "endpoint_env": "DMF_V3_QDRANT_URL",
            "request_timeout_seconds": 7,
        },
        "qdrant": {"endpoint_env": "QDRANT_URL"},
    }
    adapter = DmfQdrantFrameworkAdapter.from_experiment(config, client_factory=client_factory)
    assert adapter.qdrant_client is client
    assert calls == [("http://v3-qdrant:6333", None, 7.0)]

    del config["storage"]
    DmfQdrantFrameworkAdapter.from_experiment(config, client_factory=client_factory)
    assert calls[-1] == ("http://v2-qdrant:6333", None, 10.0)


def test_factory_rejects_wrong_storage_kind_before_client_creation(
    tmp_path: Path,
) -> None:
    config_path = Path(__file__).resolve().parents[2] / "config" / "locomo_dmf_qdrant_settings.toml"
    config = {
        "framework_config": {"path": str(config_path)},
        "runtime": {"cache_dir": str(tmp_path / "cache")},
        "storage": {"kind": "chroma-embedded", "endpoint_env": "QDRANT_URL"},
    }
    with pytest.raises(DmfRuntimeError, match="storage.kind"):
        DmfQdrantFrameworkAdapter.from_experiment(
            config,
            client_factory=lambda *_args: pytest.fail("client must not be created"),
        )
