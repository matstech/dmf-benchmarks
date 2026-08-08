from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dmf_bench.frameworks.dmf_context import DmfNativeContextSurface
from dmf_bench.adapters.base import BenchmarkUnit, FrameworkRunContext
from dmf_bench.adapters.dmf import (
    DefaultDmfEngineBuilder,
    DmfEngineBundle,
    DmfQdrantFrameworkAdapter,
    DmfRuntimeError,
    dmf_framework_factories,
)
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.adapters.qdrant_lifecycle import CollectionRole
from dmf.utils.config_loader import load_dmf_config


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"
CONFIG_DIR = Path(__file__).parents[2] / "config"


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

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        del vectors_config
        self.collections[collection_name] = 0
        self.created.append(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)
        self.deleted.append(collection_name)

    def count(self, collection_name: str, exact: bool = True) -> FakeCount:
        assert exact is True
        return FakeCount(self.collections[collection_name])


class FakePipeline:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.texts: list[str] = []
        self.roles: list[str] = []

    def analyze_interaction_with_vector(
        self,
        *,
        text: str,
        is_system: bool,
        provenance: Any,
    ) -> tuple[Any, list[float]]:
        assert is_system is False
        self.texts.append(text)
        self.roles.append(str(provenance.role))
        if self.fail_at is not None and len(self.texts) == self.fail_at:
            raise RuntimeError("injected ingestion failure")
        return SimpleNamespace(raw_metadata={}), [0.25, 0.5, 0.75, 1.0]


class FakeScoring:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def calculate_score(self, report: Any, *, text: str) -> None:
        report.raw_metadata["scored"] = True
        self.texts.append(text)


class FakeEmbedding:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def get_embedding(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.25, 0.5, 0.75, 1.0]


class FakeMemoryEngine:
    def __init__(self) -> None:
        self.entries: list[Any] = []
        self._diagnostics: dict[str, list[dict[str, Any]]] = {
            "raw_candidates": [],
            "ranked_candidates": [],
            "final_candidates": [],
            "suppressed": [],
        }

    @property
    def size(self) -> int:
        return len(self.entries)

    def add_interaction(self, text: str, report: Any, vector: Any) -> Any:
        del report, vector
        entry = SimpleNamespace(
            record_id=f"record-{len(self.entries) + 1}",
            timestamp=1000.0 + len(self.entries),
            text=text,
        )
        self.entries.append(entry)
        return entry

    def get_raw_recall_hits(self, query_vector: Any, k: int) -> list[Any]:
        del query_vector
        return self.entries[:k]

    def contextualize_raw_recall_hits(self, hits: list[Any]) -> list[Any]:
        return hits

    def rerank_contextualized_recall_candidates(self, candidates: list[Any]) -> list[Any]:
        serialized = [
            {
                "record": {
                    "record_id": entry.record_id,
                    "text": entry.text,
                    "created_at": entry.timestamp,
                },
                "similarity_score": 0.8,
                "recall_score": 0.9,
            }
            for entry in candidates
        ]
        self._diagnostics = {
            "raw_candidates": list(serialized),
            "ranked_candidates": list(serialized),
            "final_candidates": list(serialized),
            "suppressed": [],
        }
        return candidates

    def get_recall_diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)

    def get_context_metrics(self) -> dict[str, int]:
        return {"active_entries": len(self.entries)}


@dataclass
class FakeEngineBuilder:
    fail_at: int | None = None
    build_count: int = 0
    last_bundle: DmfEngineBundle | None = None
    cards_path: Path | None = None

    def build(
        self,
        *,
        dmf_config: Any,
        cleanup_manifest: Any,
        qdrant_client: Any,
        cards_path: Path,
    ) -> DmfEngineBundle:
        del dmf_config, cleanup_manifest, qdrant_client
        self.build_count += 1
        self.cards_path = cards_path
        cards_path.parent.mkdir(parents=True, exist_ok=True)
        cards_path.write_text("fixture-card\n", encoding="utf-8")
        bundle = DmfEngineBundle(
            pipeline=FakePipeline(fail_at=self.fail_at),
            scoring=FakeScoring(),
            memory_engine=FakeMemoryEngine(),
            embedding_engine=FakeEmbedding(),
            memory_api=object(),
        )
        self.last_bundle = bundle
        return bundle


def runtime_config(tmp_path: Path, *, benchmark: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "framework": "dmf",
        "runtime": {
            "root": str(tmp_path),
            "runs_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "framework_config": {
            "path": str(CONFIG_DIR / f"{benchmark}_dmf_qdrant_settings.toml")
        },
        "qdrant": {
            "endpoint_env": "QDRANT_URL",
            "request_timeout_seconds": 10,
        },
    }


def run_context(tmp_path: Path, suffix: str = "a") -> FrameworkRunContext:
    return FrameworkRunContext(
        run_id=f"run-{suffix}",
        scientific_fingerprint=suffix * 64,
        run_dir=tmp_path / "runs" / f"run-{suffix}",
    )


def locomo_case() -> tuple[BenchmarkUnit, dict[str, Any], tuple[Any, ...]]:
    config = {
        "dataset": {
            "path": str(FIXTURE_DIR / "locomo-mini.json"),
        },
        "selection": {
            "ordered_item_ids": ["conversation-0001"],
            "filters": {"categories": [1, 2]},
        },
    }
    adapter = LoCoMoAdapter()
    unit = adapter.enumerate_units(config)[0]
    _index, conversation, questions = adapter.selected_conversations_by_id(config)[
        unit.unit_id
    ]
    return unit, conversation, questions


def longmemeval_case() -> tuple[BenchmarkUnit, dict[str, Any]]:
    config = {
        "dataset": {
            "path": str(FIXTURE_DIR / "longmemeval-mini.json"),
        },
        "selection": {
            "ordered_item_ids": ["lme-001"],
        },
    }
    adapter = LongMemEvalAdapter()
    unit = adapter.enumerate_units(config)[0]
    return unit, adapter.selected_questions_by_id(config)[unit.unit_id]


def adapter_for(
    *,
    benchmark: str,
    client: FakeQdrantClient,
    builder: FakeEngineBuilder,
    native_surface_builder: Any | None = None,
) -> DmfQdrantFrameworkAdapter:
    kwargs: dict[str, Any] = {}
    if native_surface_builder is not None:
        kwargs["native_surface_builder"] = native_surface_builder
    return DmfQdrantFrameworkAdapter(
        vector_size=768,
        qdrant_client=client,
        dmf_config=load_dmf_config(
            CONFIG_DIR / f"{benchmark}_dmf_qdrant_settings.toml"
        ),
        engine_builder=builder,
        **kwargs,
    )


def test_locomo_runtime_ingests_once_retrieves_many_and_cleans_owned_resources(
    tmp_path: Path,
) -> None:
    unit, conversation, questions = locomo_case()
    config = runtime_config(tmp_path, benchmark="locomo")
    context = run_context(tmp_path)
    client = FakeQdrantClient()
    builder = FakeEngineBuilder()
    native_queries: list[str] = []

    def native_surface(**kwargs: Any) -> DmfNativeContextSurface:
        native_queries.append(kwargs["query_text"])
        source_unit_id = "D1:1" if len(native_queries) == 1 else "D2:1"
        search_results = [
            {
                "memory": "fixture memory",
                "score": 1.0,
                "id": f"record-{len(native_queries)}",
                "metadata": {
                    "source_unit_id": source_unit_id,
                    "source_unit_ids": [source_unit_id],
                },
            }
        ]
        return DmfNativeContextSurface(
            native_context="=== ACTIVE CONVERSATION ===\nfixture",
            query_vector=None,
            surface_marker="dmf_render_context",
            recalled_section_present=False,
            active_section_present=True,
            raw_retrieval_outputs={
                "retrieval_stack": "dmf_structured_native",
                "search_results": search_results,
                "retrieved_evidence": search_results,
            },
            result_count=1,
            context_metrics={"active_entries": 3},
        )

    adapter = adapter_for(
        benchmark="locomo",
        client=client,
        builder=builder,
        native_surface_builder=native_surface,
    )

    prepared = adapter.prepare_unit(
        unit,
        conversation,
        config,
        run_context=context,
    )
    bundle = builder.last_bundle
    assert bundle is not None
    assert builder.build_count == 1
    assert len(bundle.memory_engine.entries) == 3
    assert prepared["qdrant_commit_barrier"] == {
        "verified": True,
        "ingested_count": 3,
        "active_count": 3,
        "collection_counts": {"primary": 0, "cards": 0},
    }
    assert [collection["role"] for collection in prepared["cleanup_manifest"]["collections"]] == [
        "primary",
        "cards",
    ]

    first = adapter.retrieve_question(
        unit,
        conversation,
        questions[0],
        config,
        prepared,
        run_context=context,
    )
    second = adapter.retrieve_question(
        unit,
        conversation,
        questions[1],
        config,
        prepared,
        run_context=context,
    )

    assert len(bundle.memory_engine.entries) == 3
    assert bundle.pipeline.texts[2] == (
        "I moved Pixel's bed near the kitchen window. "
        "Image: query cat bed. The image shows a blue pet bed."
    )
    assert native_queries == [
        "What is Alice's cat called?",
        "Where did Alice move the bed?",
    ]
    assert first.search_results[0]["metadata"]["source_unit_id"] == "D1:1"
    assert second.recall_diagnostics["final_candidates_canonical"] == list(
        second.search_results
    )

    owned_names = set(client.created)
    client.collections["foreign_collection"] = 11
    adapter.cleanup_unit(unit, conversation, config, run_context=context)

    assert set(client.deleted) == owned_names
    assert client.collections == {"foreign_collection": 11}
    assert builder.cards_path is not None
    assert not builder.cards_path.exists()


def test_dmf_resources_are_isolated_by_run_and_unit(tmp_path: Path) -> None:
    unit, conversation, _questions = locomo_case()
    config = runtime_config(tmp_path, benchmark="locomo")
    first = adapter_for(
        benchmark="locomo",
        client=FakeQdrantClient(),
        builder=FakeEngineBuilder(),
    )
    second = adapter_for(
        benchmark="locomo",
        client=FakeQdrantClient(),
        builder=FakeEngineBuilder(),
    )

    first_prepared = first.prepare_unit(
        unit,
        conversation,
        config,
        run_context=run_context(tmp_path, "a"),
    )
    second_prepared = second.prepare_unit(
        unit,
        conversation,
        config,
        run_context=run_context(tmp_path, "b"),
    )
    first_names = {
        item["name"] for item in first_prepared["cleanup_manifest"]["collections"]
    }
    second_names = {
        item["name"] for item in second_prepared["cleanup_manifest"]["collections"]
    }

    assert first_names.isdisjoint(second_names)


def test_longmemeval_runtime_preserves_order_and_normalizes_native_surface(
    tmp_path: Path,
) -> None:
    unit, question = longmemeval_case()
    config = runtime_config(tmp_path, benchmark="longmemeval")
    context = run_context(tmp_path)
    client = FakeQdrantClient()
    builder = FakeEngineBuilder()

    def native_surface(**kwargs: Any) -> DmfNativeContextSurface:
        assert kwargs["memory"] is builder.last_bundle.memory_api
        return DmfNativeContextSurface(
            native_context="=== ACTIVE CONVERSATION ===\nfixture",
            query_vector=None,
            surface_marker="dmf_render_context",
            recalled_section_present=False,
            active_section_present=True,
            raw_retrieval_outputs={
                "retrieval_stack": "dmf_structured_native",
                "search_results": [
                    {
                        "memory": "I prefer almonds as a snack.",
                        "score": 1.0,
                        "id": "record-1",
                        "metadata": {"source_unit_ids": ["session-a"]},
                    }
                ],
            },
            result_count=1,
            context_metrics={"active_entries": 4},
        )

    adapter = adapter_for(
        benchmark="longmemeval",
        client=client,
        builder=builder,
        native_surface_builder=native_surface,
    )
    prepared = adapter.prepare_unit(unit, question, config, run_context=context)
    bundle = builder.last_bundle
    assert bundle is not None
    assert bundle.pipeline.texts == [
        "I prefer almonds as a snack.",
        "I will remember that.",
        "I bought tea.",
        "Noted.",
    ]

    native = adapter.retrieve(unit, question, config, prepared, run_context=context)
    assert native.native_context.startswith("=== ACTIVE CONVERSATION ===")
    assert native.search_results[0]["metadata"]["source_unit_ids"] == ["session-a"]
    assert native.recall_diagnostics["diagnostics_available"] is True
    assert native.recall_diagnostics["diagnostic_source"] == (
        "dmf_structured_native_final_projection"
    )
    assert native.recall_diagnostics["raw_stage_available"] is False
    assert native.recall_diagnostics["ranked_candidates_canonical"] == list(
        native.search_results
    )
    assert native.recall_diagnostics["final_candidates_canonical"] == list(
        native.search_results
    )
    assert native.native_surface_diagnostics["raw_retrieval_outputs"][
        "retrieval_stack"
    ] == "dmf_structured_native"


def test_failed_ingestion_removes_partial_collections_and_local_cards(
    tmp_path: Path,
) -> None:
    unit, conversation, _questions = locomo_case()
    config = runtime_config(tmp_path, benchmark="locomo")
    client = FakeQdrantClient()
    builder = FakeEngineBuilder(fail_at=2)
    adapter = adapter_for(benchmark="locomo", client=client, builder=builder)

    with pytest.raises(RuntimeError, match="injected ingestion failure"):
        adapter.prepare_unit(
            unit,
            conversation,
            config,
            run_context=run_context(tmp_path),
        )

    assert client.collections == {}
    assert set(client.deleted) == set(client.created)
    assert builder.cards_path is not None
    assert not builder.cards_path.exists()


@pytest.mark.parametrize(
    ("storage_type", "qdrant_mode", "message"),
    [
        ("chroma", "server", "storage_type='qdrant'"),
        ("qdrant", "memory", "qdrant_mode='server'"),
    ],
)
def test_runtime_rejects_non_server_backends(
    storage_type: str,
    qdrant_mode: str,
    message: str,
) -> None:
    config = load_dmf_config(CONFIG_DIR / "locomo_dmf_qdrant_settings.toml")
    config = replace(
        config,
        ltm=replace(
            config.ltm,
            storage_type=storage_type,
            qdrant_mode=qdrant_mode,
        ),
    )
    adapter = DmfQdrantFrameworkAdapter(
        vector_size=768,
        qdrant_client=FakeQdrantClient(),
        dmf_config=config,
    )

    with pytest.raises(DmfRuntimeError, match=message):
        adapter.validate_runtime()


def test_factory_uses_qdrant_url_without_requiring_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(tmp_path, benchmark="locomo")
    client = FakeQdrantClient()
    calls: list[tuple[str, str | None, float]] = []

    def client_factory(endpoint: str, api_key: str | None, timeout: float) -> Any:
        calls.append((endpoint, api_key, timeout))
        return client

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    factory = dmf_framework_factories(
        engine_builder=FakeEngineBuilder(),
        client_factory=client_factory,
    )["dmf"]

    adapter = factory(config)

    assert adapter.qdrant_client is client
    assert calls == [("http://qdrant:6333", None, 10.0)]


def test_default_engine_builder_uses_runtime_cache_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(tmp_path, benchmark="locomo")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    adapter = DmfQdrantFrameworkAdapter.from_experiment(
        config,
        client_factory=lambda _endpoint, _api_key, _timeout: FakeQdrantClient(),
    )

    assert isinstance(adapter.engine_builder, DefaultDmfEngineBuilder)
    assert adapter.engine_builder.embedding_cache_dir == (
        tmp_path / "cache" / "models" / "embeddings"
    )


def test_factory_fails_before_client_creation_when_qdrant_url_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runtime_config(tmp_path, benchmark="locomo")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    client_created = False

    def client_factory(endpoint: str, api_key: str | None, timeout: float) -> Any:
        nonlocal client_created
        del endpoint, api_key, timeout
        client_created = True
        return FakeQdrantClient()

    with pytest.raises(ValueError, match="requires QDRANT_URL"):
        DmfQdrantFrameworkAdapter.from_experiment(
            config,
            engine_builder=FakeEngineBuilder(),
            client_factory=client_factory,
        )

    assert client_created is False


def test_tracked_qdrant_configs_do_not_select_chroma() -> None:
    for path in (
        CONFIG_DIR / "locomo_dmf_qdrant_settings.toml",
        CONFIG_DIR / "longmemeval_dmf_qdrant_settings.toml",
    ):
        raw = path.read_text(encoding="utf-8").lower()
        loaded = load_dmf_config(path)

        assert "chroma" not in raw
        assert loaded.ltm.storage_type == "qdrant"
        assert loaded.ltm.qdrant_mode == "server"
        assert loaded.ltm.qdrant_api_key_env == ""
