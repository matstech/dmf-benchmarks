from __future__ import annotations

import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.frameworks.mem0_config import Mem0Config, load_mem0_config
from dmf_bench.frameworks.mem0_runtime import (
    add_mem0_with_observation_timestamp,
    empty_memory_internal_usage,
)
from dmf_bench.adapters.base import BenchmarkUnit, FrameworkRunContext, ProgressUpdate
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.adapters.mem0 import (
    Mem0EngineBundle,
    Mem0QdrantFrameworkAdapter,
    Mem0RuntimeBackend,
    Mem0RuntimeError,
    _validate_mem0_qdrant_config,
    mem0_framework_factories,
)
from dmf_bench.adapters.qdrant_lifecycle import CollectionRole
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.provenance import build_run_provenance


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


class FakeMem0Backend:
    def __init__(
        self,
        *,
        client: FakeQdrantClient,
        primary_name: str,
        fail_at: int | None = None,
    ) -> None:
        self.client = client
        self.primary_name = primary_name
        self.fail_at = fail_at
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.memories: list[dict[str, Any]] = []
        self.usage = empty_memory_internal_usage(available=True, framework="mem0")

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        timestamp: int | None,
        metadata: dict[str, Any],
    ) -> int:
        call = {
            "messages": deepcopy(messages),
            "user_id": user_id,
            "timestamp": timestamp,
            "metadata": deepcopy(metadata),
        }
        self.add_calls.append(call)
        if self.fail_at is not None and len(self.add_calls) == self.fail_at:
            raise RuntimeError("injected Mem0 ingestion failure")
        memory_id = f"memory-{len(self.memories) + 1}"
        self.memories.append(
            {
                "id": memory_id,
                "memory": " ".join(message["content"] for message in messages),
                "score": 1.0,
                "created_at": timestamp,
                "metadata": deepcopy(metadata),
            }
        )
        self.client.collections[self.primary_name] += 1
        self._add_usage(prompt_tokens=3, completion_tokens=1)
        return 1

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        raw = self.search_raw(query, user_id=user_id, top_k=top_k)
        return list(raw["results"])

    def search_raw(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int,
    ) -> dict[str, list[dict[str, Any]]]:
        self.search_calls.append(
            {"query": query, "user_id": user_id, "top_k": top_k}
        )
        self._add_usage(prompt_tokens=2, completion_tokens=0)
        return {"results": deepcopy(self.memories[:top_k])}

    def get_usage(self) -> dict[str, Any]:
        return deepcopy(self.usage)

    def _add_usage(self, *, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens
        self.usage["calls"] += 1


@dataclass
class FakeMem0EngineBuilder:
    fail_at: int | None = None
    build_count: int = 0
    backend: FakeMem0Backend | None = None
    history_path: Path | None = None

    def build(
        self,
        *,
        mem0_config: Mem0Config,
        cleanup_manifest: Any,
        qdrant_client: FakeQdrantClient,
        history_path: Path,
    ) -> Mem0EngineBundle:
        del mem0_config
        self.build_count += 1
        self.history_path = history_path
        for collection in cleanup_manifest.collections:
            qdrant_client.create_collection(collection.name, object())
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(history_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS history (id TEXT)")
        primary_name = next(
            collection.name
            for collection in cleanup_manifest.collections
            if collection.role is CollectionRole.PRIMARY
        )
        backend = FakeMem0Backend(
            client=qdrant_client,
            primary_name=primary_name,
            fail_at=self.fail_at,
        )
        self.backend = backend
        return Mem0EngineBundle(memory=object(), backend=backend)  # type: ignore[arg-type]


def runtime_config(tmp_path: Path, *, benchmark: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "framework": "mem0",
        "runtime": {
            "root": str(tmp_path),
            "runs_dir": str(tmp_path / "runs"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "framework_config": {
            "path": str(CONFIG_DIR / f"{benchmark}_mem0_qdrant_settings.yaml")
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
        "dataset": {"path": str(FIXTURE_DIR / "locomo-mini.json")},
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
        "dataset": {"path": str(FIXTURE_DIR / "longmemeval-mini.json")},
        "selection": {"ordered_item_ids": ["lme-001"]},
    }
    adapter = LongMemEvalAdapter()
    unit = adapter.enumerate_units(config)[0]
    return unit, adapter.selected_questions_by_id(config)[unit.unit_id]


def adapter_for(
    *,
    benchmark: str,
    client: FakeQdrantClient,
    builder: FakeMem0EngineBuilder,
    metrics: BenchmarkMetrics | None = None,
) -> Mem0QdrantFrameworkAdapter:
    mem0_config = load_mem0_config(
        CONFIG_DIR / f"{benchmark}_mem0_qdrant_settings.yaml"
    )
    return Mem0QdrantFrameworkAdapter(
        vector_size=768,
        qdrant_client=client,
        mem0_config=mem0_config,
        engine_builder=builder,
        metrics=metrics,
    )


def test_locomo_runtime_ingests_once_reuses_memory_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    unit, conversation, questions = locomo_case()
    config = runtime_config(tmp_path, benchmark="locomo")
    updates: list[ProgressUpdate] = []
    context = FrameworkRunContext(
        run_id="run-a",
        scientific_fingerprint="a" * 64,
        run_dir=tmp_path / "runs" / "run-a",
        progress_reporter=updates.append,
    )
    client = FakeQdrantClient()
    builder = FakeMem0EngineBuilder()
    adapter = adapter_for(benchmark="locomo", client=client, builder=builder)

    prepared = adapter.prepare_unit(unit, conversation, config, run_context=context)
    backend = builder.backend
    assert backend is not None
    assert builder.build_count == 1
    assert [(update.stage, update.completed, update.total) for update in updates] == [
        ("memory_initialization", 0, 1),
        ("memory_initialization", 1, 1),
        ("memory_ingestion", 0, 3),
        ("memory_ingestion", 1, 3),
        ("memory_ingestion", 2, 3),
        ("memory_ingestion", 3, 3),
    ]
    assert [call["messages"][0]["role"] for call in backend.add_calls] == [
        "user",
        "assistant",
        "user",
    ]
    assert backend.add_calls[0]["messages"][0]["content"] == (
        "Alice: I adopted a grey cat named Pixel."
    )
    assert backend.add_calls[2]["messages"][0]["content"] == (
        "Alice: I moved Pixel's bed near the kitchen window. "
        "Shared image: query cat bed. The image shows a blue pet bed."
    )
    assert len({call["user_id"] for call in backend.add_calls}) == 1
    assert prepared["qdrant_commit_barrier"] == {
        "verified": True,
        "ingested_batches": 3,
        "persisted_memory_count": 3,
        "collection_counts": {"primary": 3, "entities": 0},
        "sqlite_history_integrity": "ok",
    }

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

    assert len(backend.add_calls) == 3
    assert [call["query"] for call in backend.search_calls] == [
        "What is Alice's cat called?",
        "Where did Alice move the bed?",
    ]
    assert first.search_results[0]["metadata"]["source_unit_id"] == "D1:1"
    assert second.cutoff_label == "native"
    assert first.memory_internal_usage["available"] is True
    assert first.memory_internal_usage["calls"] == 3


def test_longmemeval_runtime_preserves_pair_order_and_native_surface(
    tmp_path: Path,
) -> None:
    unit, question = longmemeval_case()
    config = runtime_config(tmp_path, benchmark="longmemeval")
    context = run_context(tmp_path)
    client = FakeQdrantClient()
    builder = FakeMem0EngineBuilder()
    adapter = adapter_for(benchmark="longmemeval", client=client, builder=builder)

    prepared = adapter.prepare_unit(unit, question, config, run_context=context)
    backend = builder.backend
    assert backend is not None
    assert [[message["content"] for message in call["messages"]] for call in backend.add_calls] == [
        ["I prefer almonds as a snack.", "I will remember that."],
        ["I bought tea.", "Noted."],
    ]

    native = adapter.retrieve(
        unit,
        question,
        config,
        prepared,
        run_context=context,
    )

    assert native.native_surface_diagnostics["surface_marker"] == "mem0_search_surface"
    assert isinstance(native.native_context, list)
    assert native.search_results[0]["metadata"]["source_unit_id"] == "session-a"
    assert native.memory_internal_usage["calls"] == 3

    history_path = Path(prepared["cleanup_manifest"]["local_paths"][0])
    assert history_path.is_file()
    owned_names = set(client.created)
    client.collections["foreign_collection"] = 4
    adapter.cleanup_unit(unit, question, config, run_context=context)

    assert set(client.deleted) == owned_names
    assert client.collections == {"foreign_collection": 4}
    assert not history_path.exists()


def test_mem0_internal_llm_usage_is_emitted_live_per_operation(
    tmp_path: Path,
) -> None:
    unit, conversation, questions = locomo_case()
    config = runtime_config(tmp_path, benchmark="locomo")
    context = run_context(tmp_path)
    client = FakeQdrantClient()
    builder = FakeMem0EngineBuilder()
    metrics = BenchmarkMetrics()
    adapter = adapter_for(
        benchmark="locomo",
        client=client,
        builder=builder,
        metrics=metrics,
    )

    prepared = adapter.prepare_unit(
        unit,
        conversation,
        config,
        run_context=context,
    )
    adapter.retrieve_question(
        unit,
        conversation,
        questions[0],
        config,
        prepared,
        run_context=context,
    )

    body = metrics.render().decode("utf-8")
    assert (
        'dmf_bench_llm_requests_total{outcome="completed",provider="openai",'
        'role="memory_internal"} 4.0'
    ) in body
    assert (
        'dmf_bench_llm_tokens_total{provider="openai",role="memory_internal",'
        'token_type="prompt"} 11.0'
    ) in body
    assert (
        'dmf_bench_llm_tokens_total{provider="openai",role="memory_internal",'
        'token_type="completion"} 3.0'
    ) in body
    assert (
        'dmf_bench_llm_tokens_total{provider="openai",role="memory_internal",'
        'token_type="total"} 14.0'
    ) in body


def test_mem0_resources_are_isolated_by_run_and_unit(tmp_path: Path) -> None:
    unit, question = longmemeval_case()
    config = runtime_config(tmp_path, benchmark="longmemeval")
    first = adapter_for(
        benchmark="longmemeval",
        client=FakeQdrantClient(),
        builder=FakeMem0EngineBuilder(),
    )
    second = adapter_for(
        benchmark="longmemeval",
        client=FakeQdrantClient(),
        builder=FakeMem0EngineBuilder(),
    )

    first_prepared = first.prepare_unit(
        unit, question, config, run_context=run_context(tmp_path, "a")
    )
    second_prepared = second.prepare_unit(
        unit, question, config, run_context=run_context(tmp_path, "b")
    )

    first_names = {
        item["name"] for item in first_prepared["cleanup_manifest"]["collections"]
    }
    second_names = {
        item["name"] for item in second_prepared["cleanup_manifest"]["collections"]
    }
    assert first_names.isdisjoint(second_names)
    assert set(first_prepared["cleanup_manifest"]["local_paths"]).isdisjoint(
        second_prepared["cleanup_manifest"]["local_paths"]
    )


def test_failed_ingestion_removes_partial_collections_and_history(tmp_path: Path) -> None:
    unit, conversation, _questions = locomo_case()
    config = runtime_config(tmp_path, benchmark="locomo")
    client = FakeQdrantClient()
    builder = FakeMem0EngineBuilder(fail_at=2)
    adapter = adapter_for(benchmark="locomo", client=client, builder=builder)

    with pytest.raises(RuntimeError, match="injected Mem0 ingestion failure"):
        adapter.prepare_unit(
            unit,
            conversation,
            config,
            run_context=run_context(tmp_path),
        )

    assert client.collections == {}
    assert set(client.deleted) == set(client.created)
    assert builder.history_path is not None
    assert not builder.history_path.exists()


def test_runtime_rejects_chroma_endpoints_and_dimension_mismatch() -> None:
    loaded = load_mem0_config(CONFIG_DIR / "locomo_mem0_qdrant_settings.yaml")

    chroma = deepcopy(loaded.memory_config)
    chroma["vector_store"]["provider"] = "chroma"
    with pytest.raises(Mem0RuntimeError, match="provider='qdrant'"):
        _validate_mem0_qdrant_config(Mem0Config(loaded.top_k, chroma))

    endpoint = deepcopy(loaded.memory_config)
    endpoint["vector_store"]["config"]["url"] = "http://qdrant:6333"
    with pytest.raises(Mem0RuntimeError, match="runtime-only fields: url"):
        _validate_mem0_qdrant_config(Mem0Config(loaded.top_k, endpoint))

    mismatch = deepcopy(loaded.memory_config)
    mismatch["embedder"]["config"]["embedding_dims"] = 384
    with pytest.raises(Mem0RuntimeError, match="must match"):
        _validate_mem0_qdrant_config(Mem0Config(loaded.top_k, mismatch))


def test_usage_sidecar_is_required_and_validated() -> None:
    class MissingUsage:
        pass

    class IncompatibleUsage:
        def get_llm_usage(self) -> dict[str, Any]:
            return {
                "prompt_tokens": 1,
                "completion_tokens": 0,
                "total_tokens": 1,
                "calls": 1,
            }

    with pytest.raises(Mem0RuntimeError, match="sidecar is unavailable"):
        Mem0RuntimeBackend(MissingUsage()).get_usage()
    with pytest.raises(Mem0RuntimeError, match="incompatible snapshot"):
        Mem0RuntimeBackend(IncompatibleUsage()).get_usage()


def test_add_rejects_swallowed_extraction_failure_without_usage() -> None:
    class SwallowedFailure:
        def get_llm_usage(self) -> dict[str, Any]:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "scopes": {},
            }

        def add(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            return {"results": []}

    with pytest.raises(Mem0RuntimeError, match="required LLM extraction call"):
        Mem0RuntimeBackend(SwallowedFailure()).add(
            [{"role": "user", "content": "remember this"}],
            user_id="user-1",
            timestamp=None,
            metadata={"source_unit_id": "item-1"},
        )


def test_mem0_additive_prompt_uses_dataset_observation_timestamp() -> None:
    from mem0.memory import main as mem0_memory_main

    original = mem0_memory_main.generate_additive_extraction_prompt

    class PromptCapturingMemory:
        prompt = ""

        def add(self, messages: list[dict[str, str]], **_kwargs: Any) -> dict[str, Any]:
            self.prompt = mem0_memory_main.generate_additive_extraction_prompt(
                new_messages=messages,
            )
            return {"results": [{"event": "ADD"}]}

    memory = PromptCapturingMemory()
    add_mem0_with_observation_timestamp(
        memory,
        [{"role": "user", "content": "I adopted Pixel."}],
        timestamp=1704067200,
        user_id="user-1",
    )

    observation = memory.prompt.split("## Observation Date\n", 1)[1].split(
        "\n\n", 1
    )[0]
    assert observation == "2024-01-01T00:00:00+00:00"
    assert mem0_memory_main.generate_additive_extraction_prompt is original


def test_factory_uses_qdrant_without_api_key_and_requires_disabled_telemetry(
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
    monkeypatch.setenv("MEM0_TELEMETRY", "false")
    adapter = mem0_framework_factories(
        engine_builder=FakeMem0EngineBuilder(),
        client_factory=client_factory,
    )["mem0"](config)

    assert adapter.qdrant_client is client
    assert calls == [("http://qdrant:6333", None, 10.0)]

    monkeypatch.delenv("MEM0_TELEMETRY")
    with pytest.raises(Mem0RuntimeError, match="requires MEM0_TELEMETRY=false"):
        Mem0QdrantFrameworkAdapter.from_experiment(
            config,
            engine_builder=FakeMem0EngineBuilder(),
            client_factory=client_factory,
        )


def test_tracked_mem0_qdrant_configs_have_no_local_fallback_fields() -> None:
    for path in (
        CONFIG_DIR / "locomo_mem0_qdrant_settings.yaml",
        CONFIG_DIR / "longmemeval_mem0_qdrant_settings.yaml",
    ):
        raw = path.read_text(encoding="utf-8").lower()
        loaded = load_mem0_config(path)

        assert "chroma" not in raw
        assert _validate_mem0_qdrant_config(loaded) == 768
        assert set(loaded.memory_config["vector_store"]["config"]) == {
            "embedding_model_dims",
            "on_disk",
        }


def test_mem0_provenance_records_pinned_distribution_and_fork_commit() -> None:
    config = {
        "benchmark": "locomo",
        "framework": "mem0",
        "framework_config": {
            "format": "yaml",
            "profile": "qdrant",
            "sha256": "a" * 64,
        },
    }

    provenance = build_run_provenance(config, fingerprint_inputs=config)

    assert provenance["scientific"]["framework_config"]["sha256"] == "a" * 64
    assert provenance["operational"]["framework"] == {
        "name": "mem0",
        "version": "2.0.2",
        "commit": "8db3430d20f8b76cb7f80fb30df048321863392f",
    }
