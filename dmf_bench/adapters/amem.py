"""Pinned A-MEM adapter with a private persistent Chroma index per unit."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.request import urlopen

from dmf_bench.adapters.base import (
    BenchmarkUnit, CanonicalRetrievalResult, FrameworkRunContext, MemoryEvent,
    MemoryQuery, OwnedResource, PreparedMemoryUnit, RetrievedMemory,
)
from dmf_bench.adapters.embedded_lifecycle import cleanup_owned_embedded_resource
from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.contracts import hash_canonical_json


AMEM_COMMIT = "ceffb860f0712bbae97b184d440df62bc910ca8d"
AMEM_VERSION = "0.0.1"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "host.docker.internal"})


class AMemRuntimeError(RuntimeError):
    """Raised when A-MEM cannot satisfy the retrieval lifecycle contract."""


def _load_profile(config: dict[str, Any]) -> dict[str, Any]:
    profile = json.loads(Path(config["framework_config"]["path"]).read_text(encoding="utf-8"))
    if (not isinstance(profile, dict) or profile.get("framework") != "amem"
            or profile.get("version") != "local-v1" or profile.get("source_commit") != AMEM_COMMIT):
        raise AMemRuntimeError("A-MEM profile does not match the pinned adapter.")
    if profile.get("llm_backend") != "ollama" or profile.get("embedding_dimensions") != 384:
        raise AMemRuntimeError("A-MEM requires the pinned local LLM and embedding profile.")
    for key in ("evolution_threshold", "evolution_neighbors", "direct_results", "linked_results"):
        if type(profile.get(key)) is not int or profile[key] < 1:
            raise AMemRuntimeError(f"A-MEM {key} must be a positive integer.")
    if profile["evolution_neighbors"] != 5:
        raise AMemRuntimeError("Upstream A-MEM evolution uses five neighbors.")
    return profile


def _model_path(profile: dict[str, Any]) -> Path:
    raw = os.getenv("AMEM_EMBEDDING_MODEL_PATH")
    if not raw:
        raise AMemRuntimeError("AMEM_EMBEDDING_MODEL_PATH is required.")
    root = Path(raw).resolve()
    model = profile["embedding"]
    if not root.is_dir() or model.get("weights_file") not in model.get("files", {}):
        raise AMemRuntimeError("Pinned A-MEM embedding directory is incomplete.")
    for name, expected in model["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(expected) != 64:
            raise AMemRuntimeError("Invalid A-MEM embedding manifest.")
        path = root / relative
        if not path.is_file():
            raise AMemRuntimeError(f"Missing A-MEM embedding file: {name}.")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise AMemRuntimeError(f"A-MEM embedding file differs from pin: {name}.")
    return root


def _preflight(profile: dict[str, Any]) -> tuple[Path, str]:
    if sys.version_info[:2] != (3, 11):
        raise AMemRuntimeError("A-MEM requires the dedicated Python 3.11 image.")
    if metadata.version("agentic-memory") != AMEM_VERSION or os.getenv("AMEM_SOURCE_COMMIT") != AMEM_COMMIT:
        raise AMemRuntimeError("A-MEM package version or source commit differs from the pin.")
    model_path = _model_path(profile)
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "http" or parsed.hostname not in _LOCAL_HOSTS or parsed.path not in ("", "/v1"):
        raise AMemRuntimeError("A-MEM Ollama endpoint must be local HTTP.")
    base = base.removesuffix("/v1")
    with urlopen(base + "/api/tags", timeout=10) as response:
        models = json.load(response).get("models", [])
    model = profile["llm"]
    if not any(item.get("name") == model["model"] and item.get("digest") == model["digest"] for item in models):
        raise AMemRuntimeError("Pinned A-MEM Ollama model is unavailable.")
    return model_path, base


def _build_engine(profile: dict[str, Any], root: Path, environment: tuple[Path, str]) -> Any:
    from agentic_memory.llm_controller import LLMController
    from agentic_memory.memory_system import AgenticMemorySystem
    from agentic_memory.retrievers import PersistentChromaRetriever
    from ollama import Client

    model_path, base = environment
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    class LocalAgenticMemory(AgenticMemorySystem):
        def __init__(self) -> None:
            self.memories = {}
            self.model_name = str(model_path)
            self.retriever = PersistentChromaRetriever(
                directory=str(root / "chroma"), collection_name="memories",
                model_name=self.model_name, extend=False,
            )
            self.llm_controller = LLMController("ollama", profile["llm"]["model"])
            self.evo_cnt = 0
            self.evo_threshold = profile["evolution_threshold"]

        def analyze_content(self, content: str) -> dict[str, Any]:
            schema = {"type": "object", "properties": {
                "keywords": {"type": "array", "items": {"type": "string"},
                             "minItems": 1, "maxItems": 5},
                "context": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"},
                         "minItems": 1, "maxItems": 5},
            }, "required": ["keywords", "context", "tags"], "additionalProperties": False}
            response = self.llm_controller.llm.get_completion(
                "Extract distinct keywords, a short context description, and topic tags "
                f"from this memory: {content}",
                {"type": "json_schema", "json_schema": {"name": "note", "schema": schema}},
            )
            attributes = json.loads(response)
            for name in ("keywords", "tags"):
                attributes[name] = list(dict.fromkeys(attributes[name]))
            return attributes

        def consolidate_memories(self) -> None:
            self.retriever.client.delete_collection(self.retriever.collection_name)
            self.retriever = PersistentChromaRetriever(
                directory=str(root / "chroma"), collection_name="memories",
                model_name=str(model_path), extend=False,
            )
            for note in self.memories.values():
                self.retriever.add_document(note.content, _note_metadata(note), note.id)

        def find_related_memories(self, query: str, k: int = 5) -> tuple[str, list[int]]:
            if not self.memories:
                return "", []
            try:
                results = self.retriever.search(query, min(k, len(self.memories)))
            except Exception as exc:
                self._dmf_errors.append(str(exc))
                raise
            ids = results["ids"][0]
            positions = {note_id: index for index, note_id in enumerate(self.memories)}
            lines = [
                f"memory id:{note_id}\tmemory index:{positions[note_id]}\t"
                f"memory content:{self.memories[note_id].content}\t"
                f"memory context:{self.memories[note_id].context}\t"
                f"memory tags:{self.memories[note_id].tags}"
                for note_id in ids if note_id in positions
            ]
            return "\n".join(lines), [positions[note_id] for note_id in ids if note_id in positions]

    engine = LocalAgenticMemory()
    engine._evolution_system_prompt = (
        "Decide whether the new memory should link to existing notes or update "
        "their context and tags. Use only exact memory ids shown below for links. "
        "New memory: {content}. Context: {context}. Keywords: {keywords}. "
        "Nearest notes ({neighbor_number}): {nearest_neighbors_memories}. "
        "Return JSON with should_evolve (boolean), actions (array of strengthen "
        "or update_neighbor), suggested_connections (array of exact memory ids), "
        "tags_to_update (array of strings), new_context_neighborhood (array of "
        "strings in neighbor order), and new_tags_neighborhood (array of string "
        "arrays in neighbor order)."
    )
    engine._dmf_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_ms": 0.0}
    engine._dmf_errors = []
    engine._dmf_last_evolution_decision = None
    client = Client(host=base)

    def completion(prompt: str, response_format: dict[str, Any], temperature: float = 0.7) -> str:
        started = time.perf_counter()
        try:
            schema = response_format["json_schema"]["schema"]
            if "should_evolve" in schema["properties"]:
                for field in schema["properties"].values():
                    if field.get("type") == "array":
                        field["maxItems"] = 5
                        if field.get("items", {}).get("type") == "array":
                            field["items"]["maxItems"] = 5
            result = client.chat(
                model=profile["llm"]["model"],
                messages=[{"role": "system", "content": "Respond with a JSON object."},
                          {"role": "user", "content": prompt}],
                format=schema,
                options={"temperature": 0, "seed": profile["llm"]["seed"],
                         "num_predict": profile["llm"]["max_tokens"]},
            )
            content = result["message"]["content"]
            parsed = json.loads(content)
            evolution = "should_evolve" in schema["properties"]
            if (not set(schema["properties"]).issubset(parsed)
                    or not _valid_llm_payload(parsed, evolution=evolution)):
                raise AMemRuntimeError("A-MEM LLM returned incomplete structured output.")
            if evolution:
                engine._dmf_last_evolution_decision = parsed["should_evolve"]
            engine._dmf_usage["calls"] += 1
            engine._dmf_usage["prompt_tokens"] += result.get("prompt_eval_count") or 0
            engine._dmf_usage["completion_tokens"] += result.get("eval_count") or 0
            return content
        except Exception as exc:
            engine._dmf_errors.append(str(exc))
            raise
        finally:
            engine._dmf_usage["total_ms"] += (time.perf_counter() - started) * 1000

    engine.llm_controller.llm.get_completion = completion
    return engine


def _note_metadata(note: Any) -> dict[str, Any]:
    return {key: getattr(note, key) for key in (
        "id", "content", "keywords", "links", "retrieval_count", "timestamp",
        "last_accessed", "context", "evolution_history", "category", "tags",
    )}


def _valid_llm_payload(payload: Any, *, evolution: bool) -> bool:
    if not isinstance(payload, dict):
        return False
    strings = lambda values: isinstance(values, list) and all(isinstance(value, str) for value in values)
    if evolution:
        return (
            type(payload.get("should_evolve")) is bool
            and strings(payload.get("actions"))
            and set(payload["actions"]).issubset({"strengthen", "update_neighbor"})
            and strings(payload.get("suggested_connections"))
            and strings(payload.get("tags_to_update"))
            and strings(payload.get("new_context_neighborhood"))
            and isinstance(payload.get("new_tags_neighborhood"), list)
            and all(strings(tags) for tags in payload["new_tags_neighborhood"])
        )
    return strings(payload.get("keywords")) and isinstance(payload.get("context"), str) and strings(payload.get("tags"))


def _state(engine: Any) -> dict[str, Any]:
    rows = engine.retriever.collection.get(include=["documents", "metadatas"])
    return {note_id: {"document": document, "metadata": row}
            for note_id, document, row in zip(rows["ids"], rows["documents"], rows["metadatas"])}


def _state_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _close_chroma(engine: Any, root: Path) -> None:
    client = getattr(getattr(engine, "retriever", None), "client", None)
    identifier = getattr(client, "_identifier", None)
    if identifier is None and not (root / "chroma").exists():
        return
    expected = str(root / "chroma")
    if identifier is not None and identifier != expected:
        raise AMemRuntimeError("A-MEM Chroma client is outside the owned resource.")
    from chromadb.api.client import SharedSystemClient

    system = SharedSystemClient._identifier_to_system.pop(expected, None)
    if system is not None:
        system.stop()
    if client is not None:
        engine.retriever.client = None
        engine.retriever.collection = None


@dataclass
class _Handle:
    unit_id: str
    engine: Any
    digest: str
    lineage: dict[str, str]
    timestamps: dict[str, str]


class AMemAdapter:
    name = "amem"
    adapter_version = "amem-v1"

    def __init__(
        self, config: dict[str, Any], *,
        preflight: Callable[[dict[str, Any]], tuple[Path, str]] = _preflight,
        engine_factory: Callable[[dict[str, Any], Path, tuple[Path, str]], Any] = _build_engine,
    ) -> None:
        self.profile = _load_profile(config)
        self.environment = preflight(self.profile)
        self.engine_factory = engine_factory
        self._engines: dict[str, Any] = {}

    def resources_for_unit_v3(self, unit: BenchmarkUnit, config: dict[str, Any],
                              run_context: FrameworkRunContext) -> tuple[OwnedResource, ...]:
        fragment = hashlib.sha256(unit.unit_id.encode("utf-8")).hexdigest()[:24]
        return (OwnedResource(f"{run_context.run_id}:{unit.unit_id}:amem-state", "embedded-path",
                              "primary", f"amem/{fragment}"),)

    def cleanup_unit_v3(self, unit: BenchmarkUnit, resources: tuple[OwnedResource, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if resources != self.resources_for_unit_v3(unit, config, run_context):
            raise AMemRuntimeError("A-MEM cleanup manifest differs from the owned unit.")
        engine = self._engines.pop(resources[0].resource_id, None)
        root = run_context.run_dir / resources[0].locator
        _close_chroma(engine, root)
        run_context.run_dir.mkdir(parents=True, exist_ok=True)
        evidence = cleanup_owned_embedded_resource(resources[0], root=run_context.run_dir,
                                                   owned_resources=resources)
        return {"verified": True, "cleanup": evidence}

    def prepare_unit_v3(self, unit: BenchmarkUnit, events: tuple[MemoryEvent, ...],
                        config: dict[str, Any], run_context: FrameworkRunContext) -> PreparedMemoryUnit:
        if not isinstance(events, tuple) or any(not isinstance(event, MemoryEvent) for event in events):
            raise AMemRuntimeError("A-MEM requires normalized memory events.")
        resources = self.resources_for_unit_v3(unit, config, run_context)
        root = run_context.run_dir / resources[0].locator
        if root.exists():
            raise AMemRuntimeError("A-MEM unit state already exists before ingestion.")
        root.mkdir(parents=True)
        started = time.perf_counter()
        engine: Any = None
        try:
            engine = self.engine_factory(self.profile, root, self.environment)
            lineage: dict[str, str] = {}
            timestamps: dict[str, str] = {}
            for event in events:
                content = f"[{event.role}] {event.content}"
                attributes = engine.analyze_content(content)
                if engine._dmf_errors or not isinstance(attributes, dict) or not attributes.get("keywords"):
                    raise AMemRuntimeError("A-MEM note analysis did not complete.")
                timestamp = datetime.fromisoformat((event.occurred_at or "2000-01-01T00:00:00Z").replace("Z", "+00:00"))
                previous_evolutions = engine.evo_cnt
                engine._dmf_last_evolution_decision = None
                note_id = engine.add_note(
                    content, time=timestamp.strftime("%Y%m%d%H%M"),
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL,
                                     f"{run_context.scientific_fingerprint}:{unit.unit_id}:{event.event_id}")),
                    last_accessed=timestamp.strftime("%Y%m%d%H%M"),
                    context=attributes["context"], keywords=attributes["keywords"], tags=attributes["tags"],
                    category=event.role,
                )
                if engine._dmf_errors or note_id in lineage or engine.read(note_id) is None:
                    raise AMemRuntimeError("A-MEM note evolution did not complete.")
                decision = getattr(engine, "_dmf_last_evolution_decision", None)
                if decision is not None and engine.evo_cnt - previous_evolutions != int(decision):
                    raise AMemRuntimeError("A-MEM evolution decision was not applied.")
                lineage[note_id] = event.event_id
                timestamps[note_id] = event.occurred_at or "2000-01-01T00:00:00Z"
            valid_ids = set(lineage)
            invalid_links = 0
            for note in engine.memories.values():
                before = len(note.links)
                note.links = list(dict.fromkeys(link for link in note.links if link in valid_ids and link != note.id))
                invalid_links += before - len(note.links)
            engine.consolidate_memories()
            state = _state(engine)
            if set(state) != valid_ids:
                raise AMemRuntimeError("A-MEM Chroma index is incomplete after evolution.")
            digest = hash_canonical_json(state)
            write_json_atomic(root / "lineage.json", {"source_event_ids": lineage,
                                                      "source_timestamps": timestamps,
                                                      "state_sha256": digest})
            handle = _Handle(unit.unit_id, engine, digest, lineage, timestamps)
            self._engines[resources[0].resource_id] = engine
            return PreparedMemoryUnit(
                handle, resources, ingestion_usage={"amem": dict(engine._dmf_usage)},
                ingestion_timing={"total_ms": (time.perf_counter() - started) * 1000},
                diagnostics={"event_count": len(events), "indexed_notes": len(state),
                             "linked_notes": sum(bool(note.links) for note in engine.memories.values()),
                             "invalid_links_removed": invalid_links, "evolution_decisions": engine.evo_cnt,
                             "state_bytes": _state_bytes(root), "state_sha256": digest,
                             "source_commit": AMEM_COMMIT},
            )
        except Exception:
            if engine is not None:
                self._engines[resources[0].resource_id] = engine
            self.cleanup_unit_v3(unit, resources, config, run_context)
            raise

    def verify_prepared_v3(self, unit: BenchmarkUnit, prepared: PreparedMemoryUnit,
                           config: dict[str, Any], run_context: FrameworkRunContext) -> dict[str, Any]:
        if prepared.resources != self.resources_for_unit_v3(unit, config, run_context):
            raise AMemRuntimeError("A-MEM resource manifest mismatch.")
        handle = prepared.handle
        if not isinstance(handle, _Handle) or handle.unit_id != unit.unit_id:
            raise AMemRuntimeError("A-MEM unit handle mismatch.")
        current = _state(handle.engine)
        if hash_canonical_json(current) != handle.digest or set(current) != set(handle.lineage):
            raise AMemRuntimeError("A-MEM committed index differs from prepared state.")
        root = run_context.run_dir / prepared.resources[0].locator
        manifest = json.loads((root / "lineage.json").read_text(encoding="utf-8"))
        if manifest != {"source_event_ids": handle.lineage,
                        "source_timestamps": handle.timestamps,
                        "state_sha256": handle.digest}:
            raise AMemRuntimeError("A-MEM lineage evidence changed.")
        return {"verified": True, "entry_count": len(current), "state_sha256": handle.digest}

    def retrieve_v3(self, unit: BenchmarkUnit, query: MemoryQuery, prepared: PreparedMemoryUnit,
                    config: dict[str, Any], run_context: FrameworkRunContext) -> CanonicalRetrievalResult:
        if not isinstance(query, MemoryQuery) or query.query_id not in unit.item_ids:
            raise AMemRuntimeError("A-MEM query does not belong to prepared unit.")
        self.verify_prepared_v3(unit, prepared, config, run_context)
        started = time.perf_counter()
        handle: _Handle = prepared.handle
        limit = config["retrieval"]["max_results"]
        direct_count = min(self.profile["direct_results"], limit, len(handle.lineage))
        if not direct_count:
            return CanonicalRetrievalResult((), raw_payload={"ids": [[]], "distances": [[]]})
        results = handle.engine.retriever.search(query.text, direct_count)
        ranked: list[tuple[str, float | None, bool]] = []
        for note_id, distance in zip(results["ids"][0], results["distances"][0]):
            if note_id in handle.lineage and note_id not in {row[0] for row in ranked}:
                ranked.append((note_id, float(distance), False))
        direct_len = len(ranked)
        for note_id, _, _ in tuple(ranked):
            note = handle.engine.read(note_id)
            for neighbor_id in note.links:
                if len(ranked) >= limit or len(ranked) - direct_len >= self.profile["linked_results"]:
                    break
                if neighbor_id in handle.lineage and neighbor_id not in {row[0] for row in ranked}:
                    ranked.append((neighbor_id, None, True))
        items = []
        for rank, (note_id, distance, linked) in enumerate(ranked, 1):
            note = handle.engine.read(note_id)
            items.append(RetrievedMemory(
                memory_id=note_id, content=note.content, rank=rank,
                native_score=distance, native_score_kind="chroma-l2-distance" if distance is not None else None,
                occurred_at=handle.timestamps[note_id],
                source_event_ids=(handle.lineage[note_id],),
                metadata={"context": note.context, "keywords": note.keywords, "tags": note.tags,
                          "linked": linked, "lineage_status": "exact"},
            ))
        return CanonicalRetrievalResult(
            tuple(items), raw_payload={"ids": results["ids"], "distances": results["distances"]},
            diagnostics={"amem": {"indexed_notes": len(handle.lineage), "source_commit": AMEM_COMMIT}},
            usage={"amem": dict(handle.engine._dmf_usage)},
            timing={"retrieval_pipeline_ms": (time.perf_counter() - started) * 1000},
        )
