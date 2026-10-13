"""MemoryAgentBench adapter with isolated references and inject-once units."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dmf_bench.adapters.base import (
    AnswererRequest, BenchmarkCase, BenchmarkUnit, EvaluationReference,
    MemoryEvent, MemoryQuery,
)
from dmf_bench.contracts import V3_PREDICTION_SCHEMA_VERSION, sha256_file

from .spec import CODE_REVISION, DATASET_REVISION, PROFILES, format_for_task, task_for_source


def _chunks(text: str, size: int = 1024) -> tuple[str, ...]:
    """Partition the exact source text without dropping or rewriting a character."""
    return tuple(text[index:index + size] for index in range(0, len(text), size))


class MemoryAgentBenchAdapter:
    name = "memoryagentbench"
    atomic_unit = "memoryagentbench-context"

    def _manifest(self, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        profile = config.get("scientific_profile")
        if profile not in PROFILES:
            raise ValueError(f"Unsupported MemoryAgentBench profile: {profile!r}")
        dataset = config["dataset"]
        path = Path(dataset["path"])
        if not path.is_file() or sha256_file(path) != dataset["sha256"]:
            raise ValueError("MemoryAgentBench dataset manifest is absent or has a SHA-256 mismatch.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("benchmark") != self.name
                or manifest.get("dataset_revision") != DATASET_REVISION
                or manifest.get("code_revision") != CODE_REVISION):
            raise ValueError("MemoryAgentBench manifest release does not match the pinned adapter.")
        return path, manifest

    def enumerate_units(self, config: dict[str, Any]) -> list[BenchmarkUnit]:
        _path, manifest = self._manifest(config)
        records = manifest.get("records")
        if not isinstance(records, list):
            raise ValueError("MemoryAgentBench manifest requires ordered records.")
        by_id = {record["unit_id"]: record for record in records}
        if len(by_id) != len(records):
            raise ValueError("MemoryAgentBench manifest has duplicate unit IDs.")
        selection = config["selection"]
        selected_ids = selection["ordered_item_ids"]
        missing = [unit_id for unit_id in selected_ids if unit_id not in by_id]
        if missing:
            raise ValueError(f"MemoryAgentBench selection has missing units: {missing}")
        filters = selection.get("filters", {})
        if set(filters) - {"max_queries_per_unit"}:
            raise ValueError("MemoryAgentBench selection has unsupported filters.")
        limit = filters.get("max_queries_per_unit")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_queries_per_unit must be a positive integer.")
        units = []
        for unit_id in selected_ids:
            record = by_id[unit_id]
            task, _metric = task_for_source(record["source"])
            if config["scientific_profile"] == "memoryagentbench-ar-pilot-v1" and record["split"] != "Accurate_Retrieval":
                raise ValueError("AR pilot profile only accepts Accurate Retrieval units.")
            count = min(record["query_count"], limit or record["query_count"])
            query_ids = tuple(f"{unit_id}--q{index:03d}" for index in range(count))
            units.append(BenchmarkUnit(unit_id, query_ids, {
                "benchmark": self.name,
                "unit_type": self.atomic_unit,
                "source": record["source"],
                "task": task,
                "split": record["split"],
            }))
        return units

    def expected_item_ids(self, config: dict[str, Any]) -> tuple[str, ...]:
        return tuple(query_id for unit in self.enumerate_units(config) for query_id in unit.item_ids)

    def load_case(self, unit: BenchmarkUnit, config: dict[str, Any]) -> BenchmarkCase:
        manifest_path, manifest = self._manifest(config)
        record = next((item for item in manifest["records"] if item["unit_id"] == unit.unit_id), None)
        if record is None:
            raise ValueError(f"Unknown MemoryAgentBench unit: {unit.unit_id}")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("MemoryAgentBench unit path must be relative and contained.")
        path = manifest_path.parent / relative
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"MemoryAgentBench unit SHA-256 mismatch: {unit.unit_id}")
        row = json.loads(path.read_text(encoding="utf-8"))
        context = row["context"]
        if len(context) != record["context_chars"]:
            raise ValueError("MemoryAgentBench context length differs from manifest.")
        questions, answers = row["questions"], row["answers"]
        metadata = row["metadata"]
        if (len(questions) != record["query_count"] or len(answers) != len(questions)
                or len(metadata["qa_pair_ids"]) != len(questions)):
            raise ValueError("MemoryAgentBench question/reference alignment differs from manifest.")
        task, metric = task_for_source(record["source"])
        chunks = _chunks(context)
        events = tuple(MemoryEvent(
            event_id=f"{unit.unit_id}--e{index:05d}",
            session_id=unit.unit_id,
            sequence=index,
            role="observation",
            content=chunk,
            occurred_at=None,
            source_refs=(f"{unit.unit_id}--e{index:05d}",),
            metadata={"source": record["source"], "chunk_index": index},
        ) for index, chunk in enumerate(chunks))
        queries = []
        references = {}
        entity_names = None
        if task == "recsys":
            entity_path = manifest_path.parent / "entity2id.json"
            if sha256_file(entity_path) != manifest["entity2id_sha256"]:
                raise ValueError("MemoryAgentBench entity mapping SHA-256 mismatch.")
            entities = json.loads(entity_path.read_text(encoding="utf-8"))
            entity_names = {str(value): _movie_name(key) for key, value in entities.items()}
        for index, query_id in enumerate(unit.item_ids):
            query = MemoryQuery(query_id, str(questions[index]), None,
                                {"source": record["source"], "task": task})
            queries.append(query)
            expected: Any = answers[index]
            if task == "recsys":
                expected = {
                    "ids": expected,
                    "names": [entity_names[str(value)] for value in expected],
                }
            references[query_id] = EvaluationReference(
                query_id=query_id,
                expected_answer=expected,
                evidence_refs=(),
                strata={
                    "competence": record["split"],
                    "task": task,
                    "source": record["source"],
                    "official_metric": metric,
                    "official_qa_pair_id": metadata["qa_pair_ids"][index],
                    "context_chars": len(context),
                    "context_chunks": len(chunks),
                    "answer_format": format_for_task(task),
                    **({
                        "question_type": metadata["question_types"][index],
                        "question_date": metadata["question_dates"][index],
                        "is_abstention": str(metadata["question_ids"][index]).endswith("_abs"),
                    } if task == "longmemeval" else {}),
                },
            )
        return BenchmarkCase(unit, events, tuple(queries), references)

    def build_answerer_request_v3(self, query: MemoryQuery, packed_context: Any, config: dict[str, Any]) -> AnswererRequest:
        del config
        task = query.metadata["task"]
        instruction = {
            "eventqa": "Based on the memory, identify the event that happens next. Give only the event.",
            "ruler_qa": "Answer from the memorized documents. Give only the answer.",
            "longmemeval": "Answer from the relevant chat history as concisely as possible.",
            "icl": 'Use the examples to assign a numerical label. Output only "label: N".',
            "recsys": "Recommend 20 movies in a numbered list based on the memorized dialogues.",
            "detective_qa": "Answer from the story using the question's strict output format.",
            "factconsolidation": "Answer only from the facts. Newer facts have larger serial numbers and resolve conflicts. Give only the concise answer.",
        }[task]
        return AnswererRequest(
            system_prompt="You are a helpful assistant that can read the context and memorize it for future retrieval.",
            user_prompt=f"{instruction}\n\nMemory:\n{packed_context.text}\n\nQuestion: {query.text}\nAnswer:",
            metadata={"question_id": query.query_id, "task": task},
        )

    def build_prediction_v3(
        self, case: BenchmarkCase, query: MemoryQuery, retrieval: Any,
        packed_context: Any, answer: dict[str, Any],
    ) -> dict[str, Any]:
        reference = case.references[query.query_id]
        return {
            "schema_version": V3_PREDICTION_SCHEMA_VERSION,
            "benchmark": self.name,
            "question_id": query.query_id,
            "question": query.text,
            "ground_truth_answer": reference.expected_answer,
            **reference.strata,
            "generated_answer": str(answer.get("generated_answer", answer.get("answer", ""))),
            "answerer_usage": answer.get("answerer_usage", answer.get("usage", {})),
            "answerer_provider": answer.get("answerer_provider", answer.get("provider", "")),
            "answerer_requested_model": answer.get("answerer_requested_model", ""),
            "answerer_model": answer.get("answerer_model", answer.get("model", "")),
            "answerer_finish_reason": answer.get("answerer_finish_reason"),
            "retrieval": retrieval.to_dict(),
            "packed_context": packed_context.to_dict(),
            "protocol_label": "MemoryAgentBench retrieval-controlled adaptation",
        }


def _movie_name(value: str) -> str:
    filename = value.split("/")[-1]
    filename = filename.replace("_", " ").replace("-", " ").replace(">", " ")
    import re
    return " ".join(re.sub(r"\([^()]*\)", "", filename).split())
