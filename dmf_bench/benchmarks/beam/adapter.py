"""BEAM conversation adapter with evaluator-only rubric isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dmf_bench.adapters.base import (
    AnswererRequest, BenchmarkCase, BenchmarkUnit, EvaluationReference,
    MemoryEvent, MemoryQuery,
)
from dmf_bench.contracts import V3_PREDICTION_SCHEMA_VERSION, sha256_file

from .spec import ABILITIES, CODE_REVISION, DATASET_REVISIONS, PROFILES, PROTOCOL_LABEL


class BeamAdapter:
    name = "beam"
    atomic_unit = "beam-conversation"

    def _manifest(self, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        profile = config.get("scientific_profile")
        if profile not in PROFILES:
            raise ValueError(f"Unsupported BEAM profile: {profile!r}")
        dataset = config["dataset"]
        path = Path(dataset["path"])
        if not path.is_file() or sha256_file(path) != dataset["sha256"]:
            raise ValueError("BEAM manifest is absent or has a SHA-256 mismatch.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("benchmark") != self.name
                or manifest.get("code_revision") != CODE_REVISION
                or manifest.get("dataset_revisions") != DATASET_REVISIONS
                or manifest.get("abilities") != list(ABILITIES)):
            raise ValueError("BEAM manifest differs from the pinned release.")
        return path, manifest

    def enumerate_units(self, config: dict[str, Any]) -> list[BenchmarkUnit]:
        _, manifest = self._manifest(config)
        records = manifest["records"]
        by_id = {record["unit_id"]: record for record in records}
        if len(by_id) != len(records):
            raise ValueError("BEAM manifest has duplicate unit IDs.")
        selection = config["selection"]
        selected = selection["ordered_item_ids"]
        if selected == ["*"]:
            selected = [record["unit_id"] for record in records if record["scale"] == PROFILES[config["scientific_profile"]]]
        limit = selection.get("filters", {}).get("max_queries_per_unit")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_queries_per_unit must be a positive integer.")
        if set(selection.get("filters", {})) - {"max_queries_per_unit"}:
            raise ValueError("Unsupported BEAM selection filter.")
        units = []
        for unit_id in selected:
            if unit_id not in by_id:
                raise ValueError(f"Unknown BEAM unit: {unit_id}")
            record = by_id[unit_id]
            if record["scale"] != PROFILES[config["scientific_profile"]]:
                raise ValueError("BEAM unit scale differs from scientific profile.")
            query_ids = tuple(
                f"{unit_id}--q-{ability}-{index + 1:02d}"
                for ability in ABILITIES for index in range(2)
            )[:limit]
            units.append(BenchmarkUnit(unit_id, query_ids, {
                "benchmark": self.name, "unit_type": self.atomic_unit,
                "scale": record["scale"], "category": record["category"],
            }))
        return units

    def expected_item_ids(self, config: dict[str, Any]) -> tuple[str, ...]:
        return tuple(query_id for unit in self.enumerate_units(config) for query_id in unit.item_ids)

    def load_case(self, unit: BenchmarkUnit, config: dict[str, Any]) -> BenchmarkCase:
        manifest_path, manifest = self._manifest(config)
        record = next((record for record in manifest["records"] if record["unit_id"] == unit.unit_id), None)
        if record is None:
            raise ValueError(f"Unknown BEAM unit: {unit.unit_id}")

        def load(name: str) -> list[dict[str, Any]]:
            relative = Path(record[f"{name}_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("BEAM materialized path must stay inside the dataset root.")
            path = manifest_path.parent / relative
            if not path.is_file() or sha256_file(path) != record[f"{name}_sha256"]:
                raise ValueError(f"BEAM {name} SHA-256 mismatch: {unit.unit_id}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"BEAM {name} file must contain a list.")
            return payload

        raw_events = load("events")
        raw_queries = load("queries")
        raw_references = load("references")
        if (len(raw_events) != record["event_count"]
                or len(raw_queries) != record["question_count"]
                or len(raw_references) != len(raw_queries)):
            raise ValueError("BEAM materialized counts differ from manifest.")
        query_by_id = {row["query_id"]: row for row in raw_queries}
        reference_by_id = {row["query_id"]: row for row in raw_references}
        if len(query_by_id) != len(raw_queries) or len(reference_by_id) != len(raw_references):
            raise ValueError("BEAM materialized query IDs are not unique.")
        events = tuple(MemoryEvent(**{**row, "source_refs": tuple(row["source_refs"])})
                       for row in raw_events)
        queries = []
        references = {}
        for query_id in unit.item_ids:
            row = query_by_id[query_id]
            reference = reference_by_id[query_id]
            queries.append(MemoryQuery(query_id, row["text"], None, {"ability": row["ability"]}))
            references[query_id] = EvaluationReference(
                query_id, reference["expected_answer"], (),
                {"ability": row["ability"], "difficulty": row["difficulty"],
                 "scale": record["scale"], "category": record["category"],
                 "context_chars": record["context_chars"],
                 "rubric": reference["rubric"],
                 "source_chat_ids": reference["source_chat_ids"]},
            )
        return BenchmarkCase(unit, events, tuple(queries), references)

    def build_answerer_request_v3(self, query: MemoryQuery, packed_context: Any,
                                  config: dict[str, Any]) -> AnswererRequest:
        del config
        has_memory = bool(packed_context.text)
        return AnswererRequest(
            system_prompt=("Answer the user's question using the available conversation memory. "
                           "Follow any applicable user preferences and instructions.")
            if has_memory else "Answer the user's question.",
            user_prompt=(f"Conversation memory:\n{packed_context.text}\n\n"
                         f"Question: {query.text}\nAnswer:") if has_memory else query.text,
            metadata={"question_id": query.query_id, "ability": query.metadata["ability"]},
        )

    def build_prediction_v3(self, case: BenchmarkCase, query: MemoryQuery,
                            retrieval: Any, packed_context: Any,
                            answer: dict[str, Any]) -> dict[str, Any]:
        reference = case.references[query.query_id]
        strata = dict(reference.strata)
        rubric = strata.pop("rubric")
        source_chat_ids = strata.pop("source_chat_ids")
        return {
            "schema_version": V3_PREDICTION_SCHEMA_VERSION,
            "benchmark": self.name, "question_id": query.query_id,
            "question": query.text, "ground_truth_answer": reference.expected_answer,
            "ground_truth_rubric": rubric, "source_chat_ids": source_chat_ids,
            **strata,
            "generated_answer": str(answer.get("generated_answer", answer.get("answer", ""))),
            "answerer_usage": answer.get("answerer_usage", answer.get("usage", {})),
            "answerer_provider": answer.get("answerer_provider", answer.get("provider", "")),
            "answerer_requested_model": answer.get("answerer_requested_model", ""),
            "answerer_model": answer.get("answerer_model", answer.get("model", "")),
            "answerer_finish_reason": answer.get("answerer_finish_reason"),
            "retrieval": retrieval.to_dict(), "packed_context": packed_context.to_dict(),
            "protocol_label": PROTOCOL_LABEL,
        }
