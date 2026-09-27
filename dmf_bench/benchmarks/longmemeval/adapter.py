"""LongMemEval benchmark adapter for the v2 lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dmf_bench.adapters.base import (
    BenchmarkCase,
    BenchmarkUnit,
    EvaluationReference,
    MemoryEvent,
    MemoryQuery,
)
from . import prompts
from .dataset import (
    filter_questions_by_ids,
    load_dataset,
    parse_longmemeval_date,
    sample_questions_stratified,
    sort_sessions_chronologically,
)

from dmf_bench.contracts import PREDICTION_SCHEMA_VERSION, V3_PREDICTION_SCHEMA_VERSION, sha256_file



@dataclass(frozen=True)
class LongMemEvalReference:
    path: Path
    sha256: str
    questions: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LongMemEvalAnswererInput:
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


class LongMemEvalAdapter:
    name = "longmemeval"
    atomic_unit = "longmemeval-question"

    def build_answerer_request_v3(self, query: MemoryQuery, packed_context: Any, config: dict[str, Any]) -> Any:
        """Build the shared LongMemEval answer prompt from canonical packed memory."""
        from dmf_bench.adapters.base import AnswererRequest

        del config
        return AnswererRequest(
            system_prompt=prompts.build_answerer_system_prompt(),
            user_prompt=(
                "Use the provided memory to answer the LongMemEval question. "
                "If the answer is unsupported, say you do not know.\n\n"
                f"Memory:\n{packed_context.text}\n\n"
                f"Question date: {query.as_of or '(not specified)'}\n"
                f"Question: {query.text}\nAnswer:"
            ),
            metadata={"question_id": query.query_id},
        )

    def build_prediction_v3(
        self, case: BenchmarkCase, query: MemoryQuery, retrieval: Any,
        packed_context: Any, answer: dict[str, Any],
    ) -> dict[str, Any]:
        """Add the official reference only after answer generation."""
        reference = case.references[query.query_id]
        return {
            "schema_version": V3_PREDICTION_SCHEMA_VERSION,
            "benchmark": self.name,
            "question_id": query.query_id,
            "question_type": reference.strata["question_type"],
            "question": query.text,
            "ground_truth_answer": reference.expected_answer,
            "question_date": query.as_of,
            "is_abstention": reference.strata["is_abstention"],
            "answer_session_ids": list(reference.evidence_refs),
            "generated_answer": str(answer.get("generated_answer", answer.get("answer", ""))),
            "answerer_usage": answer.get("answerer_usage", answer.get("usage", {})),
            "answerer_provider": answer.get("answerer_provider", answer.get("provider", "")),
            "answerer_requested_model": answer.get("answerer_requested_model", ""),
            "answerer_model": answer.get("answerer_model", answer.get("model", "")),
            "answerer_finish_reason": answer.get("answerer_finish_reason"),
            "retrieval": retrieval.to_dict(),
            "packed_context": packed_context.to_dict(),
        }

    def load_case(self, unit: BenchmarkUnit, config: dict[str, Any]) -> BenchmarkCase:
        """Materialize one question with its answer held only in the reference."""
        selected = self.selected_questions_by_id(config)
        if unit.unit_id not in selected:
            raise ValueError(f"Unknown selected LongMemEval unit: {unit.unit_id}")
        question = selected[unit.unit_id]
        if unit.item_ids != (unit.unit_id,):
            raise ValueError("LongMemEval unit must contain its single question ID.")
        session_ids = question["haystack_session_ids"]
        dates = question["haystack_dates"]
        sessions = question["haystack_sessions"]
        if not isinstance(session_ids, list) or not isinstance(dates, list) or not isinstance(sessions, list):
            raise ValueError("LongMemEval haystack fields must be lists.")
        if len(session_ids) != len(dates) or len(dates) != len(sessions):
            raise ValueError("LongMemEval haystack fields must have equal lengths.")

        events: list[MemoryEvent] = []
        for session_id, raw_date, turns in sort_sessions_chronologically(question):
            if not isinstance(turns, list):
                raise ValueError("LongMemEval haystack session must be a list.")
            epoch = parse_longmemeval_date(raw_date)
            occurred_at = datetime.fromtimestamp(epoch, UTC).isoformat().replace("+00:00", "Z") if epoch is not None else None
            for sequence, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError("LongMemEval turn must be an object.")
                role = str(turn.get("role", "") or "").strip().lower()
                content = str(turn.get("content", "") or "").strip()
                if not content:
                    continue
                events.append(MemoryEvent(
                    event_id=f"{unit.unit_id}:{session_id}:{sequence}",
                    session_id=str(session_id),
                    sequence=sequence,
                    role=role,
                    content=content,
                    occurred_at=occurred_at,
                    source_refs=(str(session_id),),
                ))

        question_epoch = parse_longmemeval_date(str(question.get("question_date", "")))
        as_of = datetime.fromtimestamp(question_epoch, UTC).isoformat().replace("+00:00", "Z") if question_epoch is not None else None
        query = MemoryQuery(
            query_id=unit.unit_id,
            text=_string(question, "question"),
            as_of=as_of,
        )
        reference = EvaluationReference(
            query_id=unit.unit_id,
            expected_answer=question["answer"],
            evidence_refs=tuple(str(value) for value in question.get("answer_session_ids", [])),
            strata={
                "question_type": str(question["question_type"]),
                "is_abstention": unit.unit_id.endswith("_abs"),
            },
        )
        safe_unit = BenchmarkUnit(unit.unit_id, unit.item_ids, {
            "benchmark": self.name,
            "unit_type": self.atomic_unit,
            "question_id": unit.unit_id,
        })
        return BenchmarkCase(safe_unit, tuple(events), (query,), {unit.unit_id: reference})

    def materialize_reference(self, config: dict[str, Any]) -> LongMemEvalReference:
        """Load a local pinned dataset; never downloads implicitly."""
        dataset_config = _mapping(config, "dataset")
        dataset_path = Path(_string(dataset_config, "path"))
        if not dataset_path.is_file():
            raise FileNotFoundError(f"LongMemEval dataset file not found: {dataset_path}")

        actual_sha256 = sha256_file(dataset_path)
        expected_sha256 = dataset_config.get("sha256")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                "LongMemEval dataset SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        questions = load_dataset(str(dataset_path))
        if not isinstance(questions, list):
            raise ValueError("LongMemEval dataset root must be a JSON array.")
        for question in questions:
            _validate_question(question)
        return LongMemEvalReference(
            path=dataset_path,
            sha256=actual_sha256,
            questions=tuple(questions),
        )

    def enumerate_units(self, config: dict[str, Any]) -> list[BenchmarkUnit]:
        reference = self.materialize_reference(config)
        return [
            BenchmarkUnit(
                unit_id=str(question["question_id"]),
                item_ids=(str(question["question_id"]),),
                metadata=self.metadata_for_question(question),
            )
            for question in self.select_questions(reference.questions, config)
        ]

    def expected_item_ids(self, config: dict[str, Any]) -> tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.enumerate_units(config))

    def selected_questions_by_id(self, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        reference = self.materialize_reference(config)
        return {
            str(question["question_id"]): question
            for question in self.select_questions(reference.questions, config)
        }

    def select_questions(
        self,
        questions: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        selection = _mapping(config, "selection")
        ordered_item_ids = tuple(str(item) for item in selection.get("ordered_item_ids", ()))
        if not ordered_item_ids:
            raise ValueError("selection.ordered_item_ids must contain at least one question id.")

        if ordered_item_ids != ("*",):
            selected = filter_questions_by_ids(list(questions), list(ordered_item_ids))
            selected_ids = {str(question["question_id"]) for question in selected}
            missing = [item_id for item_id in ordered_item_ids if item_id not in selected_ids]
            if missing:
                raise ValueError(f"LongMemEval selection references missing question ids: {missing}")
            return selected

        filtered = list(questions)
        filters = selection.get("filters") or {}
        if not isinstance(filters, dict):
            raise ValueError("selection.filters must be an object when provided.")

        question_types = filters.get("question_types") or filters.get("types")
        if question_types is not None:
            if not isinstance(question_types, list) or not all(
                isinstance(item, str) and item for item in question_types
            ):
                raise ValueError("selection.filters.question_types must be a list of strings.")
            allowed_types = set(question_types)
            filtered = [
                question
                for question in filtered
                if str(question.get("question_type")) in allowed_types
            ]

        sample_per_type = selection.get("sample_per_type", filters.get("sample_per_type"))
        if sample_per_type is not None:
            seed = selection.get("seed", 42)
            seed = 42 if seed is None else int(seed)
            filtered = sample_questions_stratified(
                filtered,
                per_type=int(sample_per_type),
                seed=seed,
                selected_types=list(question_types) if question_types else None,
            )

        if not filtered:
            raise ValueError("LongMemEval selection produced no questions.")
        return filtered

    def metadata_for_question(self, question: dict[str, Any]) -> dict[str, Any]:
        return {
            "benchmark": self.name,
            "unit_type": self.atomic_unit,
            "question_id": str(question["question_id"]),
            "question_type": str(question["question_type"]),
            "question": str(question["question"]),
            "ground_truth_answer": str(question["answer"]),
            "question_date": str(question.get("question_date", "")),
            "answer_session_ids": [str(item) for item in question.get("answer_session_ids", [])],
        }

    def build_answerer_input(
        self,
        *,
        question: dict[str, Any],
        framework_name: str,
        retrieval: dict[str, Any],
    ) -> LongMemEvalAnswererInput:
        question_text = str(question["question"])
        question_date = str(question.get("question_date", ""))
        native_context = retrieval.get("native_context", "")
        user_prompt = prompts.build_answerer_user_prompt(
            native_context,
            question_text,
            question_date,
        )
        metadata = {
            "framework": framework_name,
            "question_id": str(question["question_id"]),
            "native_context": native_context,
            "native_surface_diagnostics": retrieval.get("native_surface_diagnostics", {}),
        }
        return LongMemEvalAnswererInput(
            prompts.build_answerer_system_prompt(),
            user_prompt,
            metadata,
        )

    def build_prediction(
        self,
        *,
        question: dict[str, Any],
        framework_name: str,
        retrieval: dict[str, Any],
        answerer_input: LongMemEvalAnswererInput,
        answerer_output: dict[str, Any],
    ) -> dict[str, Any]:
        generated_answer = str(
            answerer_output.get("generated_answer", answerer_output.get("answer", ""))
        )
        provider = str(answerer_output.get("answerer_provider", answerer_output.get("provider", "")))
        model = str(answerer_output.get("answerer_model", answerer_output.get("model", "")))
        usage = answerer_output.get("answerer_usage", answerer_output.get("usage", {}))
        if not isinstance(usage, dict):
            usage = {}

        result = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "benchmark": self.name,
            "framework": framework_name,
            "question_id": str(question["question_id"]),
            "question_type": str(question["question_type"]),
            "question": str(question["question"]),
            "ground_truth_answer": str(question["answer"]),
            "question_date": str(question.get("question_date", "")),
            "is_abstention": str(question["question_id"]).endswith("_abs"),
            "answer_session_ids": [str(item) for item in question.get("answer_session_ids", [])],
            "generated_answer": generated_answer,
            "answerer_provider": provider,
            "answerer_requested_model": str(answerer_output.get("answerer_requested_model", "")),
            "answerer_model": model,
            "answerer_finish_reason": answerer_output.get("answerer_finish_reason"),
            "answerer_usage": usage,
            "cutoff_label": str(retrieval.get("cutoff_label", "top_unknown")),
            "memory_internal_usage": dict(
                retrieval.get("memory_internal_usage", {})
            ),
            "pipeline_timing": dict(retrieval.get("timing", {})),
            "prompt": {
                "system": answerer_input.system_prompt,
                "user": answerer_input.user_prompt,
            },
        }

        search_results = list(retrieval.get("search_results", []))
        result["retrieval"] = {
            "search_query": str(question["question"]),
            "search_results": search_results,
            "total_results": len(search_results),
            "memories_evaluated": int(
                retrieval.get("memories_evaluated", len(search_results))
            ),
            "recall_diagnostics": dict(
                retrieval.get("recall_diagnostics", {})
            ),
        }
        result["native"] = {
            "native_context": retrieval.get("native_context", ""),
            "native_surface_diagnostics": retrieval.get("native_surface_diagnostics", {}),
        }

        return result


def _validate_question(question: Any) -> None:
    if not isinstance(question, dict):
        raise ValueError("LongMemEval question must be a JSON object.")
    for key in (
        "question_id",
        "question_type",
        "question",
        "answer",
        "haystack_session_ids",
        "haystack_dates",
        "haystack_sessions",
    ):
        if key not in question:
            raise ValueError(f"LongMemEval question missing required field: {key}")


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object.")
    return value


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value
