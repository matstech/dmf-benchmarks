"""LongMemEval benchmark adapter for the new local runner boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from longmemeval import native_prompts
from longmemeval.prompts import build_answerer_user_prompt, format_strict_sessions_as_history_chats
from longmemeval.utils import (
    build_longmemeval_strict_session_substrate,
    dedupe_longmemeval_strict_sessions_by_session_id,
    filter_questions_by_ids,
    load_dataset,
    sample_questions_stratified,
    sort_longmemeval_strict_sessions_chronologically,
)

from dmf_bench.contracts import sha256_file

from .base import BenchmarkUnit


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
    supported_protocols = ("strict", "native")
    atomic_unit = "longmemeval-question"

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
        protocol: str,
        framework_name: str,
        retrieval: dict[str, Any],
    ) -> LongMemEvalAnswererInput:
        question_text = str(question["question"])
        question_date = str(question.get("question_date", ""))
        if protocol == "strict":
            selected_session_ids, strict_context = build_strict_reader_context(
                question,
                list(retrieval.get("search_results", [])),
            )
            user_prompt = build_answerer_user_prompt(
                strict_context,
                question_text,
                question_date,
            )
            metadata = {
                "protocol": protocol,
                "framework": framework_name,
                "question_id": str(question["question_id"]),
                "strict_session_ids": selected_session_ids,
                "strict_context": strict_context,
            }
            return LongMemEvalAnswererInput("", user_prompt, metadata)

        if protocol == "native":
            native_context = retrieval.get("native_context", "")
            user_prompt = native_prompts.build_answerer_user_prompt(
                native_context,
                question_text,
                question_date,
            )
            metadata = {
                "protocol": protocol,
                "framework": framework_name,
                "question_id": str(question["question_id"]),
                "native_context": native_context,
                "native_surface_diagnostics": retrieval.get("native_surface_diagnostics", {}),
            }
            return LongMemEvalAnswererInput(
                native_prompts.build_answerer_system_prompt(),
                user_prompt,
                metadata,
            )

        raise ValueError(f"Unsupported LongMemEval protocol: {protocol!r}")

    def build_prediction(
        self,
        *,
        question: dict[str, Any],
        protocol: str,
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
            "schema_version": 1,
            "benchmark": self.name,
            "protocol": protocol,
            "protocol_label": f"{protocol}/longmemeval",
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
            "answerer_model": model,
            "answerer_usage": usage,
            "cutoff_label": str(retrieval.get("cutoff_label", "top_unknown")),
            "prompt": {
                "system": answerer_input.system_prompt,
                "user": answerer_input.user_prompt,
            },
        }

        if protocol == "strict":
            strict_context = str(answerer_input.metadata.get("strict_context", ""))
            search_results = list(retrieval.get("search_results", []))
            result["retrieval"] = {
                "search_query": str(question["question"]),
                "search_results": search_results,
                "total_results": len(search_results),
                "memories_evaluated": int(retrieval.get("memories_evaluated", len(search_results))),
                "strict_session_ids": list(answerer_input.metadata.get("strict_session_ids", [])),
                "strict_context": strict_context,
                "context": strict_context,
            }
        else:
            result["native"] = {
                "native_context": retrieval.get("native_context", ""),
                "native_surface_diagnostics": retrieval.get("native_surface_diagnostics", {}),
            }

        return result


def build_strict_reader_context(
    question: dict[str, Any],
    search_results: list[dict[str, Any]],
) -> tuple[list[str], str]:
    strict_substrate = build_longmemeval_strict_session_substrate(question)
    selected_sessions = [
        strict_substrate[session_id]
        for session_id in _selected_session_ids_from_search_results(search_results)
        if session_id in strict_substrate
    ]
    deduped_sessions = dedupe_longmemeval_strict_sessions_by_session_id(selected_sessions)
    ordered_sessions = sort_longmemeval_strict_sessions_chronologically(deduped_sessions)
    mapped_session_ids = [str(session["session_id"]) for session in deduped_sessions]
    return mapped_session_ids, format_strict_sessions_as_history_chats(ordered_sessions)


def _selected_session_ids_from_search_results(search_results: list[dict[str, Any]]) -> list[str]:
    session_ids: list[str] = []
    for item in search_results:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        if not isinstance(metadata, dict):
            continue
        raw_ids = metadata.get("source_unit_ids")
        if isinstance(raw_ids, list):
            session_ids.extend(str(session_id) for session_id in raw_ids if str(session_id).strip())
            continue
        session_id = metadata.get("source_unit_id") or metadata.get("session_id")
        if session_id is not None and str(session_id).strip():
            session_ids.append(str(session_id))

    deduped: list[str] = []
    seen: set[str] = set()
    for session_id in session_ids:
        if session_id in seen:
            continue
        seen.add(session_id)
        deduped.append(session_id)
    return deduped


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
