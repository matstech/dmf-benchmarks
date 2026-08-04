"""LoCoMo benchmark adapter for conversation-atomic prediction runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from locomo import native_prompts
from locomo.prompts import official_ground_truth_answer
from locomo.qa import category_name, normalize_generated_answer_for_category

from dmf_bench.contracts import PREDICTION_SCHEMA_VERSION, sha256_file

from .base import BenchmarkUnit


@dataclass(frozen=True)
class LoCoMoReference:
    path: Path
    sha256: str
    conversations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LoCoMoQuestion:
    question_id: str
    conversation_id: str
    conversation_idx: int
    question_idx: int
    qa_item: dict[str, Any]


@dataclass(frozen=True)
class LoCoMoAnswererInput:
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any]


class LoCoMoAdapter:
    name = "locomo"
    atomic_unit = "locomo-conversation"

    def materialize_reference(self, config: dict[str, Any]) -> LoCoMoReference:
        """Load a local pinned LoCoMo dataset; never downloads implicitly."""
        dataset_config = _mapping(config, "dataset")
        dataset_path = Path(_string(dataset_config, "path"))
        if not dataset_path.is_file():
            raise FileNotFoundError(f"LoCoMo dataset file not found: {dataset_path}")

        actual_sha256 = sha256_file(dataset_path)
        expected_sha256 = dataset_config.get("sha256")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                "LoCoMo dataset SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("LoCoMo dataset root must be a JSON array.")
        for conversation in payload:
            _validate_conversation(conversation)
        return LoCoMoReference(
            path=dataset_path,
            sha256=actual_sha256,
            conversations=tuple(payload),
        )

    def enumerate_units(self, config: dict[str, Any]) -> list[BenchmarkUnit]:
        reference = self.materialize_reference(config)
        return [
            BenchmarkUnit(
                unit_id=conversation_id,
                item_ids=tuple(question.question_id for question in questions),
                metadata={
                    "benchmark": self.name,
                    "unit_type": self.atomic_unit,
                    "conversation_id": conversation_id,
                    "conversation_idx": conversation_idx,
                    "question_ids": [question.question_id for question in questions],
                    "categories": [int(question.qa_item.get("category", 0) or 0) for question in questions],
                },
            )
            for conversation_id, conversation_idx, _conversation, questions
            in self.select_conversations(reference.conversations, config)
        ]

    def expected_question_ids(self, config: dict[str, Any]) -> tuple[str, ...]:
        reference = self.materialize_reference(config)
        return tuple(
            question.question_id
            for _conversation_id, _conversation_idx, _conversation, questions
            in self.select_conversations(reference.conversations, config)
            for question in questions
        )

    def selected_conversations_by_id(
        self,
        config: dict[str, Any],
    ) -> dict[str, tuple[int, dict[str, Any], tuple[LoCoMoQuestion, ...]]]:
        reference = self.materialize_reference(config)
        return {
            conversation_id: (conversation_idx, conversation, tuple(questions))
            for conversation_id, conversation_idx, conversation, questions
            in self.select_conversations(reference.conversations, config)
        }

    def select_conversations(
        self,
        conversations: tuple[dict[str, Any], ...] | list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[tuple[str, int, dict[str, Any], tuple[LoCoMoQuestion, ...]]]:
        selection = _mapping(config, "selection")
        ordered_item_ids = tuple(str(item) for item in selection.get("ordered_item_ids", ()))
        if not ordered_item_ids:
            raise ValueError("selection.ordered_item_ids must contain at least one conversation id.")

        filters = selection.get("filters") or {}
        if not isinstance(filters, dict):
            raise ValueError("selection.filters must be an object when provided.")
        allowed_categories = _optional_int_list(filters.get("categories"))
        allowed_conversation_ids = set(_optional_string_list(filters.get("conversation_ids")) or ())

        selected: list[tuple[str, int, dict[str, Any], tuple[LoCoMoQuestion, ...]]] = []
        explicit_wildcard = ordered_item_ids == ("*",)
        requested_ids = set(ordered_item_ids)
        for conversation_idx, conversation in enumerate(conversations):
            conversation_id = conversation_unit_id(conversation_idx)
            if not explicit_wildcard and conversation_id not in requested_ids:
                continue
            if allowed_conversation_ids and conversation_id not in allowed_conversation_ids:
                continue
            questions = tuple(
                self.enumerate_questions(
                    conversation_id=conversation_id,
                    conversation_idx=conversation_idx,
                    conversation=conversation,
                    allowed_categories=allowed_categories,
                )
            )
            if questions:
                selected.append((conversation_id, conversation_idx, conversation, questions))

        if not explicit_wildcard:
            found_ids = {conversation_id for conversation_id, *_rest in selected}
            missing = [item_id for item_id in ordered_item_ids if item_id not in found_ids]
            if missing:
                raise ValueError(f"LoCoMo selection references missing conversation ids: {missing}")
        if not selected:
            raise ValueError("LoCoMo selection produced no conversations.")
        return selected

    def enumerate_questions(
        self,
        *,
        conversation_id: str,
        conversation_idx: int,
        conversation: dict[str, Any],
        allowed_categories: tuple[int, ...] | None = None,
    ) -> list[LoCoMoQuestion]:
        qa_items = conversation.get("qa", [])
        if not isinstance(qa_items, list):
            raise ValueError("LoCoMo conversation.qa must be a list.")
        allowed = set(allowed_categories or ())
        questions: list[LoCoMoQuestion] = []
        for question_idx, qa_item in enumerate(qa_items):
            category = int(qa_item.get("category", 0) or 0)
            if allowed and category not in allowed:
                continue
            questions.append(
                LoCoMoQuestion(
                    question_id=question_id(conversation_idx, question_idx),
                    conversation_id=conversation_id,
                    conversation_idx=conversation_idx,
                    question_idx=question_idx,
                    qa_item=qa_item,
                )
            )
        return questions

    def build_answerer_input(
        self,
        *,
        conversation: dict[str, Any],
        question: LoCoMoQuestion,
        framework_name: str,
        retrieval: dict[str, Any],
    ) -> LoCoMoAnswererInput:
        qa_item = question.qa_item
        question_text = str(qa_item.get("question", ""))
        category = int(qa_item.get("category", 0) or 0)
        native_context = retrieval.get("native_context", "")
        user_prompt = native_prompts.build_answerer_user_prompt(
            native_context,
            question_text,
            category=category,
        )
        return LoCoMoAnswererInput(
            native_prompts.build_answerer_system_prompt(),
            user_prompt,
            {
                "framework": framework_name,
                "conversation_id": question.conversation_id,
                "conversation_idx": question.conversation_idx,
                "question_id": question.question_id,
                "question_idx": question.question_idx,
                "native_context": native_context,
                "native_surface_diagnostics": retrieval.get("native_surface_diagnostics", {}),
            },
        )

    def build_prediction(
        self,
        *,
        conversation: dict[str, Any],
        question: LoCoMoQuestion,
        framework_name: str,
        retrieval: dict[str, Any],
        answerer_input: LoCoMoAnswererInput,
        answerer_output: dict[str, Any],
    ) -> dict[str, Any]:
        del conversation
        qa_item = question.qa_item
        category = int(qa_item.get("category", 0) or 0)
        raw_generated_answer = str(
            answerer_output.get("generated_answer", answerer_output.get("answer", ""))
        )
        generated_answer = normalize_generated_answer_for_category(
            category=category,
            generated_answer=raw_generated_answer,
            ground_truth_answer=str(qa_item.get("answer", "")),
        )
        usage = answerer_output.get("answerer_usage", answerer_output.get("usage", {}))
        if not isinstance(usage, dict):
            usage = {}

        result = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "benchmark": self.name,
            "framework": framework_name,
            "question_id": question.question_id,
            "conversation_id": question.conversation_id,
            "conversation_idx": question.conversation_idx,
            "question_idx": question.question_idx,
            "question": str(qa_item.get("question", "")),
            "ground_truth_answer": official_ground_truth_answer(
                category,
                str(qa_item.get("answer", "")),
            ),
            "category": category,
            "category_name": category_name(category),
            "evidence": [str(item) for item in qa_item.get("evidence", [])],
            "generated_answer": generated_answer,
            "answerer_provider": str(answerer_output.get("answerer_provider", answerer_output.get("provider", ""))),
            "answerer_requested_model": str(answerer_output.get("answerer_requested_model", "")),
            "answerer_model": str(answerer_output.get("answerer_model", answerer_output.get("model", ""))),
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
            "search_query": str(qa_item.get("question", "")),
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


def conversation_unit_id(conversation_idx: int) -> str:
    return f"conversation-{conversation_idx + 1:04d}"


def question_id(conversation_idx: int, question_idx: int) -> str:
    return f"conv{conversation_idx}_q{question_idx}"


def _validate_conversation(conversation: Any) -> None:
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo conversation item must be a JSON object.")
    if not isinstance(conversation.get("conversation"), dict):
        raise ValueError("LoCoMo item missing conversation object.")
    if not isinstance(conversation.get("qa"), list):
        raise ValueError("LoCoMo item missing qa list.")


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


def _optional_int_list(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Expected a list of integers.")
    return tuple(int(item) for item in value)


def _optional_string_list(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings.")
    return tuple(str(item) for item in value)
