"""Dataset-specific ingestion retained only for historical v2 runs."""

from __future__ import annotations

from typing import Any, Callable

from dmf_bench.benchmarks.locomo import dataset as locomo_utils
from dmf_bench.benchmarks.longmemeval.dataset import (
    normalize_longmemeval_haystack,
    render_longmemeval_pair_for_context,
    serialize_longmemeval_pair_for_mem0,
)
from dmf_bench.adapters.base import ProgressReporter, ProgressUpdate
from dmf_bench.adapters.mem0 import Mem0RuntimeError, _mapping, _required_string


def _ingest_locomo(
    conversation: dict[str, Any],
    add_memory: Callable[..., int],
    *,
    user_id: str,
    conversation_idx: int,
    progress: ProgressReporter,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    conversation_data = _mapping(conversation.get("conversation"), "conversation")
    speaker_a = str(conversation_data.get("speaker_a", "") or "")
    record_index: dict[str, dict[str, Any]] = {}
    session_rows: list[tuple[float, str, str]] = []
    for session_key, session_value in conversation_data.items():
        if not session_key.startswith("session_") or session_key.endswith("_date_time"):
            continue
        if not isinstance(session_value, list):
            raise Mem0RuntimeError(f"LoCoMo {session_key} must be a list.")
        time_str = _required_string(conversation_data, f"{session_key}_date_time")
        session_rows.append(
            (locomo_utils.parse_locomo_date(date_str=time_str), session_key, time_str)
        )
    session_rows.sort(key=lambda row: row[0])

    work_rows: list[tuple[float, str, str, dict[str, Any], str]] = []
    for current_ts, session_key, time_str in session_rows:
        for turn in conversation_data[session_key]:
            if not isinstance(turn, dict):
                raise Mem0RuntimeError("LoCoMo turn must be an object.")
            ingest_text = locomo_utils.serialize_locomo_turn_for_mem0(turn)
            if not ingest_text:
                continue
            work_rows.append((current_ts, session_key, time_str, turn, ingest_text))

    progress(
        ProgressUpdate(
            stage="memory_ingestion",
            label="Ingesting source memory",
            completed=0,
            total=len(work_rows),
            item_label="conversation turns",
        )
    )
    ingested_batches = 0
    persisted_memory_count = 0
    for current_ts, session_key, time_str, turn, ingest_text in work_rows:
        dia_id = _required_string(turn, "dia_id")
        role = "user" if str(turn.get("speaker", "")) == speaker_a else "assistant"
        persisted_memory_count += add_memory(
            [{"role": role, "content": ingest_text}],
            user_id=user_id,
            timestamp=int(current_ts),
            metadata={
                "benchmark": "locomo",
                "conversation_idx": conversation_idx,
                "source_unit_type": "dia",
                "source_unit_id": dia_id,
                "framework": "mem0",
            },
        )
        ingested_batches += 1
        record_index[dia_id] = {
            "benchmark": "locomo",
            "conversation_idx": conversation_idx,
            "source_unit_type": "dia",
            "source_unit_id": dia_id,
            "source_unit_ids": [dia_id],
            "session_key": session_key,
            "session_datetime_raw": time_str,
            "speaker": str(turn.get("speaker", "")),
            "text": locomo_utils.render_locomo_turn_for_context(turn),
            "ingest_text": ingest_text,
            "raw_text": str(turn.get("text", "") or ""),
            "query": str(turn.get("query", "") or ""),
            "blip_caption": str(turn.get("blip_caption", "") or ""),
        }
        progress(
            ProgressUpdate(
                stage="memory_ingestion",
                label="Ingesting source memory",
                completed=ingested_batches,
                total=len(work_rows),
                item_label="conversation turns",
            )
        )
    return record_index, ingested_batches, persisted_memory_count


def _ingest_longmemeval(
    question: dict[str, Any],
    add_memory: Callable[..., int],
    *,
    user_id: str,
    progress: ProgressReporter,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    question_id = _required_string(question, "question_id")
    record_index: dict[str, dict[str, Any]] = {}
    work_rows: list[
        tuple[str, int | None, str, dict[str, Any], list[dict[str, Any]]]
    ] = []
    for session in normalize_longmemeval_haystack(question):
        session_id = session["session_id"]
        session_ts = session["session_timestamp"]
        session_date_raw = session["session_date_raw"]
        for pair in session["pairs"]:
            messages = serialize_longmemeval_pair_for_mem0(pair)
            if not messages:
                continue
            work_rows.append(
                (session_id, session_ts, session_date_raw, pair, messages)
            )

    progress(
        ProgressUpdate(
            stage="memory_ingestion",
            label="Ingesting source memory",
            completed=0,
            total=len(work_rows),
            item_label="message batches",
        )
    )
    ingested_batches = 0
    persisted_memory_count = 0
    for session_id, session_ts, session_date_raw, pair, messages in work_rows:
        metadata_payload = {
            "benchmark": "longmemeval",
            "question_id": question_id,
            "source_unit_type": "session",
            "source_unit_id": session_id,
        }
        persisted_memory_count += add_memory(
            messages,
            user_id=user_id,
            timestamp=session_ts,
            metadata=metadata_payload,
        )
        ingested_batches += 1
        record_id = f"{session_id}:pair:{pair['pair_index']}"
        record_index[record_id] = {
            "benchmark": "longmemeval",
            "question_id": question_id,
            "source_unit_type": "session",
            "source_unit_id": session_id,
            "source_unit_ids": [session_id],
            "session_id": session_id,
            "session_date_raw": session_date_raw,
            "session_timestamp": session_ts,
            "pair_index": pair["pair_index"],
            "text": render_longmemeval_pair_for_context(pair),
        }
        progress(
            ProgressUpdate(
                stage="memory_ingestion",
                label="Ingesting source memory",
                completed=ingested_batches,
                total=len(work_rows),
                item_label="message batches",
            )
        )
    return record_index, ingested_batches, persisted_memory_count
