"""Dataset-specific ingestion retained only for historical v2 runs."""

from __future__ import annotations

from typing import Any

from dmf_bench.adapters.base import ProgressReporter, ProgressUpdate
from dmf_bench.adapters.dmf import (
    DmfEngineBundle, DmfRuntimeError, _mapping, _required_string,
)


def _ingest_locomo(
    conversation: dict[str, Any],
    engine: DmfEngineBundle,
    *,
    conversation_idx: int,
    progress: ProgressReporter,
) -> tuple[dict[str, dict[str, Any]], int]:
    from dmf_bench.benchmarks.locomo import dataset as locomo_utils
    from dmf.runtime.pipeline import InteractionProvenance

    conversation_data = _mapping(conversation.get("conversation"), "conversation")
    record_index: dict[str, dict[str, Any]] = {}
    session_rows: list[tuple[float, str, str]] = []
    for session_key, session_value in conversation_data.items():
        if not session_key.startswith("session_") or session_key.endswith("_date_time"):
            continue
        if not isinstance(session_value, list):
            raise DmfRuntimeError(f"LoCoMo {session_key} must be a list.")
        time_str = _required_string(conversation_data, f"{session_key}_date_time")
        session_rows.append(
            (locomo_utils.parse_locomo_date(date_str=time_str), session_key, time_str)
        )
    session_rows.sort(key=lambda row: row[0])

    work_rows: list[tuple[float, str, str, dict[str, Any], str, str]] = []
    for current_ts, session_key, time_str in session_rows:
        for turn in conversation_data[session_key]:
            if not isinstance(turn, dict):
                raise DmfRuntimeError("LoCoMo turn must be an object.")
            text = locomo_utils.serialize_locomo_turn_for_dmf(turn)
            if not text:
                continue
            context_text = locomo_utils.render_locomo_turn_for_context(turn)
            work_rows.append(
                (current_ts, session_key, time_str, turn, text, context_text)
            )

    progress(
        ProgressUpdate(
            stage="memory_ingestion",
            label="Ingesting source memory",
            completed=0,
            total=len(work_rows),
            item_label="conversation turns",
        )
    )
    ingested_count = 0
    for current_ts, session_key, time_str, turn, text, context_text in work_rows:
        report, vector = engine.pipeline.analyze_interaction_with_vector(
            text=text,
            is_system=False,
            provenance=InteractionProvenance(
                role=str(turn.get("speaker", "")).lower()
            ),
        )
        dia_id = _required_string(turn, "dia_id")
        report.raw_metadata.update(
            {
                "benchmark": "locomo",
                "conversation_idx": conversation_idx,
                "source_unit_type": "dia",
                "source_unit_id": dia_id,
                "session_key": session_key,
                "session_datetime_raw": time_str,
                "framework": "dmf",
            }
        )
        engine.scoring.calculate_score(report, text=text)
        entry = engine.memory_engine.add_interaction(text, report, vector)
        entry.timestamp = current_ts
        record_index[entry.record_id] = {
            "benchmark": "locomo",
            "conversation_idx": conversation_idx,
            "source_unit_type": "dia",
            "source_unit_id": dia_id,
            "source_unit_ids": [dia_id],
            "session_key": session_key,
            "session_datetime_raw": time_str,
            "speaker": str(turn.get("speaker", "")),
            "text": context_text,
            "analysis_text": text,
            "raw_text": str(turn.get("text", "") or ""),
            "query": str(turn.get("query", "") or ""),
            "blip_caption": str(turn.get("blip_caption", "") or ""),
        }
        ingested_count += 1
        progress(
            ProgressUpdate(
                stage="memory_ingestion",
                label="Ingesting source memory",
                completed=ingested_count,
                total=len(work_rows),
                item_label="conversation turns",
            )
        )
    return record_index, ingested_count


def _ingest_longmemeval(
    question: dict[str, Any],
    engine: DmfEngineBundle,
    *,
    progress: ProgressReporter,
) -> tuple[dict[str, dict[str, Any]], int]:
    from dmf_bench.benchmarks.longmemeval.dataset import (
        pair_turns,
        parse_longmemeval_date,
        sort_sessions_chronologically,
    )
    from dmf.runtime.pipeline import InteractionProvenance

    question_id = _required_string(question, "question_id")
    record_index: dict[str, dict[str, Any]] = {}
    work_rows: list[tuple[str, str, float | None, str, str]] = []
    for session_id, date_str, session in sort_sessions_chronologically(question):
        session_ts = parse_longmemeval_date(date_str)
        for pair in pair_turns(session):
            for message in pair:
                text = str(message.get("content", ""))
                role = str(message.get("role", ""))
                if not text.strip():
                    continue
                work_rows.append((session_id, date_str, session_ts, text, role))

    progress(
        ProgressUpdate(
            stage="memory_ingestion",
            label="Ingesting source memory",
            completed=0,
            total=len(work_rows),
            item_label="messages",
        )
    )
    ingested_count = 0
    for session_id, date_str, session_ts, text, role in work_rows:
        report, vector = engine.pipeline.analyze_interaction_with_vector(
            text=text,
            is_system=False,
            provenance=InteractionProvenance(role=role),
        )
        report.raw_metadata.update(
            {
                "benchmark": "longmemeval",
                "question_id": question_id,
                "source_unit_type": "session",
                "source_unit_id": session_id,
                "session_date_raw": date_str,
            }
        )
        engine.scoring.calculate_score(report, text=text)
        entry = engine.memory_engine.add_interaction(text, report, vector)
        if session_ts is not None:
            entry.timestamp = session_ts
        record_index[entry.record_id] = {
            "benchmark": "longmemeval",
            "question_id": question_id,
            "source_unit_type": "session",
            "source_unit_id": session_id,
            "source_unit_ids": [session_id],
            "session_id": session_id,
            "session_date_raw": date_str,
            "session_timestamp": session_ts,
            "role": role,
            "text": text,
        }
        ingested_count += 1
        progress(
            ProgressUpdate(
                stage="memory_ingestion",
                label="Ingesting source memory",
                completed=ingested_count,
                total=len(work_rows),
                item_label="messages",
            )
        )
    return record_index, ingested_count
