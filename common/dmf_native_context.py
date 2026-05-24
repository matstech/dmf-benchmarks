# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DMF_NATIVE_SURFACE_MARKER = "dmf_render_context"
DMF_RECALL_HEADER = "=== LONG-TERM MEMORY (RECALLED) ==="
DMF_ACTIVE_HEADER = "=== ACTIVE CONVERSATION ==="


@dataclass(frozen=True)
class DmfNativeContextSurface:
    """DMF native context built through the structured retrieval facade."""

    native_context: str
    query_vector: Any
    surface_marker: str
    recalled_section_present: bool
    active_section_present: bool
    raw_retrieval_outputs: dict[str, Any]
    result_count: int
    context_metrics: dict[str, int]


def inspect_dmf_native_context(native_context: str) -> tuple[bool, bool]:
    """Return whether the canonical recalled and active sections are present."""
    recalled_present = DMF_RECALL_HEADER in native_context
    active_present = DMF_ACTIVE_HEADER in native_context

    if not active_present:
        raise ValueError(
            "DMF native context is missing the canonical active conversation header."
        )

    if recalled_present and native_context.index(DMF_RECALL_HEADER) > native_context.index(
        DMF_ACTIVE_HEADER
    ):
        raise ValueError(
            "DMF native context has recalled memory after active conversation."
        )

    return recalled_present, active_present


def build_dmf_native_context_surface(
    *,
    memory: Any,
    query_text: str,
    record_index: dict[str, dict[str, Any]] | None = None,
) -> DmfNativeContextSurface:
    """Build the canonical DMF native answerer surface for one benchmark query.

    The recalled section is produced by the new two-stage retrieval stack via a
    single ``Memory.retrieve(query_text)`` call. The resulting evidence is then
    rendered locally and also projected into benchmark-canonical
    ``search_results`` artifacts. The active section is produced by
    ``TemporalMemory.get_full_context(query_vector=None)``, which emits only the
    active conversation block. The two are composed under the canonical
    benchmark headers so that inspection contracts are preserved.
    """
    from dmf.memory.evidence_assembly import render_evidence_context

    final_evidence = memory.retrieve(query_text)
    recalled_body = render_evidence_context(final_evidence)
    temporal_memory = memory._temporal_memory
    active_block = temporal_memory.get_full_context(query_vector=None)

    parts: list[str] = []
    if recalled_body.strip():
        parts.append(DMF_RECALL_HEADER)
        parts.append(recalled_body)
        parts.append("")
    parts.append(active_block)
    native_context = "\n".join(parts)
    recalled_present, active_present = inspect_dmf_native_context(native_context)

    raw_retrieval_outputs = _build_native_retrieval_outputs(
        final_evidence,
        record_index=record_index or {},
    )
    context_metrics = (
        temporal_memory.get_context_metrics()
        if hasattr(temporal_memory, "get_context_metrics")
        else {}
    )

    return DmfNativeContextSurface(
        native_context=native_context,
        query_vector=None,
        surface_marker=DMF_NATIVE_SURFACE_MARKER,
        recalled_section_present=recalled_present,
        active_section_present=active_present,
        raw_retrieval_outputs=raw_retrieval_outputs,
        result_count=len(final_evidence),
        context_metrics=dict(context_metrics),
    )


def _build_native_retrieval_outputs(
    final_evidence: list[Any],
    *,
    record_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    search_results = [
        _canonical_search_result(item, record_index=record_index)
        for item in final_evidence
    ]
    return {
        "retrieval_stack": "dmf_structured_native",
        "search_results": search_results,
        "retrieved_evidence": [
            item.to_dict() if hasattr(item, "to_dict") else item
            for item in final_evidence
        ],
    }


def _canonical_search_result(
    evidence: Any,
    *,
    record_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = getattr(evidence, "render_payload", {})
    provenance = getattr(evidence, "provenance", {})
    score = _candidate_score(evidence)
    evidence_type = str(getattr(evidence, "evidence_type", ""))
    evidence_id = str(getattr(evidence, "evidence_id", ""))
    channels = _dedupe_strings(provenance.get("channels", []))

    if evidence_type == "raw_turn":
        record = payload.get("record", {})
        record_id = (
            _record_id(record)
            or str(provenance.get("source_record_id", ""))
            or evidence_id
        )
        metadata = _metadata_for_record(record_id, record_index=record_index)
        support_records = _support_records(evidence)
        support_record_ids = _support_record_ids(
            support_records=support_records,
            source_record_id=record_id,
        )
        metadata.update(
            {
                "evidence_type": evidence_type,
                "source_record_id": record_id,
                "source_record_ids": support_record_ids,
                "support_record_ids": support_record_ids,
            }
        )
        if channels:
            metadata["channels"] = channels
        source_unit_ids = _source_unit_ids_for_records(
            support_record_ids,
            record_index=record_index,
        )
        if len(source_unit_ids) == 1:
            metadata["source_unit_id"] = source_unit_ids[0]
            metadata["source_unit_ids"] = list(source_unit_ids)
            if metadata.get("source_unit_type") == "session":
                metadata.setdefault("session_id", source_unit_ids[0])
        elif source_unit_ids:
            metadata["source_unit_id"] = source_unit_ids[0]
            metadata["source_unit_ids"] = list(source_unit_ids)
        return {
            "memory": str(record.get("text", "")),
            "score": score,
            "id": record_id,
            "created_at": record.get("created_at"),
            "metadata": metadata,
        }

    support_records = _support_records(evidence)
    source_record_id = str(provenance.get("source_record_id", "")) or _record_id(
        payload.get("record", {})
    )
    support_record_ids = _support_record_ids(
        support_records=support_records,
        source_record_id=source_record_id,
    )
    metadata = _card_metadata(
        evidence=evidence,
        record_index=record_index,
        source_record_id=source_record_id,
        support_record_ids=support_record_ids,
    )
    primary_record = _primary_record(
        support_records=support_records,
        source_record_id=source_record_id,
    )
    card_payload = payload.get("card", {})
    return {
        "memory": _evidence_memory_text(
            primary_record=primary_record,
            card_payload=card_payload,
        ),
        "score": score,
        "id": evidence_id,
        "created_at": _created_at(primary_record),
        "metadata": metadata,
    }


def _card_metadata(
    *,
    evidence: Any,
    record_index: dict[str, dict[str, Any]],
    source_record_id: str,
    support_record_ids: list[str],
) -> dict[str, Any]:
    provenance = getattr(evidence, "provenance", {})
    evidence_type = str(getattr(evidence, "evidence_type", ""))
    metadata = _metadata_for_record(source_record_id, record_index=record_index)
    if not metadata and support_record_ids:
        metadata = _metadata_for_record(support_record_ids[0], record_index=record_index)
    metadata.update(
        {
            "evidence_type": evidence_type,
            "card_id": str(provenance.get("card_id", getattr(evidence, "evidence_id", ""))),
            "source_record_id": source_record_id,
            "source_record_ids": support_record_ids,
            "support_record_ids": support_record_ids,
        }
    )
    channels = _dedupe_strings(provenance.get("channels", []))
    if channels:
        metadata["channels"] = channels

    source_unit_ids = _source_unit_ids_for_records(
        support_record_ids,
        record_index=record_index,
    )
    if len(source_unit_ids) == 1:
        metadata["source_unit_id"] = source_unit_ids[0]
        metadata["source_unit_ids"] = list(source_unit_ids)
        if metadata.get("source_unit_type") == "session":
            metadata.setdefault("session_id", source_unit_ids[0])
    elif source_unit_ids:
        metadata["source_unit_id"] = source_unit_ids[0]
        metadata["source_unit_ids"] = list(source_unit_ids)

    return metadata


def _metadata_for_record(
    record_id: str,
    *,
    record_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = dict(record_index.get(record_id, {}))
    source_unit_ids = _dedupe_strings(metadata.get("source_unit_ids", []))
    source_unit_id = metadata.get("source_unit_id")
    if not source_unit_ids and isinstance(source_unit_id, str) and source_unit_id:
        source_unit_ids = [source_unit_id]

    if len(source_unit_ids) == 1:
        metadata["source_unit_id"] = source_unit_ids[0]
        metadata["source_unit_ids"] = list(source_unit_ids)
        if metadata.get("source_unit_type") == "session":
            metadata.setdefault("session_id", source_unit_ids[0])
    elif source_unit_ids:
        metadata["source_unit_ids"] = list(source_unit_ids)
        metadata.pop("source_unit_id", None)

    return metadata


def _source_unit_ids_for_records(
    record_ids: list[str],
    *,
    record_index: dict[str, dict[str, Any]],
) -> list[str]:
    source_unit_ids: list[str] = []
    for record_id in record_ids:
        metadata = record_index.get(record_id, {})
        raw_ids = metadata.get("source_unit_ids", [])
        if isinstance(raw_ids, list):
            source_unit_ids.extend(str(value) for value in raw_ids if str(value).strip())
            continue
        single_id = metadata.get("source_unit_id")
        if isinstance(single_id, str) and single_id:
            source_unit_ids.append(single_id)
    return _dedupe_strings(source_unit_ids)


def _support_record_ids(
    *,
    support_records: list[dict[str, Any]],
    source_record_id: str,
) -> list[str]:
    record_ids = [
        _record_id(support_item.get("record", {}))
        for support_item in support_records
        if isinstance(support_item, dict)
    ]
    if source_record_id:
        record_ids.insert(0, source_record_id)
    return _dedupe_strings(record_ids)


def _support_records(evidence: Any) -> list[dict[str, Any]]:
    payload = getattr(evidence, "render_payload", {})
    support = payload.get("support_records", [])
    if not isinstance(support, list):
        return []
    return [item for item in support if isinstance(item, dict)]


def _primary_record(
    *,
    support_records: list[dict[str, Any]],
    source_record_id: str,
) -> dict[str, Any]:
    for support_item in support_records:
        record = support_item.get("record")
        if not isinstance(record, dict):
            continue
        if _record_id(record) == source_record_id:
            return record
    for support_item in support_records:
        record = support_item.get("record")
        if isinstance(record, dict):
            return record
    return {}


def _record_id(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    value = record.get("record_id")
    return str(value) if value is not None else ""


def _created_at(record: dict[str, Any]) -> Any:
    if not isinstance(record, dict):
        return None
    return record.get("created_at")


def _evidence_memory_text(
    *,
    primary_record: dict[str, Any],
    card_payload: Any,
) -> str:
    if isinstance(primary_record, dict):
        text = str(primary_record.get("text", "")).strip()
        if text:
            return text
    if isinstance(card_payload, dict):
        pieces = [
            card_payload.get("subject"),
            card_payload.get("predicate"),
            card_payload.get("object"),
        ]
        return " ".join(str(piece) for piece in pieces if piece).strip()
    return ""


def _candidate_score(evidence: Any) -> float:
    features = getattr(evidence, "answerability_features", {})
    answerability = features.get("answerability_score")
    if isinstance(answerability, (int, float)):
        return float(answerability)
    for attr in ("semantic_score", "symbolic_score", "temporal_score"):
        value = getattr(evidence, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _dedupe_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
