from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dmf_bench.adapters.base import (
    BenchmarkCase,
    BenchmarkUnit,
    CanonicalRetrievalResult,
    EvaluationReference,
    MemoryEvent,
    MemoryQuery,
    OwnedResource,
    PreparedMemoryUnit,
    RetrievedMemory,
    RetrievalResult,
)


def event(**changes: object) -> MemoryEvent:
    fields = {"event_id": "e-1", "session_id": "s-1", "sequence": 0,
              "role": "user", "content": "Remember this", "occurred_at": "2026-01-02T03:04:05Z"}
    fields.update(changes)
    return MemoryEvent(**fields)


def query(**changes: object) -> MemoryQuery:
    fields = {"query_id": "q-1", "text": "What happened?", "as_of": "2026-01-03T00:00:00+00:00"}
    fields.update(changes)
    return MemoryQuery(**fields)


def memory(**changes: object) -> RetrievedMemory:
    fields = {"memory_id": "m-1", "content": "A memory", "rank": 1}
    fields.update(changes)
    return RetrievedMemory(**fields)


def test_case_preserves_order_and_keeps_reference_out_of_framework_inputs() -> None:
    answer = "ANSWER_SENTINEL_SECRET"
    evidence = "EVIDENCE_SENTINEL_SECRET"
    rubric = "RUBRIC_SENTINEL_SECRET"
    unit = BenchmarkUnit("u-1", ("q-2", "q-1"), {})
    events = (event(), event(event_id="e-2", sequence=1, role="assistant", content="I remember"))
    queries = (query(query_id="q-2"), query())
    references = {
        "q-2": EvaluationReference("q-2", answer, (evidence,), {"rubric": rubric}),
        "q-1": EvaluationReference("q-1", "other", (), {}),
    }
    case = BenchmarkCase(unit, events, queries, references)

    assert tuple(item.query_id for item in case.queries) == unit.item_ids
    assert tuple(item.event_id for item in case.events) == ("e-1", "e-2")
    framework_input = json.dumps([item.to_dict() for item in case.events] +
                                 [item.to_dict() for item in case.queries])
    for sentinel in (answer, evidence, rubric):
        assert sentinel not in framework_input
        assert sentinel in json.dumps(case.to_dict())


@pytest.mark.parametrize("bad", [
    {"event_id": " "}, {"session_id": ""}, {"sequence": -1},
    {"sequence": True}, {"role": "tool"}, {"content": " "},
    {"occurred_at": "2026-01-02T03:04:05+02:00"},
    {"occurred_at": "2026-02-30T03:04:05Z"},
    {"source_refs": ("s", "s")}, {"metadata": {"ground_truth_answer": "SECRET"}},
    {"metadata": {"nested": [{"evidence_refs": ["SECRET"]}]}},
    {"metadata": {"valid": float("nan")}},
    {"metadata": {1: "non-string key"}},
])
def test_event_rejects_invalid_or_leaking_fields(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        event(**bad)


@pytest.mark.parametrize("bad", [
    {"query_id": ""}, {"text": " "}, {"as_of": "yesterday"},
    {"metadata": {"expected_answer": "SECRET"}},
    {"metadata": {"judge": {"rubric": "SECRET"}}},
])
def test_query_rejects_invalid_or_leaking_fields(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        query(**bad)


def test_case_rejects_order_ids_and_reference_mismatches() -> None:
    unit = BenchmarkUnit("u-1", ("q-1", "q-2"), {})
    events = (event(), event(event_id="e-2", sequence=1))
    queries = (query(), query(query_id="q-2"))
    refs = {item.query_id: EvaluationReference(item.query_id, "answer", (), {}) for item in queries}
    BenchmarkCase(unit, events, queries, refs)
    invalid = [
        (events[::-1], queries, refs),
        ((event(), event()), queries, refs),
        (events, queries[::-1], refs),
        (events, queries, {"q-1": refs["q-1"]}),
        (events, queries, {"q-1": refs["q-2"], "q-2": refs["q-1"]}),
    ]
    for bad_events, bad_queries, bad_refs in invalid:
        with pytest.raises(ValueError):
            BenchmarkCase(unit, bad_events, bad_queries, bad_refs)
    with pytest.raises(ValueError):
        BenchmarkCase(replace(unit, metadata={"answer": "SECRET"}), events, queries, refs)


def test_retrieval_rank_score_provenance_and_json() -> None:
    first = memory(native_score=0.8, native_score_kind="similarity",
                   occurred_at="2026-01-02T03:04:05Z", source_event_ids=("e-1",),
                   metadata={"provider": "fixture"})
    second = memory(memory_id="m-2", rank=2)
    result = CanonicalRetrievalResult((first, second), raw_payload={"raw": [1, 2]},
                                      diagnostics={"dmf": {"selected": 2}},
                                      usage={"tokens": 3}, timing={"seconds": 0.1})
    data = result.to_dict()
    assert [item["memory_id"] for item in data["items"]] == ["m-1", "m-2"]
    assert [item["rank"] for item in data["items"]] == [1, 2]
    assert data["items"][0]["source_event_ids"] == ["e-1"]
    assert data["items"][0]["native_score_kind"] == "similarity"
    assert json.loads(json.dumps(data)) == data


@pytest.mark.parametrize("bad", [
    {"memory_id": ""}, {"content": ""}, {"rank": 0}, {"rank": True},
    {"native_score": float("nan"), "native_score_kind": "similarity"},
    {"native_score": float("inf"), "native_score_kind": "similarity"},
    {"native_score": 0.5}, {"native_score_kind": "similarity"},
    {"occurred_at": "2026-01-02T03:04:05-05:00"},
    {"source_event_ids": ("e-1", "e-1")},
    {"metadata": {"expected_answer": "SECRET"}},
])
def test_retrieved_memory_rejects_invalid_fields(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        memory(**bad)


def test_retrieval_result_rejects_duplicate_ids_rank_gaps_and_non_json() -> None:
    invalid = [
        (memory(), memory(memory_id="m-1", rank=2)),
        (memory(rank=2),),
        (memory(), memory(memory_id="m-2", rank=3)),
    ]
    for items in invalid:
        with pytest.raises(ValueError):
            CanonicalRetrievalResult(items)
    with pytest.raises(ValueError):
        CanonicalRetrievalResult((memory(),), raw_payload={"unserializable": object()})
    with pytest.raises(ValueError):
        CanonicalRetrievalResult((memory(),), diagnostics={"bad": float("inf")})


def test_prepared_unit_excludes_handle_and_validates_resources() -> None:
    handle = object()
    prepared = PreparedMemoryUnit(handle, (OwnedResource("r-1", "qdrant", "primary", "collection-1"),),
                                  {"tokens": 2}, {"seconds": 0.1}, {"backend": "fixture"})
    data = prepared.to_dict()
    assert "handle" not in data
    assert data["resources"][0]["resource_id"] == "r-1"
    json.dumps(data)
    with pytest.raises(ValueError):
        PreparedMemoryUnit(handle, (OwnedResource("r-1", "qdrant", "primary", "collection-1"),
                                    OwnedResource("r-1", "qdrant", "primary", "collection-2")), {}, {}, {})
    with pytest.raises(ValueError):
        OwnedResource("", "qdrant", "primary", "collection-1")
    with pytest.raises(ValueError):
        PreparedMemoryUnit(handle, (), {}, {}, {"bad": object()})


def test_legacy_retrieval_result_remains_available_during_cutover() -> None:
    assert RetrievalResult(cutoff_label="fixture").to_dict()["cutoff_label"] == "fixture"
