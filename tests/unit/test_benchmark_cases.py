"""Offline contract checks for the v3 benchmark case boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.benchmarks.longmemeval.adapter import LongMemEvalAdapter
from dmf_bench.adapters.base import CanonicalRetrievalResult
from dmf_bench.context import pack_context


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _config(name: str, ordered_ids: list[str]) -> dict:
    return {
        "dataset": {"path": str(FIXTURES / f"{name}-mini.json")},
        "selection": {"ordered_item_ids": ordered_ids},
    }


def _exposed(case: object) -> str:
    return json.dumps({
        "unit": case.unit.metadata,
        "events": [event.to_dict() for event in case.events],
        "queries": [query.to_dict() for query in case.queries],
    }, sort_keys=True)


def test_locomo_case_keeps_order_caption_and_references_separate() -> None:
    adapter = LoCoMoAdapter()
    config = _config("locomo", ["conversation-0001"])
    unit = adapter.enumerate_units(config)[0]
    case = adapter.load_case(unit, config)

    assert case.unit.item_ids == ("conv0_q0", "conv0_q1")
    assert [event.event_id for event in case.events] == ["D1:1", "D1:2", "D2:1"]
    assert [event.session_id for event in case.events] == ["session_1", "session_1", "session_2"]
    assert [event.sequence for event in case.events] == [0, 1, 0]
    assert case.events[0].occurred_at == "2024-01-01T09:00:00Z"
    assert case.events[2].occurred_at == "2024-01-05T10:00:00Z"
    assert "Shared image: query cat bed. The image shows a blue pet bed." in case.events[2].content
    assert [query.text for query in case.queries] == [
        "What is Alice's cat called?", "Where did Alice move the bed?",
    ]
    assert case.references["conv0_q0"].expected_answer == "Pixel"
    assert case.references["conv0_q0"].evidence_refs == ("D1:1",)
    assert "ground_truth_answer" not in _exposed(case)


def test_longmemeval_case_sorts_sessions_and_isolates_answer() -> None:
    adapter = LongMemEvalAdapter()
    config = _config("longmemeval", ["lme-001"])
    unit = adapter.enumerate_units(config)[0]
    case = adapter.load_case(unit, config)

    assert [event.session_id for event in case.events] == [
        "session-a", "session-a", "session-b", "session-b",
    ]
    assert [event.role for event in case.events] == ["user", "assistant", "user", "assistant"]
    assert case.events[0].occurred_at == "2024-02-01T08:00:00Z"
    assert case.events[2].occurred_at == "2024-02-09T08:00:00Z"
    assert case.queries[0].as_of == "2024-02-10T09:00:00Z"
    assert case.references["lme-001"].expected_answer == "almonds"
    assert case.references["lme-001"].evidence_refs == ("session-a",)
    assert "ground_truth_answer" not in _exposed(case)
    assert "answer_session_ids" not in _exposed(case)


@pytest.mark.parametrize("adapter,name,unit_id", [
    (LoCoMoAdapter(), "locomo", "conversation-0001"),
    (LongMemEvalAdapter(), "longmemeval", "lme-001"),
])
def test_reference_sentinel_cannot_reach_memory_or_query(
    tmp_path: Path, adapter: object, name: str, unit_id: str,
) -> None:
    data = json.loads((FIXTURES / f"{name}-mini.json").read_text())
    sentinel = "EVALUATOR_ONLY_SENTINEL_7F31"
    if name == "locomo":
        data[0]["qa"][0]["answer"] = sentinel
        data[0]["qa"][0]["evidence"] = [sentinel]
    else:
        data[0]["answer"] = sentinel
        data[0]["answer_session_ids"] = [sentinel]
    dataset_path = tmp_path / "sentinel.json"
    dataset_path.write_text(json.dumps(data))
    config = {
        "dataset": {"path": str(dataset_path)},
        "selection": {"ordered_item_ids": [unit_id]},
    }
    case = adapter.load_case(adapter.enumerate_units(config)[0], config)
    assert sentinel not in _exposed(case)
    assert sentinel in json.dumps({key: ref.to_dict() for key, ref in case.references.items()})


def test_longmemeval_rejects_truncated_haystack(tmp_path: Path) -> None:
    data = json.loads((FIXTURES / "longmemeval-mini.json").read_text())
    data[0]["haystack_dates"].pop()
    dataset_path = tmp_path / "truncated.json"
    dataset_path.write_text(json.dumps(data))
    config = {
        "dataset": {"path": str(dataset_path)},
        "selection": {"ordered_item_ids": ["lme-001"]},
    }
    adapter = LongMemEvalAdapter()
    with pytest.raises(ValueError, match="equal lengths"):
        adapter.load_case(adapter.enumerate_units(config)[0], config)


@pytest.mark.parametrize("adapter,name,unit_id", [
    (LoCoMoAdapter(), "locomo", "conversation-0001"),
    (LongMemEvalAdapter(), "longmemeval", "lme-001"),
])
def test_v3_answerer_prompt_excludes_reference(
    tmp_path: Path, adapter: object, name: str, unit_id: str,
) -> None:
    data = json.loads((FIXTURES / f"{name}-mini.json").read_text())
    sentinel = "EXPECTED_ANSWER_ONLY_8C12"
    if name == "locomo":
        data[0]["qa"][0]["answer"] = sentinel
    else:
        data[0]["answer"] = sentinel
    dataset_path = tmp_path / "sentinel.json"
    dataset_path.write_text(json.dumps(data))
    config = {
        "dataset": {"path": str(dataset_path)},
        "selection": {"ordered_item_ids": [unit_id]},
    }
    case = adapter.load_case(adapter.enumerate_units(config)[0], config)
    query = case.queries[0]
    packed = pack_context(CanonicalRetrievalResult(()), 0)
    request = adapter.build_answerer_request_v3(query, packed, config)
    assert sentinel not in request.user_prompt
    assert sentinel not in json.dumps(request.metadata)
    prediction = adapter.build_prediction_v3(
        case, query, CanonicalRetrievalResult(()), packed,
        {"generated_answer": "fixture"},
    )
    assert sentinel in json.dumps(prediction)
