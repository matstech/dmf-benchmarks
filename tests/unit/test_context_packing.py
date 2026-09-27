from __future__ import annotations

import hashlib
import json

import pytest

from dmf_bench.adapters.base import CanonicalRetrievalResult, RetrievedMemory
from dmf_bench.context import (
    PACKING_ID,
    RENDERER_ID,
    TOKENIZER_ID,
    pack_context,
    render_memory,
)


def fixture_result() -> CanonicalRetrievalResult:
    return CanonicalRetrievalResult((
        RetrievedMemory("m-1", "Alpha beta gamma", 1,
                        occurred_at="2026-01-02T03:04:05Z",
                        source_event_ids=("e-1", "e-2")),
        RetrievedMemory("m-2", "Delta epsilon", 2),
    ), raw_payload={"native": "not prompt material"},
        diagnostics={"dmf": {"private": "not prompt material"}})


def test_golden_rendered_bytes_tokens_and_digest() -> None:
    packed = pack_context(fixture_result(), 100)
    expected = (
        "[Memory 1]\n"
        "ID: m-1\n"
        "Occurred at: 2026-01-02T03:04:05Z\n"
        "Source events: e-1, e-2\n"
        "Content:\n"
        "Alpha beta gamma\n\n"
        "[Memory 2]\n"
        "ID: m-2\n"
        "Content:\n"
        "Delta epsilon"
    )
    assert packed.text.encode("utf-8") == expected.encode("utf-8")
    assert packed.original_tokens == packed.included_tokens == 61
    assert packed.sha256 == "82818a15cf21097f31d45676d39777674dbd3f9e3d46e3e89758cc154964ede3"
    assert (packed.tokenizer_id, packed.renderer_id, packed.packing_id) == (
        TOKENIZER_ID, RENDERER_ID, PACKING_ID)
    assert packed.returned_ids == packed.included_ids == ("m-1", "m-2")
    assert packed.excluded_ids == packed.cuts == ()
    assert "not prompt material" not in packed.text
    assert json.loads(json.dumps(packed.to_dict())) == packed.to_dict()


def test_whole_item_policy_stops_at_first_nonfitting_item_without_reordering() -> None:
    first = RetrievedMemory("m-1", "short", 1)
    second = RetrievedMemory("m-2", "long " * 100, 2)
    third = RetrievedMemory("m-3", "tiny", 3)
    result = CanonicalRetrievalResult((first, second, third))
    first_tokens = pack_context(CanonicalRetrievalResult((first,)), 100).included_tokens
    packed = pack_context(result, first_tokens + 2)
    assert packed.text == render_memory(first)
    assert packed.returned_ids == ("m-1", "m-2", "m-3")
    assert packed.included_ids == ("m-1",)
    assert packed.excluded_ids == ("m-2", "m-3")
    assert packed.cuts == ()
    assert packed.included_tokens <= first_tokens + 2


def test_truncation_applies_only_to_first_item_content() -> None:
    packed = pack_context(fixture_result(), 43)
    assert packed.text.endswith("Content:\nAlpha")
    assert packed.included_ids == ("m-1",)
    assert packed.excluded_ids == ("m-2",)
    assert packed.original_tokens == 61
    assert packed.included_tokens == 43
    assert packed.cuts == ({"memory_id": "m-1", "original_content_tokens": 3,
                            "included_content_tokens": 1, "removed_content_tokens": 2},)
    assert packed.sha256 == hashlib.sha256(packed.text.encode("utf-8")).hexdigest()


def test_labels_timestamp_sources_and_separator_consume_budget() -> None:
    result = fixture_result()
    packed = pack_context(result, 60)
    assert packed.included_ids == ("m-1",)
    assert packed.included_tokens == 45
    assert packed.original_tokens == 61
    assert "Occurred at:" in packed.text
    assert "Source events:" in packed.text
    with pytest.raises(ValueError, match="labels and provenance"):
        pack_context(result, 0)


def test_unicode_truncation_keeps_complete_utf8_characters() -> None:
    result = CanonicalRetrievalResult((RetrievedMemory("m-1", "😀 café", 1),))
    packed = pack_context(result, 15)
    assert packed.text.endswith("Content:\n😀")
    assert "�" not in packed.text
    assert packed.included_tokens == 15
    assert packed.cuts[0]["included_content_tokens"] == 2


def test_empty_result_and_invalid_versions_or_budget() -> None:
    empty = pack_context(CanonicalRetrievalResult(()), 0)
    assert empty.text == ""
    assert empty.original_tokens == empty.included_tokens == 0
    assert empty.returned_ids == empty.included_ids == empty.excluded_ids == ()
    assert empty.sha256 == hashlib.sha256(b"").hexdigest()
    result = fixture_result()
    with pytest.raises(ValueError):
        pack_context(result, -1)
    with pytest.raises(ValueError):
        pack_context(result, True)
    with pytest.raises(ValueError, match="Unsupported tokenizer"):
        pack_context(result, 100, tokenizer_id="unknown")
    with pytest.raises(ValueError, match="Unsupported context"):
        pack_context(result, 100, renderer_id="unknown")
    with pytest.raises(ValueError, match="Unsupported context"):
        pack_context(result, 100, packing_id="unknown")
