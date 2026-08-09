"""Pure LoCoMo timestamp and framework turn serialization utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict


class LocomoNormalizedTurn(TypedDict):
    speaker_name: str
    utterance_text: str
    image_query: str
    image_caption: str


def parse_locomo_date(date_str: str) -> float:
    """Parse LoCoMo's human timestamp into a UTC epoch."""
    normalized = str(date_str or "").strip()
    for pattern in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M%p on %d %B, %Y",
        "%d %B, %Y",
    ):
        try:
            parsed = datetime.strptime(normalized, pattern).replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            continue
    raise ValueError(f"Unsupported LoCoMo date: {date_str!r}")


def normalize_locomo_turn(turn: dict[str, Any]) -> LocomoNormalizedTurn:
    return {
        "speaker_name": str(turn.get("speaker", "") or "").strip(),
        "utterance_text": str(turn.get("text", "") or "").strip(),
        "image_query": str(turn.get("query", "") or "").strip(),
        "image_caption": str(turn.get("blip_caption", "") or "").strip(),
    }


def _build_locomo_image_text(
    turn: LocomoNormalizedTurn,
    *,
    prefix: str,
) -> str:
    query = turn["image_query"]
    caption = turn["image_caption"]
    if query and caption:
        return f"{prefix}: query {query}. The image shows {caption}."
    if query:
        return f"{prefix}: query {query}."
    if caption:
        return f"{prefix}: {caption}."
    return ""


def serialize_locomo_turn_for_dmf(turn: dict[str, Any]) -> str:
    normalized = normalize_locomo_turn(turn)
    return " ".join(
        part
        for part in (
            normalized["utterance_text"],
            _build_locomo_image_text(normalized, prefix="Image"),
        )
        if part
    ).strip()


def serialize_locomo_turn_for_mem0(turn: dict[str, Any]) -> str:
    normalized = normalize_locomo_turn(turn)
    content = " ".join(
        part
        for part in (
            normalized["utterance_text"],
            _build_locomo_image_text(normalized, prefix="Shared image"),
        )
        if part
    ).strip()
    if not content:
        return ""
    if normalized["speaker_name"]:
        return f"{normalized['speaker_name']}: {content}"
    return content


def render_locomo_turn_for_context(turn: dict[str, Any]) -> str:
    """Return the auditable context text stored alongside a framework record."""
    return serialize_locomo_turn_for_mem0(turn)
