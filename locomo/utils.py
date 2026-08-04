"""LoCoMo dataset and framework-native turn serialization utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import requests


LOCOMO_DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_DATASET_DIR = "datasets/locomo"
LOCOMO_DATASET_FILENAME = "locomo10.json"


class LocomoNormalizedTurn(TypedDict):
    speaker_name: str
    utterance_text: str
    image_query: str
    image_caption: str


def get_locomo_dataset_path(
    dest_dir: str = LOCOMO_DATASET_DIR,
    filename: str = LOCOMO_DATASET_FILENAME,
) -> Path:
    return Path(dest_dir) / filename


def load_locomo_dataset(
    dest_dir: str = LOCOMO_DATASET_DIR,
    filename: str = LOCOMO_DATASET_FILENAME,
) -> list[Any]:
    path = get_locomo_dataset_path(dest_dir=dest_dir, filename=filename)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("LoCoMo dataset root must be a JSON array.")
    return payload


def download_locomo_dataset(
    url: str = LOCOMO_DATASET_URL,
    dest_dir: str = LOCOMO_DATASET_DIR,
    filename: str = LOCOMO_DATASET_FILENAME,
) -> list[Any]:
    path = get_locomo_dataset_path(dest_dir=dest_dir, filename=filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    return load_locomo_dataset(dest_dir=dest_dir, filename=filename)


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


def build_locomo_image_tag(turn: dict[str, Any]) -> str:
    query = str(turn.get("query", "") or "").strip()
    caption = str(turn.get("blip_caption", "") or "").strip()
    if query and caption:
        return f"Image: {query}. {caption}"
    if query:
        return f"Image: {query}."
    if caption:
        return caption
    return ""


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
