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

import requests
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, TypedDict

LOCOMO_DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_DATASET_DIR = "datasets/locomo"
LOCOMO_DATASET_FILENAME = "locomo10.json"


class LocomoNormalizedTurn(TypedDict):
    """Structured representation of one LoCoMo turn."""

    speaker_name: str
    utterance_text: str
    image_query: str
    image_caption: str


class LocomoStrictDialog(TypedDict):
    """Canonical dataset-side dialog substrate for LoCoMo strict."""

    dialog_id: str
    session_key: str
    session_date_raw: str
    speaker_a: str
    speaker_b: str
    turn_dia_ids: list[str]
    turns: list[dict[str, Any]]
    serialized_dialog: str


def get_locomo_dataset_path(
    dest_dir: str = LOCOMO_DATASET_DIR,
    filename: str = LOCOMO_DATASET_FILENAME,
) -> Path:
    return Path(dest_dir) / filename


def load_locomo_dataset(
    dest_dir: str = LOCOMO_DATASET_DIR,
    filename: str = LOCOMO_DATASET_FILENAME,
) -> list:
    dataset_path = get_locomo_dataset_path(dest_dir=dest_dir, filename=filename)
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_locomo_dataset(
    url: str = LOCOMO_DATASET_URL,
    dest_dir: str = LOCOMO_DATASET_DIR,
    filename: str = LOCOMO_DATASET_FILENAME
) -> list:
    """
    Scarica il dataset LoCoMo e lo salva localmente.
    Ritorna il dataset caricato come lista di dizionari.
    """
    dest_path = get_locomo_dataset_path(dest_dir=dest_dir, filename=filename)
    # 1. Creazione della directory se non esiste
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # 2. Download solo se il file non esiste localmente
    if not dest_path.exists():
        print(f"Download in corso da {url}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Solleva eccezione in caso di errore HTTP
            # Salvataggio su disco
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=2, ensure_ascii=False)
            print(f"Dataset salvato correttamente in: {dest_path}")
        except requests.exceptions.RequestException as e:
            print(f"Errore durante il download: {e}")
            raise
    else:
        print(f"Dataset già presente localmente: {dest_path}")
    # 3. Caricamento dei dati
    return load_locomo_dataset(dest_dir=dest_dir, filename=filename)

def parse_locomo_date(date_str: str) -> float:
    """
    Converte il formato data di LoCoMo in Unix Timestamp.
    Formato LoCoMo: "4:04 pm on 20 January, 2023"
    """
    # 1. Normalizzazione: Python strptime spesso richiede AM/PM in maiuscolo
    # a seconda della locale del sistema.
    normalized_str = date_str.replace("am", "AM").replace("pm", "PM")
    # 2. Definizione del pattern:
    # %I:%M %p -> Ore:Minuti AM/PM (es: 4:04 PM)
    # on       -> Testo statico nel dataset
    # %d %B,   -> Giorno nomeMese, (es: 20 January,)
    # %Y       -> Anno (es: 2023)
    date_pattern = "%I:%M %p on %d %B, %Y"
    try:
        dt = datetime.strptime(normalized_str, date_pattern).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError as e:
        print(f"Errore nel parsing della data '{date_str}': {e}")
        # In un paper scientifico, è meglio sollevare l'errore che ritornare un valore nullo
        raise


def build_locomo_image_tag(turn: dict[str, Any]) -> str:
    """Format LoCoMo image-sharing metadata consistently with the Mem0 reference."""
    query = str(turn.get("query", "") or "").strip()
    blip_caption = str(turn.get("blip_caption", "") or "").strip()

    if query and blip_caption:
        return f"[Sharing image - query: {query}. The image shows: {blip_caption}]"
    if query:
        return f"[Sharing image - query for: {query}]"
    if blip_caption:
        return f"[Sharing image that shows: {blip_caption}]"
    return ""


def normalize_locomo_turn(turn: dict[str, Any]) -> LocomoNormalizedTurn:
    """Return the framework-agnostic structured representation of one turn."""
    return {
        "speaker_name": str(turn.get("speaker", "") or "").strip(),
        "utterance_text": str(turn.get("text", "") or "").strip(),
        "image_query": str(turn.get("query", "") or "").strip(),
        "image_caption": str(turn.get("blip_caption", "") or "").strip(),
    }


def _build_locomo_image_text(
    normalized_turn: LocomoNormalizedTurn,
    *,
    include_prefix: bool,
) -> str:
    query = normalized_turn["image_query"]
    caption = normalized_turn["image_caption"]
    if query and caption:
        prefix = "Shared image" if include_prefix else "Image"
        return f"{prefix}: query {query}. The image shows {caption}."
    if query:
        prefix = "Shared image" if include_prefix else "Image"
        return f"{prefix}: query {query}."
    if caption:
        prefix = "Shared image" if include_prefix else "Image"
        return f"{prefix}: {caption}."
    return ""


def serialize_locomo_turn_for_dmf(turn: dict[str, Any]) -> str:
    """Return the DMF-specific turn text used for analysis and storage.

    The speaker name stays out of the analysed text to avoid contaminating
    deterministic NLP signals with benchmark-only identity markers.
    """
    normalized_turn = normalize_locomo_turn(turn)
    parts: list[str] = []
    if normalized_turn["utterance_text"]:
        parts.append(normalized_turn["utterance_text"])
    image_text = _build_locomo_image_text(normalized_turn, include_prefix=False)
    if image_text:
        parts.append(image_text)
    return " ".join(parts).strip()


def serialize_locomo_turn_for_mem0(turn: dict[str, Any]) -> str:
    """Return the Mem0-specific turn text used for ingestion.

    Mem0 receives a natural conversational rendering that keeps the speaker
    name explicit in the message content without adding benchmark-only tags.
    """
    normalized_turn = normalize_locomo_turn(turn)
    parts: list[str] = []
    if normalized_turn["utterance_text"]:
        parts.append(normalized_turn["utterance_text"])
    image_text = _build_locomo_image_text(normalized_turn, include_prefix=True)
    if image_text:
        parts.append(image_text)
    content = " ".join(parts).strip()
    if not content:
        return ""
    if normalized_turn["speaker_name"]:
        return f"{normalized_turn['speaker_name']}: {content}"
    return content


def render_locomo_turn_for_context(turn: dict[str, Any]) -> str:
    """Return the common display form used in benchmark diagnostics/context."""
    normalized_turn = normalize_locomo_turn(turn)
    parts: list[str] = []
    if normalized_turn["utterance_text"]:
        parts.append(normalized_turn["utterance_text"])
    image_text = _build_locomo_image_text(normalized_turn, include_prefix=True)
    if image_text:
        parts.append(image_text)
    content = " ".join(parts).strip()
    if not content:
        return ""
    if normalized_turn["speaker_name"]:
        return f"{normalized_turn['speaker_name']}: {content}"
    return content


def build_locomo_canonical_text(turn: dict[str, Any]) -> str:
    """Backward-compatible alias for the legacy shared serialization."""
    return serialize_locomo_turn_for_mem0(turn)


def locomo_dialog_id_from_dia_id(dia_id: str) -> str:
    """Return the dialog identifier (`D<n>`) encoded in a LoCoMo `dia_id`."""
    normalized_dia_id = dia_id.strip()
    if not normalized_dia_id:
        raise ValueError("LoCoMo dia_id cannot be empty.")

    dialog_id, separator, _turn_id = normalized_dia_id.partition(":")
    if separator != ":" or not dialog_id:
        raise ValueError(f"Invalid LoCoMo dia_id: {dia_id!r}")
    return dialog_id


def locomo_session_key_from_dia_id(dia_id: str) -> str:
    """Map a LoCoMo `dia_id` like `D12:4` to its dataset session key."""
    dialog_id = locomo_dialog_id_from_dia_id(dia_id)
    session_number = dialog_id.removeprefix("D")
    if not session_number.isdigit():
        raise ValueError(f"Invalid LoCoMo dialog id in dia_id: {dia_id!r}")
    return f"session_{int(session_number)}"


def _locomo_session_sort_key(session_key: str) -> tuple[int, str]:
    suffix = session_key.removeprefix("session_")
    if suffix.isdigit():
        return (0, f"{int(suffix):08d}")
    return (1, session_key)


def list_locomo_session_keys(conversation_data: dict[str, Any]) -> list[str]:
    """Return LoCoMo session keys in canonical dataset order."""
    session_keys = [
        key
        for key in conversation_data
        if key.startswith("session_") and not key.endswith("_date_time")
    ]
    return sorted(session_keys, key=_locomo_session_sort_key)


def _build_locomo_strict_image_metadata(turn: dict[str, Any]) -> str:
    query = str(turn.get("query", "") or "").strip()
    blip_caption = str(turn.get("blip_caption", "") or "").strip()
    image_urls = [
        str(image_url).strip()
        for image_url in turn.get("img_url", []) or []
        if str(image_url).strip()
    ]

    image_fields: list[str] = []
    if query:
        image_fields.append(f"query: {query}")
    if blip_caption:
        image_fields.append(f"caption: {blip_caption}")
    if image_urls:
        image_fields.append(f"urls: {', '.join(image_urls)}")

    if not image_fields:
        return ""
    return f"[Image shared | {' | '.join(image_fields)}]"


def serialize_locomo_turn_for_strict_dialog(turn: dict[str, Any]) -> str:
    """Serialize one turn for the dataset-side LoCoMo strict substrate."""
    normalized_turn = normalize_locomo_turn(turn)
    speaker_name = normalized_turn["speaker_name"]
    utterance_text = normalized_turn["utterance_text"]
    image_metadata = _build_locomo_strict_image_metadata(turn)

    content_parts: list[str] = []
    if utterance_text:
        content_parts.append(utterance_text)
    if image_metadata:
        content_parts.append(image_metadata)

    content = " ".join(content_parts).strip()
    if not content:
        return f"{speaker_name}:" if speaker_name else ""
    if speaker_name:
        return f"{speaker_name}: {content}"
    return content


def serialize_locomo_dialog_for_strict(
    conversation_data: dict[str, Any],
    session_key: str,
) -> str:
    """Serialize one original LoCoMo dialog in a stable dataset-faithful form."""
    session_date_raw = str(conversation_data.get(f"{session_key}_date_time", "") or "").strip()
    turns = conversation_data.get(session_key, []) or []

    lines: list[str] = []
    if session_date_raw:
        lines.append(f"Session Date: {session_date_raw}")

    for turn in turns:
        rendered_turn = serialize_locomo_turn_for_strict_dialog(turn)
        if rendered_turn:
            lines.append(rendered_turn)

    return "\n".join(lines).strip()


def build_locomo_dialog_substrate(
    conversation_data: dict[str, Any],
) -> dict[str, LocomoStrictDialog]:
    """Build the canonical dialog substrate keyed by dialog id (`D<n>`)."""
    speaker_a = str(conversation_data.get("speaker_a", "") or "").strip()
    speaker_b = str(conversation_data.get("speaker_b", "") or "").strip()

    dialog_substrate: dict[str, LocomoStrictDialog] = {}
    for session_key in list_locomo_session_keys(conversation_data):
        turns = [dict(turn) for turn in conversation_data.get(session_key, []) or []]
        if not turns:
            continue

        first_dia_id = str(turns[0].get("dia_id", "") or "").strip()
        dialog_id = locomo_dialog_id_from_dia_id(first_dia_id)
        dialog_substrate[dialog_id] = {
            "dialog_id": dialog_id,
            "session_key": session_key,
            "session_date_raw": str(conversation_data.get(f"{session_key}_date_time", "") or "").strip(),
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "turn_dia_ids": [
                str(turn.get("dia_id", "") or "").strip()
                for turn in turns
                if str(turn.get("dia_id", "") or "").strip()
            ],
            "turns": turns,
            "serialized_dialog": serialize_locomo_dialog_for_strict(
                conversation_data,
                session_key,
            ),
        }

    return dialog_substrate


def map_locomo_dia_ids_to_strict_dialogs(
    conversation_data: dict[str, Any],
) -> dict[str, LocomoStrictDialog]:
    """Return the canonical strict dialog substrate keyed by original `dia_id`."""
    dialog_substrate = build_locomo_dialog_substrate(conversation_data)
    dialog_by_dia_id: dict[str, LocomoStrictDialog] = {}
    for dialog in dialog_substrate.values():
        for dia_id in dialog["turn_dia_ids"]:
            dialog_by_dia_id[dia_id] = dialog
    return dialog_by_dia_id


def temporal_memory_debug_snapshot(
    memory_engine: Any,
    max_entries: int | None = None,
    write_jsonl: bool = False,
    output_path: str | Path = "temporal_memory_debug.jsonl",
) -> dict[str, Any] | list[dict[str, Any]]:
    def build_single_snapshot(single_memory_engine: Any) -> dict[str, Any]:
        entries = []
        for entry in single_memory_engine.queue:
            entry_payload = entry.to_dict()
            if "vector" in entry_payload:
                entry_payload["vector"] = "..."
            entries.append(entry_payload)

        if max_entries is not None:
            entries = entries[:max_entries]

        return {
            "entries_count": len(single_memory_engine.queue),
            "total_tokens": single_memory_engine.get_total_tokens(),
            "next_id": single_memory_engine._next_id,
            "turn_counter": single_memory_engine._turn_counter,
            "entries": entries,
            "context_metrics": single_memory_engine.get_context_metrics(),
            "recall_diagnostics": single_memory_engine.get_recall_diagnostics(),
        }

    if isinstance(memory_engine, dict):
        snapshots = []
        for conversation_idx, single_memory_engine in memory_engine.items():
            snapshot = build_single_snapshot(single_memory_engine)
            snapshot["conversation_idx"] = conversation_idx
            snapshots.append(snapshot)
    else:
        snapshots = build_single_snapshot(memory_engine)

    if write_jsonl:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "a", encoding="utf-8") as f:
            if isinstance(snapshots, list):
                for snapshot in snapshots:
                    f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(snapshots, ensure_ascii=False) + "\n")
    else:
        print(json.dumps(snapshots, indent=2, ensure_ascii=False))

    return snapshots
