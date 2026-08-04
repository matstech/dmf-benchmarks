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

"""LongMemEval dataset loading, date parsing, and sampling utilities."""

import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import requests

QUESTION_TYPES = [
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]

DEFAULT_DATASET_DIR = "datasets/longmemeval"
DEFAULT_DATASET_FILE = "longmemeval_s_cleaned.json"
DEFAULT_DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
    "resolve/main/longmemeval_s_cleaned.json"
)


class LongMemEvalNormalizedTurn(TypedDict):
    """Framework-agnostic representation of one LongMemEval turn."""

    role: str
    speaker_label: str
    content: str


class LongMemEvalMem0Message(TypedDict):
    """Mem0-native message payload for one conversational turn."""

    role: str
    content: str


class LongMemEvalNormalizedPair(TypedDict):
    """Framework-agnostic representation of one haystack pair."""

    pair_index: int
    source_unit_type: str
    source_unit_id: str
    session_id: str
    session_date_raw: str
    session_timestamp: int | None
    turns: list[LongMemEvalNormalizedTurn]


class LongMemEvalNormalizedSession(TypedDict):
    """Framework-agnostic representation of one haystack session."""

    source_unit_type: str
    source_unit_id: str
    session_id: str
    session_date_raw: str
    session_timestamp: int | None
    pairs: list[LongMemEvalNormalizedPair]


def get_default_dataset_path() -> Path:
    return Path(DEFAULT_DATASET_DIR) / DEFAULT_DATASET_FILE


def download_longmemeval_dataset(
    *,
    url: str = DEFAULT_DATASET_URL,
    dest_dir: str = DEFAULT_DATASET_DIR,
    filename: str = DEFAULT_DATASET_FILE,
) -> list[dict]:
    """Download the official LongMemEval cleaned dataset and return it."""
    dataset_path = Path(dest_dir) / filename
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        print(f"Downloading LongMemEval dataset from {url}...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        dataset_path.write_text(response.text, encoding="utf-8")
        print(f"Dataset saved to: {dataset_path}")
    else:
        print(f"Dataset already exists locally: {dataset_path}")

    return load_dataset(str(dataset_path))


def ensure_longmemeval_dataset(
    path: str | None = None,
) -> tuple[str, list[dict]]:
    """Return a usable LongMemEval dataset path and contents, downloading if needed."""
    dataset_path = Path(path) if path is not None else get_default_dataset_path()
    if dataset_path.exists():
        print(f"Dataset already exists locally: {dataset_path}")
        return str(dataset_path), load_dataset(str(dataset_path))

    if path is not None:
        raise FileNotFoundError(
            f"LongMemEval dataset file not found: {dataset_path}"
        )

    dataset = download_longmemeval_dataset()
    return str(dataset_path), dataset


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_longmemeval_date(date_str: str) -> int | None:
    """'2023/05/01 (Mon) 21:05' -> Unix epoch int (UTC)."""
    try:
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
        dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


def parse_longmemeval_date_human(date_str: str) -> str:
    """'2023/05/01 (Mon) 21:05' -> 'Monday, May 01, 2023'."""
    try:
        cleaned = re.sub(r"\s*\([A-Za-z]+\)\s*", " ", date_str).strip()
        dt = datetime.strptime(cleaned, "%Y/%m/%d %H:%M")
        return dt.strftime("%A, %B %d, %Y")
    except (ValueError, TypeError):
        return date_str


def sort_sessions_chronologically(
    question: dict,
) -> list[tuple[str, str, list[dict]]]:
    """Zip and sort (session_id, date_str, session) by parsed date."""
    paired = list(zip(
        question["haystack_session_ids"],
        question["haystack_dates"],
        question["haystack_sessions"],
    ))

    def sort_key(item: tuple) -> tuple:
        parsed = parse_longmemeval_date(item[1])
        if parsed is not None:
            return (0, parsed, item[1])
        return (1, 0, item[1])

    paired.sort(key=sort_key)
    return paired


def pair_turns(session: list[dict]) -> list[list[dict]]:
    """Group consecutive turns into user/assistant pairs."""
    cleaned = [{"role": t["role"], "content": t["content"]} for t in session]
    pairs = []
    for i in range(0, len(cleaned), 2):
        pairs.append(cleaned[i : i + 2])
    return pairs


def _normalize_longmemeval_role(role: str) -> str:
    normalized = str(role or "").strip().lower()
    return normalized or "unknown"


def _longmemeval_speaker_label(role: str) -> str:
    normalized = _normalize_longmemeval_role(role)
    if normalized == "user":
        return "User"
    if normalized == "assistant":
        return "Assistant"
    return normalized.replace("_", " ").title()


def normalize_longmemeval_turn(turn: dict) -> LongMemEvalNormalizedTurn:
    """Return the framework-agnostic structured representation of one turn."""
    role = _normalize_longmemeval_role(turn.get("role", ""))
    return {
        "role": role,
        "speaker_label": _longmemeval_speaker_label(role),
        "content": str(turn.get("content", "") or "").strip(),
    }


def normalize_longmemeval_pair(
    turns: list[dict],
    *,
    session_id: str,
    session_date_raw: str,
    pair_index: int,
) -> LongMemEvalNormalizedPair:
    """Return the structured representation of one pair within a session."""
    return {
        "pair_index": pair_index,
        "source_unit_type": "session",
        "source_unit_id": session_id,
        "session_id": session_id,
        "session_date_raw": session_date_raw,
        "session_timestamp": parse_longmemeval_date(session_date_raw),
        "turns": [normalize_longmemeval_turn(turn) for turn in turns],
    }


def normalize_longmemeval_session(
    session: list[dict],
    *,
    session_id: str,
    session_date_raw: str,
) -> LongMemEvalNormalizedSession:
    """Return the structured representation of one haystack session."""
    pairs = [
        normalize_longmemeval_pair(
            turns,
            session_id=session_id,
            session_date_raw=session_date_raw,
            pair_index=pair_index,
        )
        for pair_index, turns in enumerate(pair_turns(session))
    ]
    return {
        "source_unit_type": "session",
        "source_unit_id": session_id,
        "session_id": session_id,
        "session_date_raw": session_date_raw,
        "session_timestamp": parse_longmemeval_date(session_date_raw),
        "pairs": pairs,
    }


def normalize_longmemeval_haystack(
    question: dict,
) -> list[LongMemEvalNormalizedSession]:
    """Return the question haystack in chronological structured form."""
    return [
        normalize_longmemeval_session(
            session,
            session_id=session_id,
            session_date_raw=date_str,
        )
        for session_id, date_str, session in sort_sessions_chronologically(question)
    ]


def serialize_longmemeval_pair_for_dmf(pair: LongMemEvalNormalizedPair) -> str:
    """Return DMF-oriented analysis text for one pair."""
    return " ".join(
        turn["content"]
        for turn in pair["turns"]
        if turn["content"]
    ).strip()


def serialize_longmemeval_pair_for_mem0(
    pair: LongMemEvalNormalizedPair,
) -> list[LongMemEvalMem0Message]:
    """Return Mem0-native messages for one pair without benchmark-only tags."""
    return [
        {
            "role": turn["role"],
            "content": turn["content"],
        }
        for turn in pair["turns"]
        if turn["content"]
    ]


def render_longmemeval_pair_for_context(pair: LongMemEvalNormalizedPair) -> str:
    """Return the common display form used for answerer/context rendering."""
    lines = [
        f"{turn['speaker_label']}: {turn['content']}"
        for turn in pair["turns"]
        if turn["content"]
    ]
    return "\n".join(lines)


def serialize_longmemeval_session_for_dmf(
    session: LongMemEvalNormalizedSession,
) -> list[str]:
    """Return DMF-oriented texts for all pairs in a session."""
    return [
        serialized
        for serialized in (
            serialize_longmemeval_pair_for_dmf(pair) for pair in session["pairs"]
        )
        if serialized
    ]


def serialize_longmemeval_session_for_mem0(
    session: LongMemEvalNormalizedSession,
) -> list[list[LongMemEvalMem0Message]]:
    """Return Mem0-native message pairs for one session."""
    return [
        messages
        for messages in (
            serialize_longmemeval_pair_for_mem0(pair) for pair in session["pairs"]
        )
        if messages
    ]


def render_longmemeval_session_for_context(
    session: LongMemEvalNormalizedSession,
) -> str:
    """Return a chronological context block for one session."""
    pair_blocks = [
        pair_block
        for pair_block in (
            render_longmemeval_pair_for_context(pair) for pair in session["pairs"]
        )
        if pair_block
    ]
    header = f"Session {session['session_id']} ({session['session_date_raw']})"
    if not pair_blocks:
        return header
    return f"{header}\n" + "\n".join(pair_blocks)


def sample_questions_stratified(
    questions: list[dict],
    per_type: int = 5,
    seed: int = 42,
    selected_types: list[str] | None = None,
) -> list[dict]:
    """Sample up to per_type questions for each question_type, deterministically."""
    type_filter = set(selected_types) if selected_types else set(QUESTION_TYPES)

    groups: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        if q["question_type"] in type_filter:
            groups[q["question_type"]].append(q)

    for qtype in groups:
        groups[qtype].sort(key=lambda q: q["question_id"])

    rng = random.Random(seed)
    sampled = []
    for qtype in sorted(groups.keys()):
        group = groups[qtype]
        n = min(per_type, len(group))
        sampled.extend(rng.sample(group, n))

    sampled.sort(key=lambda q: q["question_id"])
    return sampled


def filter_questions_by_ids(
    questions: list[dict],
    question_ids: list[str],
) -> list[dict]:
    """Keep only questions whose question_id is in the given list."""
    ids = set(question_ids)
    return [q for q in questions if q["question_id"] in ids]
