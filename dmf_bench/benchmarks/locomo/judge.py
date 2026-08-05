"""Pure parser for the LoCoMo judge response contract."""

from __future__ import annotations

import json
import re


def parse_judge_response(text: str) -> tuple[str, float, str]:
    """Parse JSON or text judge output into verdict, score, and reason."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        raw_label = str(payload.get("label", "")).strip().upper()
        verdict = raw_label if raw_label in {"CORRECT", "WRONG"} else "WRONG"
        reasoning = str(payload.get("reasoning", "")).strip()
        reason = reasoning if reasoning else text.strip()
        score = 1.0 if verdict == "CORRECT" else 0.0
        return verdict, score, reason

    verdict_match = re.search(r"VERDICT:\s*(CORRECT|WRONG)", text, flags=re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)

    verdict = verdict_match.group(1).upper() if verdict_match else "WRONG"
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    score = 1.0 if verdict == "CORRECT" else 0.0
    return verdict, score, reason
