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

"""LongMemEval judge response parser.

Provider execution belongs to the provider adapter; this module contains only
the benchmark-specific deterministic parser.
"""

from __future__ import annotations

import json
import re


def parse_judge_response(text: str) -> tuple[str, float, str]:
    """Parse a judge response into `(judgment, score, reason)`.

    Accepted normalized labels:
    - `CORRECT`
    - `WRONG`
    - `PASS`
    - `FAIL`

    The returned `judgment` uses `CORRECT|WRONG`, and `score` is `1.0|0.0`.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        raw_label = str(payload.get("label", payload.get("judgment", ""))).strip().upper()
        if raw_label in {"CORRECT", "PASS"}:
            judgment = "CORRECT"
            score = 1.0
        else:
            judgment = "WRONG"
            score = 0.0
        reasoning = str(payload.get("reasoning", payload.get("reason", ""))).strip()
        reason = reasoning if reasoning else text.strip()
        return judgment, score, reason

    verdict_match = re.search(
        r"(?:VERDICT|JUDGMENT)\s*:\s*(CORRECT|WRONG|PASS|FAIL)",
        text,
        flags=re.IGNORECASE,
    )
    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    raw_label = verdict_match.group(1).upper() if verdict_match else "WRONG"
    if raw_label in {"CORRECT", "PASS"}:
        judgment = "CORRECT"
        score = 1.0
    else:
        judgment = "WRONG"
        score = 0.0
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    return judgment, score, reason
