"""Render a self-contained, read-only view of an offline analysis result."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


TEMPLATE_PATH = Path(__file__).with_name("static") / "analysis-dashboard.html"
DATA_MARKER = "__DMF_ANALYSIS_BASE64__"


def render_dashboard(result: dict[str, Any]) -> str:
    """Embed the exact analysis payload without external services or requests."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count(DATA_MARKER) != 1:
        raise ValueError("Analysis dashboard template has an invalid data marker.")
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return template.replace(DATA_MARKER, encoded)
