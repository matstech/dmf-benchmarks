"""Runtime secret-file loading without persisting secret values in artifacts."""

from __future__ import annotations

import os
from pathlib import Path


SECRET_FILE_ENVIRONMENT = {
    "OPENAI_API_KEY": "OPENAI_API_KEY_FILE",
    "OPENROUTER_API_KEY": "OPENROUTER_API_KEY_FILE",
}


def load_runtime_secret_files() -> None:
    """Populate provider variables from mounted files when not already set."""

    for secret_name, file_name in SECRET_FILE_ENVIRONMENT.items():
        if os.getenv(secret_name):
            continue
        raw_path = os.getenv(file_name, "").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(
                f"Cannot read provider secret file declared by {file_name}: {exc}"
            ) from exc
        if not value:
            raise ValueError(f"Provider secret file declared by {file_name} is empty.")
        os.environ[secret_name] = value
