"""Structured JSON logging for benchmark runtime events."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from dmf_bench.contracts import STRUCTURED_LOG_SCHEMA_VERSION

SECRET_KEY_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")
ALLOWED_EVENTS = {
    "run.preflight.started",
    "run.preflight.completed",
    "run.started",
    "run.completed",
    "run.failed",
    "run.interrupt.requested",
    "run.interrupted",
    "unit.prediction.started",
    "unit.prediction.committed",
    "judge.started",
    "judge.completed",
    "evaluation.started",
    "evaluation.completed",
    "report.written",
    "artifact.publish.started",
    "artifact.publish.completed",
    "artifact.publish.failed",
    "metrics.snapshot.written",
}
LOG_EVENT_FIELDS = {
    "schema_version",
    "timestamp",
    "level",
    "event",
    "message",
    "run_id",
    "attempt_id",
    "benchmark",
    "framework",
    "phase",
    "unit_index",
    "unit_id_hash",
    "elapsed_seconds",
    "outcome",
    "error_type",
    "retryable",
}


class JsonEventFormatter(logging.Formatter):
    """Format LogRecord instances as schema-compatible one-line JSON events."""

    def format(self, record: logging.LogRecord) -> str:
        payload = build_log_event(
            level=record.levelname,
            event=str(getattr(record, "event", "run.failed")),
            message=record.getMessage(),
            run_id=getattr(record, "run_id", None),
            attempt_id=getattr(record, "attempt_id", None),
            benchmark=getattr(record, "benchmark", None),
            framework=getattr(record, "framework", None),
            phase=str(getattr(record, "phase", "UNKNOWN")),
            unit_index=getattr(record, "unit_index", None),
            unit_id_hash=getattr(record, "unit_id_hash", None),
            elapsed_seconds=getattr(record, "elapsed_seconds", None),
            outcome=getattr(record, "outcome", None),
            error_type=getattr(record, "error_type", None),
            retryable=getattr(record, "retryable", None),
        )
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class JsonEventLogger:
    """Small dual-sink logger used by container runtime and tests."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        file_path: str | Path | None = None,
        level: str = "INFO",
    ) -> None:
        self._history: list[str] = []
        self._file_path: Path | None = None
        self.logger = logging.getLogger(f"dmf_bench.events.{id(self)}")
        self.logger.setLevel(level)
        self.logger.propagate = False
        self.logger.handlers.clear()

        formatter = JsonEventFormatter()
        history_handler = _HistoryHandler(self._history)
        history_handler.setFormatter(formatter)
        self.logger.addHandler(history_handler)
        stream_handler = logging.StreamHandler(stream or sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        if file_path is not None:
            self.bind_file(file_path)

    def event(
        self,
        event: str,
        message: str,
        *,
        level: str = "INFO",
        **fields: Any,
    ) -> None:
        log_event(self.logger, event, message, level=level, **fields)

    def bind_file(self, file_path: str | Path) -> Path:
        path = Path(file_path)
        if self._file_path == path:
            return path
        if self._file_path is not None:
            raise RuntimeError(f"JSON event logger is already bound to {self._file_path}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_size = path.stat().st_size if path.exists() else 0
        if self._history:
            with path.open("a", encoding="utf-8") as file:
                if existing_size and not path.read_bytes().endswith(b"\n"):
                    file.write("\n")
                file.write("\n".join(self._history))
                file.write("\n")
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(JsonEventFormatter())
        self.logger.addHandler(file_handler)
        self._file_path = path
        return path

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)


class _HistoryHandler(logging.Handler):
    def __init__(self, history: list[str]) -> None:
        super().__init__()
        self.history = history

    def emit(self, record: logging.LogRecord) -> None:
        self.history.append(self.format(record))


def configure_json_logging(
    *,
    log_file: str | Path | None = None,
    stream: TextIO | None = None,
    level: str = "INFO",
) -> logging.Logger:
    event_logger = JsonEventLogger(stream=stream, file_path=log_file, level=level)
    return event_logger.logger


def log_event(
    logger: logging.Logger,
    event: str,
    message: str,
    *,
    level: str = "INFO",
    **fields: Any,
) -> None:
    if event not in ALLOWED_EVENTS:
        raise ValueError(f"Unsupported log event: {event!r}")
    extra = {key: redact(value) for key, value in fields.items() if key in LOG_EVENT_FIELDS}
    extra["event"] = event
    logger.log(getattr(logging, level.upper()), redact(message), extra=extra)


def build_log_event(
    *,
    level: str,
    event: str,
    message: str,
    run_id: str | None = None,
    attempt_id: str | None = None,
    benchmark: str | None = None,
    framework: str | None = None,
    phase: str = "UNKNOWN",
    unit_index: int | None = None,
    unit_id_hash: str | None = None,
    elapsed_seconds: float | None = None,
    outcome: str | None = None,
    error_type: str | None = None,
    retryable: bool | None = None,
) -> dict[str, Any]:
    if event not in ALLOWED_EVENTS:
        raise ValueError(f"Unsupported log event: {event!r}")
    return {
        "schema_version": STRUCTURED_LOG_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": level,
        "event": event,
        "message": str(redact(message)),
        "run_id": run_id,
        "attempt_id": attempt_id,
        "benchmark": benchmark,
        "framework": framework,
        "phase": phase,
        "unit_index": unit_index,
        "unit_id_hash": unit_id_hash,
        "elapsed_seconds": elapsed_seconds,
        "outcome": outcome,
        "error_type": error_type,
        "retryable": retryable,
    }


def redact(value: Any) -> Any:
    """Recursively redact secret-like fields without mutating the input."""
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if is_secret_key(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        redacted = value
        for marker in ("api_key=", "apikey=", "password=", "secret=", "token="):
            lower = redacted.lower()
            index = lower.find(marker)
            if index >= 0:
                end = redacted.find(" ", index)
                if end < 0:
                    end = len(redacted)
                redacted = redacted[: index + len(marker)] + "<redacted>" + redacted[end:]
        return redacted
    return value


def is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SECRET_KEY_PARTS)
