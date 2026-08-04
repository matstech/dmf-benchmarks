"""Cooperative process-signal cancellation at atomic lifecycle boundaries."""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Iterator


class RunInterrupted(RuntimeError):
    """Raised when a requested process signal reaches a safe boundary."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        try:
            signal_name = signal.Signals(self.signum).name
        except ValueError:
            signal_name = str(self.signum)
        super().__init__(f"Benchmark interrupted by {signal_name}.")

    @property
    def exit_code(self) -> int:
        return 128 + self.signum


class CancellationController:
    """Record SIGINT/SIGTERM and raise only when the runner checks a boundary."""

    def __init__(self) -> None:
        self._requested = threading.Event()
        self._signum = int(signal.SIGTERM)

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    @property
    def signum(self) -> int:
        return self._signum

    def request(self, signum: int) -> None:
        self._signum = int(signum)
        self._requested.set()

    def check(self) -> None:
        if self.requested:
            raise RunInterrupted(self.signum)

    @contextmanager
    def installed(self) -> Iterator["CancellationController"]:
        if threading.current_thread() is not threading.main_thread():
            yield self
            return
        previous = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }

        def handler(signum: int, _frame: FrameType | None) -> None:
            self.request(signum)

        for signum in previous:
            signal.signal(signum, handler)
        try:
            yield self
        finally:
            for signum, old_handler in previous.items():
                signal.signal(signum, old_handler)
