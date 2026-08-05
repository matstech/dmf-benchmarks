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

"""Process resource sampling and machine characteristics for native pipelines."""

from __future__ import annotations

import os
import platform
import threading
from statistics import mean
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only when psutil is unavailable.
    psutil = None


_BYTES_PER_MB = 1024 * 1024


def _bytes_to_mb(value: int | float | None) -> float:
    if value is None:
        return 0.0
    return float(value) / _BYTES_PER_MB


def _fallback_total_memory_bytes() -> int | None:
    page_size_names = ("SC_PAGE_SIZE", "SC_PAGESIZE")
    page_size = next(
        (os.sysconf(name) for name in page_size_names if name in os.sysconf_names),
        None,
    )
    if page_size is None or "SC_PHYS_PAGES" not in os.sysconf_names:
        return None
    return int(page_size) * int(os.sysconf("SC_PHYS_PAGES"))


def detect_machine_characteristics() -> dict[str, Any]:
    """Return a stable summary of the host machine for run metadata/logging."""
    total_memory_bytes = None
    physical_cpu_count = None
    cpu_freq_mhz = None

    if psutil is not None:
        total_memory_bytes = int(psutil.virtual_memory().total)
        physical_cpu_count = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq is not None:
            cpu_freq_mhz = float(cpu_freq.max or cpu_freq.current or 0.0)

    if total_memory_bytes is None:
        total_memory_bytes = _fallback_total_memory_bytes()

    return {
        "hostname": platform.node() or "unknown",
        "platform": platform.system() or "unknown",
        "platform_release": platform.release() or "unknown",
        "platform_version": platform.version() or "unknown",
        "architecture": platform.machine() or "unknown",
        "python_version": platform.python_version(),
        "logical_cpu_count": int(os.cpu_count() or 0),
        "physical_cpu_count": (
            int(physical_cpu_count) if physical_cpu_count is not None else None
        ),
        "cpu_max_frequency_mhz": cpu_freq_mhz,
        "total_memory_gb": (
            round(float(total_memory_bytes) / (1024**3), 2)
            if total_memory_bytes is not None
            else None
        ),
    }


def format_machine_characteristics(machine: dict[str, Any]) -> list[str]:
    """Render machine characteristics as concise human-readable lines."""
    platform_line = (
        f"{machine.get('platform', 'unknown')} "
        f"{machine.get('platform_release', 'unknown')} "
        f"({machine.get('architecture', 'unknown')})"
    )
    logical = machine.get("logical_cpu_count", 0)
    physical = machine.get("physical_cpu_count")
    cpu_line = f"logical={logical}"
    if physical is not None:
        cpu_line += f", physical={physical}"
    total_memory_gb = machine.get("total_memory_gb")
    memory_line = (
        f"{total_memory_gb:.2f} GiB"
        if isinstance(total_memory_gb, (int, float))
        else "unknown"
    )
    freq = machine.get("cpu_max_frequency_mhz")
    freq_line = f", max_freq={freq:.0f} MHz" if isinstance(freq, (int, float)) and freq > 0 else ""

    return [
        f"Machine: {machine.get('hostname', 'unknown')}",
        f"  Platform: {platform_line}",
        f"  Python: {machine.get('python_version', 'unknown')}",
        f"  CPUs: {cpu_line}{freq_line}",
        f"  RAM: {memory_line}",
    ]


class ProcessResourceSampler:
    """Best-effort sampler for process RSS and CPU usage during one phase."""

    def __init__(self, *, sample_interval_s: float = 0.1) -> None:
        self.sample_interval_s = sample_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu_percent_samples: list[float] = []
        self._cpu_user_s_start = 0.0
        self._cpu_system_s_start = 0.0
        self._rss_start_bytes = 0
        self._rss_peak_bytes = 0
        self._process = psutil.Process(os.getpid()) if psutil is not None else None

    def _sample_loop(self) -> None:
        if self._process is None:
            return
        while not self._stop_event.wait(self.sample_interval_s):
            try:
                rss = int(self._process.memory_info().rss)
                self._rss_peak_bytes = max(self._rss_peak_bytes, rss)
                self._cpu_percent_samples.append(
                    float(self._process.cpu_percent(interval=None))
                )
            except Exception:
                return

    def start(self) -> None:
        if self._process is None:
            return
        cpu_times = self._process.cpu_times()
        self._cpu_user_s_start = float(cpu_times.user)
        self._cpu_system_s_start = float(cpu_times.system)
        self._rss_start_bytes = int(self._process.memory_info().rss)
        self._rss_peak_bytes = self._rss_start_bytes
        self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self, *, scope: str) -> dict[str, Any]:
        if self._process is None:
            return {
                "scope": scope,
                "rss_start_mb": 0.0,
                "rss_end_mb": 0.0,
                "rss_peak_mb": 0.0,
                "cpu_user_s": 0.0,
                "cpu_system_s": 0.0,
                "cpu_percent_avg": 0.0,
            }

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.sample_interval_s * 2, 0.1))
        rss_end_bytes = int(self._process.memory_info().rss)
        self._rss_peak_bytes = max(self._rss_peak_bytes, rss_end_bytes)
        cpu_times_end = self._process.cpu_times()

        return {
            "scope": scope,
            "rss_start_mb": round(_bytes_to_mb(self._rss_start_bytes), 2),
            "rss_end_mb": round(_bytes_to_mb(rss_end_bytes), 2),
            "rss_peak_mb": round(_bytes_to_mb(self._rss_peak_bytes), 2),
            "cpu_user_s": round(float(cpu_times_end.user) - self._cpu_user_s_start, 4),
            "cpu_system_s": round(
                float(cpu_times_end.system) - self._cpu_system_s_start,
                4,
            ),
            "cpu_percent_avg": round(mean(self._cpu_percent_samples), 2)
            if self._cpu_percent_samples
            else 0.0,
        }
