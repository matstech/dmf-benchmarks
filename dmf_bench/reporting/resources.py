"""Portable, best-effort resource measurements for official run artifacts."""

from __future__ import annotations

import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmf_bench.contracts import REPORT_SCHEMA_VERSION


THREAD_ENVIRONMENT = (
    "DMF_BENCH_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
)


@dataclass(frozen=True)
class ResourceUsageTracker:
    """Capture resource deltas for one active benchmark attempt."""

    started_monotonic: float
    process_user_seconds: float
    process_system_seconds: float
    cgroup_cpu_usage_seconds: float | None

    @classmethod
    def start(cls) -> "ResourceUsageTracker":
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return cls(
            started_monotonic=time.perf_counter(),
            process_user_seconds=float(usage.ru_utime),
            process_system_seconds=float(usage.ru_stime),
            cgroup_cpu_usage_seconds=_cgroup_cpu_usage_seconds(),
        )

    def report(self, *, attempt_id: str, resume: bool) -> dict[str, Any]:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        wall_seconds = max(time.perf_counter() - self.started_monotonic, 0.0)
        user_seconds = max(float(usage.ru_utime) - self.process_user_seconds, 0.0)
        system_seconds = max(float(usage.ru_stime) - self.process_system_seconds, 0.0)
        total_seconds = user_seconds + system_seconds
        cgroup_end = _cgroup_cpu_usage_seconds()
        cgroup_delta = (
            max(cgroup_end - self.cgroup_cpu_usage_seconds, 0.0)
            if cgroup_end is not None and self.cgroup_cpu_usage_seconds is not None
            else None
        )
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "scope": {
                "name": "active_attempt_through_reporting",
                "includes": "prediction, judging, evaluation and report generation",
                "excludes": "CLI preflight and post-report artifact publication/verification",
            },
            "attempt": {
                "attempt_id": attempt_id,
                "resume": resume,
                "wall_seconds": round(wall_seconds, 6),
            },
            "process": {
                "cpu_user_seconds": round(user_seconds, 6),
                "cpu_system_seconds": round(system_seconds, 6),
                "cpu_total_seconds": round(total_seconds, 6),
                "average_cpu_percent": round(
                    (total_seconds / wall_seconds) * 100 if wall_seconds else 0.0,
                    3,
                ),
                "current_rss_bytes": _current_rss_bytes(),
                "peak_rss_bytes": _peak_rss_bytes(float(usage.ru_maxrss)),
            },
            "container": {
                "cpu_usage_seconds": (
                    round(cgroup_delta, 6) if cgroup_delta is not None else None
                ),
                "memory_current_bytes": _read_int(
                    Path("/sys/fs/cgroup/memory.current")
                ),
                "memory_peak_bytes": _read_int(Path("/sys/fs/cgroup/memory.peak")),
                "memory_limit_bytes": _read_limit(
                    Path("/sys/fs/cgroup/memory.max")
                ),
                "cpu_limit_cores": _cgroup_cpu_limit_cores(),
            },
            "configuration": {
                "host_logical_cpu_count": os.cpu_count(),
                "declared_cpu_limit": _number_or_text(os.getenv("DMF_BENCH_CPUS")),
                "threads": {
                    name: os.getenv(name)
                    for name in THREAD_ENVIRONMENT
                    if os.getenv(name) is not None
                },
            },
        }


def _peak_rss_bytes(value: float) -> int:
    # Darwin reports bytes; Linux and the other supported Unix targets report KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def _current_rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return _peak_rss_bytes(float(usage.ru_maxrss))


def _cgroup_cpu_usage_seconds() -> float | None:
    try:
        fields = dict(
            line.split(maxsplit=1)
            for line in Path("/sys/fs/cgroup/cpu.stat")
            .read_text(encoding="utf-8")
            .splitlines()
            if " " in line
        )
        return int(fields["usage_usec"]) / 1_000_000
    except (OSError, ValueError, KeyError):
        return None


def _cgroup_cpu_limit_cores() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(
            encoding="utf-8"
        ).split()[:2]
        if quota == "max":
            return None
        return round(int(quota) / int(period), 3)
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        return None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_limit(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _number_or_text(value: str | None) -> float | str | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return value
