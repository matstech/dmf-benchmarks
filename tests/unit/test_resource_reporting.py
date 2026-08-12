from pathlib import Path

from dmf_bench.reporting.resources import ResourceUsageTracker


def test_resource_tracker_reports_process_and_declared_limits(monkeypatch) -> None:
    monkeypatch.setenv("DMF_BENCH_CPUS", "3.5")
    monkeypatch.setenv("DMF_BENCH_THREADS", "3")
    tracker = ResourceUsageTracker.start()

    report = tracker.report(attempt_id="attempt-1", resume=False)

    assert report["schema_version"] == 2
    assert report["attempt"]["attempt_id"] == "attempt-1"
    assert report["attempt"]["wall_seconds"] >= 0
    assert report["process"]["cpu_total_seconds"] >= 0
    assert report["process"]["current_rss_bytes"] > 0
    assert report["process"]["peak_rss_bytes"] > 0
    assert report["configuration"]["declared_cpu_limit"] == 3.5
    assert report["configuration"]["threads"]["DMF_BENCH_THREADS"] == "3"
    assert set(report["container"]) == {
        "cpu_usage_seconds",
        "memory_current_bytes",
        "memory_peak_bytes",
        "memory_limit_bytes",
        "cpu_limit_cores",
    }


def test_resource_tracker_works_without_linux_procfs(monkeypatch) -> None:
    original_read_text = Path.read_text

    def fail_proc_statm(path: Path, *args, **kwargs):
        if str(path) == "/proc/self/statm":
            raise OSError("missing procfs")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_proc_statm)

    report = ResourceUsageTracker.start().report(
        attempt_id="attempt-portable",
        resume=True,
    )

    assert report["attempt"]["resume"] is True
    assert report["process"]["current_rss_bytes"] > 0
