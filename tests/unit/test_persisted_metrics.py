import os
from pathlib import Path

from dmf_bench.atomic_io import write_json_atomic
from dmf_bench.persisted_metrics import render_persisted_run_metrics


def test_persisted_metrics_project_progress_usage_timing_and_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-001"
    write_json_atomic(
        run_dir / "run-manifest.json",
        {
            "fingerprint_inputs": {
                "benchmark": "locomo",
                "framework": "dmf",
            }
        },
    )
    write_json_atomic(
        run_dir / "run-status.json",
        {
            "state": "COMPLETED",
            "items": {"expected": 10, "committed": 10},
            "units": {"expected": 2, "committed": 2},
            "current_activity": {
                "stage": "memory_ingestion",
                "completed": 3,
                "total": 12,
            },
        },
    )
    write_json_atomic(
        run_dir / "reports" / "usage.json",
        {
            "components": {
                "answerer": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
                "judge": {
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40,
                },
            }
        },
    )
    write_json_atomic(
        run_dir / "reports" / "timing.json",
        {
            "attempt": {"execution_seconds": 12.5},
            "pipeline": {"judge": {"total_ms": 2500}},
        },
    )
    write_json_atomic(
        run_dir / "reports" / "resources.json",
        {
            "process": {
                "cpu_total_seconds": 31.25,
                "average_cpu_percent": 250.0,
                "peak_rss_bytes": 536870912,
            }
        },
    )
    write_json_atomic(
        run_dir / "evaluations" / "primary_judge_score.json",
        {"metrics": {"overall": {"judge_pass_rate": 0.7}}},
    )
    write_json_atomic(
        run_dir / "evaluations" / "rigorous_report.json",
        {
            "evaluator_version": "locomo-evaluator-v1",
            "metrics": {"overall": {"ndcg_at_k": 1.5}},
        },
    )
    write_json_atomic(
        tmp_path
        / ".derived-evaluations"
        / "run-001"
        / "locomo-evaluator-v2"
        / "rigorous_report.json",
        {
            "evaluator_version": "locomo-evaluator-v2",
            "metrics": {"overall": {"ndcg_at_k": 0.5}},
        },
    )

    body = render_persisted_run_metrics(tmp_path).decode("utf-8")

    assert 'dmf_bench_persisted_run_progress_ratio{benchmark="locomo",framework="dmf"} 1.0' in body
    assert 'dmf_bench_persisted_run_current_activity_expected_items{benchmark="locomo",framework="dmf",stage="memory_ingestion"} 12.0' in body
    assert 'dmf_bench_persisted_run_current_activity_completed_items{benchmark="locomo",framework="dmf",stage="memory_ingestion"} 3.0' in body
    assert 'dmf_bench_persisted_run_current_activity_progress_ratio{benchmark="locomo",framework="dmf",stage="memory_ingestion"} 0.25' in body
    assert 'role="answerer",token_type="total"} 120.0' in body
    assert 'dmf_bench_persisted_run_execution_seconds{benchmark="locomo",framework="dmf"} 12.5' in body
    assert 'stage="judge"} 2.5' in body
    assert 'dmf_bench_persisted_run_process_cpu_seconds{benchmark="locomo",framework="dmf"} 31.25' in body
    assert 'dmf_bench_persisted_run_average_cpu_percent{benchmark="locomo",framework="dmf"} 250.0' in body
    assert 'dmf_bench_persisted_run_peak_rss_bytes{benchmark="locomo",framework="dmf"} 5.36870912e+08' in body
    assert 'evaluator="judge",framework="dmf",metric="judge_pass_rate"} 0.7' in body
    assert 'evaluator="locomo-evaluator-v2",framework="dmf",metric="ndcg_at_k"} 0.5' in body
    assert "locomo-evaluator-v1" not in body


def test_persisted_metrics_select_latest_run_per_benchmark_framework(tmp_path: Path) -> None:
    for run_id, committed in (("older", 1), ("newer", 2)):
        run_dir = tmp_path / run_id
        write_json_atomic(
            run_dir / "run-manifest.json",
            {"fingerprint_inputs": {"benchmark": "locomo", "framework": "dmf"}},
        )
        write_json_atomic(
            run_dir / "run-status.json",
            {
                "state": "RUNNING",
                "items": {"expected": 10, "committed": committed},
                "units": {"expected": 2, "committed": 1},
            },
        )
    older_status = tmp_path / "older" / "run-status.json"
    os.utime(older_status, ns=(1_000_000_000, 1_000_000_000))
    newer_status = tmp_path / "newer" / "run-status.json"
    os.utime(newer_status, ns=(2_000_000_000, 2_000_000_000))

    body = render_persisted_run_metrics(tmp_path).decode("utf-8")

    assert body.count("dmf_bench_persisted_run_completed_evaluation_items{") == 1
    assert 'dmf_bench_persisted_run_completed_evaluation_items{benchmark="locomo",framework="dmf"} 2.0' in body
