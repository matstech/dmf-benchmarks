from __future__ import annotations

import json
import io
import signal
import shutil
import threading
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from dmf_bench.adapters.base import (
    AnswererRequest,
    BenchmarkUnit,
    FrameworkRunContext,
    JudgeRequest,
    RetrievalResult,
)
from dmf_bench.atomic_io import read_json
from dmf_bench.cli import (
    EXIT_INCOMPLETE,
    EXIT_RUNTIME,
    EXIT_SCIENTIFIC_FAILURE,
    main,
)
from dmf_bench.contracts import sha256_file
from dmf_bench.execution import CancellationController, RunInterrupted
from dmf_bench.fingerprints import judge_fingerprint
from dmf_bench.logging_config import JsonEventLogger
from dmf_bench.metrics import BenchmarkMetrics, start_metrics_endpoint, stop_metrics_endpoint
from dmf_bench.registry import supported_combinations
from dmf_bench.runtime import RuntimeFactories, assemble_application, benchmark_factories


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


class FixtureFramework:
    def __init__(self, name: str) -> None:
        self.name = name

    def cleanup_unit(
        self,
        _unit: BenchmarkUnit,
        _item: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> None:
        del run_context

    def prepare_unit(
        self,
        unit: BenchmarkUnit,
        _item: dict[str, Any],
        _config: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> dict[str, Any]:
        del run_context
        return {
            "qdrant_commit_barrier": True,
            "cleanup_manifest": {
                "framework": self.name,
                "unit_hash": unit.unit_id,
                "collections": [],
                "local_paths": [],
            },
        }

    def retrieve(
        self,
        unit: BenchmarkUnit,
        question: dict[str, Any],
        config: dict[str, Any],
        _prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        del run_context
        source_ids = [str(item) for item in question["answer_session_ids"]]
        search_results = (
            {"metadata": {"source_unit_ids": source_ids}},
        )
        diagnostics = (
            {
                "ranked_candidates_canonical": list(search_results),
                "final_candidates_canonical": list(search_results),
            }
            if self.name == "dmf"
            else {}
        )
        if config["protocol"] == "native":
            return RetrievalResult(
                native_context={"question_id": unit.unit_id},
                native_surface_diagnostics={"result_count": 1},
                cutoff_label="native",
                search_results=search_results,
                recall_diagnostics=diagnostics,
                memories_evaluated=1,
            )
        return RetrievalResult(
            search_results=search_results,
            recall_diagnostics=diagnostics,
            cutoff_label="top_1",
            memories_evaluated=1,
        )

    def retrieve_question(
        self,
        unit: BenchmarkUnit,
        _conversation: dict[str, Any],
        question: Any,
        config: dict[str, Any],
        _prepared: dict[str, Any],
        *,
        run_context: FrameworkRunContext,
    ) -> RetrievalResult:
        del run_context
        source_ids = [str(item) for item in question.qa_item.get("evidence", [])]
        search_results = (
            {"metadata": {"source_unit_ids": source_ids}},
        )
        diagnostics = (
            {
                "ranked_candidates_canonical": list(search_results),
                "final_candidates_canonical": list(search_results),
            }
            if self.name == "dmf"
            else {}
        )
        if config["protocol"] == "native":
            return RetrievalResult(
                native_context={
                    "conversation_id": unit.unit_id,
                    "question_id": question.question_id,
                },
                native_surface_diagnostics={"result_count": 1},
                cutoff_label="native",
                search_results=search_results,
                recall_diagnostics=diagnostics,
                memories_evaluated=1,
            )
        return RetrievalResult(
            search_results=search_results,
            recall_diagnostics=diagnostics,
            cutoff_label="top_1",
            memories_evaluated=1,
        )


class FixtureAnswerer:
    name = "fixture-answerer"

    def __init__(
        self,
        *,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.entered = entered
        self.release = release

    def generate(self, _request: AnswererRequest) -> dict[str, Any]:
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and not self.release.wait(timeout=5):
            raise RuntimeError("fixture answerer release timed out")
        return {
            "generated_answer": "fixture answer",
            "answerer_provider": "fixture",
            "answerer_requested_model": "fixture-answerer",
            "answerer_model": "fixture-answerer",
            "answerer_finish_reason": "stop",
            "answerer_usage": {"total_tokens": 1},
        }


class FixtureJudge:
    name = "fixture-judge"

    def judge(self, request: JudgeRequest) -> dict[str, Any]:
        return {
            "judgment": "CORRECT",
            "score": 1.0,
            "reason": "deterministic offline fixture",
            "judge_provider": "fixture",
            "judge_requested_model": "fixture-judge",
            "judge_model": "fixture-judge",
            "judge_finish_reason": "stop",
            "judge_usage": {"total_tokens": 1},
            "judge_fingerprint": judge_fingerprint(str(request.metadata["benchmark"])),
        }


class InterruptingAnswerer(FixtureAnswerer):
    def generate(self, _request: AnswererRequest) -> dict[str, Any]:
        raise RunInterrupted(signal.SIGTERM)


class FailingJudge(FixtureJudge):
    def judge(self, _request: JudgeRequest) -> dict[str, Any]:
        raise RuntimeError("scientific fixture failure")


def fixture_factories(
    *,
    answerer: FixtureAnswerer | None = None,
    judge: FixtureJudge | None = None,
) -> RuntimeFactories:
    return RuntimeFactories(
        benchmarks=benchmark_factories(),
        frameworks={
            "dmf": lambda _config: FixtureFramework("dmf"),
            "mem0": lambda _config: FixtureFramework("mem0"),
        },
        answerers={"fixture": lambda _config: answerer or FixtureAnswerer()},
        judges={
            ("locomo", "fixture"): lambda _config: judge or FixtureJudge(),
            ("longmemeval", "fixture"): lambda _config: judge or FixtureJudge(),
        },
    )


def application_builder(config: dict[str, Any], **kwargs: Any) -> Any:
    return assemble_application(config, factories=fixture_factories(), **kwargs)


def write_config(
    root: Path,
    *,
    benchmark: str,
    framework: str,
    protocol: str,
    run_id: str,
) -> Path:
    root.mkdir(parents=True)
    dataset_path = root / f"{benchmark}.json"
    shutil.copy2(FIXTURE_DIR / f"{benchmark}-mini.json", dataset_path)
    suffix = "toml" if framework == "dmf" else "yaml"
    framework_path = root / f"{framework}.{suffix}"
    framework_path.write_text(
        '[ltm]\nstorage_type = "qdrant"\n' if framework == "dmf" else "vector_store: qdrant\n",
        encoding="utf-8",
    )
    selection = (
        {"ordered_item_ids": ["conversation-0001"], "filters": {"categories": [1, 2]}, "seed": 7}
        if benchmark == "locomo"
        else {"ordered_item_ids": ["lme-001", "lme-002"], "filters": {}, "seed": 7}
    )
    answerer_model = {
        "provider": "fixture",
        "requested_model": "fixture-answerer",
        "parameters": {"temperature": 0, "max_tokens": 256},
        "runtime": {"timeout_seconds": 30, "rpm": 100, "max_retries": 0},
    }
    judge_model = {
        "provider": "fixture",
        "requested_model": "fixture-judge",
        "parameters": {"temperature": 0, "max_tokens": 256},
        "runtime": {"timeout_seconds": 30, "rpm": 100, "max_retries": 0},
    }
    config = {
        "schema_version": 1,
        "experiment_id": run_id,
        "benchmark": benchmark,
        "framework": framework,
        "protocol": protocol,
        "runtime": {
            "root": str(root),
            "runs_dir": str(root / "runs"),
            "cache_dir": str(root / "cache"),
            "metrics_port": 9464,
            "log_level": "INFO",
        },
        "framework_config": {
            "path": str(framework_path),
            "sha256": sha256_file(framework_path),
            "format": suffix,
            "profile": "fixture",
        },
        "qdrant": {
            "endpoint_env": "QDRANT_URL",
            "retention": "keep",
            "request_timeout_seconds": 10,
        },
        "dataset": {
            "name": benchmark,
            "path": str(dataset_path),
            "source": "fixture",
            "revision": "fixture-v1",
            "sha256": sha256_file(dataset_path),
        },
        "selection": selection,
        "models": {"answerer": answerer_model, "judge": judge_model},
        "evaluation": {"required": ["primary_judge_score"], "optional": []},
        "artifact_store": {"type": "local", "uri": str(root / "runs")},
    }
    path = root / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_cli_runs_full_offline_lifecycle_for_all_supported_combinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DMF_BENCH_METRICS_PORT", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak-runtime")

    executed: list[tuple[str, str, str]] = []
    for index, (benchmark, framework, protocol) in enumerate(supported_combinations()):
        root = tmp_path / f"case-{index}"
        run_id = f"cli-{benchmark}-{framework}-{protocol}"
        config_path = write_config(
            root,
            benchmark=benchmark,
            framework=framework,
            protocol=protocol,
            run_id=run_id,
        )

        assert main(["run", "--config", str(config_path)], application_builder=application_builder) == 0
        run_dir = root / "runs" / run_id
        final_dir = run_dir / "final"
        assert read_json(run_dir / "run-status.json")["state"] == "COMPLETED"
        assert read_json(final_dir / "COMPLETED.json")["state"] == "COMPLETED"

        evaluations = read_json(final_dir / "evaluations" / "evaluations.json")
        rigorous = read_json(final_dir / "evaluations" / "rigorous_report.json")
        ablation = read_json(final_dir / "evaluations" / "ablation_report.json")
        assert rigorous["metrics"]["overall"]["recall_at_k"] == 1.0
        assert ablation["status"] == (
            "COMPLETED" if framework == "dmf" else "NOT_APPLICABLE"
        )
        if framework == "dmf":
            assert ablation["metrics"]["stats"]["questions_without_diagnostics"] == 0

        executed.append((benchmark, framework, protocol))

    assert executed == supported_combinations()

    output = capsys.readouterr().out
    assert '"event":"run.preflight.started"' in output
    assert '"event":"run.completed"' in output
    assert "must-not-leak-runtime" not in output


def test_cli_predict_only_status_and_resume_use_persisted_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DMF_BENCH_METRICS_PORT", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak-runtime")
    root = tmp_path / "resume"
    run_id = "cli-resume"
    config_path = write_config(
        root,
        benchmark="longmemeval",
        framework="dmf",
        protocol="native",
        run_id=run_id,
    )

    assert main(
        ["run", "--config", str(config_path), "--predict-only"],
        application_builder=application_builder,
    ) == 0
    assert main(["status", "--runs-dir", str(root / "runs"), "--run-id", run_id]) == EXIT_INCOMPLETE

    source = json.loads(config_path.read_text(encoding="utf-8"))
    source["models"]["answerer"]["provider"] = "not-registered"
    config_path.write_text(json.dumps(source), encoding="utf-8")

    assert main(
        ["resume", "--runs-dir", str(root / "runs"), "--run-id", run_id],
        application_builder=application_builder,
    ) == 0
    run_dir = root / "runs" / run_id
    before = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert main(["status", "--runs-dir", str(root / "runs"), "--run-id", run_id]) == 0
    assert main(["health", "--runs-dir", str(root / "runs")]) == 0
    after = {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
    events = (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "run.preflight.started" in events
    assert "run.completed" in events
    assert "not-registered" not in events
    persisted = (run_dir / "resolved-config.json").read_text(encoding="utf-8")
    output = capsys.readouterr().out
    assert "must-not-leak-runtime" not in events
    assert "must-not-leak-runtime" not in persisted
    assert "must-not-leak-runtime" not in output


def test_invalid_run_config_fails_before_builder_or_run_creation(tmp_path: Path) -> None:
    root = tmp_path / "invalid"
    config_path = write_config(
        root,
        benchmark="locomo",
        framework="dmf",
        protocol="native",
        run_id="invalid-config",
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["protocol"] = "unsupported"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    builder_called = False

    def forbidden_builder(_config: dict[str, Any], **_kwargs: Any) -> Any:
        nonlocal builder_called
        builder_called = True
        raise AssertionError("builder must not be called")

    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--config", str(config_path)], application_builder=forbidden_builder)

    assert exc_info.value.code == 2
    assert builder_called is False
    assert not (root / "runs").exists()


def test_missing_openai_key_is_reported_in_json_log_and_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DMF_BENCH_METRICS_PORT", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = tmp_path / "missing-openai-key"
    config_path = write_config(
        root,
        benchmark="locomo",
        framework="dmf",
        protocol="native",
        run_id="missing-openai-key",
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for role in ("answerer", "judge"):
        payload["models"][role]["provider"] = "openai"
        payload["models"][role]["requested_model"] = "gpt-4.1-mini"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["run", "--config", str(config_path)]) == EXIT_RUNTIME

    captured = capsys.readouterr()
    expected = "Provider 'openai' requires OPENAI_API_KEY in the environment or .env file."
    failure_events = [
        json.loads(line)
        for line in captured.out.splitlines()
        if '"event":"run.failed"' in line
    ]
    assert len(failure_events) == 1
    assert failure_events[0]["protocol"] == "native"
    assert expected in failure_events[0]["message"]
    assert f"ValueError: {expected}" in captured.err
    assert not (root / "runs").exists()


def test_third_party_exception_is_reported_without_escaping_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class ThirdPartyFailure(Exception):
        pass

    monkeypatch.setenv("DMF_BENCH_METRICS_PORT", "0")
    root = tmp_path / "third-party-failure"
    config_path = write_config(
        root,
        benchmark="locomo",
        framework="dmf",
        protocol="native",
        run_id="third-party-failure",
    )

    def broken_builder(_config: dict[str, Any], **_kwargs: Any) -> Any:
        raise ThirdPartyFailure("ONNX model file is missing")

    assert main(
        ["run", "--config", str(config_path)],
        application_builder=broken_builder,
    ) == EXIT_RUNTIME

    captured = capsys.readouterr()
    failure_events = [
        json.loads(line)
        for line in captured.out.splitlines()
        if '"event":"run.failed"' in line
    ]
    assert len(failure_events) == 1
    assert failure_events[0]["error_type"] == "ThirdPartyFailure"
    assert "ONNX model file is missing" in failure_events[0]["message"]
    assert "ThirdPartyFailure: ONNX model file is missing" in captured.err
    assert not (root / "runs").exists()


def test_cli_exit_codes_distinguish_runtime_scientific_and_interrupt_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DMF_BENCH_METRICS_PORT", "0")

    runtime_root = tmp_path / "runtime"
    runtime_config = write_config(
        runtime_root,
        benchmark="longmemeval",
        framework="dmf",
        protocol="native",
        run_id="runtime-failure",
    )

    def broken_builder(_config: dict[str, Any], **_kwargs: Any) -> Any:
        raise RuntimeError("runtime fixture failure")

    assert main(
        ["run", "--config", str(runtime_config)],
        application_builder=broken_builder,
    ) == EXIT_RUNTIME
    assert not (runtime_root / "runs").exists()

    scientific_root = tmp_path / "scientific"
    scientific_config = write_config(
        scientific_root,
        benchmark="longmemeval",
        framework="dmf",
        protocol="native",
        run_id="scientific-failure",
    )

    def scientific_builder(config: dict[str, Any], **kwargs: Any) -> Any:
        return assemble_application(
            config,
            factories=fixture_factories(judge=FailingJudge()),
            **kwargs,
        )

    assert main(
        ["run", "--config", str(scientific_config)],
        application_builder=scientific_builder,
    ) == EXIT_SCIENTIFIC_FAILURE
    assert read_json(
        scientific_root / "runs" / "scientific-failure" / "run-status.json"
    )["state"] == "FAILED_JUDGING"

    interrupt_root = tmp_path / "interrupt-exit"
    interrupt_config = write_config(
        interrupt_root,
        benchmark="longmemeval",
        framework="dmf",
        protocol="native",
        run_id="interrupt-exit",
    )

    def interrupt_builder(config: dict[str, Any], **kwargs: Any) -> Any:
        return assemble_application(
            config,
            factories=fixture_factories(answerer=InterruptingAnswerer()),
            **kwargs,
        )

    assert main(
        ["run", "--config", str(interrupt_config)],
        application_builder=interrupt_builder,
    ) == 128 + signal.SIGTERM
    assert read_json(
        interrupt_root / "runs" / "interrupt-exit" / "run-status.json"
    )["state"] == "INTERRUPTED"
    assert main(["health", "--runs-dir", str(tmp_path / "missing")]) == EXIT_RUNTIME
    capsys.readouterr()


def test_metrics_are_scrapable_while_answerer_is_blocked(tmp_path: Path) -> None:
    root = tmp_path / "metrics"
    config_path = write_config(
        root,
        benchmark="longmemeval",
        framework="dmf",
        protocol="native",
        run_id="metrics-live",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entered = threading.Event()
    release = threading.Event()
    metrics = BenchmarkMetrics()
    application = assemble_application(
        config,
        metrics=metrics,
        factories=fixture_factories(answerer=FixtureAnswerer(entered=entered, release=release)),
    )
    server = start_metrics_endpoint(metrics=metrics, port=0, address="127.0.0.1")
    failure: list[BaseException] = []

    def execute() -> None:
        try:
            application.run(config)
        except BaseException as exc:  # pragma: no cover - asserted below
            failure.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        assert read_json(root / "runs" / "metrics-live" / "run-status.json")["state"] == "RUNNING"
        body = urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/metrics",
            timeout=2,
        ).read().decode("utf-8")
        assert "dmf_bench_run_progress_ratio" in body
        assert "dmf_bench_run_expected_items" in body
    finally:
        release.set()
        worker.join(timeout=10)
        stop_metrics_endpoint(server)

    assert not worker.is_alive()
    assert not failure


def test_signal_cancellation_interrupts_at_boundary_and_resume_completes(tmp_path: Path) -> None:
    root = tmp_path / "interrupt"
    config_path = write_config(
        root,
        benchmark="longmemeval",
        framework="dmf",
        protocol="native",
        run_id="signal-resume",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entered = threading.Event()
    release = threading.Event()
    controller = CancellationController()
    application = assemble_application(
        config,
        factories=fixture_factories(answerer=FixtureAnswerer(entered=entered, release=release)),
        events=JsonEventLogger(stream=io.StringIO()),
    )
    failures: list[BaseException] = []

    def execute() -> None:
        try:
            application.run(config, cancel_check=controller.check)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    assert entered.wait(timeout=5)
    controller.request(signal.SIGTERM)
    release.set()
    worker.join(timeout=10)

    assert len(failures) == 1
    assert isinstance(failures[0], RunInterrupted)
    run_dir = root / "runs" / "signal-resume"
    assert read_json(run_dir / "run-status.json")["state"] == "INTERRUPTED"
    checkpoint = read_json(run_dir / "checkpoints" / "lme-001" / "checkpoint.json")
    assert checkpoint["status"] != "COMMITTED"

    resumed = assemble_application(config, factories=fixture_factories()).run(
        config,
        run_id="signal-resume",
        resume=True,
    )
    assert resumed.state == "COMPLETED"


def test_installed_signal_handler_records_sigterm_without_terminating_process() -> None:
    controller = CancellationController()

    with controller.installed():
        signal.raise_signal(signal.SIGTERM)
        with pytest.raises(RunInterrupted) as exc_info:
            controller.check()

    assert exc_info.value.exit_code == 128 + signal.SIGTERM
