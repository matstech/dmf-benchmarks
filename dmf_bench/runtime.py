"""Explicit runtime component assembly without provider-side effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from dmf_bench.adapters.base import AnswererAdapter, BenchmarkAdapter, JudgeAdapter
from dmf_bench.adapters.dmf import dmf_framework_factories
from dmf_bench.benchmarks.locomo.adapter import LoCoMoAdapter
from dmf_bench.adapters.longmemeval import LongMemEvalAdapter
from dmf_bench.adapters.mem0 import mem0_framework_factories
from dmf_bench.adapters.providers import answerer_factories, judge_factories
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.evaluation import OfflineFullLifecycleRunner
from dmf_bench.logging_config import JsonEventLogger
from dmf_bench.metrics import BenchmarkMetrics
from dmf_bench.registry import validate_combination
from dmf_bench.runner import LoCoMoPredictOnlyRunner, LongMemEvalPredictOnlyRunner


ComponentFactory = Callable[[dict[str, Any]], Any]


class RuntimeAssemblyError(RuntimeError):
    """Raised when an explicit runtime component is not registered."""


@dataclass(frozen=True)
class RuntimeRequest:
    benchmark: str
    framework: str
    answerer_provider: str
    judge_provider: str

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RuntimeRequest":
        models = _mapping(config.get("models"))
        answerer = _mapping(models.get("answerer"))
        judge = _mapping(models.get("judge"))
        request = cls(
            benchmark=_required_string(config, "benchmark"),
            framework=_required_string(config, "framework"),
            answerer_provider=_required_string(answerer, "provider"),
            judge_provider=_required_string(judge, "provider"),
        )
        validate_combination(request.benchmark, request.framework)
        return request


@dataclass(frozen=True)
class RuntimeFactories:
    benchmarks: Mapping[str, ComponentFactory] = field(default_factory=dict)
    frameworks: Mapping[str, ComponentFactory] = field(default_factory=dict)
    answerers: Mapping[str, ComponentFactory] = field(default_factory=dict)
    judges: Mapping[tuple[str, str], ComponentFactory] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeComponents:
    request: RuntimeRequest
    benchmark: BenchmarkAdapter
    framework: Any
    answerer: AnswererAdapter
    judge: JudgeAdapter


@dataclass(frozen=True)
class RuntimeApplication:
    """Fully assembled benchmark process for one resolved experiment config."""

    components: RuntimeComponents
    artifact_store: LocalArtifactStore
    prediction_runner: Any
    full_runner: OfflineFullLifecycleRunner
    metrics: BenchmarkMetrics

    def run(
        self,
        config: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
        predict_only: bool = False,
        cancel_check: Callable[[], None] | None = None,
        on_run_ready: Callable[[Path, Any], None] | None = None,
    ) -> Any:
        if predict_only:
            return self.prediction_runner.run(
                config,
                run_id=run_id,
                resume=resume,
                cancel_check=cancel_check,
                _on_run_ready=on_run_ready,
            )
        return self.full_runner.run(
            config,
            run_id=run_id,
            resume=resume,
            cancel_check=cancel_check,
            on_run_ready=on_run_ready,
        )


def assemble_runtime(
    config: dict[str, Any],
    *,
    factories: RuntimeFactories,
) -> RuntimeComponents:
    """Build one explicitly registered component set without hidden fallback."""
    request = RuntimeRequest.from_config(config)
    benchmark = _build(factories.benchmarks, request.benchmark, config, "benchmark")
    answerer = _build(
        factories.answerers,
        request.answerer_provider,
        config,
        "answerer provider",
    )
    judge = _build(
        factories.judges,
        (request.benchmark, request.judge_provider),
        config,
        "judge",
    )
    framework = _build(factories.frameworks, request.framework, config, "framework")
    return RuntimeComponents(
        request=request,
        benchmark=benchmark,
        framework=framework,
        answerer=answerer,
        judge=judge,
    )


def benchmark_factories() -> dict[str, ComponentFactory]:
    """Return built-in benchmark factories; runtime/provider factories arrive later."""
    return {
        "locomo": lambda _config: LoCoMoAdapter(),
        "longmemeval": lambda _config: LongMemEvalAdapter(),
    }


def default_runtime_factories(
    *,
    metrics: BenchmarkMetrics | None = None,
) -> RuntimeFactories:
    return RuntimeFactories(
        benchmarks=benchmark_factories(),
        frameworks={
            **dmf_framework_factories(metrics=metrics),
            **mem0_framework_factories(metrics=metrics),
        },
        answerers=answerer_factories(metrics=metrics),
        judges=judge_factories(metrics=metrics),
    )


def assemble_application(
    config: dict[str, Any],
    *,
    metrics: BenchmarkMetrics | None = None,
    factories: RuntimeFactories | None = None,
    events: JsonEventLogger | None = None,
) -> RuntimeApplication:
    """Build the real benchmark application without hidden runtime fallbacks."""
    metrics_registry = metrics or BenchmarkMetrics()
    artifact_config = _mapping(config.get("artifact_store"))
    if artifact_config.get("type") != "local":
        raise RuntimeAssemblyError("The executable runtime supports only local artifact storage.")
    runtime_config = _mapping(config.get("runtime"))
    runs_dir = Path(_required_string(runtime_config, "runs_dir"))
    artifact_uri = Path(_required_string(artifact_config, "uri"))
    if not runs_dir.is_absolute() or not artifact_uri.is_absolute():
        raise RuntimeAssemblyError("runtime.runs_dir and artifact_store.uri must be absolute.")
    if runs_dir != artifact_uri:
        raise RuntimeAssemblyError(
            "artifact_store.uri must match runtime.runs_dir for local-only publication."
        )

    components = assemble_runtime(
        config,
        factories=factories or default_runtime_factories(metrics=metrics_registry),
    )
    store = LocalArtifactStore(runs_dir)
    if components.request.benchmark == "longmemeval":
        prediction_runner: Any = LongMemEvalPredictOnlyRunner(
            benchmark=components.benchmark,
            artifact_store=store,
            framework=components.framework,
            answerer=components.answerer,
            metrics=metrics_registry,
            events=events,
        )
    elif components.request.benchmark == "locomo":
        prediction_runner = LoCoMoPredictOnlyRunner(
            benchmark=components.benchmark,
            artifact_store=store,
            framework=components.framework,
            answerer=components.answerer,
            metrics=metrics_registry,
            events=events,
        )
    else:  # pragma: no cover - guarded by RuntimeRequest registry validation
        raise RuntimeAssemblyError(
            f"Unsupported executable benchmark: {components.request.benchmark!r}."
        )
    full_runner = OfflineFullLifecycleRunner(
        prediction_runner=prediction_runner,
        artifact_store=store,
        judge=components.judge,
        metrics=metrics_registry,
        events=events,
    )
    return RuntimeApplication(
        components=components,
        artifact_store=store,
        prediction_runner=prediction_runner,
        full_runner=full_runner,
        metrics=metrics_registry,
    )


def _build(
    registry: Mapping[Any, ComponentFactory],
    key: Any,
    config: dict[str, Any],
    component: str,
) -> Any:
    try:
        factory = registry[key]
    except KeyError as exc:
        raise RuntimeAssemblyError(
            f"No {component} factory registered for {key!r}."
        ) from exc
    return factory(config)


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
