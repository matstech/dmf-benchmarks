"""BEAM lifecycle checks with a portable miniature materialized release."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dmf_bench.adapters.baselines import FullContextAdapter, ModelOnlyAdapter, VectorRagAdapter
from dmf_bench.adapters.base import FrameworkRunContext
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.benchmarks.beam.adapter import BeamAdapter
from dmf_bench.benchmarks.beam.judge import parse_rubric_score
from dmf_bench.benchmarks.beam.spec import ABILITIES, CODE_REVISION, DATASET_REVISIONS
from dmf_bench.contracts import sha256_file
from dmf_bench.providers.openai_compatible import ProviderResponseError
from dmf_bench.retrieval_qa import RetrievalQAPredictOnlyRunner
from dmf_bench.runner import InjectedInterrupt
from dmf_bench.state import plan_resume
from scripts.prepare_beam_configs import ROOT, build_config


MANIFEST = ROOT / "datasets/beam/manifest.json"


def fixture_manifest(tmp_path: Path) -> Path:
    root = ROOT / ".dmf-bench" / "beam-tests" / tmp_path.parent.name / tmp_path.name / "dataset"
    unit_id = "beam-100k-001"
    unit_dir = root / "units" / unit_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    events = [{
        "event_id": f"{unit_id}--e{index:07d}", "session_id": f"{unit_id}--p00-b0000",
        "sequence": index, "role": "user" if index % 2 == 0 else "assistant",
        "content": f"Fact {index}: " + ("a memory detail " * 8),
        "occurred_at": None, "source_refs": [f"{unit_id}--e{index:07d}"],
        "metadata": {},
    } for index in range(100)]
    queries = [{
        "query_id": f"{unit_id}--q-{ability}-{index + 1:02d}",
        "text": f"What is the {ability} fact {index + 1}?",
        "ability": ability, "difficulty": "easy",
    } for ability in ABILITIES for index in range(2)]
    references = [{
        "query_id": row["query_id"], "expected_answer": "fixture",
        "rubric": ["The answer is relevant."], "source_chat_ids": [],
    } for row in queries]
    files = {}
    for name, payload in (("events", events), ("queries", queries),
                          ("references", references)):
        path = unit_dir / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        files[f"{name}_path"] = path.relative_to(root).as_posix()
        files[f"{name}_sha256"] = sha256_file(path)
        if name == "events":
            files["events_bytes"] = path.stat().st_size
    context_chars = sum(len(event["content"]) for event in events)
    manifest = {
        "schema_version": 1, "benchmark": "beam", "code_revision": CODE_REVISION,
        "dataset_revisions": DATASET_REVISIONS, "abilities": list(ABILITIES),
        "records": [{"unit_id": unit_id, "scale": "100K", "category": "fixture",
                     "event_count": len(events), "question_count": len(queries),
                     "context_chars": context_chars,
                     "estimated_tokens_chars_div_4": (context_chars + 3) // 4,
                     **files}],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class CountingMemory(ModelOnlyAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.preparations = 0

    def prepare_unit_v3(self, unit, events, config, run_context):
        self.preparations += 1
        return super().prepare_unit_v3(unit, events, config, run_context)


class CountingAnswerer:
    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return {"generated_answer": "fixture", "answerer_usage": {"total_tokens": 1}}


def setup(tmp_path: Path, *, queries: int = 2):
    dataset_manifest = fixture_manifest(tmp_path)
    config = build_config(profile="beam-128k-pilot-v1", framework="model-only",
                          smoke=True, runtime_root=ROOT, dataset_manifest=dataset_manifest)
    runs_dir = ROOT / ".dmf-bench" / "beam-tests" / tmp_path.parent.name / tmp_path.name / "runs"
    config["runtime"]["runs_dir"] = str(runs_dir)
    config["artifact_store"]["uri"] = str(runs_dir)
    config["selection"]["filters"]["max_queries_per_unit"] = queries
    config["storage"]["retention"] = "keep"
    framework = CountingMemory(config)
    answerer = CountingAnswerer()
    runner = RetrievalQAPredictOnlyRunner(
        benchmark=BeamAdapter(), framework=framework, answerer=answerer,
        artifact_store=LocalArtifactStore(runs_dir),
    )
    return config, framework, answerer, runner


def test_case_rubric_isolation(tmp_path):
    config, _, _, _ = setup(tmp_path)
    case = BeamAdapter().load_case(BeamAdapter().enumerate_units(config)[0], config)
    assert len(case.queries) == 2
    assert len(case.events) > 0
    assert "rubric" not in json.dumps([event.to_dict() for event in case.events])
    assert "rubric" not in json.dumps([query.to_dict() for query in case.queries])
    assert case.references[case.queries[0].query_id].strata["rubric"]
    request = BeamAdapter().build_answerer_request_v3(
        case.queries[0], type("EmptyContext", (), {"text": ""})(), config,
    )
    assert request.user_prompt == case.queries[0].text


@pytest.mark.skipif(not MANIFEST.is_file(), reason="BEAM source data is local only")
def test_pinned_materialized_counts():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 100
    assert sum(row["question_count"] for row in manifest["records"]) == 2000


def test_full_context_is_bounded_on_pilot_conversation(tmp_path):
    dataset_manifest = fixture_manifest(tmp_path)
    config = build_config(profile="beam-128k-pilot-v1", framework="full-context",
                          smoke=True, runtime_root=ROOT, dataset_manifest=dataset_manifest)
    benchmark = BeamAdapter()
    case = benchmark.load_case(benchmark.enumerate_units(config)[0], config)
    framework = FullContextAdapter(config)
    context = FrameworkRunContext("local-check", "0" * 64, tmp_path)
    prepared = framework.prepare_unit_v3(case.unit, case.events, config, context)
    retrieval = framework.retrieve_v3(case.unit, case.queries[0], prepared, config, context)
    assert retrieval.diagnostics["baseline_family"] == "bounded-full-context"
    assert len(retrieval.diagnostics["dropped_event_ids"]) > 0


def test_ingested_resume_reuses_answer_and_refuses_changed_fingerprint(tmp_path):
    config, framework, answerer, runner = setup(tmp_path)
    with pytest.raises(InjectedInterrupt):
        runner.run(config, interrupt_at="after-first-prediction")
    run_dir = Path(config["runtime"]["runs_dir"]) / config["experiment_id"]
    assert plan_resume(run_dir).next_phase == "RUNNING"
    assert framework.preparations == 1
    assert len(answerer.requests) == 1
    result = runner.run(config, resume=True)
    assert result.committed_unit_ids == result.expected_unit_ids
    assert framework.preparations == 1
    assert len(answerer.requests) == 2
    changed = copy.deepcopy(config)
    changed["context_budget"]["max_tokens"] += 1
    with pytest.raises(Exception, match="manifest mismatch"):
        runner.run(changed, resume=True)


def test_ingested_barrier_and_corrupt_question_trigger_safe_recompute(tmp_path):
    config, framework, answerer, runner = setup(tmp_path)
    with pytest.raises(InjectedInterrupt):
        runner.run(config, interrupt_at="after-ingested")
    assert framework.preparations == 1
    assert not answerer.requests
    with pytest.raises(InjectedInterrupt):
        runner.run(config, resume=True, interrupt_at="after-first-prediction")
    assert framework.preparations == 1
    assert len(answerer.requests) == 1
    run_dir = Path(config["runtime"]["runs_dir"]) / config["experiment_id"]
    question = BeamAdapter().enumerate_units(config)[0].item_ids[0]
    prediction = run_dir / "items" / "beam-100k-001" / "questions" / f"{question}.json"
    prediction.write_text("{}", encoding="utf-8")
    runner.run(config, resume=True)
    assert framework.preparations == 1
    assert len(answerer.requests) == 3


def test_interrupt_before_ingestion_restarts_only_unit(tmp_path):
    config, framework, answerer, runner = setup(tmp_path, queries=1)
    with pytest.raises(InjectedInterrupt):
        runner.run(config, interrupt_at="before-ingestion")
    assert framework.preparations == 0
    runner.run(config, resume=True)
    assert framework.preparations == 1
    assert len(answerer.requests) == 1


def test_vector_index_restores_and_cleanup_stays_owned(tmp_path):
    dataset_manifest = fixture_manifest(tmp_path)
    config = build_config(profile="beam-128k-pilot-v1", framework="vector-rag",
                          smoke=True, runtime_root=ROOT, dataset_manifest=dataset_manifest)
    runs_dir = ROOT / ".dmf-bench" / "beam-tests" / tmp_path.parent.name / tmp_path.name / "vector-runs"
    config["runtime"]["runs_dir"] = str(runs_dir)
    config["artifact_store"]["uri"] = str(runs_dir)
    config["selection"]["filters"]["max_queries_per_unit"] = 1
    config["storage"]["retention"] = "keep"
    embed_calls = []

    def embed(model, texts, base_url):
        embed_calls.append(len(texts))
        return [[float(len(text) % 11), 1.0] for text in texts]

    framework = VectorRagAdapter(config, embed=embed)
    answerer = CountingAnswerer()
    runner = RetrievalQAPredictOnlyRunner(
        benchmark=BeamAdapter(), framework=framework, answerer=answerer,
        artifact_store=LocalArtifactStore(runs_dir),
    )
    with pytest.raises(InjectedInterrupt):
        runner.run(config, interrupt_at="after-ingested")
    run_dir = runs_dir / config["experiment_id"]
    indices = list((run_dir / "baseline" / "vector-rag").glob("*/index.json"))
    assert len(indices) == 1
    index = indices[0]
    initial_inode = index.stat().st_ino
    outside = run_dir / "baseline" / "unowned.txt"
    outside.write_text("keep", encoding="utf-8")
    assert len(embed_calls) == 1
    with pytest.raises(InjectedInterrupt):
        runner.run(config, resume=True, interrupt_at="after-first-prediction")
    assert index.stat().st_ino == initial_inode
    assert len(embed_calls) == 2
    index.write_text("{}", encoding="utf-8")
    runner.run(config, resume=True)
    assert len(embed_calls) == 4
    assert outside.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("raw,expected", [
    ('{"score": 0, "reason": "no"}', 0.0),
    ('```json\n{"score": 0.5, "reason": "partial"}\n```', 0.5),
    ('{"score": 1, "reason": "yes"}', 1.0),
])
def test_golden_rubric_parser(raw, expected):
    assert parse_rubric_score(raw)[0] == expected


@pytest.mark.parametrize("raw", ["", "not json", '{"score": 0}',
                                     '{"score": 2, "reason": "bad"}',
                                     '{"score": 0, "reason": ""}'])
def test_bad_rubric_response_raises(raw):
    with pytest.raises(ProviderResponseError):
        parse_rubric_score(raw)
