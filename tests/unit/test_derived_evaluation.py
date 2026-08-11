from pathlib import Path

from fastapi.testclient import TestClient

from dmf_bench.api import create_app
from dmf_bench.artifacts import LocalArtifactStore
from dmf_bench.atomic_io import read_json, write_json_atomic
from dmf_bench.contracts import RunManifest
from dmf_bench.derived_evaluation import derive_locomo_evaluation


def _commit_locomo_run(runs_dir: Path) -> None:
    store = LocalArtifactStore(runs_dir)
    run_dir = store.create_run(
        RunManifest(
            run_id="locomo-source",
            scientific_fingerprint="d" * 64,
            fingerprint_inputs={"benchmark": "locomo", "framework": "dmf"},
            expected_item_ids=("conversation-1",),
            atomic_unit="conversation",
        )
    )
    write_json_atomic(
        run_dir / "evaluations" / "evaluations.json",
        [
            {
                "generated_answer": "Pixel",
                "ground_truth_answer": "Pixel",
                "category_name": "single-hop",
                "cutoff_label": "native",
                "evidence": ["D1:1"],
                "retrieval": {
                    "memories_evaluated": 1,
                    "search_results": [
                        {
                            "metadata": {
                                "source_unit_id": "D1:1",
                                "source_unit_ids": ["D1:1", "D1:2"],
                            }
                        }
                    ],
                    "recall_diagnostics": {},
                },
            }
        ],
    )
    staged = store.stage("locomo-source", run_dir)
    receipt = store.verify(staged)
    store.commit(staged, receipt)


def test_derive_locomo_evaluation_is_immutable_and_idempotent(tmp_path: Path) -> None:
    _commit_locomo_run(tmp_path)

    first = derive_locomo_evaluation(run_id="locomo-source", runs_dir=tmp_path)
    second = derive_locomo_evaluation(run_id="locomo-source", runs_dir=tmp_path)

    assert first["reused"] is False
    assert second["reused"] is True
    output = Path(str(first["output_dir"]))
    rigorous = read_json(output / "rigorous_report.json")
    assert rigorous["evaluator_version"] == "locomo-evaluator-v2"
    assert rigorous["metrics"]["overall"]["ndcg_at_k"] == 1.0
    derivation = read_json(output / "derivation.json")
    assert derivation["source_run_id"] == "locomo-source"
    assert len(derivation["source_final_manifest_sha256"]) == 64


def test_artifact_api_serves_derived_evaluation_reports(tmp_path: Path) -> None:
    _commit_locomo_run(tmp_path)
    derive_locomo_evaluation(run_id="locomo-source", runs_dir=tmp_path)
    client = TestClient(create_app(tmp_path))

    listing = client.get("/runs/locomo-source/derived-evaluations")
    report = client.get(
        "/runs/locomo-source/derived-evaluations/"
        "locomo-evaluator-v2/rigorous_report.json"
    )

    assert listing.json()["evaluator_versions"] == ["locomo-evaluator-v2"]
    assert report.status_code == 200
    assert report.json()["metrics"]["overall"]["ndcg_at_k"] == 1.0
