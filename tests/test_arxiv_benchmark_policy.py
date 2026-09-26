"""Production scope and evaluation barriers; no CPU model computation."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.aggregation_comparison import final_test, runner
from experiments.aggregation_comparison.benchmark_policy import require_benchmark_datasets
from experiments.sampled_inductive import runner as retired


@pytest.mark.parametrize("dataset", ["ppi", "cora", "citeseer", "pubmed"])
def test_production_rejects_old_datasets_before_io(dataset, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("rejected benchmark must not load data or probe a GPU")

    monkeypatch.setattr(runner.standalone, "check_dependencies", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(
            [
                "--run-id",
                "debug-reject",
                "--datasets",
                dataset,
                "--profiles",
                "reference",
                "--dry-run",
            ]
        )
    assert error.value.code == 2
    with pytest.raises(ValueError, match="ogbn-arxiv"):
        require_benchmark_datasets([dataset])


def test_retired_ppi_entry_point_cannot_start_training():
    with pytest.raises(ValueError, match="retired"):
        retired.main([])


def test_default_plan_has_arxiv_all_baselines_and_original_internal_arms():
    args = runner.parser().parse_args(["--run-id", "debug-plan", "--profiles", "reference"])
    assert args.datasets == ["ogbn-arxiv"] and args.hardware_profile == "portable"
    assert {"gcn", "graphsage", "gatv2"} <= set(args.arms)
    assert sum(arm.startswith("incidence") for arm in args.arms) == 16
    jobs = runner.make_jobs(args, Path("results/debug-plan"))
    assert len(jobs) == 21
    assert all(j["dataset"] == "ogbn-arxiv" for j in jobs)
    assert "--evaluate-test" in runner.parser().format_help()
    assert "ppi" not in runner.parser().format_help()


@pytest.mark.parametrize("defect", ["missing", "duplicate", "later_audit", "later_disk"])
def test_entire_matrix_must_pass_before_test_data_access(monkeypatch, tmp_path, defect):
    args = SimpleNamespace(
        datasets=["ogbn-arxiv"], profiles=["reference"], model_seeds=[0], arms=["gcn", "graphsage"]
    )
    log = tmp_path / "explicit-metadata-fixture.log"
    log.write_text("mock audit evidence, not model output", encoding="utf-8")
    hashes = final_test.engine.implementation_source_hashes()
    jobs = [
        {
            "job_id": f"reference/ogbn-arxiv/model-seed-0/{arm}",
            "status": "passed",
            "result": {"checkpoint_sha256": arm},
            "audit": {
                "status": "passed",
                "checkpoint_sha256": arm,
                "log_path": str(log),
                "log_sha256": final_test.engine.base.sha256_file(log),
                "evaluator_source_sha256": hashes,
            },
        }
        for arm in args.arms
    ]
    manifest = {"jobs": copy.deepcopy(jobs)}
    if defect == "missing":
        manifest["jobs"].pop()
    elif defect == "duplicate":
        manifest["jobs"][1] = copy.deepcopy(manifest["jobs"][0])
    elif defect == "later_audit":
        manifest["jobs"][1]["audit"]["status"] = "failed"

    def read(job):
        if defect == "later_disk" and job["job_id"].endswith("graphsage"):
            raise ValueError("injected later disk corruption")
        return job["result"]

    def forbidden(*args, **kwargs):
        pytest.fail("test data touched before all cells were validated")

    monkeypatch.setattr(runner, "_read_result", read)
    monkeypatch.setattr(final_test.engine.base, "load_dataset", forbidden)
    with pytest.raises(ValueError):
        final_test.evaluate_matrix(args, manifest, lambda: None)
    assert "official_test_checkpoint_lock" not in manifest
