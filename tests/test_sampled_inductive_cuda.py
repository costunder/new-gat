"""Explicit synthetic CUDA study execution; not PPI benchmark evidence."""

import pytest
import torch

from experiments.sampled_inductive import runner, train
from tests.test_aggregation_comparison_cuda import cuda_required  # noqa: F401
from tests.test_sampled_inductive import synthetic_payload


@pytest.fixture(scope="module")
def trained_matrix(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("debug-synthetic-inductive-matrix")
    payload = synthetic_payload()
    options = runner.parser().parse_args(
        [
            "--run-id",
            "debug-synthetic-only",
            "--context-seeds",
            "16",
            "--context-batches",
            "4",
            "8",
            "--epochs",
            "4",
            "--num-neighbors",
            "2",
            "--edge-chunk-size",
            "128",
        ]
    )
    protocol = {"explicit_synthetic_debug": True}
    manifest = {
        "identity": {"configuration": runner.configuration(options)},
        "results": {},
        "selected_resources": {m: {"batch": 4, "workers": 0} for m in ("full", "sampled")},
    }
    for seed, mode, arm in runner.cells(options):
        args = runner.child_arguments(options, arm, seed, batch=4, workers=0)
        key = f"seed-{seed}/{mode}/{arm}"
        manifest["results"][key] = train.train_cell(
            payload, protocol, args, mode, 4, 0, torch.device("cuda:0"), tmp_path / key
        )
    return payload, protocol, options, manifest, tmp_path


@pytest.mark.parametrize("mode", ["full", "sampled"])
@pytest.mark.parametrize("arm", ["incidence_fixed", "incidence"])
def test_full_and_sampled_gpu_training_complete_and_read_only_audit(mode, arm, trained_matrix):
    _, _, _, manifest, root = trained_matrix
    key = f"seed-0/{mode}/{arm}"
    result, folder = manifest["results"][key], root / key
    assert result["status"] == "passed" and len(result["audit"]) == 5
    assert all(r["coverage"]["supervised_nodes"] == 284 for r in result["history"])
    assert result["validation"]["graph_ids"] == [4]
    assert not result["test_evaluated"]
    assert all(
        "c_solver_including_checkpoint_recompute" in r["stage_seconds"]["cuda_event_seconds"]
        for r in result["history"]
    )
    assert train.completed(folder) == result


def test_heldout_test_matrix_is_frozen_after_validation_selection(trained_matrix):
    import copy

    from experiments.sampled_inductive.report import summarize

    payload, protocol, options, manifest, root = trained_matrix
    partial = copy.deepcopy(manifest)
    partial["results"].pop(next(iter(partial["results"])))
    with pytest.raises(ValueError, match="every validation-selected cell"):
        runner.evaluate_test(
            options, payload, protocol, partial, root, torch.device("cuda:0"), lambda: None
        )
    original = {key: row["last_sha256"] for key, row in manifest["results"].items()}
    runner.evaluate_test(
        options, payload, protocol, manifest, root, torch.device("cuda:0"), lambda: None
    )
    assert manifest["test_checkpoint_lock"] == original
    assert len(manifest["test"]) == 4
    assert all(row["evaluation"]["graph_ids"] == [5] for row in manifest["test"].values())
    report = summarize(manifest)
    assert report["seeds"][0]["status"] == "passed"
    assert len(report["seeds"][0]["effects"]) == 5
    # A completed rerun verifies stored test rows without rerunning inference.
    runner.evaluate_test(
        options, payload, protocol, manifest, root, torch.device("cuda:0"), lambda: None
    )
    altered = copy.deepcopy(manifest)
    altered["test"][next(iter(altered["test"]))]["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint identity"):
        runner.evaluate_test(
            options, payload, protocol, altered, root, torch.device("cuda:0"), lambda: None
        )


def test_gpu_probe_covers_real_updates_validation_and_coverage():
    options = runner.parser().parse_args(
        [
            "--run-id",
            "debug-probe",
            "--context-seeds",
            "16",
            "--context-batches",
            "4",
            "8",
            "--epochs",
            "4",
            "--num-neighbors",
            "2",
            "--edge-chunk-size",
            "128",
        ]
    )
    args = runner.child_arguments(options, "incidence", 0, batch=4, workers=0)
    result = train.probe(synthetic_payload(), args, "sampled", 4, 0, torch.device("cuda:0"))
    assert result["status"] == "passed" and result["optimizer_steps"] >= 5
    assert result["peak_allocated_bytes"] > 0 and result["validation_seconds"] > 0
    assert all(r["coverage"]["seed_coverage"] == 1 for r in result["epoch_evidence"])


def test_midrun_failure_resumes_identical_gpu_parameters(monkeypatch, tmp_path):
    payload = synthetic_payload()
    options = runner.parser().parse_args(
        [
            "--run-id",
            "debug-resume",
            "--context-seeds",
            "16",
            "--context-batches",
            "4",
            "8",
            "--epochs",
            "4",
            "--num-neighbors",
            "2",
            "--edge-chunk-size",
            "128",
        ]
    )
    args = runner.child_arguments(options, "incidence_fixed", 0, batch=4, workers=0)
    device = torch.device("cuda:0")
    protocol = {"explicit_synthetic_debug": True}
    reference = train.train_cell(
        payload, protocol, args, "sampled", 4, 0, device, tmp_path / "reference"
    )
    original = train.train_epoch

    def fail(*values):
        if values[-1] == 3:
            raise RuntimeError("explicit interruption before epoch 3")
        return original(*values)

    monkeypatch.setattr(train, "train_epoch", fail)
    with pytest.raises(RuntimeError, match="explicit interruption"):
        train.train_cell(payload, protocol, args, "sampled", 4, 0, device, tmp_path / "resume")
    monkeypatch.setattr(train, "train_epoch", original)
    resumed = train.train_cell(
        payload, protocol, args, "sampled", 4, 0, device, tmp_path / "resume"
    )
    assert resumed["validation"]["counts"] == reference["validation"]["counts"]
    assert resumed["resumed"] and not resumed["cost_comparable"]
    a = train.engine.base.load_checkpoint_on_cpu(tmp_path / "reference" / "last.pt")
    b = train.engine.base.load_checkpoint_on_cpu(tmp_path / "resume" / "last.pt")
    for key, value in a["model_state"].items():
        torch.testing.assert_close(value, b["model_state"][key], rtol=0, atol=0)
