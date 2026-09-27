"""CUDA baselines and official-mask evaluation with explicit synthetic debug inputs."""

import copy
import json

import pytest
import torch
from torch.nn import functional as F

from experiments.aggregation_comparison import audit, calibration, engine, final_test, runner
from experiments.aggregation_comparison.model import AggregationClassifier
from tests.test_aggregation_comparison_cuda import cuda_required  # noqa: F401
from tests.test_aggregation_sampling_path_cuda import transductive_payload


@pytest.mark.parametrize("arm", ["gcn", "graphsage"])
def test_reference_baseline_matches_independent_dense_operator_and_gradients(arm):
    from types import SimpleNamespace

    torch.manual_seed(772)
    graph = SimpleNamespace(
        x=torch.randn(9, 5, device="cuda", requires_grad=True),
        incidence_edge_index=torch.tensor([[0, 0, 1, 4, 5], [1, 2, 3, 5, 6]], device="cuda"),
        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2], device="cuda"),
        _v5_num_graphs=3,
    )

    def make():
        return AggregationClassifier(
            5,
            3,
            arm=arm,
            selection_config={"condition": "full"},
            hidden_channels=256,
            layers=8,
            heads=8,
            dropout=0,
        ).cuda()

    actual, reference = make(), make()
    reference.load_state_dict(actual.state_dict())
    dense = torch.zeros(9, 9, device="cuda")
    edges = graph.incidence_edge_index
    dense[edges[0], edges[1]] = 1
    dense[edges[1], edges[0]] = 1
    if arm == "gcn":
        dense += torch.eye(9, device="cuda")
        degree = dense.sum(1).sqrt()
        support = dense / degree[:, None] / degree[None, :]
    else:
        support = dense / dense.sum(1).clamp_min(1)[:, None]
    value = reference.encoder(graph.x)
    for layer in reference.layers:
        if arm == "gcn":
            value = support @ F.linear(value, layer.lin.weight) + layer.bias
        else:
            value = F.linear(support @ value, layer.lin_l.weight, layer.lin_l.bias) + F.linear(
                value,
                layer.lin_r.weight,
            )
        value = value.relu()
    expected, output = reference.decoder(value), actual(graph)
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
    left = torch.autograd.grad(output.square().sum(), (graph.x, *actual.parameters()))
    right = torch.autograd.grad(expected.square().sum(), (graph.x, *reference.parameters()))
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    cache = getattr(graph, "_comparison_local_" + arm)
    actual(graph)
    assert getattr(graph, "_comparison_local_" + arm) is cache
    graph.incidence_edge_index = edges.clone()
    actual(graph)
    assert getattr(graph, "_comparison_local_" + arm) is not cache


def test_arxiv_full_training_validation_selection_and_frozen_test_matrix(monkeypatch, tmp_path):
    torch.manual_seed(10)
    payload = transductive_payload()
    protocol = {
        "explicit_synthetic_debug": True,
        "data_sha256": engine.base.tensor_hash(payload["graphs"][0]["x"]),
        "split_sha256": {k: engine.base.tensor_hash(v) for k, v in payload["splits"].items()},
    }
    options = runner.parser().parse_args(
        [
            "--run-id",
            "debug-arxiv-baselines",
            "--profiles",
            "reference",
            "--arms",
            "gcn",
            "graphsage",
            "gatv2",
            "incidence",
            "--epochs",
            "4",
            "--patience",
            "4",
            "--edge-chunk-size",
            "128",
            "--data-root",
            str(tmp_path / "debug-data"),
            "--evaluate-test",
        ]
    )
    jobs = runner.make_jobs(options, tmp_path / "explicit-synthetic-debug")
    monkeypatch.setattr(engine.base, "load_dataset", lambda *a, **k: (payload, protocol))
    for job in jobs:
        child = calibration.parse_job(job)
        root = child.output_dir
        engine.train_model(payload, protocol, child, torch.device("cuda:0"), root)
        report = audit.audit(root, child.data_root, torch.device("cuda:0"), 5)
        log = root / "synthetic-audit.json"
        log.write_text(json.dumps(report), encoding="utf-8")
        job.update(
            status="passed",
            result=runner._read_result(job),
            audit={
                "status": "passed",
                "checkpoint_sha256": report["checkpoint_sha256"],
                "evaluator_source_sha256": engine.implementation_source_hashes(),
                "log_path": str(log),
                "log_sha256": engine.base.sha256_file(log),
            },
        )
    manifest = {"jobs": jobs}
    from contextvars import Context

    for job in jobs:
        metrics = engine.inspect_completed(job["output_dir"])
        assert metrics["debug"] is True
        assert metrics["resume_identity"]["evidence_origin"]["debug"] is True
        with pytest.raises(ValueError, match="synthetic debug evidence"):
            Context().run(engine.inspect_completed, job["output_dir"])
    persisted = []
    final_test.evaluate_matrix(options, manifest, lambda: persisted.append(copy.deepcopy(manifest)))
    assert persisted[0]["official_test_checkpoint_lock"]
    assert "official_test" not in persisted[0]
    assert manifest["test_evaluated"]
    assert len(manifest["official_test"]["results"]) == 4
    for row in manifest["official_test"]["results"].values():
        assert row["evaluation"]["counts"]["total"] == 32
        assert row["evaluation"]["metric_kind"] == "accuracy"
    assert "unavailable" in "\n".join(final_test.markdown(manifest["official_test"]))
    args = calibration.parse_job(jobs[0])
    inputs = engine.PreparedInputs(payload, args)
    test_inputs = final_test.HeldoutInputs(inputs, payload["splits"]["test"])
    batch = next(test_inputs.validation_batches(torch.device("cuda:0")))
    torch.testing.assert_close(batch.selected_indices, torch.arange(128, 160, device="cuda"))
    assert torch.equal(inputs.validation_record.selected_indices, torch.arange(96, 128))

    def forbidden(*args, **kwargs):
        pytest.fail("completed frozen test results must not be recomputed")

    monkeypatch.setattr(engine, "evaluate", forbidden)
    final_test.evaluate_matrix(options, manifest, lambda: None)
    for defect in ("score", "negative_time", "nan_time", "deleted", "unknown_state"):
        changed = copy.deepcopy(manifest)
        report = changed["official_test"]
        key = next(iter(report["results"]))
        row = report["results"][key]
        if defect == "score":
            counts = row["evaluation"]["counts"]
            counts["correct"] = (counts["correct"] + 1) % (counts["total"] + 1)
            row["evaluation"]["metric"] = counts["correct"] / counts["total"]
        elif defect == "negative_time":
            row["evaluation_seconds"] = -100
        elif defect == "nan_time":
            row["evaluation_seconds"] = float("nan")
        elif defect == "deleted":
            del report["results"][key]
        else:
            report["status"] = "unknown"
        with pytest.raises(ValueError):
            final_test.evaluate_matrix(options, changed, lambda: None)
    # Crash between artifact publish and manifest commit: validate and adopt,
    # never recompute a test prediction that is already on disk.
    interrupted = copy.deepcopy(manifest)
    interrupted["official_test"].update(status="running", results={})
    interrupted["test_evaluated"] = False
    final_test.evaluate_matrix(options, interrupted, lambda: None)
    assert interrupted == manifest
    wrong = copy.deepcopy(manifest)
    next(iter(wrong["official_test"]["results"].values()))["arm"] = "other"
    with pytest.raises(ValueError, match="saved official test"):
        final_test.evaluate_matrix(options, wrong, lambda: None)
    changed_protocol = {**protocol, "data_sha256": "changed"}
    monkeypatch.setattr(engine.base, "load_dataset", lambda *a, **k: (payload, changed_protocol))
    with pytest.raises(ValueError, match="current cache"):
        final_test.evaluate_matrix(options, manifest, lambda: None)
