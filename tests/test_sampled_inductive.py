"""CPU sampling/control tests; no CPU model forward/backward."""

import copy

import pytest
import torch

from experiments.sampled_inductive import runner
from experiments.sampled_inductive.data import Coverage, Inputs, validate_splits


def synthetic_payload():
    graphs = []
    for n in (65, 73, 67, 79, 71, 69):
        ids = torch.arange(n)
        edges = torch.stack((ids, (ids + 1) % n)).sort(0).values
        edges = edges[:, torch.argsort(edges[0] * n + edges[1])]
        graphs.append(
            {
                "x": torch.randn(n, 50),
                "y": (torch.arange(n * 7).reshape(n, 7) % 3 == 0).float(),
                "incidence_edge_index": edges,
            }
        )
    return {
        "dataset": "ppi",
        "classes": 7,
        "graphs": graphs,
        "splits": {"train": [0, 1, 2, 3], "validation": [4], "test": [5]},
    }


def test_sampling_seed_coverage_context_labels_and_real_edges():
    payload = synthetic_payload()
    inputs = Inputs(
        payload, "sampled", physical_batch=4, workers=0, context_seeds=8, fanouts=[2], seed=9
    )
    try:
        evidence = Coverage(inputs.datasets["train"])
        for batch in inputs.loaders["train"]:
            evidence.add(batch.observations)
            assert batch.seed_mask.sum() <= 32
            assert batch.seed_mask.numel() == batch.graph.x.shape[0]
            assert batch.graph.x.shape[0] > batch.seed_mask.sum()
            for obs in batch.observations:
                source = payload["graphs"][obs["source_graph_id"]]["incidence_edge_index"]
                assert torch.isin(source[:, obs["edges"]], obs["nodes"]).all()
                assert obs["source_graph_id"] < 4
        report = evidence.finish()
        assert report["seed_coverage"] == 1 and report["supervised_nodes"] == 284
        assert report["saturated_contexts"] == 0
        assert inputs.datasets["validation"].ids == [4]
    finally:
        inputs.close()


def test_split_overlap_and_missing_coverage_fail():
    payload = synthetic_payload()
    bad = copy.deepcopy(payload)
    bad["splits"]["test"] = [0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_splits(bad)
    inputs = Inputs(
        payload, "sampled", physical_batch=4, workers=0, context_seeds=8, fanouts=[2], seed=9
    )
    with pytest.raises(ValueError, match="lost or repeated"):
        Coverage(inputs.datasets["train"]).finish()
    inputs.close()


def test_four_cells_preserve_reference_architecture_and_budget(tmp_path):
    options = runner.parser().parse_args(
        ["--run-id", "debug-plan", "--context-seeds", "32", "--context-batches", "32", "64"]
    )
    runner.validate(options)
    assert len(runner.cells(options)) == 4
    for seed, _mode, arm in runner.cells(options):
        args = runner.child_arguments(options, arm, seed)
        assert (args.layers, args.hidden_channels, args.heads, args.epochs) == (8, 256, 8, 200)
    options.dry_run = True
    options.results_root = tmp_path
    assert not list(tmp_path.iterdir())


def test_multiple_workers_preserve_contexts_and_epoch_seed_coverage():
    payload = synthetic_payload()
    collected = []
    for workers in (0, 2):
        inputs = Inputs(
            payload,
            "sampled",
            physical_batch=4,
            workers=workers,
            context_seeds=8,
            fanouts=[2],
            seed=13,
        )
        epochs = []
        try:
            for epoch in (1, 2):
                inputs.batchers["train"].epoch = epoch
                coverage = Coverage(inputs.datasets["train"])
                signatures = []
                for batch in inputs.loaders["train"]:
                    coverage.add(batch.observations)
                    signatures.extend(
                        (o["source_graph_id"], o["nodes"].tolist(), o["seeds"].tolist())
                        for o in batch.observations
                    )
                assert coverage.finish()["seed_coverage"] == 1
                epochs.append(signatures)
        finally:
            inputs.close()
        assert epochs[0] != epochs[1]
        collected.append(epochs)
    assert collected[0] == collected[1]


def test_incidence_only_transductive_sampling_is_now_plannable():
    from pathlib import Path

    from experiments.aggregation_comparison import engine
    from experiments.aggregation_comparison import runner as comparison

    for mode in ("cluster", "cluster_disjoint"):
        argv = [
            "--run-id",
            "debug-sampling-plan",
            "--datasets",
            "ogbn-arxiv",
            "--profiles",
            "reference",
            "--arms",
            "incidence",
            "--sampling",
            mode,
            "--hardware-profile",
            "portable",
        ]
        if mode == "cluster_disjoint":
            argv.extend(["--sample-context-seed-batch-size", "32"])
        opts = comparison.parser().parse_args(argv)
        comparison.validate_args(opts)
        job = comparison.make_jobs(opts, Path("results/debug-plan"))[0]
        args = engine.build_parser().parse_args(job["command"][job["command"].index("-m") + 2 :])
        engine.validate_args(args)
        opts.arms.append("dualformer")
        with pytest.raises(ValueError, match="full graph"):
            comparison.validate_args(opts)


@pytest.mark.parametrize("corruption", ["selected", "measurement", "missing"])
def test_calibration_resume_verifies_measured_plan(monkeypatch, corruption):
    """Synthetic control-plane records only; no fabricated GPU evidence saved."""
    options = runner.parser().parse_args(
        ["--run-id", "debug-control", "--context-seeds", "8", "--context-batches", "4", "8"]
    )
    monkeypatch.setattr(runner, "allocated_cpu_count", lambda: 8)
    calls = []

    def probe(payload, args, mode, batch, workers, device):
        calls.append((mode, batch, workers, args.ablation_arm))
        return {
            "status": "passed",
            "safe": True,
            "epoch_seconds": 100 / batch + workers,
            "validation_seconds": 2 if args.ablation_arm == "incidence" else 1,
        }

    monkeypatch.setattr(runner.train, "probe", probe)
    manifest = {"calibration": {}, "selected_resources": {}}
    payload = synthetic_payload()
    runner.calibrate(options, payload, manifest, lambda: None, None)
    assert len(calls) == 16  # 2 modes x 2 batches x 2 workers x 2 C regimes
    assert manifest["selected_resources"]["full"] == {
        "batch": 4,
        "workers": 2,
        "projected_epoch_seconds": 29.0,
    }
    assert manifest["selected_resources"]["sampled"] == {
        "batch": 8,
        "workers": 2,
        "projected_epoch_seconds": 16.5,
    }
    runner.calibrate(options, payload, manifest, lambda: None, None)
    assert len(calls) == 16
    if corruption == "selected":
        manifest["selected_resources"]["full"]["batch"] = 2
    elif corruption == "measurement":
        manifest["calibration"][next(iter(manifest["calibration"]))]["epoch_seconds"] = 0
    else:
        manifest["calibration"].pop(next(iter(manifest["calibration"])))
    with pytest.raises(ValueError, match="evidence"):
        runner.calibrate(options, payload, manifest, lambda: None, None)
