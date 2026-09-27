"""Full-size reference network, explicitly synthetic four-cell integration test."""

import copy
import json

import pytest
import torch

from experiments.aggregation_comparison import audit, calibration, core, engine, runner
from tests.test_aggregation_comparison_cuda import cuda_required  # noqa: F401
from tests.test_aggregation_sampling_path_cuda import transductive_payload


@pytest.mark.parametrize("visibility", ["official_transductive", "arxiv_node_year_views_v1"])
def test_four_cells_complete_exposure_pairing_and_global_freeze(monkeypatch, tmp_path, visibility):
    torch.manual_seed(100)
    payload = transductive_payload()
    protocol = {
        "explicit_synthetic_debug": True,
        "data_sha256": engine.base.tensor_hash(payload["graphs"][0]["x"]),
        "split_sha256": {k: engine.base.tensor_hash(v) for k, v in payload["splits"].items()},
    }
    monkeypatch.setattr(engine.base, "load_dataset", lambda *a, **k: (payload, protocol))
    options = core.parser().parse_args(
        [
            "--run-id",
            "debug-core",
            "--profiles",
            "reference",
            "--cuda-allocator-limit-gib",
            "7",
            "--epochs",
            "4",
            "--patience",
            "1",
            "--sample-seed-batch-size",
            "32",
            "--sample-context-seed-batch-size",
            "8",
            "--sample-context-workers",
            "2",
            "--num-neighbors",
            "2",
            "--edge-chunk-size",
            "128",
            "--data-root",
            str(tmp_path / "data"),
            "--evaluate-test",
        ]
    )
    core.validate(options)
    options.visibility_protocol = visibility
    if visibility != "official_transductive":
        import numpy as np

        from experiments.aggregation_comparison import temporal

        raw = tmp_path / "synthetic-raw"
        raw.mkdir()
        graph = payload["graphs"][0]
        for name, values, fmt in (
            ("node-feat", graph["x"].numpy(), "%.9g"),
            ("node-label", graph["y"].numpy(), "%d"),
            ("edge", graph["incidence_edge_index"].T.numpy(), "%d"),
            ("node-year", np.array([2017] * 96 + [2018] * 32 + [2019] * 16 + [2020] * 16), "%d"),
        ):
            np.savetxt(raw / (name + ".csv.gz"), values, delimiter=",", fmt=fmt)
        temporal.prepare_years(options.data_root, raw)
    groups = {}
    for mode in core.MODES:
        args = runner.parser().parse_args(core.child_argv(options, tmp_path, mode))
        root = core.child_root(options, tmp_path, mode)
        jobs = runner.make_jobs(args, root)
        for job in jobs:
            child = calibration.parse_job(job)
            loaded, identity = engine.load_dataset(child)
            engine.train_model(loaded, identity, child, torch.device("cuda:0"), child.output_dir)
            report = audit.audit(child.output_dir, child.data_root, torch.device("cuda:0"), 5)
            log = child.output_dir / "synthetic-audit.json"
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
        groups[mode] = {"args": args, "manifest": {"jobs": jobs}, "path": root / "manifest.json"}
    master, saves = {}, []
    report = core.freeze_and_test(
        options, groups, master, lambda: saves.append(copy.deepcopy(master))
    )
    assert len(saves[0]["checkpoint_lock"]) == 4
    assert "official_test" not in saves[0]
    assert len(report["cells"]) == 4
    assert {r["epochs"] for r in report["cells"]} == {4}
    assert {r["supervised_exposures"] for r in report["cells"]} == {384}
    assert {r["optimizer_steps"] for r in report["cells"]} == {4, 12}
    assert len(master["official_test"]) == 2

    def forbidden(*a, **k):
        pytest.fail("completed four-cell test must not run inference again")

    monkeypatch.setattr(engine, "evaluate", forbidden)
    core.freeze_and_test(options, groups, master, lambda: None)
    partial = {
        **groups,
        "sampled": {
            **groups["sampled"],
            "manifest": {"jobs": groups["sampled"]["manifest"]["jobs"][:1]},
        },
    }
    with pytest.raises(ValueError, match="every F0/F1/S0/S1"):
        core.verify_and_report(partial)
