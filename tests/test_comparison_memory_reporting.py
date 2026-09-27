"""Control-plane memory rejection diagnostics; no CPU model execution."""

from types import SimpleNamespace

import pytest
import torch

from experiments.aggregation_comparison import calibration, core, engine, memory, runner


def test_headroom_rejection_is_distinguished_from_cuda_oom(monkeypatch):
    gib = 1024**3
    passed = {
        "status": "passed",
        "condition": "incidence_fixed",
        "model_seed": 0,
        "peak_allocated_bytes": 5 * gib,
        "peak_reserved_bytes": 9 * gib,
        "free_bytes_before": 10 * gib,
        "total_memory_bytes": 10 * gib,
    }
    oom = {
        "status": "oom",
        "condition": "incidence",
        "model_seed": 0,
        "error": "actual CUDA allocation failure",
        "failure_memory": passed,
    }
    monkeypatch.setattr(calibration, "_score", lambda *args: None)
    with pytest.raises(RuntimeError) as failure:
        calibration._choose([{"batch_size": 1, "workers": 0, "measurements": [passed, oom]}], {})
    message = str(failure.value)
    assert "incidence_fixed seed=0 physical=1 workers=0: headroom rejected" in message
    assert "shortfall=1.000GiB" in message and "peak allocated=5.000GiB" in message
    assert "incidence seed=0 physical=1 workers=0: CUDA OOM" in message
    assert "actual CUDA allocation failure" in message


def test_oom_keeps_telemetry_and_traceback(monkeypatch):
    def fail(*args, **kwargs):
        error = torch.OutOfMemoryError("debug injected OOM")
        error.calibration_memory = {"peak_reserved_bytes": 123}
        error.calibration_resource_observability = {"cpu_seconds": 2.5}
        raise error

    monkeypatch.setattr(engine, "run_calibration_candidate", fail)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = calibration._measure(
        {"variant_id": "incidence", "model_seed": 0},
        {},
        SimpleNamespace(device="cuda:0", negative_loss_weight=0, selection_mode="full"),
        1,
        0,
    )
    assert result["status"] == "oom"
    assert result["failure_memory"] == {"peak_reserved_bytes": 123}
    assert result["resource_observability"] == {"cpu_seconds": 2.5}
    assert "debug injected OOM" in result["traceback"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_allocator_budget_rejects_invalid_limits(value):
    with pytest.raises(ValueError, match="finite and positive"):
        memory.validate(SimpleNamespace(cuda_allocator_limit_gib=value))


def test_allocator_budget_reaches_all_four_core_cells(tmp_path):
    args = core.parser().parse_args(
        [
            "--run-id",
            "debug-memory-forwarding",
            "--profiles",
            "reference",
            "--sample-context-seed-batch-size",
            "2048",
            "--sample-seed-batch-size",
            "2048",
            "--cuda-allocator-limit-gib",
            "7",
        ]
    )
    core.validate(args)
    for mode in core.MODES:
        child = runner.parser().parse_args(core.child_argv(args, tmp_path, mode))
        runner.validate_args(child)
        jobs = runner.make_jobs(child, tmp_path / mode)
        assert len(jobs) == 2
        for job in jobs:
            config = engine.configuration(calibration.parse_job(job))
            assert config["cuda_allocator_limit_gib"] == 7
