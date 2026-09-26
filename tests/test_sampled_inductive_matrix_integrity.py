"""Synthetic metadata adversarial checks; no model computation or saved benchmark evidence."""

from pathlib import Path

import pytest

from experiments.sampled_inductive import runner


def test_all_checkpoints_are_validated_before_any_test_inputs(monkeypatch):
    options = runner.parser().parse_args(
        ["--run-id", "debug-lock", "--context-seeds", "32", "--context-batches", "32", "64"]
    )
    keys = [f"seed-{s}/{m}/{a}" for s, m, a in runner.cells(options)]
    manifest = {
        "results": {k: {"last_sha256": str(i)} for i, k in enumerate(keys)},
        "selected_resources": {m: {"batch": 2, "workers": 2} for m in ("full", "sampled")},
    }

    def completed(path):
        key = path.as_posix().removeprefix("debug-root/")
        if key == keys[-1]:
            raise ValueError("last checkpoint invalid")
        return manifest["results"][key]

    def test_input(*args, **kwargs):
        pytest.fail("test inputs touched before every checkpoint was validated")

    monkeypatch.setattr(runner.train, "completed", completed)
    monkeypatch.setattr(runner.train, "make_inputs", test_input)
    with pytest.raises(ValueError, match="last checkpoint invalid"):
        runner.evaluate_test(options, {}, {}, manifest, Path("debug-root"), None, lambda: None)
    assert "test_checkpoint_lock" not in manifest
