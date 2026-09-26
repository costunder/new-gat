"""Control-plane negative tests; no model training or invented benchmark data."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from experiments.aggregation_comparison import audit, validation


@pytest.mark.parametrize(
    "kind,counts",
    [
        ("accuracy", {"correct": 7, "total": 10}),
        ("micro_f1", {"tp": 7, "fp": 3, "fn": 3, "total": 20}),
    ],
)
def test_reproduction_requires_counts_even_when_score_is_unchanged(kind, counts):
    a = {"metric_kind": kind, "metric": 0.7, "counts": counts}
    validation.require_reproduction(a, copy.deepcopy(a), label="debug")
    b = copy.deepcopy(a)
    b["counts"] = {key: value * 2 for key, value in counts.items()}
    with pytest.raises(ValueError, match="reproduction failed"):
        validation.require_reproduction(a, b, label="debug")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, -1, 1.1, 0.1])
def test_invalid_or_nonreproduced_scores_fail(bad):
    with pytest.raises(ValueError):
        validation.require_score(0.7, bad, label="debug")


@pytest.mark.parametrize("bad_repeat", [0, 4])
def test_audit_rejects_review_counterexample_on_any_repeat(monkeypatch, tmp_path, bad_repeat):
    # Explicitly mocked control flow, never reported as a GPU evaluation.
    good = {"metric": 0.7, "metric_kind": "accuracy", "counts": {"correct": 7, "total": 10}}
    wrong = {"metric": 0.1, "metric_kind": "accuracy", "counts": {"correct": 1, "total": 10}}
    metrics = {
        "resume_identity": {
            "source_sha256": {"x": "y"},
            "dataset_protocol": {},
            "input_provenance": [],
        },
        "selected_validation_evidence": good,
        "validation_evidence": good,
    }
    args = SimpleNamespace(dataset="cora", data_root=tmp_path, model_seed=0)
    monkeypatch.setattr(audit.train, "inspect_completed", lambda root: metrics)
    monkeypatch.setattr(audit.train, "implementation_source_hashes", lambda: {"x": "y"})
    monkeypatch.setattr(audit.train, "restore_arguments", lambda *a: args)
    monkeypatch.setattr(audit.train.base, "_require_cuda", lambda *a: None)
    monkeypatch.setattr(audit.train.base, "configure_compute", lambda *a: None)
    monkeypatch.setattr(audit.train.base, "load_dataset", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(audit.train.base, "_seed", lambda *a: None)
    monkeypatch.setattr(audit.train, "PreparedInputs", lambda *a: SimpleNamespace(provenance=[]))
    model = SimpleNamespace(
        load_state_dict=lambda *a, **k: None, clear_auxiliary_cache=lambda: None
    )
    monkeypatch.setattr(audit.train, "make_model", lambda *a: model)
    monkeypatch.setattr(audit.train.base, "load_checkpoint_on_cpu", lambda *a: {"model_state": {}})
    monkeypatch.setattr(audit.train, "validate_identity", lambda *a: None)
    monkeypatch.setattr(audit.train.base, "state_sha256", lambda *a: "same")
    monkeypatch.setattr(audit, "_isolated_execution_state", lambda *a: nullcontext())
    monkeypatch.setattr(audit, "_synchronize", lambda *a: None)
    monkeypatch.setattr(
        audit,
        "RuntimeResourceMonitor",
        lambda *a: SimpleNamespace(start=lambda: None, finish=lambda **k: {}),
    )
    monkeypatch.setattr(audit.torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(audit.torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(audit.torch.cuda, "max_memory_reserved", lambda *a: 0)
    evaluations = iter([good] * bad_repeat + [wrong] + [good] * (4 - bad_repeat))
    monkeypatch.setattr(audit.train, "evaluate", lambda *a: next(evaluations))
    with pytest.raises(ValueError, match=f"audit repeat {bad_repeat + 1}.*reproduction failed"):
        audit.audit(tmp_path, tmp_path, "cuda:0", 5)
