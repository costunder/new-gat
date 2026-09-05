"""CPU-only debug checkpoints for explicit, non-destructive V5 C replacement."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.train import make_optimizer
from research.conductance_gat.v5.transition import (
    C_NAMESPACE,
    canonical_sha256,
    inspect_transition_source,
    prepare_transition_state,
)

# Reviewed 76e514a Linux/LF source hashes, not test aliases accepted in production.
LEGACY_SOURCE = dict(
    zip(
        [
            "research/conductance_gat/ablation/train.py",
            "research/conductance_gat/benchmark.py",
            "research/conductance_gat/benchmark_data.py",
            "research/conductance_gat/v5/__init__.py",
            "research/conductance_gat/v5/batch_calibration.py",
            "research/conductance_gat/v5/diagnostics.py",
            "research/conductance_gat/v5/model.py",
            "research/conductance_gat/v5/operator.py",
            "research/conductance_gat/v5/protocol.py",
            "research/conductance_gat/v5/report.py",
            "research/conductance_gat/v5/sampling.py",
            "research/conductance_gat/v5/train.py",
            "src/chartgat/cache.py",
            "src/chartgat/observability.py",
        ],
        [
            "522df8ee0c7d40acb2db6814bcf6065648f8674fb3669d299ffb9872c66dbff5",
            "c441a65e22f0431e0c362d4eeadb3036aaceabb5b89863bd8709e6b7680b5d13",
            "082e4a1a94bb7beffddfa2d5f59a922c806f15093d5ffc97fb3240b0bb844dc7",
            "8b313a477bbd51eedc0861fad06cffdf7f7a0aed5d9fcddc55fa89c333a3bd46",
            "46cb2d2632c1283089f3684e32bcc22d97d6e5b81809646ffe21a5eb2ab7148f",
            "cd7b7bb6f72bae2d8282a8cb33e48a0f0a25872b2c0ffd27008528e029fe9182",
            "fd2256d0815c2eed40a3a5cefaca3a1b786049a7fa799e8cf63b12ac4419b57f",
            "28e252ac76116217beaf801ccbb6308c37c85cef7334125322043e326cbfe0b3",
            "77bd54d053172a82bcd0b07f22d70eda8f9a5c0c9204f50291b4ea2f0ffd81e3",
            "ad8964970d3372cc631c49f412057941a067c65c0b09d160139780350629c0f7",
            "0f4d5226cd91202fdcca1e269c03de36aee56f1ed95992b9da14046ed108a7dd",
            "e0f68e4ecb73e93018af1a12d63d36050228eef58661234c687b98fa846db6ad",
            "b9feeef3c3e033a064677a612790b036d78d09eecddce773a44273e007cd7cab",
            "53b1b037f84371ad51abb9524c808b5d72cf27f3c66eca8fcc97a825e81755b1",
        ],
        strict=True,
    )
)


def _model(*, backend="mlp", fixed=False):
    return GraphConditionedConductanceNodeClassifier(
        5,
        3,
        hidden_channels=8,
        layers=2,
        heads=2,
        ffn_multiplier=2,
        dropout=0,
        conductance_mode="fixed_one" if fixed else "dynamic",
        conductance_backend=backend,
        activation_checkpoint=False,
        edge_chunk_size=3,
    )


def _graph():
    return SimpleNamespace(
        x=torch.randn(7, 5),
        incidence_edge_index=torch.tensor([[0, 0, 1, 2, 2, 3, 4, 5], [1, 6, 2, 3, 5, 4, 5, 6]]),
    )


def _save(tmp_path, *, fixed=False, epoch=2, complete=False):
    torch.manual_seed(56)
    model = _model(fixed=fixed)
    optimizer = make_optimizer(model)
    graph = _graph()
    for _ in range(epoch):
        optimizer.zero_grad(set_to_none=True)
        model(graph).square().mean().backward()
        optimizer.step()
    configuration = {
        "optimizer": "AdamW",
        "model_seed": 0,
        "epochs": 10,
        "hidden_channels": 8,
        "layers": 2,
        "heads": 2,
        "ffn_multiplier": 2,
        "dropout": 0,
        "activation_checkpoint": False,
        "precision": "fp32",
        "amp": False,
        "sampling": "cluster",
        "sample_seed_batch_size": 8,
        "num_neighbors": [15, 10],
        "batch_size": 1,
        "workers": 0,
        "loader_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None,
        "worker_configuration_source": "debug_fixture",
        "lr": 0.0005,
        "beta_initial": 0.1,
        "beta_parameterization": "sigmoid",
        "patience": 5,
    }
    protocol = {"data_sha256": "a" * 64, "dataset": "cora", "debug_fixture": True}
    identity = {
        "schema_version": 1,
        "research_suite": "conductance_graph_conditioned_v5",
        "dataset": "cora",
        "condition": "fixed_c" if fixed else "shared_dynamic_c",
        "configuration": configuration,
        "schedule": [
            {"name": "spatial_warmup", "start_epoch": 1, "end_epoch": 1, "length": 1},
            {"name": "conductance_calibration", "start_epoch": 2, "end_epoch": 2, "length": 1},
            {"name": "alternating", "start_epoch": 3, "end_epoch": 4, "length": 2},
            {"name": "joint", "start_epoch": 5, "end_epoch": 10, "length": 6},
        ],
        "dataset_protocol": protocol,
        "dataset_protocol_sha256": canonical_sha256(protocol),
        "cache_sha256": protocol["data_sha256"],
        "initial_state_sha256": "b" * 64,
        "source_sha256": copy.deepcopy(LEGACY_SOURCE),
        "runtime_versions": {"torch": str(torch.__version__), "test_scope": "CPU debug fixture"},
        "resume_semantics": "epoch-boundary deterministic resume",
    }
    saved = {
        "schema_version": 3,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "resume_identity": identity,
        "resume_identity_sha256": canonical_sha256(identity),
        "epoch": epoch,
        "complete": complete,
        "history": [{"epoch": i, "validation": 0.4} for i in range(1, epoch + 1)],
        "best_epoch": 1,
        "best_metric": 0.4,
        "best_checkpoint_sha256": "c" * 64,
        "global_best_epoch": 1,
        "global_best_metric": 0.4,
        "joint_best_epoch": 0,
        "joint_best_metric": -math.inf,
        "optimizer_steps": epoch,
        "effective_optimizer_steps_by_group": {
            "backbone": epoch,
            "spatial_w": epoch,
            "beta": epoch,
            "conductance": 0 if fixed else epoch,
        },
        "first_c_gradient": None if fixed else 0.01,
        "cpu_rng_state": torch.get_rng_state(),
        # This is a typed debug placeholder for format tests, not a CUDA claim.
        "cuda_rng_state": torch.get_rng_state(),
        "elapsed_seconds": 12.5,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
        "resume_source_compatibility": [],
    }
    path = tmp_path / "legacy-last.pt"
    torch.save(saved, path)
    return path, saved


def _target(saved, *, fixed=False, extra=0):
    model = _model(backend="mlp" if fixed else "optimization", fixed=fixed)
    optimizer = make_optimizer(model)
    identity = copy.deepcopy(saved["resume_identity"])
    identity["configuration"].update(
        conductance_backend="mlp" if fixed else "optimization",
        training_schedule="staged" if fixed else "joint",
        solver_steps=8,
        solver_step_size=0.25,
        solver_entropy=1.0,
        solver_degree_barrier=0.1,
        epochs=identity["configuration"]["epochs"] + extra,
    )
    if fixed:
        identity["schedule"][-1]["end_epoch"] += extra
        identity["schedule"][-1]["length"] += extra
    else:
        identity["schedule"] = [
            {
                "name": "joint",
                "start_epoch": 1,
                "end_epoch": 10 + extra,
                "length": 10 + extra,
            }
        ]
    identity["source_sha256"]["research/conductance_gat/v5/model.py"] = "d" * 64
    identity["source_sha256"]["research/conductance_gat/v5/optimization.py"] = "e" * 64
    return model, optimizer, identity


def _prepare(path, target, **kwargs):
    model, optimizer, identity = target
    return prepare_transition_state(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        target_model=model,
        target_optimizer=optimizer,
        target_identity=identity,
        **kwargs,
    )


def _assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert first == second


def test_c_only_transition_preserves_all_shared_weights_moments_and_progress(tmp_path):
    path, saved = _save(tmp_path)
    original_bytes = path.read_bytes()
    target = _target(saved)
    model, optimizer, _ = target
    before = copy.deepcopy(model.state_dict())
    result = _prepare(path, target)
    assert path.read_bytes() == original_bytes
    _assert_nested_equal(model.state_dict(), before)  # Preparation is pure.
    assert optimizer.state_dict()["state"] == {}
    for name, value in result["model_state"].items():
        expected = before[name] if C_NAMESPACE.match(name) else saved["model_state"][name]
        assert torch.equal(value, expected), name
    assert result["start_epoch"] == 3 and result["total_epochs"] == 10
    assert result["epoch_offset"] == 2 and result["history"] == []
    assert result["source_history"] == saved["history"]
    assert result["selection_state"]["best_metric"] == -math.inf
    assert result["selection_state"]["joint_best_epoch"] == 0
    assert result["counters"]["optimizer_steps"] == 2
    assert result["counters"]["effective_optimizer_steps_by_group"]["conductance"] == 0
    for old, new in zip(
        saved["optimizer_state"]["param_groups"],
        result["optimizer_state"]["param_groups"],
        strict=True,
    ):
        if old["name"] == "conductance":
            assert all(index not in result["optimizer_state"]["state"] for index in new["params"])
            continue
        assert {k: v for k, v in old.items() if k != "params"} == {
            k: v for k, v in new.items() if k != "params"
        }
        for old_id, new_id in zip(old["params"], new["params"], strict=True):
            _assert_nested_equal(
                saved["optimizer_state"]["state"][old_id],
                result["optimizer_state"]["state"][new_id],
            )
    for name in ("cpu_rng_state", "cuda_rng_state"):
        assert torch.equal(result["rng_state"][name], saved[name])
    assert result["provenance"]["test_labels_used"] is False
    json.dumps(result["provenance"], allow_nan=False)
    assert result["provenance"]["source_selection_state"]["joint_best_metric"] is None
    assert result["provenance"]["source_selection_applicability"]["joint_best_metric"].startswith(
        "not_yet"
    )
    assert all(C_NAMESPACE.match(name) for name in result["provenance"]["dropped_c_tensor_names"])
    model.load_state_dict(result["model_state"])
    optimizer.load_state_dict(result["optimizer_state"])
    optimizer.zero_grad(set_to_none=True)
    model(_graph()).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(
        float(state["step"]) == (1 if C_NAMESPACE.match(name) else 3)
        for group in optimizer.param_groups
        for name, parameter in zip(group["parameter_names"], group["params"], strict=True)
        for state in (optimizer.state[parameter],)
    )


def test_fixed_continuation_preserves_entire_state_history_and_best(tmp_path):
    path, saved = _save(tmp_path, fixed=True)
    target = _target(saved, fixed=True)
    result = _prepare(path, target, mode="continue_fixed")
    _assert_nested_equal(result["model_state"], saved["model_state"])
    _assert_nested_equal(result["optimizer_state"], saved["optimizer_state"])
    assert result["history"] == saved["history"]
    assert result["epoch_offset"] == 0
    assert result["selection_state"]["best_metric"] == 0.4
    assert result["selection_state"]["best_checkpoint_sha256"] == "c" * 64
    assert result["provenance"]["dropped_c_tensor_names"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_seed", 1),
        ("hidden_channels", 16),
        ("layers", 3),
        ("heads", 4),
        ("dropout", 0.5),
        ("precision", "bf16"),
        ("sampling", "neighbor"),
        ("sample_seed_batch_size", 4),
        ("beta_initial", 0.2),
        ("lr", 0.01),
    ],
)
def test_shared_configuration_drift_fails_without_modifying_source_or_target(
    tmp_path, field, value
):
    path, saved = _save(tmp_path)
    original = path.read_bytes()
    target = _target(saved)
    before = copy.deepcopy(target[0].state_dict())
    target[2]["configuration"][field] = value
    with pytest.raises(ValueError, match="shared identity"):
        _prepare(path, target)
    assert path.read_bytes() == original
    _assert_nested_equal(target[0].state_dict(), before)
    assert target[1].state_dict()["state"] == {}


@pytest.mark.parametrize(
    "kind",
    [
        "hash",
        "identity",
        "source",
        "data",
        "schema",
        "history",
        "rng",
        "model_nan",
        "moment_nan",
        "moment_shape",
        "names",
        "negative_step",
        "resources",
        "legacy_c_names",
    ],
)
def test_corrupt_checkpoint_fails_closed(tmp_path, kind):
    path, saved = _save(tmp_path)
    old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if kind == "hash":
        with path.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="SHA-256"):
            inspect_transition_source(path, expected_sha256=old_hash)
        return
    if kind == "identity":
        saved["resume_identity"]["configuration"]["model_seed"] = 3
    elif kind == "source":
        saved["resume_identity"]["source_sha256"]["research/conductance_gat/v5/model.py"] = "f" * 64
        saved["resume_identity_sha256"] = canonical_sha256(saved["resume_identity"])
    elif kind == "data":
        saved["resume_identity"]["cache_sha256"] = "f" * 64
        saved["resume_identity_sha256"] = canonical_sha256(saved["resume_identity"])
    elif kind == "schema":
        saved["schema_version"] = 2
    elif kind == "history":
        saved["history"][1]["epoch"] = 1
    elif kind == "rng":
        saved["cpu_rng_state"] = torch.ones(4)
    elif kind == "model_nan":
        saved["model_state"]["encoder.weight"][0, 0] = math.nan
    elif kind == "moment_nan":
        next(iter(saved["optimizer_state"]["state"].values()))["exp_avg"][0] = math.nan
    elif kind == "moment_shape":
        next(iter(saved["optimizer_state"]["state"].values()))["exp_avg"] = torch.zeros(1)
    elif kind == "names":
        saved["optimizer_state"]["param_groups"][0]["parameter_names"][0] = "missing.weight"
    elif kind == "negative_step":
        next(iter(saved["optimizer_state"]["state"].values()))["step"] = torch.tensor(-1.0)
    elif kind == "resources":
        saved["elapsed_seconds"] = math.nan
    elif kind == "legacy_c_names":
        saved["model_state"]["blocks.0.operator.estimator.mystery"] = torch.ones(1)
    torch.save(saved, path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        inspect_transition_source(path)
    assert path.read_bytes() == before


def test_complete_dynamic_requires_explicit_remaining_budget(tmp_path):
    path, saved = _save(tmp_path, epoch=10, complete=True)
    with pytest.raises(ValueError, match="no remaining epochs"):
        _prepare(path, _target(saved))
    target = _target(saved, extra=4)
    with pytest.raises(ValueError, match="epoch budget"):
        _prepare(path, target)
    result = _prepare(path, target, additional_epochs=4)
    assert result["start_epoch"] == 11 and result["total_epochs"] == 14
    assert result["provenance"]["additional_epochs"] == 4


def test_complete_fixed_is_reference_not_retraining_and_fixed_extension_keeps_phases(tmp_path):
    path, saved = _save(tmp_path, fixed=True, complete=True)
    with pytest.raises(ValueError, match="historical reference"):
        _prepare(path, _target(saved, fixed=True), mode="continue_fixed")
    path, saved = _save(tmp_path, fixed=True)
    target = _target(saved, fixed=True, extra=5)
    result = _prepare(path, target, mode="continue_fixed", additional_epochs=5)
    assert result["total_epochs"] == 15
    target[2]["schedule"][0]["length"] += 1
    with pytest.raises(ValueError, match="retain phases"):
        _prepare(path, target, mode="continue_fixed", additional_epochs=5)


def test_explicit_nondecreasing_execution_change_is_audited(tmp_path):
    path, saved = _save(tmp_path)
    target = _target(saved)
    target[2]["configuration"]["sample_seed_batch_size"] = 16
    declared = {"sample_seed_batch_size": {"before": 8, "after": 16}}
    result = _prepare(path, target, declared_execution_changes=declared)
    assert result["provenance"]["declared_execution_changes"] == declared
    target[2]["configuration"]["sample_seed_batch_size"] = 4
    with pytest.raises(ValueError, match="reduce physical"):
        _prepare(
            path,
            target,
            declared_execution_changes={"sample_seed_batch_size": {"before": 8, "after": 4}},
        )


def test_target_request_must_match_verified_action_and_source(tmp_path):
    path, saved = _save(tmp_path)
    target = _target(saved)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    request = {
        "source_checkpoint_sha256": sha,
        "mode": "replace_c",
        "additional_epochs": 0,
        "declared_execution_changes": {},
        "source_path": str(path),
        "resource_certificate_sha256": "d" * 64,
        "resource_certificate_path": "debug-only.json",
    }
    target[2]["transition_request"] = request
    assert _prepare(path, target)["provenance"]["source_checkpoint_sha256"] == sha
    request["source_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="request"):
        _prepare(path, target)


def test_undeclared_shared_operator_and_dataset_runtime_changes_rejected(tmp_path):
    path, saved = _save(tmp_path)
    for field in ("dataset", "runtime_versions", "dataset_protocol"):
        target = _target(saved)
        target[2][field] = "changed"
        with pytest.raises(ValueError, match="shared identity"):
            _prepare(path, target)
    target = _target(saved)
    target[2]["source_sha256"]["research/conductance_gat/v5/operator.py"] = "f" * 64
    with pytest.raises(ValueError, match="shared operator"):
        _prepare(path, target)


def test_shared_tensor_dtype_shape_and_busy_target_optimizer_are_rejected(tmp_path):
    path, saved = _save(tmp_path)
    target = _target(saved)
    target[0].encoder.weight = torch.nn.Parameter(torch.zeros(9, 5))
    with pytest.raises(ValueError, match="shape/dtype"):
        _prepare(path, target)
    target = _target(saved)
    target[0].double()
    with pytest.raises(ValueError, match="shape/dtype"):
        _prepare(path, target)
    target = _target(saved)
    target[0](_graph()).square().mean().backward()
    target[1].step()
    with pytest.raises(ValueError, match="fresh"):
        _prepare(path, target)


@pytest.mark.parametrize(
    "kind", ["missing_moments", "zero_step", "step_above_group", "group_counter"]
)
def test_missing_reusable_adam_state_cannot_silently_restart_optimizer(tmp_path, kind):
    path, saved = _save(tmp_path)
    if kind == "missing_moments":
        saved["optimizer_state"]["state"].clear()
    elif kind == "zero_step":
        next(iter(saved["optimizer_state"]["state"].values()))["step"] = torch.tensor(0.0)
    elif kind == "step_above_group":
        saved["effective_optimizer_steps_by_group"]["backbone"] = 1
    else:
        saved["effective_optimizer_steps_by_group"]["backbone"] = 0
    torch.save(saved, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="moments/steps"):
        inspect_transition_source(path)
    assert path.read_bytes() == original


def test_later_skipped_gradient_preserves_smaller_actual_adam_step(tmp_path):
    path, saved = _save(tmp_path)
    old_model = _model()
    old_optimizer = make_optimizer(old_model)
    old_model.load_state_dict(saved["model_state"])
    old_optimizer.load_state_dict(saved["optimizer_state"])
    old_optimizer.zero_grad(set_to_none=True)
    old_model(_graph()).square().mean().backward()
    skipped_name, skipped_parameter = next(iter(old_model.named_parameters()))
    # Model-dependent later branches can yield grad=None for an initialized
    # parameter; AdamW correctly leaves that parameter's step/moments intact.
    skipped_parameter.grad = None
    old_optimizer.step()
    saved["model_state"] = old_model.state_dict()
    saved["optimizer_state"] = old_optimizer.state_dict()
    saved["epoch"] = saved["optimizer_steps"] = 3
    saved["history"].append({"epoch": 3, "validation": 0.4})
    saved["effective_optimizer_steps_by_group"] = dict.fromkeys(
        saved["effective_optimizer_steps_by_group"], 3
    )
    torch.save(saved, path)
    result = _prepare(path, _target(saved))
    for group in result["optimizer_state"]["param_groups"]:
        for name, identifier in zip(group["parameter_names"], group["params"], strict=True):
            if name == skipped_name:
                assert float(result["optimizer_state"]["state"][identifier]["step"]) == 2.0
                _assert_nested_equal(
                    result["optimizer_state"]["state"][identifier],
                    old_optimizer.state[skipped_parameter],
                )
                break
        else:
            continue
        break
    else:
        pytest.fail("skipped shared parameter was not retained")


def test_reviewed_8963821_snapshot_is_accepted_without_source_waiver(tmp_path):
    path, saved = _save(tmp_path)
    source = saved["resume_identity"]["source_sha256"]
    changes = dict(
        [
            (
                "research/conductance_gat/v5/report.py",
                "5c417d340ff299eb589c2044b975717285e56425441bf401094ff3abdca4e249",
            ),
            (
                "research/conductance_gat/v5/train.py",
                "5bda72fb2a947f521d57337e2dab3645879b256d8d187171c1031d3396c60705",
            ),
            (
                "scripts/resume_compatibility_v1.json",
                "4c8695580f8723e6960db591e65f24777a0dddbf7295b01498a62a0548b8a6ec",
            ),
            (
                "src/chartgat/resume_compat.py",
                "379e4691c24fa888c937a25420c37f8ef9ab542551eae406ed81e608856afa35",
            ),
        ]
    )
    source.update(changes)
    saved["resume_identity_sha256"] = canonical_sha256(saved["resume_identity"])
    torch.save(saved, path)
    inspected = inspect_transition_source(path)
    assert inspected["legacy_revision"] == "89638212e2188746905abe92de9d8545dd624915"
