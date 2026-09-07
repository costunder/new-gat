"""CPU debug integration fixtures, never GPU performance or final-training evidence."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5 import batch_calibration, train
from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.optimization import GraphOptimizedConductance
from research.conductance_gat.v5.protocol import conductance_configuration


@pytest.fixture(autouse=True)
def preserve_debug_rng():
    with torch.random.fork_rng(devices=[]):
        yield


def _args(*extra):
    args = train.build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "shared_dynamic_c",
            "--output-dir",
            "unused-debug-optimization",
            "--hidden-channels",
            "16",
            "--layers",
            "2",
            "--heads",
            "4",
            "--ffn-multiplier",
            "2",
            "--epochs",
            "12",
            "--dropout",
            "0.2",
            "--no-activation-checkpoint",
            *extra,
        ]
    )
    train.validate_args(args)
    return args


class DebugGraph(SimpleNamespace):
    def items(self):
        return vars(self).items()

    def clone(self):
        return type(self)(**{key: value.clone() for key, value in self.items()})

    def to(self, device):
        for key, value in list(self.items()):
            setattr(self, key, value.to(device))
        return self


def _debug_graph(*, labels=True):
    generator = torch.Generator().manual_seed(109)
    values = {
        "x": torch.randn(9, 6, generator=generator),
        "incidence_edge_index": torch.tensor(
            [[0, 0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7], [1, 2, 3, 2, 3, 4, 4, 5, 6, 7, 7, 8]],
            dtype=torch.long,
        ),
    }
    if labels:
        values["y"] = torch.arange(9) % 3
    return DebugGraph(**values)


def _model(args):
    return GraphConditionedConductanceNodeClassifier(
        6,
        3,
        **train.architecture_configuration(args),
        conductance_mode=train.CONDITIONS[args.condition]["conductance_mode"],
        max_log_conductance=train.COMMON["max_log_conductance"],
        edge_chunk_size=args.edge_chunk_size,
    )


def _identity(args):
    return train.build_resume_identity(
        args,
        {"data_sha256": "d" * 64, "classification": "synthetic_debug_fixture"},
        train.phase_schedule(args.epochs, list(args.phase_fractions), args.training_schedule),
        initial_state_sha256="a" * 64,
        source_sha256={"debug_fixture.py": "b" * 64},
        runtime_versions={"torch": str(torch.__version__)},
    )


def _step(model, optimizer, graph):
    phase = train.configure_phase(model, "joint", 0)
    optimizer.zero_grad(set_to_none=True)
    loss, count = train.training_loss(model(graph), graph, torch.arange(6))
    loss.backward()
    train.validate_active_gradient_connectivity(model, phase["active_parameter_groups"])
    gradient = train.require_first_step_conductance_gradient(model)
    assert gradient["passed"] and count == 6
    optimizer.step()
    return loss.detach().clone()


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_tree_equal(value, reference)
    else:
        assert actual == expected


def test_default_cli_uses_optimization_and_joint_for_every_requested_epoch():
    args = train.build_parser().parse_args(
        ["--dataset", "cora", "--condition", "shared_dynamic_c", "--output-dir", "unused"]
    )
    train.validate_args(args)
    assert args.conductance_backend == "optimization"
    assert args.training_schedule == "joint"
    assert (
        args.solver_steps,
        args.solver_step_size,
        args.solver_entropy,
        args.solver_degree_barrier,
    ) == (8, 0.25, 1.0, 0.1)
    schedule = train.phase_schedule(args.epochs, list(args.phase_fractions), args.training_schedule)
    assert schedule == [
        {"name": "joint", "start_epoch": 1, "end_epoch": args.epochs, "length": args.epochs}
    ]
    assert [train.phase_at(schedule, epoch)[0] for epoch in range(1, args.epochs + 1)] == (
        ["joint"] * args.epochs
    )


def test_fixed_solver_is_parameter_free_and_shared_initial_state_is_paired():
    args = _args()
    torch.manual_seed(311)
    dynamic = _model(args)
    fixed_args = copy.deepcopy(args)
    fixed_args.condition = "fixed_c"
    torch.manual_seed(311)
    fixed = _model(fixed_args)
    assert dynamic.conductance_backend == fixed.conductance_backend == "optimization"
    for operator in fixed.operators:
        assert isinstance(operator.estimator, GraphOptimizedConductance)
        assert list(operator.estimator.parameters()) == []
        assert not hasattr(operator.estimator, "score_network")
    assert all(list(operator.estimator.parameters()) for operator in dynamic.operators)
    assert train.shared_initial_state_sha256(fixed) == train.shared_initial_state_sha256(dynamic)
    for name, value in fixed.state_dict().items():
        if ".operator.estimator." not in name:
            torch.testing.assert_close(value, dynamic.state_dict()[name], rtol=0, atol=0)


def test_real_joint_task_backward_updates_each_solver_and_owns_all_parameters():
    torch.manual_seed(23)
    model = _model(_args())
    optimizer = train.make_optimizer(model)
    train.validate_optimizer_parameter_ownership(model, optimizer)
    before = [copy.deepcopy(operator.estimator.state_dict()) for operator in model.operators]
    loss = _step(model, optimizer, _debug_graph())
    assert torch.isfinite(loss)
    conductance_group = next(
        group for group in optimizer.param_groups if group["name"] == "conductance"
    )
    expected = {
        id(value) for operator in model.operators for value in operator.estimator.parameters()
    }
    assert {id(value) for value in conductance_group["params"]} == expected
    for operator, old in zip(model.operators, before, strict=True):
        solver = operator.estimator
        assert isinstance(solver, GraphOptimizedConductance) and not hasattr(
            solver, "score_network"
        )
        assert solver.override is None
        assert solver.last_c.shape == (_debug_graph().incidence_edge_index.shape[1],)
        assert torch.isfinite(solver.last_c).all() and (solver.last_c > 0).all()
        assert torch.std(solver.last_c, correction=0) > 0
        assert all(
            value.grad is not None and torch.isfinite(value.grad).all()
            for value in solver.parameters()
        )
        assert sum(float(value.grad.square().sum()) for value in solver.parameters()) > 0
        assert any(not torch.equal(value, old[name]) for name, value in solver.state_dict().items())
    diagnostics = train.layer_diagnostics(model, gradients=True)
    assert all(row["conductance_backend"] == "optimization" for row in diagnostics)


def test_label_free_eval_is_deterministic_and_does_not_mutate_model_state_or_rng():
    torch.manual_seed(41)
    model = _model(_args()).eval()
    graph = _debug_graph(labels=False)
    assert not hasattr(graph, "y")
    state, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    with torch.no_grad():
        first = model(graph)
        first_c = [operator.estimator.last_c.clone() for operator in model.operators]
        second = model(graph)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    _assert_tree_equal(model.state_dict(), state)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for operator, expected in zip(model.operators, first_c, strict=True):
        torch.testing.assert_close(operator.estimator.last_c, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("conductance_backend", "mlp"),
        ("solver_steps", 9),
        ("solver_step_size", 0.125),
        ("solver_entropy", 0.5),
        ("solver_degree_barrier", 0.2),
        ("training_schedule", "staged"),
    ],
)
def test_changed_solver_or_schedule_is_not_eligible_for_source_repair_waiver(
    monkeypatch,
    field,
    replacement,
):
    args = _args()
    original = _identity(args)
    changed_args = copy.deepcopy(args)
    setattr(changed_args, field, replacement)
    train.validate_args(changed_args)
    changed = _identity(changed_args)

    def forbid_source_waiver(*_args):
        raise AssertionError("architecture/schedule changes must not consult a source waiver")

    monkeypatch.setattr(train, "snapshots_match", forbid_source_waiver)
    train.validate_resume_identity(original, original, train._canonical_sha256(original))
    with pytest.raises(ValueError, match="resume identity mismatch"):
        train.validate_resume_identity(original, changed, train._canonical_sha256(original))
    selected = {
        "resume_identity": original,
        "resume_identity_sha256": train._canonical_sha256(original),
        "epoch": 1,
        "validation": 0.5,
        "selection_role": "primary",
    }
    with pytest.raises(ValueError, match="resume identity mismatch"):
        train.validate_selected_checkpoint(
            selected,
            expected_identity=changed,
            expected_identity_sha256=train._canonical_sha256(changed),
            expected_epoch=1,
            expected_metric=0.5,
        )


def test_legacy_identity_without_solver_contract_cannot_resume_new_architecture(monkeypatch):
    current = _identity(_args())
    historical = copy.deepcopy(current)
    for name in (*conductance_configuration(), "training_schedule"):
        historical["configuration"].pop(name)
    historical["schedule"] = train.phase_schedule(12, [0.1, 0.1, 0.4, 0.4], "staged")

    def forbid_source_waiver(*_args):
        raise AssertionError("legacy model identity is not a numerical bugfix")

    monkeypatch.setattr(train, "snapshots_match", forbid_source_waiver)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        train.validate_resume_identity(historical, current, train._canonical_sha256(historical))


def test_new_solver_cpu_epoch_boundary_resume_matches_next_task_update(tmp_path, monkeypatch):
    args, graph = _args(), _debug_graph()
    torch.manual_seed(97)
    model = _model(args)
    optimizer = train.make_optimizer(model)
    _step(model, optimizer, graph)
    identity = _identity(args)
    checkpoint = tmp_path / "debug-last.pt"
    train._save(
        checkpoint,
        {
            "schema_version": 3,
            "epoch": 1,
            "resume_identity": identity,
            "resume_identity_sha256": train._canonical_sha256(identity),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.get_rng_state(),
        },
    )
    original_bytes = checkpoint.read_bytes()
    expected_loss = _step(model, optimizer, graph)
    expected_model, expected_optimizer = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
    )
    expected_draw = torch.rand(9)
    resumed = _model(args)
    resumed_optimizer = train.make_optimizer(resumed)
    saved = train.load_checkpoint_on_cpu(checkpoint)
    train.validate_resume_identity(
        saved["resume_identity"], _identity(args), saved["resume_identity_sha256"]
    )
    resumed.load_state_dict(saved["model_state"])
    resumed_optimizer.load_state_dict(saved["optimizer_state"])
    restored_devices = []

    def record_cuda_rng(state, device):
        assert state.device.type == "cpu" and state.dtype == torch.uint8
        restored_devices.append(device)

    monkeypatch.setattr(torch.cuda, "set_rng_state", record_cuda_rng)
    train.restore_checkpoint_rng(saved, torch.device("cpu"))
    actual_loss = _step(resumed, resumed_optimizer, graph)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    _assert_tree_equal(resumed.state_dict(), expected_model)
    _assert_tree_equal(resumed_optimizer.state_dict(), expected_optimizer)
    torch.testing.assert_close(torch.rand(9), expected_draw, rtol=0, atol=0)
    assert restored_devices == [torch.device("cpu")]
    assert checkpoint.read_bytes() == original_bytes


def test_debug_calibration_forwards_solver_arguments_and_runs_real_joint_updates(monkeypatch):
    args = _args(
        "--sampling",
        "neighbor",
        "--sample-seed-batch-size",
        "32",
        "--solver-steps",
        "5",
        "--solver-step-size",
        "0.2",
        "--solver-entropy",
        "0.8",
        "--solver-degree-barrier",
        "0.15",
    )
    original_args = copy.deepcopy(vars(args))
    graph = _debug_graph()
    indices = {"train": torch.arange(6), "validation": torch.arange(6, 9)}
    payload = {"dataset": args.dataset, "graphs": [vars(graph)], "classes": 3}
    sampler = SimpleNamespace(graph=graph, metadata=lambda: {"mode": "neighbor", "scope": "debug"})
    created_models, epochs = [], []
    constructor = train.GraphConditionedConductanceNodeClassifier

    def make_model(*positional, **kwargs):
        model = constructor(*positional, **kwargs)
        created_models.append(model)
        assert {key: kwargs[key] for key in conductance_configuration()} == {
            key: getattr(args, key) for key in conductance_configuration()
        }
        return model

    def batches(data, split, actual_sampler, epoch, device, seed, candidate, *, timing):
        assert candidate.sample_seed_batch_size == 64 and actual_sampler is sampler
        epochs.append(epoch)
        yield graph, torch.arange(6)

    class DebugMonitor:
        def __init__(self, device):
            assert device.type == "cpu"

        def start(self):
            return {"debug_fixture": True}

        def finish(self, **kwargs):
            return {"debug_fixture": True, **kwargs}

    monkeypatch.setattr(train, "GraphConditionedConductanceNodeClassifier", make_model)
    monkeypatch.setattr(train, "_require_cuda", lambda _device: None)
    monkeypatch.setattr(train, "validate_hardware_runtime", lambda *_args: {"debug_fixture": True})
    monkeypatch.setattr(train, "_prepare_data", lambda *_args: (graph, indices, sampler))
    monkeypatch.setattr(train, "_training_batches", batches)
    monkeypatch.setattr(batch_calibration, "RuntimeResourceMonitor", DebugMonitor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (10000, 20000))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 1000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 2000)
    report = batch_calibration.run_training_candidate(
        payload,
        args,
        torch.device("cpu"),
        physical_batch_size=64,
        workers=0,
        warmup_steps=1,
        measurement_steps=1,
        minimum_measure_seconds=0.000001,
    )
    assert report["status"] == "passed" and report["calibration_not_final"] is True
    assert report["parameter_update_verified"] is True and report["optimizer_state_bytes"] > 0
    assert report["model_phase"] == "joint_all_condition_parameter_groups_active"
    assert report["configuration"]["training_schedule"] == "joint"
    assert report["configuration"]["solver_steps"] == 5
    assert epochs == [1, 2] and len(created_models) == 1
    assert all(
        isinstance(op.estimator, GraphOptimizedConductance) and op.estimator.override is None
        for op in created_models[0].operators
    )
    assert vars(args) == original_args
