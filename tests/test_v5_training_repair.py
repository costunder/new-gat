"""Explicit CPU/synthetic debug checks; not official-data or GPU performance evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from test_v5_transition_state import _assert_nested_equal
from test_v5_transition_training import (
    DebugCrash,
    DebugGraph,
    _args,
    _debug_data,
    _install_cpu_debug_environment,
    _run,
)

from research.conductance_gat.v5 import train
from research.conductance_gat.v5.learning_budget import plan_learning_budget
from research.conductance_gat.v5.model import (
    _finite_standard_deviation,
    graph_context_features,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_variance_forward_preserved_and_gradient_finite(dtype):
    variance = torch.tensor([-1.0, 0.0, 1.0, 4.0], dtype=dtype, requires_grad=True)
    output = _finite_standard_deviation(variance)
    assert torch.equal(output, variance.clamp_min(0).sqrt())
    output.sum().backward()
    torch.testing.assert_close(variance.grad, torch.tensor([0.0, 0.0, 0.5, 0.25], dtype=dtype))


def test_safe_std_does_not_hide_nonfinite_input():
    value = _finite_standard_deviation(torch.tensor([float("nan"), float("inf")]))
    assert torch.isnan(value[0]) and torch.isposinf(value[1])


def test_constant_channel_graph_context_has_finite_task_gradient():
    state = torch.tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0], [1.0, 6.0, 7.0]], requires_grad=True)
    edges = torch.tensor([[0, 1], [1, 2]])
    context, _, _ = graph_context_features(state, edges, torch.zeros(3, dtype=torch.long), 1)
    context.sum().backward()
    assert torch.isfinite(state.grad).all()
    assert context[0, 3] == 0


@pytest.mark.parametrize("batch,planned", [(8, 200), (16, 300), (20, 600)])
def test_actual_ppi_counts_resolve_explicit_update_budget(batch, planned):
    args = SimpleNamespace(
        learning_budget_policy="reference_updates",
        budget_reference_batch_size=None,
        training_schedule="joint",
        hardware_profile="a6000-48gb",
        epochs=200,
        patience=50,
    )
    loader = DebugLoader([None] * 20, batch)
    budget = train.resolve_learning_budget({"train": loader}, None, None, args)
    assert budget["planned_epochs"] == planned
    assert budget["planned_maximum_optimizer_steps"] == 600
    assert budget["patience_optimizer_steps"] == 150
    assert args.epochs == 200 and args.patience == 50


def test_update_patience_uses_restored_actual_steps_not_epoch_age():
    budget = plan_learning_budget(200, 2, 3, 1, "reference_updates")
    args = SimpleNamespace(condition="shared_dynamic_c")
    history = [
        {"epoch": 1, "optimizer_steps": 4, "phase": {"phase": "joint"}},
        {"epoch": 5, "optimizer_steps": 9, "phase": {"phase": "joint"}},
    ]
    assert not train.budget_should_stop(
        args, budget, history, primary_best_epoch=1, joint_best_epoch=1
    )
    history.append({"epoch": 6, "optimizer_steps": 10, "phase": {"phase": "joint"}})
    assert train.budget_should_stop(args, budget, history, primary_best_epoch=1, joint_best_epoch=1)


class DebugLoader:
    """Small disjoint-union CPU fixture, not a replacement production DataLoader."""

    def __init__(self, graphs, batch_size):
        self.dataset, self.batch_size = graphs, batch_size
        self.generator = torch.Generator().manual_seed(21)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        for start in range(0, len(order), self.batch_size):
            graphs = [self.dataset[index] for index in order[start : start + self.batch_size]]
            offsets, offset = [], 0
            for graph in graphs:
                offsets.append(offset)
                offset += graph.x.shape[0]
            yield DebugGraph(
                x=torch.cat([graph.x for graph in graphs]),
                y=torch.cat([graph.y for graph in graphs]),
                incidence_edge_index=torch.cat(
                    [
                        graph.incidence_edge_index + shift
                        for graph, shift in zip(graphs, offsets, strict=True)
                    ],
                    dim=1,
                ),
                batch=torch.cat(
                    [
                        torch.full((graph.x.shape[0],), index, dtype=torch.long)
                        for index, graph in enumerate(graphs)
                    ]
                ),
                num_graphs=len(graphs),
            )


def _repair_fixture(tmp_path, monkeypatch):
    graph, indices, _, protocol = _debug_data()
    _install_cpu_debug_environment(monkeypatch, graph, indices)
    generator = torch.Generator().manual_seed(312)
    graphs = []
    for _index in range(8):
        item = graph.clone()
        item.x = torch.randn(item.x.shape, generator=generator)
        item.y = torch.randint(2, (item.x.shape[0], 3), generator=generator).float()
        graphs.append(item)
    data = {"train": DebugLoader(graphs[:6], 3), "validation": DebugLoader(graphs[6:], 2)}
    monkeypatch.setattr(train, "_prepare_data", lambda *_args: (data, None, None))
    # Fixed validation order: the production loader also does not shuffle validation.
    data["validation"].generator = torch.Generator().manual_seed(0)
    payload = {
        "dataset": "ppi",
        "classes": 3,
        "graphs": [vars(item) for item in graphs],
        "classification": "synthetic_cpu_debug_fixture_not_official_ppi",
    }
    protocol = {**protocol, "dataset": "ppi"}
    args = _args(tmp_path)
    args.dataset, args.batch_size, args.workers = "ppi", 3, 0
    args.solver_cost_scaling, args.beta_initial = "width_scaled", 0.5
    args.learning_budget_policy, args.budget_reference_batch_size = "reference_updates", 2
    train.validate_args(args)
    return args, payload, protocol


def test_real_cpu_training_extended_budget_and_completed_resume_do_not_retrain(
    tmp_path, monkeypatch
):
    args, payload, protocol = _repair_fixture(tmp_path / "continuous", monkeypatch)
    result = _run(args, payload, protocol)
    assert result["epochs_run"] == 6 and result["optimizer_steps"] == 12
    assert result["configuration"]["epochs"] == 4
    assert result["learning_budget"]["planned_epochs"] == 6
    assert result["first_active_conductance_gradient"]["passed"]
    saved = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    for name in ("conductance", "spatial_w", "beta", "backbone"):
        assert result["effective_optimizer_steps_by_group"][name] == 12

    def forbidden_update(*_args, **_kwargs):
        raise AssertionError("completed repaired run must not train again")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_update)
    resumed = _run(args, payload, protocol)
    assert resumed["optimizer_steps"] == 12
    final = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    _assert_nested_equal(final["model_state"], saved["model_state"])


def test_repaired_cpu_training_interruption_preserves_model_and_optimizer(tmp_path, monkeypatch):
    args, payload, protocol = _repair_fixture(tmp_path / "expected", monkeypatch)
    _run(args, payload, protocol)
    expected = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    args, payload, protocol = _repair_fixture(tmp_path / "resumed", monkeypatch)
    save = train._save

    def crash_after_epoch(path, value):
        save(path, value)
        if path.name == "last.pt" and value["epoch"] == 3:
            raise DebugCrash("CPU debug interruption at completed epoch boundary")

    monkeypatch.setattr(train, "_save", crash_after_epoch)
    with pytest.raises(DebugCrash):
        _run(args, payload, protocol)
    monkeypatch.setattr(train, "_save", save)
    _run(args, payload, protocol)
    actual = train.load_checkpoint_on_cpu(args.output_dir / "last.pt")
    for field in ("model_state", "optimizer_state", "learning_budget", "optimizer_steps"):
        _assert_nested_equal(actual[field], expected[field])


def test_recipe_change_is_not_silently_treated_as_same_checkpoint(tmp_path):
    args = _args(tmp_path)
    before = train.configuration(args)
    args.solver_cost_scaling, args.beta_initial = "width_scaled", 0.5
    args.learning_budget_policy = "reference_updates"
    after = train.configuration(args)
    assert before != after
    assert "solver_cost_scaling" not in before
    assert after["solver_cost_scaling"] == "width_scaled"
    assert "learning_budget_policy" not in train.architecture_configuration(args)


def test_legacy_transition_does_not_silently_ignore_new_budget(tmp_path):
    from scripts import run_v5_transition

    args = run_v5_transition.parser().parse_args(
        [
            "--source-manifest",
            str(tmp_path / "unread-source.json"),
            "--output-dir",
            str(tmp_path / "untouched-output"),
            "--learning-budget-policy",
            "reference_updates",
        ]
    )
    with pytest.raises(ValueError, match="not a legacy transition policy"):
        run_v5_transition.build_plan(args)
    assert not (tmp_path / "untouched-output").exists()
