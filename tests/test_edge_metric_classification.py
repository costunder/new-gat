"""DEBUG regression tests for capacity, packing, baselines and locked budgets."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from research.edge_metric_relations.classification.common import CONDITIONS,read_config,validate_config
from research.edge_metric_relations.classification.model import PackedClassifier
from research.edge_metric_relations.classification.verify import fixture,run_checks
from research.edge_metric_relations.classification import training


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_all_active_classifiers_complete_ce_gradient_update_path():
    record=run_checks(include_core=False)
    assert record["passed"] and len(record["classifiers"])==15


@pytest.mark.parametrize("condition",CONDITIONS)
def test_matched_projection_and_dropout_initialization(condition):
    model=PackedClassifier(condition,6,3,(11,23),hidden=64).double()
    other=PackedClassifier("G1",6,3,(11,23),hidden=64).double()
    assert all(torch.equal(a,b) for a,b in zip(model.weights,other.weights))
    assert torch.equal(model.dropout_keys,other.dropout_keys)
    variant=model.variant
    extra=2*(385*int(variant in ("D1","F2","DA"))+641*int(variant in ("F1","F2","DA")))+6*int(variant=="P2")
    assert model.parameters_per_seed==6*64+64*3+extra


@pytest.mark.parametrize("condition",("unit__F2","local_degree__DA","G1","G2","P2"))
def test_seed_packing_preserves_each_independent_adam_update(condition):
    graph=fixture();packed=PackedClassifier(condition,6,3,(11,23),path_chunk=11).double()
    independent=[PackedClassifier(condition,6,3,(seed,),path_chunk=11).double() for seed in (11,23)]
    def update(model):
        optimizer=torch.optim.Adam(model.weight_decay_groups(.0005),lr=.003,foreach=False)
        for epoch in range(2):
            optimizer.zero_grad();out,_=model(graph,epoch=epoch)
            ce=F.cross_entropy(out.flatten(0,1),graph.y.repeat(len(model.seeds)),reduction="none").reshape(len(model.seeds),-1).mean(1)
            ce.sum().backward();optimizer.step()
    update(packed)
    for model in independent:update(model)
    for name,value in packed.named_parameters():
        for i,model in enumerate(independent):
            torch.testing.assert_close(value[i],dict(model.named_parameters())[name][0],rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize("condition", ("unit__F2", "local_degree__DA"))
def test_exact_relation_chunks_preserve_logits_gradients_and_adam_updates(condition):
    """Changing allocation size preserves every pair and the CE learning path."""
    graph = fixture()
    small = PackedClassifier(condition, 6, 3, (11, 23), path_chunk=2).double()
    whole = PackedClassifier(condition, 6, 3, (11, 23), path_chunk=100000).double()
    optimizers = [torch.optim.Adam(model.weight_decay_groups(.0005), lr=.003, foreach=False)
                  for model in (small, whole)]
    for epoch in range(2):
        outputs = []
        for model, optimizer in zip((small, whole), optimizers):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(graph, epoch=epoch)
            loss = F.cross_entropy(logits.flatten(0, 1), graph.y.repeat(2), reduction="none")
            loss.reshape(2, -1).mean(1).sum().backward()
            outputs.append(logits.detach())
        torch.testing.assert_close(*outputs, rtol=1e-10, atol=1e-12)
        for (name, first), (other_name, second) in zip(small.named_parameters(), whole.named_parameters()):
            assert name == other_name and first.grad is not None and second.grad is not None
            torch.testing.assert_close(first.grad, second.grad, rtol=1e-9, atol=1e-11)
        for optimizer in optimizers:
            optimizer.step()
        for first, second in zip(small.parameters(), whole.parameters()):
            torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-11)


def test_gcn_and_two_hop_baselines_match_dense_self_loop_normalization():
    graph=fixture();z=graph.x[None]
    p=z.new_zeros(7,7);p[graph.gcn_edges[1],graph.gcn_edges[0]]=graph.gcn_weights
    for condition,power in (("G1",1),("G2",2)):
        model=PackedClassifier(condition,6,3,(11,)).double()
        torch.testing.assert_close(model._baseline(graph,z,0),torch.linalg.matrix_power(p,power)@z)


def test_polynomial_supremum_uses_interior_extremum_and_finite_small_coefficients():
    graph=fixture();model=PackedClassifier("P2",6,3,(11,)).double()
    with torch.no_grad():model.polynomial[0].copy_(torch.tensor([[0.,4.,-2.]],dtype=torch.float64))
    p=graph.x.new_zeros(7,7);p[graph.gcn_edges[1],graph.gcn_edges[0]]=graph.gcn_weights
    l=torch.eye(7,dtype=p.dtype)-p
    expected=(4*l-2*l@l)@graph.x/2
    torch.testing.assert_close(model._baseline(graph,graph.x[None],0)[0],expected)
    with torch.no_grad():model.polynomial[0].copy_(torch.tensor([[1.,1.,1e-310]],dtype=torch.float64))
    output=model._baseline(graph,graph.x[None],0);output.sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(model.polynomial[0].grad).all()


def test_full_budget_and_config_reduction_rejected():
    config=read_config(profile="full")
    assert config["training"]["total_runs"]==630
    assert config["training"]["total_updates"]==315000
    changed=copy.deepcopy(config);changed["backbone"]["hidden_dim"]=8
    with pytest.raises(ValueError,match="locked"):validate_config(changed)


def test_calibration_estimate_screens_allocations_without_changing_the_scientific_contract():
    graph, config = fixture(), read_config(profile="full")
    original = copy.deepcopy(config)
    small = training._estimated_trial_increment_bytes(graph, "unit__F2", 1, config, 2)
    whole = training._estimated_trial_increment_bytes(graph, "unit__F2", 1, config, 100000)
    packed = training._estimated_trial_increment_bytes(graph, "unit__F2", 5, config, 2)
    assert 0 < small["increment_bytes"] < whole["increment_bytes"]
    assert packed["increment_bytes"] > small["increment_bytes"]
    assert config == original
    assert small["increment_bytes"] == sum(small["components_bytes"].values())


def test_constant_pair_gate_screens_cross_flows_and_baselines_use_their_actual_edges():
    graph, config = fixture(), read_config(profile="full")
    f0 = training._estimated_trial_increment_bytes(graph, "unit__F0", 5, config, 100000)
    f1 = training._estimated_trial_increment_bytes(graph, "unit__F1", 5, config, 100000)
    assert 0 < f0["components_bytes"]["pair_chunk_intermediates"] < f1["components_bytes"]["pair_chunk_intermediates"]
    baseline = training._estimated_trial_increment_bytes(graph, "G1", 5, config, 100000)
    actual_edges = graph.gcn_edges.shape[1]
    expected = 5*actual_edges*4*config["backbone"]["hidden_dim"]*graph.x.element_size()
    assert baseline["components_bytes"]["baseline_chunk_intermediates"] == expected
    assert baseline["components_bytes"]["pair_chunk_intermediates"] == 0


def _fake_cuda_calibration_graph():
    graph = fixture()
    geometry = graph.geometry_for("unit")
    graph.x = SimpleNamespace(device=torch.device("cuda"), shape=(graph.num_nodes, 6), element_size=lambda: 4)
    graph.topology = SimpleNamespace(num_local_edges=geometry.num_occurrences)
    return graph


def test_packing_rejects_reserved_peak_and_selects_a_measured_safe_candidate(monkeypatch):
    graph, config = _fake_cuda_calibration_graph(), read_config(profile="debug")
    config["runtime"]["relation_chunk_candidates"] = [2, 4]
    config["resources"]["parallel_final_run_candidates"] = [1, 2]
    before = {"allocation_budget_bytes": 10**9, "allocated_bytes": 100,
              "reserved_bytes": 200, "allocated_ceiling_bytes": 10**9+100,
              "reserved_ceiling_bytes": 10**9+200}
    monkeypatch.setattr(training.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(training, "memory_window", lambda device, policy: before)
    def measured(graph, condition, seeds, lr, config, chunk):
        unsafe_reserved = len(seeds) == 2
        return .1, {"peak_allocated_bytes": 1000,
                    "peak_reserved_bytes": 2*10**9 if unsafe_reserved else 2000}, 100
    monkeypatch.setattr(training, "benchmark_trial", measured)
    choice = training.choose_packing(graph, "unit__F2", [11, 23], .003, config,
                                     "final", hardware_profile="a100-mig-10gb")
    assert choice["packed_runs"] == 1
    assert choice["hardware_profile"] == "a100-mig-10gb"
    assert any(row["packed_runs"] == 2 and row["status"] == "memory_safety_rejected"
               and row["measured"] for row in choice["trials"])
    assert choice["selected_peak_reserved_bytes"] == 2000


def test_oversized_candidate_is_rejected_before_model_allocation(monkeypatch):
    graph, config = _fake_cuda_calibration_graph(), read_config(profile="debug")
    config["runtime"]["relation_chunk_candidates"] = [2, 4]
    config["resources"]["parallel_final_run_candidates"] = [1, 2]
    small = training._estimated_trial_increment_bytes(graph, "unit__F2", 1, config, 2)["increment_bytes"]
    whole = training._estimated_trial_increment_bytes(graph, "unit__F2", 1, config, 100000)["increment_bytes"]
    budget = (small+whole)//2
    before = {"allocation_budget_bytes": budget, "allocated_bytes": 0, "reserved_bytes": 0,
              "allocated_ceiling_bytes": budget, "reserved_ceiling_bytes": budget}
    calls = []
    monkeypatch.setattr(training.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(training, "memory_window", lambda device, policy: before)
    def measured(graph, condition, seeds, lr, config, chunk):
        calls.append((len(seeds), chunk))
        return .1, {"peak_allocated_bytes": small, "peak_reserved_bytes": small}, 100
    monkeypatch.setattr(training, "benchmark_trial", measured)
    choice = training.choose_packing(graph, "unit__F2", [11, 23], .003, config,
                                     "final", hardware_profile="a100-mig-10gb")
    rejects = [row for row in choice["trials"] if row.get("rejection_reason") == "estimated_before_allocation"]
    assert rejects and all(not row["measured"] for row in rejects)
    assert all((row["packed_runs"], row["path_chunk"]) not in calls for row in rejects)
    assert choice["packed_runs"] >= 1


def test_checked_resume_packing_cannot_silently_move_to_an_undersized_allocation(monkeypatch):
    graph, config = _fake_cuda_calibration_graph(), read_config(profile="debug")
    monkeypatch.setattr(training.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(training, "memory_window", lambda device, policy: {"allocation_budget_bytes": 1})
    with pytest.raises(RuntimeError, match="checkpoint and Adam packing preserved"):
        training._validate_calibration_memory(graph, "unit__F2", {"packed_runs": 2, "path_chunk": 4},
                                              config, hardware_profile="a100-mig-10gb")


def test_actual_debug_calibration_reports_memory_and_the_complete_epoch_path():
    graph, config = fixture(), read_config(profile="debug")
    graph.num_features = graph.x.shape[1]
    graph.train_mask = torch.tensor([True, True, True, True, False, False, False])
    graph.val_mask = torch.tensor([False, False, False, False, True, True, False])
    graph.test_mask = ~(graph.train_mask | graph.val_mask)
    seconds, memory, parameters = training.benchmark_trial(graph, "unit__F2", (11, 23), .003, config, 2)
    assert seconds > 0 and parameters > 0
    assert memory == {"peak_allocated_bytes": None, "peak_reserved_bytes": None}


def test_resume_revalidates_same_packing_and_preserves_original_calibration(tmp_path, monkeypatch):
    graph, config = fixture(), read_config(profile="debug")
    source, destination = tmp_path/"old.json", tmp_path/"new.json"
    selection = {"packed_runs": 2, "path_chunk": 4, "trials": []}
    training.write_json(source, {"dataset":graph.name, "condition":"unit__F2",
        **training._condition_fields("unit__F2"), "phase":"final",
        "config_digest":training.digest(config), "selection":selection})
    calls = []
    def measured(graph, condition, seeds, lr, config, chunk):
        calls.append((condition, tuple(seeds), chunk))
        return .1, {"peak_allocated_bytes":None, "peak_reserved_bytes":None}, 100
    monkeypatch.setattr(training, "benchmark_trial", measured)
    reused = training.preserve_resume_calibration(source, destination, config, "final", graph.name,
        "unit__F2", graph=graph, hardware_profile="a100-mig-10gb")
    assert calls == [("unit__F2", (11, 23), 4)]
    assert destination.read_bytes() == source.read_bytes()
    assert reused["packed_runs"] == selection["packed_runs"] and reused["path_chunk"] == selection["path_chunk"]
    assert reused["device_revalidation"]["status"] == "same_packing_remeasured_safe"
    assert (tmp_path/"new_device_revalidation.json").is_file()
