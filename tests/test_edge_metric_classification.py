"""DEBUG regression tests for capacity, packing, baselines and locked budgets."""
import copy

import pytest
import torch
from torch.nn import functional as F

from research.edge_metric_relations.classification.common import CONDITIONS,read_config,validate_config
from research.edge_metric_relations.classification.model import PackedClassifier
from research.edge_metric_relations.classification.verify import fixture,run_checks


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
