from types import SimpleNamespace

import pytest
import torch

from research.conductance_gat.v5.diagnostics import require_first_step_conductance_gradient
from research.conductance_gat.v5.model import (
    GraphConditionedBeta,
    GraphConditionedConductanceNodeClassifier,
)
from research.conductance_gat.v5.operator import shared_head_diffusion
from research.conductance_gat.v5.protocol import SCALE_PROFILES, beta_configuration
from research.conductance_gat.v5.train import configure_phase, phase_schedule


def graph():
    return SimpleNamespace(
        x=torch.randn(7, 5),
        incidence_edge_index=torch.tensor(
            [[0, 0, 1, 1, 2, 3, 4, 5], [1, 2, 2, 3, 4, 4, 5, 6]], dtype=torch.long
        ),
    )


def model(mode="dynamic"):
    return GraphConditionedConductanceNodeClassifier(
        5,
        3,
        hidden_channels=32,
        layers=2,
        heads=4,
        ffn_multiplier=2,
        dropout=0.0,
        conductance_mode=mode,
        activation_checkpoint=False,
    )


def test_default_beta_is_unmargined_sigmoid_with_nominal_point_one_initialization():
    torch.manual_seed(2)
    estimator = GraphConditionedBeta(32, 4)
    context = torch.randn(5, 2 * 32 + 8)
    beta = estimator(context)
    assert estimator.beta_parameterization == "sigmoid"
    assert estimator.beta_initial == 0.1
    assert estimator.beta_min is None and estimator.beta_max is None
    assert torch.all((0 < beta) & (beta < 1))
    torch.testing.assert_close(
        estimator.network[-1].bias.sigmoid(),
        torch.full((4,), 0.1),
        atol=1e-7,
        rtol=0,
    )
    # The final weight remains small and nonzero so upstream beta features receive
    # first-step gradients; consequently beta starts near, not exactly at, beta_initial.
    assert float((beta.detach() - 0.1).abs().max()) < 0.01


def test_historical_margin_sigmoid_remains_an_explicit_ablation():
    torch.manual_seed(2)
    estimator = GraphConditionedBeta(
        32,
        4,
        beta_parameterization="margin_sigmoid",
        beta_initial=0.5,
        beta_min=0.05,
        beta_max=0.95,
    )
    beta = estimator(torch.randn(5, 2 * 32 + 8))
    assert torch.all((0.05 < beta) & (beta < 0.95))
    torch.testing.assert_close(
        estimator.network[-1].bias.sigmoid(),
        torch.full((4,), 0.5),
        atol=0,
        rtol=0,
    )
    assert float((beta.detach() - 0.5).abs().max()) < 0.01


def test_beta_configuration_omits_irrelevant_margins_and_rejects_invalid_contracts():
    assert beta_configuration() == {
        "beta_parameterization": "sigmoid",
        "beta_initial": 0.1,
    }
    with pytest.raises(ValueError, match="only valid for margin_sigmoid"):
        beta_configuration("sigmoid", 0.1, 0.05, 0.95)
    with pytest.raises(ValueError, match="requires explicit"):
        beta_configuration("margin_sigmoid", 0.5)
    with pytest.raises(ValueError, match="min < initial < max"):
        beta_configuration("margin_sigmoid", 0.05, 0.05, 0.95)


def test_dynamic_c_is_shared_positive_relative_and_not_dead_at_initialization():
    torch.manual_seed(3)
    network, data = model(), graph()
    logits = network(data)
    c = network.operators[0].estimator.last_c
    assert c.shape == (data.incidence_edge_index.shape[1],)
    assert torch.all(c > 0)
    assert float(c.mean()) == pytest.approx(1.0, abs=2e-6)
    assert float(c.std()) > 0
    logits.square().mean().backward()
    result = require_first_step_conductance_gradient(network)
    assert result["passed"] and all(row["upstream_gradient_norm"] > 0 for row in result["layers"])


def test_fixed_c_is_parameter_free_and_shared_initialization_stays_paired():
    torch.manual_seed(31)
    fixed = model("fixed_one")
    torch.manual_seed(31)
    dynamic = model("dynamic")

    for operator in fixed.operators:
        assert list(operator.estimator.parameters()) == []
        assert operator.estimator.node_projection is None
        assert operator.estimator.context_projection is None
        assert operator.estimator.score_norm is None
        assert operator.estimator.score_network is None
    assert all(list(operator.estimator.parameters()) for operator in dynamic.operators)

    fixed_shared = {
        name: value
        for name, value in fixed.state_dict().items()
        if ".operator.estimator." not in name
    }
    dynamic_shared = {
        name: value
        for name, value in dynamic.state_dict().items()
        if ".operator.estimator." not in name
    }
    assert fixed_shared.keys() == dynamic_shared.keys()
    for name in fixed_shared:
        torch.testing.assert_close(fixed_shared[name], dynamic_shared[name], rtol=0, atol=0)

    fixed(graph())
    for operator in fixed.operators:
        torch.testing.assert_close(
            operator.estimator.last_c,
            torch.ones_like(operator.estimator.last_c),
            rtol=0,
            atol=0,
        )


def test_undirected_orientation_does_not_change_prediction():
    torch.manual_seed(4)
    network, data = model(), graph()
    network.eval()
    expected = network(data)
    reversed_graph = SimpleNamespace(
        x=data.x,
        incidence_edge_index=data.incidence_edge_index.flip(0).flip(1),
    )
    actual = network(reversed_graph)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_shared_operator_is_scale_invariant_but_beta_changes_diffusion():
    torch.manual_seed(5)
    message = torch.randn(4, 2, 3)
    incidence = torch.tensor([[0, 1, 2], [1, 2, 3]])
    c = torch.tensor([0.5, 1.2, 2.0])
    batch = torch.zeros(4, dtype=torch.long)
    beta = torch.tensor([[0.2, 0.8]])
    expected = shared_head_diffusion(message, c, incidence, batch, beta)
    actual = shared_head_diffusion(message, 17 * c, incidence, batch, beta)
    torch.testing.assert_close(actual, expected)
    no_diffusion = shared_head_diffusion(message, c, incidence, batch, torch.zeros_like(beta))
    torch.testing.assert_close(no_diffusion, message)


def test_chunked_inplace_diffusion_matches_one_shot_values_and_gradients():
    torch.manual_seed(51)
    incidence = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 4]])
    batch = torch.zeros(5, dtype=torch.long)
    message = torch.randn(5, 2, 3, requires_grad=True)
    c = torch.rand(5, requires_grad=True).add(0.5)
    beta = torch.rand(1, 2, requires_grad=True)
    actual = shared_head_diffusion(message, c, incidence, batch, beta, edge_chunk_size=1)

    tail, head = incidence
    degree = c.new_zeros(5).index_add(0, tail, c).index_add(0, head, c)
    inverse = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    weight = c * inverse[tail] * inverse[head]
    propagated = message.new_zeros(message.shape)
    propagated = propagated.index_add(0, tail, weight[:, None, None] * message[head])
    propagated = propagated.index_add(0, head, weight[:, None, None] * message[tail])
    expected = message + beta[batch, :, None] * (propagated - (degree > 0)[:, None, None] * message)
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), (message, c, beta), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.square().sum(), (message, c, beta))
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_default_profile_is_research_scale_and_phase_schedule_is_nonempty():
    profile = SCALE_PROFILES["reference"]
    network = GraphConditionedConductanceNodeClassifier(128, 40, **profile)
    assert sum(parameter.numel() for parameter in network.parameters()) > 5_000_000
    schedule = phase_schedule(20, [0.1, 0.1, 0.4, 0.4])
    assert [item["length"] for item in schedule] == [2, 2, 8, 8]
    warmup = configure_phase(network, "spatial_warmup", 0)
    assert warmup["conductance_override"] == "ones"
    calibration = configure_phase(network, "conductance_calibration", 0)
    assert calibration["active_parameter_groups"] == ["conductance"]


def test_sampler_carries_original_structure_and_nontrivial_seed_batch():
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from research.conductance_gat.v5.sampling import TransductiveGraphSampler

    data = graph()
    incidence = data.incidence_edge_index
    arcs = torch.cat((incidence, incidence.flip(0)), dim=1)
    source = Data(x=data.x, y=torch.arange(7) % 3, edge_index=arcs, incidence_edge_index=incidence)
    sampler = TransductiveGraphSampler(
        source,
        torch.arange(7),
        mode="neighbor",
        seed_batch_size=4,
        fanouts=[2, 1],
        model_seed=0,
    )
    sampled = next(sampler.iter_epoch(1))
    assert sampled.graph_structure.shape == (1, 6)
    assert sampled.full_degree.shape == (sampled.num_nodes,)
    assert int(sampled.train_mask.sum()) == 4
    assert torch.all(sampled.sampling_correction >= 1)
