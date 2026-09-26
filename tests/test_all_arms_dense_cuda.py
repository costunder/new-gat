"""All 16 incidence arms against an independent dense CUDA propagation/Gram/lift path.

Only C and beta generators/context features are shared modules. Their finite-step
C derivatives are separately checked in test_deep_conductance_math_cuda.py.
Synthetic math dimensions are explicit, not production defaults.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from experiments.aggregation_comparison.model import ARMS, AggregationClassifier
from research.conductance_gat.v5.model import _static_graph_context, graph_context_features
from tests.test_aggregation_comparison_cuda import cuda_required  # noqa: F401
from tests.test_deep_conductance_math_cuda import geometry


def dense_forward(net, graph, b):
    h = net.encoder(graph.x)
    past = [h]
    static = _static_graph_context(
        graph.x.float(), graph.incidence_edge_index, graph.batch, 3, graph.full_degree, None
    )
    for index, op in enumerate(net.layers):
        context, sd, fd = graph_context_features(
            h.float(),
            graph.incidence_edge_index,
            graph.batch,
            3,
            graph.full_degree,
            None,
            static_context=static,
        )
        c = op.estimator(
            h.float(),
            graph.incidence_edge_index,
            graph.batch,
            3,
            graph_context=context,
            sample_degree=sd,
            full_degree=fd,
            edge_normalization_weight=graph.edge_normalization_weight,
        )
        beta = op.beta_estimator(context)
        weight = (c[:, None] if c.ndim == 1 else c) * graph.sampling_correction[:, None]
        weight = weight.expand(-1, net.heads)
        lap = torch.einsum("en,eh,em->hnm", b, weight, b)
        degree = lap.diagonal(dim1=1, dim2=2)
        denominator = torch.where(degree > 0, degree, torch.ones_like(degree))

        value = torch.einsum("nd,hdk->nhk", h, op.value_weight)
        if op.lift in ("linear", "pre"):
            second = value if op.lift == "linear" else value.square()
            value = torch.cat((value, second), -1)
        output = value - beta[graph.batch, :, None] * torch.einsum(
            "hnm,mhd->nhd", lap / denominator[:, :, None], value
        )
        if op.lift == "post":
            output = torch.cat((output, output.square()), -1)
        if op.lift_projection is not None:
            output = torch.einsum("nhd,hdk->nhk", output, op.lift_projection)
        output = output.flatten(1) @ op.output_projection.weight.T
        if len(net.energy_readouts):
            projected = torch.einsum("knd,hdw->knhw", torch.stack(past), op.value_weight)
            delta = torch.einsum("en,knhd->kehd", b, projected)
            count = len(past)
            pairs = (
                torch.arange(count, device="cuda").expand(2, -1)
                if net.diagonal_only
                else torch.triu_indices(count, count, device="cuda")
            )
            products = (delta[pairs[0]] * delta[pairs[1]]).sum(-1).permute(1, 2, 0)
            gamma = torch.einsum("en,ehp->nhp", b.abs(), products * weight[..., None] / 2)
            extra = torch.einsum("nhp,hpd->nhd", gamma, net.energy_readouts[index])
            output = output + extra.flatten(1) @ op.output_projection.weight.T
        h = F.relu(output)
        past.append(h)
    return net.decoder(h)


@pytest.mark.parametrize("arm", [a for a in ARMS if a.startswith("incidence")])
def test_non_neutral_all_arms_dense_forward_and_parameter_gradients(arm, record_property):
    from research.conductance_gat.edge_selection.topology import build_topology

    torch.manual_seed(221)
    edges, groups, b, omega, _, _ = geometry()
    b = b.float()
    omega = omega.float()
    plan = build_topology(9, edges.cpu(), groups.cpu()).to(torch.device("cuda:0"))
    x = torch.randn(9, 5, device="cuda", requires_grad=True)
    graph = SimpleNamespace(
        x=x,
        incidence_edge_index=edges,
        batch=groups,
        _v5_num_graphs=3,
        edge_selection_topology=plan,
        full_degree=b.abs().sum(0) + torch.arange(9, device="cuda") % 3,
        sampling_correction=omega,
        edge_normalization_weight=omega,
    )

    def make():
        return AggregationClassifier(
            5,
            3,
            arm=arm,
            selection_config={"condition": "full"},
            hidden_channels=12,
            layers=3,
            heads=3,
            dropout=0,
            edge_chunk_size=3,
            activation_checkpoint=True,
        ).cuda()

    actual_model = make()
    with torch.no_grad():
        for weight in actual_model.energy_readouts:
            weight.normal_(0, 0.07)
        for op in actual_model.layers:
            if op.lift_projection is not None:
                op.lift_projection.add_(torch.randn_like(op.lift_projection) * 0.07)
    dense_model = make()
    dense_model.load_state_dict(actual_model.state_dict())
    actual = actual_model(graph)
    expected = dense_forward(dense_model, graph, b)
    cotangent = torch.randn_like(actual)
    left = torch.autograd.grad(actual, (x, *actual_model.parameters()), cotangent)
    right = torch.autograd.grad(expected, (x, *dense_model.parameters()), cotangent)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for a, d in zip(left, right, strict=True):
        torch.testing.assert_close(a, d, rtol=2e-5, atol=2e-6)
    record_property(
        "independent_dense_error",
        json.dumps(
            {
                "output_max_abs": float((actual - expected).detach().abs().max()),
                "input_gradient_max_abs": float((left[0] - right[0]).abs().max()),
                "parameter_gradient_max_abs": max(
                    float((a - d).abs().max()) for a, d in zip(left[1:], right[1:], strict=True)
                ),
                "energy_and_lift_non_neutral": True,
                "C_beta_generators_shared": True,
            }
        ),
    )
