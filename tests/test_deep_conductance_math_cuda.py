"""Independent CUDA dense geometry and finite-step C derivatives, synthetic math only."""

import pytest
import torch
from torch.nn import functional as F

from research.conductance_gat.v5.operator import shared_head_diffusion
from research.conductance_gat.v5.optimization import GraphOptimizedConductance
from tests.test_aggregation_comparison_cuda import cuda_required  # noqa: F401


def geometry():
    edges = torch.tensor([[0, 0, 0, 1, 2, 4, 5, 5], [1, 2, 3, 2, 3, 5, 6, 7]], device="cuda")
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2], device="cuda")
    b = (F.one_hot(edges[1], 9) - F.one_hot(edges[0], 9)).double()
    omega = torch.tensor(
        [1.0, 2.5, 0.7, 1.3, 3.0, 1.2, 0.8, 2.1], device="cuda", dtype=torch.float64
    )
    edge_membership = F.one_hot(groups[edges[0]], 3).double()
    node_membership = F.one_hot(groups, 3).double()
    return edges, groups, b, omega, edge_membership, node_membership


@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("rho", [0.0, 0.1, 10.0])
def test_solver_scaled_gradient_against_independent_dense_energy(heads, rho):
    torch.manual_seed(703)
    edges, groups, b, omega, e, n = geometry()
    c = (0.4 + torch.rand(8, heads, device="cuda", dtype=torch.float64)).requires_grad_()
    delta = torch.randn_like(c)
    degree = b.abs().T @ (omega[:, None] * c)
    reference = b.abs().T @ omega
    active = reference > 0
    safe = torch.where(active[:, None], degree, torch.ones_like(degree))
    denominator = torch.where(active, reference, torch.ones_like(reference))
    mass = e.T @ omega
    count = n.T @ active.double()
    objective = (
        e.T @ (omega[:, None] * (c * delta + 0.7 * (c * c.log() - c + 1)))
    ) / mass.clamp_min(1)[:, None]
    objective = (
        objective - rho * (n.T @ (safe / denominator[:, None]).log()) / count.clamp_min(1)[:, None]
    )
    expected = torch.autograd.grad(objective.sum(), c)[0]
    module = (
        GraphOptimizedConductance(
            5, solver_entropy=0.7, solver_degree_barrier=rho, conductance_heads=heads
        )
        .cuda()
        .double()
    )
    actual, _ = module._scaled_gradient(
        c.log(), delta, edges, groups[edges[0]], omega, mass, count, 9
    )
    actual = actual * omega[:, None] / mass[groups[edges[0]], None]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("heads", [1, 3])
def test_full_c_solver_gradient_gauge_orientation_and_graph_independence(heads, record_property):
    torch.manual_seed(817)
    edges, groups, b, omega, e, n = geometry()
    x = torch.randn(9, 5, device="cuda", dtype=torch.float64, requires_grad=True)
    context = torch.randn(3, 18, device="cuda", dtype=torch.float64, requires_grad=True)
    omega.requires_grad_()
    degree = b.abs().sum(0)
    full = degree + torch.arange(9, device="cuda") % 3
    module = (
        GraphOptimizedConductance(
            5,
            conductance_heads=heads,
            solver_steps=8,
            solver_cost_scaling="width_scaled",
            edge_chunk_size=3,
        )
        .cuda()
        .double()
    )

    def forward(state=x, links=edges, ids=groups, ctx=context, sd=degree, fd=full, weights=omega):
        return module(
            state,
            links,
            ids,
            ctx.shape[0],
            graph_context=ctx,
            sample_degree=sd,
            full_degree=fd,
            edge_normalization_weight=weights,
        )

    c = forward()
    matrix = c[:, None] if c.ndim == 1 else c
    torch.testing.assert_close(
        (e.T @ (omega[:, None] * matrix)) / ((e.T @ omega).clamp_min(1)[:, None]),
        torch.tensor([[1.0], [1.0], [0.0]], device="cuda", dtype=torch.float64).expand(3, heads),
        rtol=1e-10,
        atol=1e-11,
    )
    assert torch.isfinite(c).all() and (c > 0).all()
    torch.testing.assert_close(forward(links=edges.flip(0)), c, rtol=1e-10, atol=1e-11)
    single = forward(
        state=x[:4],
        links=edges[:, :5],
        ids=groups[:4],
        ctx=context[:1],
        sd=degree[:4],
        fd=full[:4],
        weights=omega[:5],
    )
    torch.testing.assert_close(single, c[:5], rtol=1e-10, atol=1e-11)
    cotangent = torch.randn_like(c)
    variables = [x, context, omega, *module.parameters()]
    gradients = torch.autograd.grad((c * cotangent).sum(), variables)
    errors = {}
    for name, indices in [
        ("features", [0]),
        ("context", [1]),
        ("sampling_weights", [2]),
        ("theta", list(range(3, len(variables)))),
    ]:
        directions = {i: torch.randn_like(variables[i]) for i in indices}
        directions = {i: d / d.norm().clamp_min(1) for i, d in directions.items()}
        analytic = sum((gradients[i] * d).sum() for i, d in directions.items())
        originals = {i: variables[i].detach().clone() for i in indices}
        values = []
        epsilon = 1e-5
        try:
            with torch.no_grad():
                for sign in (-1, 1):
                    for i, d in directions.items():
                        variables[i].copy_(originals[i] + sign * epsilon * d)
                    values.append((forward() * cotangent).sum())
        finally:
            with torch.no_grad():
                for i, value in originals.items():
                    variables[i].copy_(value)
        finite = (values[1] - values[0]) / (2 * epsilon)
        torch.testing.assert_close(analytic, finite, rtol=2e-5, atol=2e-7)
        errors[name] = float((analytic - finite).abs())
    record_property("finite_difference_absolute_errors", str(errors))


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("normalization", ["row", "symmetric"])
def test_weighted_diffusion_and_all_gradients_against_dense_laplacian(shared, normalization):
    torch.manual_seed(149)
    edges, groups, b, omega, _, _ = geometry()
    omega.requires_grad_()
    c = (
        0.3 + torch.rand((8,) if shared else (8, 3), device="cuda", dtype=torch.float64)
    ).requires_grad_()
    v = torch.randn(9, 3, 4, device="cuda", dtype=torch.float64, requires_grad=True)
    beta = torch.rand(3, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    w = (c[:, None] if shared else c) * omega[:, None]
    w = w.expand(-1, 3)
    laplacian = torch.einsum("en,eh,em->hnm", b, w, b)
    degree = laplacian.diagonal(dim1=1, dim2=2)
    safe = torch.where(degree > 0, degree, torch.ones_like(degree))
    normalized = (
        laplacian / safe[:, :, None]
        if normalization == "row"
        else laplacian / safe.sqrt()[:, :, None] / safe.sqrt()[:, None, :]
    )
    expected = v - beta[groups, :, None] * torch.einsum("hnm,mhd->nhd", normalized, v)
    actual = shared_head_diffusion(
        v,
        c,
        edges,
        groups,
        beta,
        sampling_correction=omega,
        edge_chunk_size=3,
        propagation_normalization=normalization,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    cotangent = torch.randn_like(v)
    left = torch.autograd.grad(actual, (v, c, omega, beta), cotangent, retain_graph=True)
    right = torch.autograd.grad(expected, (v, c, omega, beta), cotangent)
    for a, d in zip(left, right, strict=True):
        torch.testing.assert_close(a, d, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(actual[8], v[8], rtol=0, atol=0)
