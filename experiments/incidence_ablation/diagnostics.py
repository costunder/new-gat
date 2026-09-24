"""Frozen-metric incidence probes, not claims of a neural-network inverse."""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def components(incidence, nodes):
    ends = incidence.detach().cpu().numpy()
    adjacency = coo_matrix((np.ones(ends.shape[1]), ends), shape=(nodes, nodes))
    count, labels = connected_components(adjacency, directed=False)
    return count, torch.as_tensor(labels, device=incidence.device, dtype=torch.long)


def component_mean(x, labels, count):
    sizes = torch.bincount(labels, minlength=count).to(x.dtype)
    sums = x.new_zeros((count, x.shape[1])).index_add_(0, labels, x)
    return sums / sizes[:, None]


def laplacian(incidence, weight, nodes):
    tail, head = incidence
    index = torch.stack((torch.cat((tail, head, tail, head)), torch.cat((tail, head, head, tail))))
    return torch.sparse_coo_tensor(
        index, torch.cat((weight, weight, -weight, -weight)), (nodes, nodes), check_invariants=True
    ).coalesce()


def solve_centered(
    incidence, weight, differences, labels, count, *, tolerance=1e-7, iterations=2000
):
    """Batched Jacobi-PCG on the component-centered subspace, every RHS.

    The solve consumes Bx, never x or its means. Nonconvergence is a reported
    audit error, not silently returned as a successful reconstruction.
    """
    nodes = labels.numel()
    tail, head = incidence
    rhs = differences.new_zeros((nodes, differences.shape[1]))
    rhs.index_add_(0, head, weight[:, None] * differences)
    rhs.index_add_(0, tail, -weight[:, None] * differences)
    return solve_rhs(
        incidence, weight, rhs, labels, count, tolerance=tolerance, iterations=iterations
    )


def solve_rhs(incidence, weight, rhs, labels, count, *, tolerance=1e-7, iterations=2000):
    nodes = labels.numel()
    tail, head = incidence
    matrix = laplacian(incidence, weight, nodes)
    rhs -= component_mean(rhs, labels, count)[labels]
    degree = weight.new_zeros(nodes).index_add_(0, tail, weight).index_add_(0, head, weight)
    degree = degree.clamp_min(torch.finfo(weight.dtype).tiny)
    result, residual = torch.zeros_like(rhs), rhs.clone()
    norm = rhs.norm(dim=0)
    target = tolerance * norm
    active = norm > 0
    z = residual / degree[:, None]
    z -= component_mean(z, labels, count)[labels]
    direction = z.clone()
    rz = (residual * z).sum(0)
    used = 0
    for iteration in range(iterations):
        if not bool(active.any()):
            break
        used = iteration + 1
        product = torch.sparse.mm(matrix, direction)
        denom = (direction * product).sum(0)
        if bool(((denom <= 0) & active).any()):
            raise RuntimeError("incidence PCG lost positive definiteness on an active RHS")
        alpha = torch.where(active, rz / denom.clamp_min(torch.finfo(weight.dtype).tiny), 0)
        result += direction * alpha
        residual -= product * alpha
        active = residual.norm(dim=0) > target
        z = residual / degree[:, None]
        z -= component_mean(z, labels, count)[labels]
        new_rz = (residual * z).sum(0)
        beta = torch.where(active, new_rz / rz.clamp_min(torch.finfo(weight.dtype).tiny), 0)
        direction = z + direction * beta
        direction[:, ~active] = 0
        rz = new_rz
    result -= component_mean(result, labels, count)[labels]
    actual = (torch.sparse.mm(matrix, result) - rhs).norm(dim=0)
    relative = torch.where(norm > 0, actual / norm.clamp_min(torch.finfo(norm.dtype).tiny), actual)
    if bool((relative > tolerance * 10).any()):
        raise RuntimeError(
            f"incidence reconstruction did not converge after {used} iterations; "
            f"max relative residual={relative.max().item():.3g}"
        )
    return result, {
        "iterations": used,
        "max_relative_residual": relative.max().item(),
        "tolerance": tolerance,
    }


def rank_summary(x):
    """All rows/features, Gram eigenspectrum; tolerance is explicitly reported."""
    return rank_from_gram(x.T @ x, x.shape)


def rank_from_gram(gram, shape):
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0)
    cutoff = eigen.max() * max(shape) * torch.finfo(gram.dtype).eps
    positive = eigen > cutoff
    probability = eigen / eigen.sum().clamp_min(torch.finfo(gram.dtype).tiny)
    entropy = -(probability * probability.clamp_min(torch.finfo(gram.dtype).tiny).log()).sum()
    return {
        "shape": list(shape),
        "gram_numerical_rank": int(positive.sum()),
        "gram_eigenvalue_cutoff": cutoff.item(),
        "spectral_entropy_rank": entropy.exp().item() if bool(eigen.sum() > 0) else 0.0,
    }


@torch.no_grad()
def reconstruction_probe(
    x,
    incidence,
    weight,
    *,
    noise_relative=0.0,
    seed=0,
    topology=None,
    edge_chunk_size=None,
    tolerance=1e-7,
    iterations=2000,
):
    """Recover component means from B[x,x²] using its lifted-manifold constraint.

    Bx and phi(Bx) share an unobservable component mean. The quadratic pre-lift
    identifies that mean only in component/features having nonzero variation.
    Oracle means are used solely in the separately labelled oracle control.
    """
    x, weight = x.double(), weight.double()
    if (
        weight.shape != (incidence.shape[1],)
        or not bool((weight > 0).all())
        or not bool(torch.isfinite(x).all())
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("probe needs strictly positive scalar edge weights")
    count, labels = components(incidence, x.shape[0]) if topology is None else topology
    if noise_relative < 0:
        raise ValueError("noise magnitude must be nonnegative")
    chunk = edge_chunk_size or max(incidence.shape[1], 1)
    if chunk <= 0:
        raise ValueError("diagnostic edge chunks must be positive")
    squares = x.new_zeros(2)
    width = x.shape[1]
    grams = [
        x.new_zeros(width, width),
        x.new_zeros(2 * width, 2 * width),
        x.new_zeros(2 * width, 2 * width),
    ]
    for start in range(0, incidence.shape[1], chunk):
        tail, head = incidence[:, start : start + chunk]
        dx, q = x[head] - x[tail], x[head].square() - x[tail].square()
        squares += torch.stack((dx.square().sum(), q.square().sum()))
        for gram, values in zip(
            grams, (dx, torch.cat((dx, q), -1), torch.cat((dx, dx.square()), -1)), strict=True
        ):
            gram += values.T @ values
    scales = (squares / max(incidence.shape[1] * x.shape[1], 1)).sqrt() * noise_relative

    def observations(power, channel):
        generator = torch.Generator(device=x.device).manual_seed(seed + channel)
        for start in range(0, incidence.shape[1], chunk):
            tail, head = incidence[:, start : start + chunk]
            value = x[head].pow(power) - x[tail].pow(power)
            if noise_relative:
                value += (
                    torch.randn(value.shape, device=x.device, dtype=x.dtype, generator=generator)
                    * scales[channel]
                )
            yield start, tail, head, value

    rhs = torch.zeros_like(x)
    for start, tail, head, dx in observations(1, 0):
        weighted = weight[start : start + chunk, None] * dx
        rhs.index_add_(0, head, weighted).index_add_(0, tail, -weighted)
    centered, solver = solve_rhs(
        incidence, weight, rhs, labels, count, tolerance=tolerance, iterations=iterations
    )
    denominator = x.new_zeros((count, x.shape[1]))
    numerator = torch.zeros_like(denominator)
    for start, tail, head, q in observations(2, 1):
        dy = centered[head] - centered[tail]
        weighted = weight[start : start + chunk, None]
        edge_labels = labels[tail]
        denominator.index_add_(0, edge_labels, weighted * dy.square())
        numerator.index_add_(
            0,
            edge_labels,
            weighted * dy * (q - (centered[head].square() - centered[tail].square())),
        )
    threshold = denominator.max(dim=0).values * 1e-12
    identifiable = denominator > threshold
    inferred_mean = numerator / (2 * denominator).clamp_min(torch.finfo(x.dtype).tiny)
    reconstructed = centered + inferred_mean[labels]
    mask = identifiable[labels]
    true_mean = component_mean(x, labels, count)
    scale = x.square().sum()

    def relative(error, target):
        size = target.square().sum()
        return (error.square().sum() / size).sqrt().item() if bool(size > 0) else None

    return {
        "nodes": x.shape[0],
        "edges": incidence.shape[1],
        "features": x.shape[1],
        "components": count,
        "incidence_rank": x.shape[0] - count,
        "laplacian_nullity": count,
        "noise_relative": noise_relative,
        "linear_minimum_norm_relative_l2": relative(centered - x, x),
        "linear_with_oracle_means_relative_l2": relative(centered + true_mean[labels] - x, x),
        "pre_quadratic_identifiable_relative_l2": relative((reconstructed - x)[mask], x[mask])
        if bool(mask.any())
        else None,
        "identifiable_component_features": int(identifiable.sum()),
        "total_component_features": identifiable.numel(),
        "identifiability_relative_energy_threshold": 1e-12,
        "mean_component_energy_fraction": (
            (true_mean[labels].square().sum() / scale).item() if bool(scale > 0) else None
        ),
        "minimum_identifiable_edge_energy": denominator[identifiable].min().item()
        if bool(identifiable.any())
        else None,
        "post_incidence_lift_mean_recoverable": False,
        "solver": solver,
        "original_feature_rank": rank_summary(x),
        "quadratic_lift_feature_rank": rank_summary(torch.cat((x, x.square()), -1)),
        "noiseless_edge_feature_ranks": {
            name: rank_from_gram(gram, (incidence.shape[1], gram.shape[0]))
            for name, gram in zip(("Bx", "B_phi_x", "phi_Bx"), grams, strict=True)
        },
    }


@torch.no_grad()
def cross_hop_gram(history, incidence, weight, edge_chunk_size):
    history = history.double()
    gram = history.new_zeros((len(history), len(history)))
    chunk = max(1, (edge_chunk_size or max(incidence.shape[1], 1)) // len(history))
    for start in range(0, incidence.shape[1], chunk):
        tail, head = incidence[:, start : start + chunk]
        delta = history[:, head] - history[:, tail]
        gram += torch.einsum("kef,e,lef->kl", delta, weight[start : start + chunk].double(), delta)
    denominator = (gram.diag()[:, None] * gram.diag()[None]).clamp_min(0).sqrt()
    cosine = torch.where(
        denominator > 0, gram / denominator.clamp_min(torch.finfo(gram.dtype).tiny), 0
    )
    return {
        "energy_gram": gram.cpu().tolist(),
        "normalized_cross_energy": cosine.cpu().tolist(),
        "nonzero_energy_hops": (gram.diag() > 0).cpu().tolist(),
    }
