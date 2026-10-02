"""Seed-batched scalar operator recovery, independent of the data classes.

``x[N, R]`` contains R independent scalar feature realizations. Parameters
have an independent leading seed axis. There is no per-seed or per-feature
Python loop in a forward, loss, or operator application. Initialization loops
once over local CPU generators so a seed is independent of seed list ordering.

The pair convention is traversal coefficients: ``g1=s1*B[e]x`` and
``g2=s2*B[f]x``. Thus the pair row has B-row coefficients ``(-s1,+s2)``.
Coefficients may include a fixed row-normalization scale.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn

from ..operators import incidence_apply, incidence_transpose, wedge_apply, wedge_transpose

__all__ = [
    "SeedBatchedGate",
    "SeedBatchedModel",
    "apply_weighted_pair",
    "apply_weighted_wedge",
    "fit_baseline",
    "make_model",
    "normalize_path_mean",
    "normalized_mse",
    "pair_gradients",
    "path_features",
    "teacher_weights",
]


@dataclass
class _IndexInfo:
    reference: weakref.ReferenceType[Tensor]
    version: int | None
    bound: int
    counts: Tensor | None
    all_groups_present: bool = True


_INDEX_CACHE: dict[tuple[str, int], _IndexInfo] = {}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_number(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _real_matrix(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] < 1:
        raise ValueError(f"{name} must have shape [count, R] with R positive")
    if not value.is_floating_point() or value.layout != torch.strided:
        raise TypeError(f"{name} must be a real strided floating-point tensor")


def _same_values(first: Tensor, second: Tensor, name: str) -> None:
    if first.shape != second.shape or first.dtype != second.dtype or first.device != second.device:
        raise ValueError(f"{name} must have the same shape, dtype, and device")


def _index_info(index: Tensor, bound: int, count: int, name: str, *, groups: bool) -> _IndexInfo:
    if not isinstance(index, Tensor) or index.dtype != torch.long:
        raise TypeError(f"{name} must be a torch.long tensor")
    expected = (2, count) if name == "pair_edges" else (count,)
    if tuple(index.shape) != expected:
        raise ValueError(f"{name} must have shape {list(expected)}")
    version = None if torch.is_inference(index) else index._version
    key = (name, id(index))
    info = _INDEX_CACHE.get(key)
    if (version is not None and info is not None and info.reference() is index
            and info.version == version and info.bound == bound):
        return info
    cpu = index.detach().to("cpu")
    if bool(((cpu < 0) | (cpu >= bound)).any()):
        raise ValueError(f"{name} contains an index outside [0, {bound})")
    if name == "pair_edges" and bool((cpu[0] == cpu[1]).any()):
        raise ValueError("pair_edges must select two distinct edges in each pair")
    cpu_counts = torch.bincount(cpu, minlength=bound) if groups else None
    counts = cpu_counts.to(index.device) if cpu_counts is not None else None
    present = cpu_counts is None or bool((cpu_counts > 0).all())

    def discard(reference: weakref.ReferenceType[Tensor]) -> None:
        current = _INDEX_CACHE.get(key)
        if current is not None and current.reference is reference:
            del _INDEX_CACHE[key]

    info = _IndexInfo(weakref.ref(index, discard), version, bound, counts, present)
    _INDEX_CACHE[key] = info
    return info


def _group_counts(index: Tensor, count: int, num_graphs: int, values: Tensor, name: str) -> Tensor:
    num_graphs = _positive_integer(num_graphs, "num_graphs")
    if index.device != values.device:
        raise ValueError(f"{name} and values must be on the same device")
    info = _index_info(index, num_graphs, count, name, groups=True)
    assert info.counts is not None
    return info.counts.to(dtype=values.dtype)


def path_features(g1: Tensor, g2: Tensor) -> Tensor:
    """Return [P,R,4] features invariant under (g1,g2)->(-g2,-g1)."""
    _real_matrix(g1, "g1")
    _real_matrix(g2, "g2")
    _same_values(g1, g2, "g1 and g2")
    return torch.stack((g1.abs() + g2.abs(), g1 * g2, (g2 - g1).abs(),
                        (g1.abs() - g2.abs()).square()), dim=-1)


def normalize_path_mean(raw: Tensor, path_graph: Tensor, num_graphs: int) -> Tensor:
    """Normalize each graph/realization independently; P=0 remains empty.

    Accept [P,R] or [S,P,R], preserving the input rank. Graphs with no paths
    have no weights to normalize; their zero operator is not a fallback.
    """
    if raw.ndim not in (2, 3) or not raw.is_floating_point() or raw.shape[-1] < 1:
        raise ValueError("raw must be a floating tensor with shape [P,R] or [S,P,R]")
    seeded = raw.unsqueeze(0) if raw.ndim == 2 else raw
    counts = _group_counts(path_graph, seeded.shape[1], num_graphs, raw, "path_graph")
    sums = raw.new_zeros((seeded.shape[0], num_graphs, raw.shape[-1]))
    sums = sums.index_add(1, path_graph, seeded)
    means = sums / counts.clamp_min(1)[None, :, None]
    normalized = seeded / means.index_select(1, path_graph)
    return normalized.squeeze(0) if raw.ndim == 2 else normalized


def teacher_weights(
    g1: Tensor,
    g2: Tensor,
    path_graph: Tensor,
    num_graphs: int,
    theta1: float = 1.0,
    theta2: float = 1.0,
    tau: float = 1.0,
    epsilon: float = 1e-8,
) -> Tensor:
    """Exact symmetric scalar teacher, with graph/realization path mean one."""
    _real_matrix(g1, "g1")
    _real_matrix(g2, "g2")
    _same_values(g1, g2, "g1 and g2")
    tau = _positive_number(tau, "tau")
    epsilon = _positive_number(epsilon, "epsilon")
    if not math.isfinite(float(theta1)) or not math.isfinite(float(theta2)):
        raise ValueError("theta1 and theta2 must be finite")
    cosine = g1 * g2 / (g1.abs() * g2.abs() + epsilon)
    difference = (g2 - g1).abs() / (g1.abs() + g2.abs() + epsilon)
    score = float(theta1) * cosine + float(theta2) * difference
    return normalize_path_mean(torch.exp(tau * score.tanh()), path_graph, num_graphs)


def _weight_shape(c: Tensor, x: Tensor, paths: int) -> None:
    if c.ndim not in (2, 3) or c.shape[-2:] != (paths, x.shape[1]):
        raise ValueError("c must have shape [P,R] or [S,P,R] matching the operator")
    if c.dtype != x.dtype or c.device != x.device:
        raise ValueError("c and x must have the same dtype and device")


def apply_weighted_wedge(wedges: Tensor, x: Tensor, c: Tensor) -> Tensor:
    """Return A.T diag(c) Ax as [N,R] or [S,N,R]."""
    _real_matrix(x, "x")
    q = wedge_apply(wedges, x)
    _weight_shape(c, x, q.shape[0])
    if c.ndim == 2:
        return wedge_transpose(wedges, c * q, x.shape[0])
    weighted = (c * q).movedim(1, 0)
    return wedge_transpose(wedges, weighted, x.shape[0]).movedim(1, 0)


def pair_gradients(
    edges: Tensor, pair_edges: Tensor, pair_coefficients: Tensor, x: Tensor
) -> tuple[Tensor, Tensor]:
    """Compute row-scaled traversal g1/g2 with one batched incidence apply."""
    _real_matrix(x, "x")
    differences = incidence_apply(edges, x)
    if not isinstance(pair_edges, Tensor) or pair_edges.ndim != 2 or pair_edges.shape[0] != 2:
        raise ValueError("pair_edges must have shape [2,P]")
    if pair_edges.device != x.device:
        raise ValueError("pair_edges and x must be on the same device")
    _index_info(pair_edges, edges.shape[1], pair_edges.shape[1], "pair_edges", groups=False)
    if pair_coefficients.shape != pair_edges.shape:
        raise ValueError("pair_coefficients must have shape [2,P]")
    if pair_coefficients.dtype != x.dtype or pair_coefficients.device != x.device:
        raise ValueError("pair_coefficients and x must have the same dtype and device")
    g1 = pair_coefficients[0, :, None] * differences.index_select(0, pair_edges[0])
    g2 = pair_coefficients[1, :, None] * differences.index_select(0, pair_edges[1])
    return g1, g2


def apply_weighted_pair(
    edges: Tensor, pair_edges: Tensor, pair_coefficients: Tensor, x: Tensor, c: Tensor
) -> Tensor:
    """Return (TB).T diag(c) (TB)x using traversal coefficients (-s1,+s2)."""
    g1, g2 = pair_gradients(edges, pair_edges, pair_coefficients, x)
    _weight_shape(c, x, pair_edges.shape[1])
    seeded = c.unsqueeze(0) if c.ndim == 2 else c
    weighted = (seeded * (g2 - g1)).movedim(1, 0)
    flows = x.new_zeros((edges.shape[1], seeded.shape[0], x.shape[1]))
    flows = flows.index_add(0, pair_edges[0], -pair_coefficients[0, :, None, None] * weighted)
    flows = flows.index_add(0, pair_edges[1], pair_coefficients[1, :, None, None] * weighted)
    result = incidence_transpose(edges, flows, x.shape[0]).movedim(1, 0)
    return result.squeeze(0) if c.ndim == 2 else result


def _seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
        raise ValueError("seeds must be a nonempty sequence of integers")
    if any(seed < -(2**63) or seed >= 2**64 for seed in seeds):
        raise ValueError("seed is outside torch.Generator.manual_seed's supported range")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct")
    return tuple(int(seed) for seed in seeds)


class SeedBatchedGate(nn.Module):
    """Independent 4->hidden->1 ReLU gates, with both biases trainable."""

    def __init__(self, seeds: Sequence[int], hidden: int = 64, tau: float = 1.0) -> None:
        super().__init__()
        self.seeds = _seeds(seeds)
        self.hidden = _positive_integer(hidden, "hidden")
        self.tau = _positive_number(tau, "tau")
        count = len(self.seeds)
        self.w1 = nn.Parameter(torch.empty(count, 4, self.hidden, dtype=torch.float64))
        self.b1 = nn.Parameter(torch.zeros(count, self.hidden, dtype=torch.float64))
        self.w2 = nn.Parameter(torch.empty(count, self.hidden, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.zeros(count, dtype=torch.float64))
        with torch.no_grad():
            for index, seed in enumerate(self.seeds):
                rng = torch.Generator(device="cpu").manual_seed(seed)
                bound1 = math.sqrt(6 / (4 + self.hidden))
                bound2 = math.sqrt(6 / (self.hidden + 1))
                self.w1[index].uniform_(-bound1, bound1, generator=rng)
                self.w2[index].uniform_(-bound2, bound2, generator=rng)

    def forward(self, g1: Tensor, g2: Tensor, path_graph: Tensor, num_graphs: int) -> Tensor:
        features = path_features(g1, g2)
        if features.dtype != self.w1.dtype or features.device != self.w1.device:
            raise ValueError("gate parameters and inputs must have the same dtype and device")
        hidden = torch.einsum("prf,sfh->sprh", features, self.w1) + self.b1[:, None, None, :]
        score = torch.einsum("sprh,sh->spr", hidden.relu(), self.w2) + self.b2[:, None, None]
        return normalize_path_mean(torch.exp(self.tau * score.tanh()), path_graph, num_graphs)


class SeedBatchedModel(nn.Module):
    """The five prescribed scalar forms for pure-operator teacher recovery.

    first: u*LX; polynomial: u*LX+v*L²X; fixed: beta*QX;
    learned/random_pair: beta*weighted_message. No unused scalar or gate
    parameters are created. This is operator recovery, not a diffusion step.
    """

    def __init__(self, kind: str, seeds: Sequence[int], hidden: int = 64, tau: float = 1.0):
        super().__init__()
        if kind not in ("first", "polynomial", "fixed", "learned", "random_pair"):
            raise ValueError(f"unknown model kind {kind!r}")
        self.kind = kind
        self.seeds = _seeds(seeds)
        count = len(self.seeds)
        if kind in ("first", "polynomial"):
            self.u = nn.Parameter(torch.zeros(count, dtype=torch.float64))
        if kind == "polynomial":
            self.v = nn.Parameter(torch.zeros(count, dtype=torch.float64))
        elif kind in ("fixed", "learned", "random_pair"):
            self.beta = nn.Parameter(torch.ones(count, dtype=torch.float64))
        if kind in ("learned", "random_pair"):
            self.gate = SeedBatchedGate(self.seeds, hidden, tau)

    def forward(self, batch: Any) -> tuple[Tensor, Tensor | None]:
        _real_matrix(batch.x, "batch.x")
        parameter = next(self.parameters())
        if parameter.dtype != batch.x.dtype or parameter.device != batch.x.device:
            raise ValueError("model parameters and batch.x must have the same dtype and device")
        if self.kind == "first":
            _same_values(batch.x, batch.lx, "batch.x and batch.lx")
            return self.u[:, None, None] * batch.lx, None
        if self.kind == "polynomial":
            _same_values(batch.x, batch.lx, "batch.x and batch.lx")
            _same_values(batch.x, batch.l2x, "batch.x and batch.l2x")
            return self.u[:, None, None] * batch.lx + self.v[:, None, None] * batch.l2x, None
        if self.kind == "fixed":
            _same_values(batch.x, batch.qx, "batch.x and batch.qx")
            return self.beta[:, None, None] * batch.qx, None
        if self.kind == "learned":
            # Canonical endpoint order has no effect because the gate is symmetric.
            g1 = batch.x.index_select(0, batch.wedges[1]) - batch.x.index_select(0, batch.wedges[0])
            g2 = batch.x.index_select(0, batch.wedges[2]) - batch.x.index_select(0, batch.wedges[1])
            c = self.gate(g1, g2, batch.path_graph, batch.num_graphs)
            message = apply_weighted_wedge(batch.wedges, batch.x, c)
        else:
            g1, g2 = pair_gradients(batch.edges, batch.pair_edges, batch.pair_coefficients, batch.x)
            c = self.gate(g1, g2, batch.path_graph, batch.num_graphs)
            message = apply_weighted_pair(
                batch.edges, batch.pair_edges, batch.pair_coefficients, batch.x, c
            )
        return self.beta[:, None, None] * message, c


def make_model(
    kind: str, seeds: Sequence[int], hidden: int = 64, tau: float = 1.0
) -> SeedBatchedModel:
    """Construct on CPU in float64; explicitly .to the batch device/dtype."""
    return SeedBatchedModel(kind, seeds, hidden, tau)


def _node_terms(target: Tensor, node_graph: Tensor, num_graphs: int, epsilon: float):
    _real_matrix(target, "target")
    counts = _group_counts(node_graph, target.shape[0], num_graphs, target, "node_graph")
    # Every graph in a node batch must be present; empty-path graphs are allowed.
    cached = _index_info(node_graph, num_graphs, target.shape[0], "node_graph", groups=True)
    assert cached.counts is not None
    if not cached.all_groups_present:
        raise ValueError("node_graph must contain nodes for every graph")
    energy = target.new_zeros((num_graphs, target.shape[1])).index_add(
        0, node_graph, target.square()
    ) / counts[:, None]
    denominator = energy + epsilon
    return counts, denominator


def normalized_mse(
    pred: Tensor, target: Tensor, node_graph: Tensor, num_graphs: int, eps: float = 1e-8
) -> Tensor:
    """Return per-seed mean_graph mean_realization relative node-MSE [S].

    Each denominator is mean_node(target²)+eps, including zero targets. The
    objective weights every graph and feature realization equally rather than
    treating all nodes as identically weighted examples.
    """
    eps = _positive_number(eps, "eps")
    _real_matrix(target, "target")
    if pred.ndim != 3 or pred.shape[1:] != target.shape:
        raise ValueError("pred must have shape [S,N,R] matching target[N,R]")
    if pred.dtype != target.dtype or pred.device != target.device:
        raise ValueError("pred and target must have the same dtype and device")
    counts, denominator = _node_terms(target, node_graph, num_graphs, eps)
    errors = pred.new_zeros((pred.shape[0], num_graphs, target.shape[1]))
    errors = errors.index_add(1, node_graph, (pred - target).square()) / counts[None, :, None]
    return (errors / denominator[None]).mean(dim=(1, 2))


def fit_baseline(
    model: SeedBatchedModel, batch: Any, target: Tensor, epsilon: float = 1e-8
) -> dict[str, float]:
    """Fit only train data by exact weighted least squares for the same loss.

    The single optimum is copied to each baseline seed. A column-scaled direct
    SVD pseudoinverse handles dependent or zero bases without squaring the
    condition number in a normal-equation solve or adding artificial data.
    Coefficients in a dependent basis need not be uniquely identifiable.
    """
    if model.kind not in ("first", "polynomial", "fixed"):
        raise ValueError("closed-form scalar fitting is only defined for fixed baselines")
    epsilon = _positive_number(epsilon, "epsilon")
    _same_values(batch.x, target, "batch.x and target")
    parameter = next(model.parameters())
    if parameter.dtype != target.dtype or parameter.device != target.device:
        raise ValueError("model and train target must have the same dtype and device")
    with torch.no_grad():
        counts, denominator = _node_terms(target, batch.node_graph, batch.num_graphs, epsilon)
        weights = 1 / (counts.index_select(0, batch.node_graph)[:, None]
                       * denominator.index_select(0, batch.node_graph)
                       * batch.num_graphs * target.shape[1])
        if model.kind == "first":
            names, actions = ["u"], [batch.lx]
        elif model.kind == "polynomial":
            names, actions = ["u", "v"], [batch.lx, batch.l2x]
        else:
            names, actions = ["beta"], [batch.qx]
        for action in actions:
            _same_values(target, action, "target and basis action")
        basis = torch.stack(actions, dim=-1).reshape(-1, len(actions))
        weights = weights.reshape(-1)
        norms = (weights[:, None] * basis.square()).sum(0).sqrt()
        scale = torch.where(norms > 0, norms, torch.ones_like(norms))
        normalized = basis / scale
        weighted_basis = weights.sqrt()[:, None] * normalized
        weighted_target = weights.sqrt() * target.reshape(-1)
        tolerance = 1e-12 if target.dtype == torch.float64 else 1e-6
        coefficients = torch.linalg.pinv(weighted_basis, rtol=tolerance) @ weighted_target / scale
        for index, name in enumerate(names):
            scalar = getattr(model, name)
            scalar.copy_(coefficients[index].expand_as(scalar))
        return {name: float(coefficients[index].cpu()) for index, name in enumerate(names)}
