"""Raw Q_theta(X) X targets, equal-graph losses and train-only linear oracles.

No classification normalizer, projection, diffusion step or hidden node state
is used here. The leading S and R axes are independent models and scalar
realizations, respectively.
"""

from __future__ import annotations

import torch
from torch import nn

TARGETS = ("diagonal", "diagonal_squared", "analytic_pair")
CONDITIONS = ("D0", "D1", "F0", "F1", "F2", "DA")
TRAINABLE = ("D1", "F1", "F2", "DA")
BASELINES = ("same_teacher_linear", "same_teacher_squared", "same_teacher_polynomial", "old_fixed_wedge")


class RawStudent(nn.Module):
    def __init__(self, condition, seeds, hidden=64, epsilon=1e-4):
        super().__init__()
        from ..gates import EdgeMetricGate

        if condition not in CONDITIONS:
            raise ValueError("unknown raw-message condition")
        self.condition = condition
        self.seeds = tuple(seeds)
        # -1 is the output-table marker for a deterministic, parameter-free
        # reference. The shared gate's random initialization API uses >=0.
        gate_seeds = self.seeds if condition in TRAINABLE else (0,)
        if condition not in TRAINABLE and self.seeds != (-1,):
            raise ValueError("fixed students use only the deterministic seed=-1 marker")
        self.gate = EdgeMetricGate(condition, gate_seeds, hidden=hidden, epsilon=epsilon,
                                   namespace="edge-metric-synthetic-raw-student-v1")
        count = sum(parameter.numel() for parameter in self.parameters())
        if (condition in TRAINABLE) != (count > 0):
            raise AssertionError("fixed/trainable condition parameter contract violated")

    def forward(self, batch, pair_chunk=None, diagnostics=False, intervention=None):
        x = batch.x.expand(len(self.seeds), -1, -1, -1)
        return self.gate.apply(batch.geometry, x, x, pair_chunk=pair_chunk,
                               intervention=intervention, diagnostics=diagnostics)


@torch.no_grad()
def target_message(geometry, x, target, pair_chunk=None):
    from ..gates import analytic_teacher
    from ..operators import diagonal_action

    if target == "diagonal":
        return diagonal_action(geometry, x)
    if target == "diagonal_squared":
        return diagonal_action(geometry, diagonal_action(geometry, x))
    if target == "analytic_pair":
        value, _ = analytic_teacher(geometry, x, pair_chunk=pair_chunk, diagnostics=False)
        return value
    raise ValueError("unknown teacher target")


def graph_draw_errors(prediction, target, geometry, epsilon=1e-8):
    if prediction.ndim != 4 or target.ndim != 4 or prediction.shape[1:] != target.shape[1:]:
        raise ValueError("predictions/targets must be [S,R,N,1] with matching scalar fields")
    if prediction.shape[-1] != 1 or target.shape[0] not in (1, prediction.shape[0]):
        raise ValueError("independent scalar input or seed broadcast contract violated")
    index = geometry.node_graph
    count = geometry.num_graphs
    error_sq = prediction.new_zeros((*prediction.shape[:2], count)).index_add(
        -1, index, (prediction - target).square().sum(-1))
    target_sq = target.new_zeros((*target.shape[:2], count)).index_add(
        -1, index, target.square().sum(-1))
    nmse = error_sq / target_sq.clamp_min(epsilon)
    return {"nmse": nmse, "relative_l2_epsilon": nmse.sqrt(),
            "absolute_l2": error_sq.sqrt(), "target_l2": target_sq.sqrt(),
            "zero_target": target_sq == 0}


def normalized_mse(prediction, target, geometry, epsilon=1e-8):
    """Return one mean over all graphs and independent fields for each seed."""
    return graph_draw_errors(prediction, target, geometry, epsilon)["nmse"].mean((1, 2))


def old_wedge_message(geometry, x):
    from ...wedge_propagation.operators import build_wedges, fixed_wedge_apply

    # Static wedge construction is cached on geometry, never repeated per epoch.
    if id(geometry) not in _WEDGE_CACHE:
        wedges = build_wedges(geometry.edges.detach().cpu(), geometry.n).to(x.device)
        # EdgeGeometry can be frozen; a namespace cache avoids modifying it.
        _WEDGE_CACHE[id(geometry)] = (geometry, wedges)
    wedges = _WEDGE_CACHE[id(geometry)][1]
    fields = x[..., 0].permute(2, 0, 1).reshape(geometry.n, -1)
    out = fixed_wedge_apply(wedges, fields)
    return out.reshape(geometry.n, *x.shape[:2]).permute(1, 2, 0)[..., None]


_WEDGE_CACHE = {}


def baseline_basis(batch, condition):
    from ..operators import diagonal_action

    lx = diagonal_action(batch.geometry, batch.x)
    if condition == "same_teacher_linear":
        return lx[..., None]
    l2x = diagonal_action(batch.geometry, lx)
    if condition == "same_teacher_squared":
        return l2x[..., None]
    if condition == "same_teacher_polynomial":
        return torch.stack((batch.x, lx, l2x), -1)
    if condition == "old_fixed_wedge":
        return old_wedge_message(batch.geometry, batch.x)[..., None]
    raise ValueError("unknown train-only least-squares baseline")


@torch.no_grad()
def fit_baseline(batches, targets, condition, epsilon=1e-8):
    """Float64 weighted least squares, using the train messages only.

    Each scalar field receives weight 1/max(||target_g,r||²,epsilon).
    Thus the analytic objective equals the gate's equal-graph/draw NMSE.
    Rank-deficient solutions use the minimum-norm Moore-Penrose solution.
    """
    gram, rhs = None, None
    for batch, target in zip(batches, targets, strict=True):
        if batch.x.dtype != torch.float64 or target.dtype != torch.float64:
            raise ValueError("analytic fits require float64 reference inputs and messages")
        basis = baseline_basis(batch, condition)
        norm_sq = target.new_zeros((*target.shape[:2], batch.num_graphs)).index_add(
            -1, batch.geometry.node_graph, target.square().sum(-1))
        weight = norm_sq.clamp_min(epsilon).reciprocal().index_select(-1, batch.geometry.node_graph)[..., None]
        matrix = basis[..., 0, :].reshape(-1, basis.shape[-1])
        vector = target[..., 0].reshape(-1)
        row_weight = weight[..., 0].reshape(-1)
        part_gram = matrix.T @ (matrix * row_weight[:, None])
        part_rhs = matrix.T @ (vector * row_weight)
        gram = part_gram if gram is None else gram + part_gram
        rhs = part_rhs if rhs is None else rhs + part_rhs
    if gram is None:
        raise ValueError("no train graphs for analytic fit")
    coefficients = torch.linalg.pinv(gram, hermitian=True) @ rhs
    return coefficients, {"condition": condition, "coefficients": coefficients.tolist(),
                          "fit_split": "train", "optimizer_updates": 0,
                          "basis_rank": int(torch.linalg.matrix_rank(gram)),
                          "objective": "equal-graph independent-scalar NMSE",
                          "reference_dtype": "float64", "normalization": "raw SAME-teacher-Ld"}


def baseline_predict(batch, condition, coefficients):
    basis = baseline_basis(batch, condition)
    return (basis * coefficients.to(batch.x)[None, None, None, None, :]).sum(-1)


def parameter_vector_stats(model):
    gradients, parameters = [], []
    seeds = len(model.seeds)
    for name, parameter in model.named_parameters():
        if parameter.shape[0] != seeds:
            raise AssertionError(f"parameter {name} does not preserve seed independence")
        if parameter.grad is None:
            raise RuntimeError(f"active gate parameter {name} is disconnected from raw message loss")
        gradients.append(parameter.grad.detach().reshape(seeds, -1))
        parameters.append(parameter.detach().reshape(seeds, -1))
    if not gradients:
        raise ValueError("fixed models do not have optimizer telemetry")
    return torch.cat(gradients, 1).norm(dim=1), torch.cat(parameters, 1).norm(dim=1)
