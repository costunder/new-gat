"""Exact dense recovery and matrix-free numerical recovery of all q/E/J targets."""
from __future__ import annotations

import torch

from .numerics import conjugate_gradient, deterministic_noise, null_projector, relative
from .targets import evaluate_targets, observation_masks


def operator_scale(action, repetitions):
    """Positive numerical rescaling, preserving the actual observation kernel.

The rescaling is reported, used for every RHS/noise/adjoint consistently, and
does not change the defined model operator or its minimum-norm solution.
"""
    g = action.geometry
    if action.name == "L0":
        degree = g.d0.new_zeros(g.n).index_add(0, g.edges[0], g.d0.new_ones(g.num_edges))
        degree = degree.index_add(0, g.edges[1], g.d0.new_ones(g.num_edges))
        bound = 2*degree.max()
    elif action.name == "Ld":
        bound = 2*g.d0.max()
    elif action.name.startswith("Q_"):
        bound = 3*g.d0.max()
    elif action.name.startswith("copy_"):
        count = action.copy.copy_counts
        bound = (count.max()/count.min()).sqrt()
    else:
        bound = g.d0.new_tensor(1.)
    return bound.clamp_min(1.).pow(repetitions)


def sparse_reconstruct(action, repetitions, source, mask, *, noise_level,
                       tolerance=1e-8, maximum_iterations=100, channel_offset=0, target_offset=0, progress=None):
    """Packed independent observer/channel RHS, with explicit true residuals.

source [R,N,F], mask [B,N]. Citation R=1. A=O F/scale, and the normal
    equation is A^T A x=A^T(A X+noise), solved from x=0. Its RHS is always in
    range(A^T), even for inconsistent noisy observations. Converged solutions
    are numerical minimum-norm least-squares estimates. Stationarity and the
    possibly irreducible observation residual are recorded separately.
"""
    scale = operator_scale(action, repetitions)
    batch, realizations, nodes, channels = mask.shape[0], *source.shape
    flat_mask = mask[:, None, :, None].expand(batch, realizations, nodes, channels).reshape(-1, nodes, channels)
    expanded = source[None].expand(batch, realizations, nodes, channels).reshape(-1, nodes, channels)
    def forward(value):
        shaped = value.reshape(batch, realizations, nodes, channels)
        return (action(shaped, repetitions)/scale).reshape(-1, nodes, channels)*flat_mask
    def adjoint(value):
        shaped = (value*flat_mask).reshape(batch, realizations, nodes, channels)
        return (action.transpose(shaped, repetitions)/scale).reshape(-1, nodes, channels)
    clean = forward(expanded)
    probe = deterministic_noise(clean.shape, clean, target_offset=target_offset*realizations,
                                channel_offset=channel_offset)*flat_mask
    probe_norm = probe.square().sum(-2).sqrt()
    signal_norm = clean.square().sum(-2).sqrt()
    ratio, nonzero = relative(signal_norm, probe_norm)
    noise = noise_level*probe*ratio[..., None, :]
    rhs = clean+noise
    normal_rhs = adjoint(rhs)
    recovered, status = conjugate_gradient(lambda x: adjoint(forward(x)), normal_rhs,
                                               tolerance=tolerance, maximum_iterations=maximum_iterations,
                                               check_every=1 if maximum_iterations <= 16 else 16, progress=progress)
    actual = forward(recovered)-rhs
    status.update(_least_squares_residuals(adjoint(actual), normal_rhs, actual, rhs, tolerance))
    status["converged"] = status["stationarity_converged"] & ~status["breakdown"]
    status.update(
                  noise_norm=noise.square().sum(-2).sqrt(), clean_rhs_norm=signal_norm,
                  numerical_operator_scale=scale)
    return recovered.reshape(batch, realizations, nodes, channels), status


def _least_squares_residuals(normal_residual, normal_rhs, observation_residual, observation_rhs, tolerance):
    normal_error = normal_residual.square().sum(-2).sqrt()
    normal_scale = normal_rhs.square().sum(-2).sqrt()
    normal_ratio, normal_available = relative(normal_error, normal_scale)
    observation_error = observation_residual.square().sum(-2).sqrt()
    observation_scale = observation_rhs.square().sum(-2).sqrt()
    observation_ratio, observation_available = relative(observation_error, observation_scale)
    normal_ok = torch.where(normal_available, normal_ratio <= tolerance, normal_error == 0)
    observation_ok = torch.where(observation_available, observation_ratio <= tolerance, observation_error == 0)
    return {"relative_residual": normal_ratio, "normal_relative_residual": normal_ratio,
            "rhs_nonzero": normal_available, "normal_rhs_nonzero": normal_available,
            "normal_rhs_norm": normal_scale, "normal_residual_norm": normal_error,
            "stationarity_converged": normal_ok,
            "observation_relative_residual": observation_ratio, "observation_rhs_nonzero": observation_available,
            "observation_rhs_norm": observation_scale, "observation_residual_norm": observation_error,
            "observation_within_tolerance": observation_ok}


def dense_reconstruct(matrix, source, mask, noise_level, *, tolerance=1e-10, solver_tolerance=None,
                      channel_offset=0, target_offset=0):
    batch, references, nodes, _ = mask.shape[0], *matrix.shape
    observed = mask[:, None, :, None]*matrix[None]
    information = null_projector(observed, tolerance)
    clean = observed@source[None]
    probe = deterministic_noise((batch*source.shape[0], nodes, source.shape[-1]), source,
                                target_offset=target_offset*source.shape[0], channel_offset=channel_offset)
    probe = probe.reshape(batch, source.shape[0], nodes, source.shape[-1])*mask[:, None, :, None]
    norm = clean.square().sum(-2).sqrt()
    ratio, _ = relative(norm, probe.square().sum(-2).sqrt())
    noise = noise_level*probe*ratio[..., None, :]
    rhs = clean+noise
    recovered = information["pseudoinverse"]@rhs
    residual = observed@recovered-rhs
    adjoint = observed.transpose(-2, -1)
    measured = _least_squares_residuals(adjoint@residual, adjoint@rhs, residual, rhs,
                                       tolerance if solver_tolerance is None else solver_tolerance)
    status = {key: value.reshape(-1, value.shape[-1]) for key, value in measured.items()}
    status.update(iterations=torch.zeros_like(status["relative_residual"], dtype=torch.long),
                  converged=status["stationarity_converged"], breakdown=torch.zeros_like(status["stationarity_converged"]),
                  noise_norm=noise.square().sum(-2).sqrt().reshape(-1, noise.shape[-1]),
                  clean_rhs_norm=norm.reshape(-1, norm.shape[-1]), numerical_operator_scale=source.new_tensor(1.))
    return recovered, status


class TargetAccumulator:
    """Sum trace terms across every original channel before comparing energies.

q norms accumulate squared vector errors. E/J accumulate signed per-channel
terms before taking an absolute error; this prevents chunking from changing
the vector-trace scientific target.
"""
    def __init__(self, topology, recipe, kind, start, stop, realizations, reference):
        self.topology, self.recipe, self.kind = topology, recipe, kind
        self.start, self.stop, self.realizations = start, stop, realizations
        self.ids = {name: torch.arange(start, stop, device=reference.device) if name in ("q", "E") else
                    torch.nonzero((topology.pair_centers[0] >= start) & (topology.pair_centers[0] < stop), as_tuple=False).flatten()
                    for name in ("q", "E", "J_shared", "J_node", "J_distinct")}
        self.observers = {name: torch.zeros_like(ids) if kind == "full" else
                          (ids-start if name in ("q", "E") else topology.pair_centers[0, ids]-start)
                          for name, ids in self.ids.items()}
        self.expected = {name: reference.new_zeros((ids.numel(), realizations)) for name, ids in self.ids.items()}
        self.recovered = {name: torch.zeros_like(value) for name, value in self.expected.items()}
        self.qerror = torch.zeros_like(self.expected["q"])

    def add(self, original, recovered):
        top = self.topology
        actual, candidate = evaluate_targets(top, original, self.recipe), evaluate_targets(top, recovered, self.recipe)
        selected = (top.local_edge_center >= self.start) & (top.local_edge_center < self.stop)
        occurrences = torch.nonzero(selected, as_tuple=False).flatten()
        centers = top.local_edge_center[occurrences]-self.start
        owner = torch.zeros_like(centers) if self.kind == "full" else centers
        q0 = actual["q"][:, occurrences].transpose(0, 1)
        qhat = candidate["q"][owner, :, occurrences]
        self.qerror.index_add_(0, centers, (qhat-q0).square().sum(-1))
        self.expected["q"].index_add_(0, centers, q0.square().sum(-1))
        for name in ("E", "J_shared", "J_node", "J_distinct"):
            ids, owner = self.ids[name], self.observers[name]
            self.expected[name] += actual[name][:, ids].transpose(0, 1)
            self.recovered[name] += candidate[name][owner, :, ids]

    def rows(self, solver_status=None):
        status = None if solver_status is None else {key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                                                    for key, value in solver_status.items()}
        for name, ids in self.ids.items():
            error = self.qerror.sqrt() if name == "q" else (self.recovered[name]-self.expected[name]).abs()
            norm = self.expected[name].sqrt() if name == "q" else self.expected[name].abs()
            ratio, available = relative(error, norm)
            host = torch.stack((error, norm, ratio, available.to(error.dtype)), -1).cpu().numpy()
            receivers = self.topology.pair_centers[1, ids].cpu().tolist() if name.startswith("J_") else [None]*ids.numel()
            observers = self.observers[name].cpu().tolist()
            for row, target in enumerate(ids.cpu().tolist()):
                for realization in range(self.realizations):
                    values = host[row, realization]
                    yield {"target_kind": name, "target": target, "receiver": receivers[row],
                           "reference_realization": realization, "absolute_error": float(values[0]),
                           "target_norm": float(values[1]), "relative_error": float(values[2]) if values[3] else None,
                           "solver_converged": None if status is None else bool(status["converged"][observers[row], realization] == status["channels"]),
                           "normal_relative_residual_max": None if status is None else float(status["maximum_relative_residual"][observers[row], realization]),
                           "observation_relative_residual_max": None if status is None else float(status["maximum_observation_relative_residual"][observers[row], realization])}
