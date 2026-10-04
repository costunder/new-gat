"""Sparse local-copy Laplacians and a matched intra/cross/intra update.

Every physical node is copied into all of its induced one-hop local graphs.
A acts inside those graphs; K couples copies of the same physical node in
adjacent local graphs. No local, node, channel, or packed-seed forward loop
is used. Preparing the static topology is a CPU operation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor

from ..local_energy_relations.operators import local_weight
from ..local_energy_relations.topology import LocalTopology


@dataclass(frozen=True)
class CouplingGeometry:
    topology: LocalTopology
    mode: str
    intra_weights: Tensor
    cross_left: Tensor
    cross_right: Tensor
    cross_graph: Tensor
    copy_graph: Tensor
    physical_graph: Tensor
    copy_counts: Tensor
    intra_degree: Tensor
    cross_degree: Tensor
    eta: Tensor
    gamma: Tensor
    max_intra_degree: Tensor
    max_cross_degree: Tensor
    metadata: dict[str, Any]

    @property
    def n(self) -> int:
        return self.topology.n

    @property
    def num_graphs(self) -> int:
        return self.topology.num_graphs

    @property
    def num_copies(self) -> int:
        return self.topology.num_local_nodes

    @property
    def num_cross_edges(self) -> int:
        return self.cross_left.numel()

    def to(
        self, device: str | torch.device, dtype: torch.dtype = torch.float64
    ) -> CouplingGeometry:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("geometry dtype must be float32 or float64")
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                values[field.name] = value.to(
                    device=device, dtype=dtype if value.is_floating_point() else value.dtype
                )
            elif field.name == "topology":
                values[field.name] = value.to(device)
            else:
                values[field.name] = value
        return CouplingGeometry(**values)


def _max_by_graph(geo: CouplingGeometry, degree: Tensor) -> Tensor:
    return degree.new_zeros(geo.num_graphs).scatter_reduce(
        0, geo.copy_graph, degree, reduce="amax", include_self=True
    )


def _safe_step(maximum: Tensor) -> Tensor:
    # Empty operators have a documented zero step, rather than an infinite step.
    positive = maximum > 0
    denominator = torch.where(positive, maximum, torch.ones_like(maximum))
    return torch.where(positive, 0.5 / denominator, torch.zeros_like(maximum))


def prepare_geometry(top: LocalTopology, mode: str = "unit") -> CouplingGeometry:
    """Prepare fixed A, unit-weight K, R/M, and graphwise safe steps on CPU.

    Canonical center endpoints select each undirected local pair once. This
    also works after batching, where each component has its own forward/reverse
    pair columns and a single global ``pair < total_edges`` mask would be wrong.
    """
    for field in fields(top):
        value = getattr(top, field.name)
        if isinstance(value, Tensor) and value.device.type != "cpu":
            raise ValueError("prepare_geometry requires CPU topology; prepare before device transfer")
    top = top.to("cpu")
    if mode not in ("unit", "local_degree"):
        raise ValueError("mode must be unit or local_degree")
    if top.n < 1 or top.num_graphs < 1:
        raise ValueError("topology must contain at least one graph and one physical node")
    lengths = top.graph_node_offsets.diff()
    if bool((lengths <= 0).any()) or int(lengths.sum()) != top.n:
        raise ValueError("graph_node_offsets must partition all physical nodes into nonempty graphs")
    physical_graph = torch.repeat_interleave(torch.arange(top.num_graphs), lengths)
    copy_graph = physical_graph[top.local_node_global]
    copy_counts = torch.bincount(top.local_node_global, minlength=top.n).to(torch.float64)
    if bool((copy_counts <= 0).any()):
        raise ValueError("every physical node must have at least one local copy")
    intra_weights = local_weight(top, mode, dtype=torch.float64)
    if not bool(torch.isfinite(intra_weights).all()) or bool((intra_weights <= 0).any()):
        raise ValueError("intra weights must be finite and positive on every local edge")
    intra_degree = torch.zeros(top.num_local_nodes, dtype=torch.float64)
    intra_degree.index_add_(0, top.local_edge_nodes[0], intra_weights)
    intra_degree.index_add_(0, top.local_edge_nodes[1], intra_weights)
    pair = top.shared_node_pair
    canonical = top.pair_centers[0, pair] < top.pair_centers[1, pair]
    cross_left = top.shared_node_left[canonical]
    cross_right = top.shared_node_right[canonical]
    if not torch.equal(top.local_node_global[cross_left], top.local_node_global[cross_right]):
        raise ValueError("cross correspondences must align the same physical node")
    cross_graph = copy_graph[cross_left]
    if not torch.equal(cross_graph, copy_graph[cross_right]):
        raise ValueError("cross correspondences cannot connect different physical graphs")
    cross_degree = torch.zeros(top.num_local_nodes, dtype=torch.float64)
    cross_degree.index_add_(0, cross_left, torch.ones_like(cross_left, dtype=torch.float64))
    cross_degree.index_add_(0, cross_right, torch.ones_like(cross_right, dtype=torch.float64))
    max_intra = torch.zeros(top.num_graphs, dtype=torch.float64).scatter_reduce_(
        0, copy_graph, intra_degree, reduce="amax", include_self=True
    )
    max_cross = torch.zeros(top.num_graphs, dtype=torch.float64).scatter_reduce_(
        0, copy_graph, cross_degree, reduce="amax", include_self=True
    )
    return CouplingGeometry(
        topology=top, mode=mode, intra_weights=intra_weights,
        cross_left=cross_left, cross_right=cross_right, cross_graph=cross_graph,
        copy_graph=copy_graph, physical_graph=physical_graph, copy_counts=copy_counts,
        intra_degree=intra_degree, cross_degree=cross_degree,
        eta=_safe_step(max_intra), gamma=_safe_step(max_cross),
        max_intra_degree=max_intra, max_cross_degree=max_cross,
        metadata={
            "weight_mode": mode, "cross_weight": "fixed_unit",
            "cross_pair_selection": "canonical_center_endpoints",
            "num_physical_nodes": top.n, "num_physical_edges": top.num_edges,
            "num_graphs": top.num_graphs, "num_copies": top.num_local_nodes,
            "num_local_edges": top.num_local_edges, "num_cross_edges": cross_left.numel(),
            "merge": "exact_physical_copy_mean", "step_rule": "0.5/max_weighted_copy_degree",
        },
    )


def _check_values(geo: CouplingGeometry, value: Tensor, count: int, name: str) -> None:
    if (
        not isinstance(value, Tensor) or value.ndim not in (2, 3)
        or value.shape[-2] != count or value.shape[-1] < 1
        or (value.ndim == 3 and value.shape[0] < 1)
    ):
        raise ValueError(f"{name} must be [{count},features] or [seeds,{count},features]")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} dtype must be float32 or float64")
    if value.device != geo.intra_weights.device or value.dtype != geo.intra_weights.dtype:
        raise ValueError(f"{name} and geometry must have identical device and dtype")


def _edge_scale(value: Tensor, weights: Tensor) -> Tensor:
    return value * weights.reshape((1,) * (value.ndim - 2) + (-1, 1))


def _copy_scale(geo: CouplingGeometry, value: Tensor, scale: Tensor) -> Tensor:
    return _edge_scale(value, scale.index_select(0, geo.copy_graph))


def _require_tensor(condition: Tensor, message: str) -> None:
    """Validate device values without copying CUDA scalars to the host."""
    if condition.device.type == "cpu":
        if not bool(condition):
            raise ValueError(message)
    else:
        torch._assert_async(condition, message)


def _cross_weights(geo: CouplingGeometry, weights: Tensor | None) -> Tensor | None:
    if weights is None:
        return None
    if (
        not isinstance(weights, Tensor) or weights.shape != (geo.num_cross_edges,)
        or weights.device != geo.intra_weights.device or weights.dtype != geo.intra_weights.dtype
    ):
        raise ValueError("cross weights must have shape [num_cross_edges] on geometry device/dtype")
    _require_tensor(
        (torch.isfinite(weights) & (weights > 0)).all(),
        "cross weights must be finite and strictly positive",
    )
    return weights


def _weighted_cross_degree(geo: CouplingGeometry, weights: Tensor | None) -> Tensor:
    if weights is None:
        return geo.cross_degree
    degree = weights.new_zeros(geo.num_copies)
    return degree.index_add(0, geo.cross_left, weights).index_add(0, geo.cross_right, weights)


def _graph_coefficient(
    geo: CouplingGeometry, value: float | Tensor | None, default: Tensor,
    name: str, *, active: Tensor | None = None, allow_zero: bool = False,
) -> Tensor:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (Real, Tensor)):
        raise ValueError(f"{name} must be a scalar or tensor [num_graphs]")
    if isinstance(value, Real) and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    result = torch.as_tensor(value, device=default.device, dtype=default.dtype)
    if result.ndim == 0:
        result = result.expand(geo.num_graphs)
    if result.shape != (geo.num_graphs,):
        raise ValueError(f"{name} must be finite with scalar or [num_graphs] shape")
    _require_tensor(
        (torch.isfinite(result) & (result >= 0)).all(), f"{name} must be finite and nonnegative",
    )
    if not allow_zero and active is not None:
        _require_tensor(
            ((result > 0) | ~active).all(),
            f"{name} must be positive for every graph with a nonzero operator",
        )
    return result


def _gain(value: float | Tensor, h: Tensor) -> Tensor:
    if isinstance(value, bool) or not isinstance(value, (Real, Tensor)):
        raise ValueError("cross_gain must be a scalar or tensor [seeds]")
    if isinstance(value, Real) and (not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("cross_gain must be finite and between zero and one")
    gain = torch.as_tensor(value, dtype=h.dtype, device=h.device)
    expected = (h.shape[0],) if h.ndim == 3 else ()
    if gain.ndim != 0 and gain.shape != expected:
        raise ValueError("cross_gain must be scalar, or [seeds] for packed inputs")
    if isinstance(value, Tensor):
        _require_tensor(
            (torch.isfinite(gain) & (gain >= 0) & (gain <= 1)).all(),
            "cross_gain must be finite and between zero and one",
        )
    return gain.reshape((-1, 1, 1)) if gain.ndim == 1 else gain


def replicate(geo: CouplingGeometry, h: Tensor) -> Tensor:
    """R h: initialize every local copy using its original physical node ID."""
    _check_values(geo, h, geo.n, "h")
    return h.index_select(-2, geo.topology.local_node_global)


def merge(geo: CouplingGeometry, y: Tensor) -> Tensor:
    """M y: sum every original-ID copy, then divide by its exact copy count."""
    _check_values(geo, y, geo.num_copies, "y")
    shape = y.shape[:-2] + (geo.n, y.shape[-1])
    result = y.new_zeros(shape).index_add(-2, geo.topology.local_node_global, y)
    return _edge_scale(result, geo.copy_counts.reciprocal())


def apply_intra(geo: CouplingGeometry, y: Tensor) -> Tensor:
    """A y = blockdiag(B_v.T C_v B_v) y."""
    _check_values(geo, y, geo.num_copies, "y")
    tail, head = geo.topology.local_edge_nodes
    flow = _edge_scale(y.index_select(-2, head) - y.index_select(-2, tail), geo.intra_weights)
    result = torch.zeros_like(y).index_add(-2, tail, -flow)
    return result.index_add(-2, head, flow)


def _apply_cross(geo: CouplingGeometry, y: Tensor, weights: Tensor | None) -> Tensor:
    difference = y.index_select(-2, geo.cross_right) - y.index_select(-2, geo.cross_left)
    flow = difference if weights is None else _edge_scale(difference, weights)
    result = torch.zeros_like(y).index_add(-2, geo.cross_left, -flow)
    return result.index_add(-2, geo.cross_right, flow)


def apply_cross(geo: CouplingGeometry, y: Tensor, weights: Tensor | None = None) -> Tensor:
    """K y: shared-physical-node copy consensus, counting each center pair once."""
    _check_values(geo, y, geo.num_copies, "y")
    return _apply_cross(geo, y, _cross_weights(geo, weights))


def intra_energy(geo: CouplingGeometry, y: Tensor) -> Tensor:
    """Half internal energy, per physical graph and feature: [G,F] / [S,G,F]."""
    _check_values(geo, y, geo.num_copies, "y")
    tail, head = geo.topology.local_edge_nodes
    values = _edge_scale((y.index_select(-2, head) - y.index_select(-2, tail)).square(), geo.intra_weights)
    shape = y.shape[:-2] + (geo.num_graphs, y.shape[-1])
    return y.new_zeros(shape).index_add(-2, geo.copy_graph[tail], 0.5 * values)


def cross_energy(geo: CouplingGeometry, y: Tensor, weights: Tensor | None = None) -> Tensor:
    """Half copy-disagreement energy, per physical graph and feature channel."""
    _check_values(geo, y, geo.num_copies, "y")
    weights = _cross_weights(geo, weights)
    values = (y.index_select(-2, geo.cross_right) - y.index_select(-2, geo.cross_left)).square()
    if weights is not None:
        values = _edge_scale(values, weights)
    shape = y.shape[:-2] + (geo.num_graphs, y.shape[-1])
    return y.new_zeros(shape).index_add(-2, geo.cross_graph, 0.5 * values)


def mixed_action(geo: CouplingGeometry, h: Tensor, *, cross_weights: Tensor | None = None) -> Tensor:
    """M A K A R h, the exact mixed operator in the sandwich difference."""
    weights = _cross_weights(geo, cross_weights)
    return merge(geo, apply_intra(geo, _apply_cross(geo, apply_intra(geo, replicate(geo, h)), weights)))


def _sandwich_stages(
    geo: CouplingGeometry, h: Tensor, *, cross: bool, cross_gain: float | Tensor,
    eta: float | Tensor | None, gamma: float | Tensor | None,
    cross_weights: Tensor | None, second_intra: bool, diagnostics: bool,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    _check_values(geo, h, geo.n, "h")
    if not isinstance(cross, bool) or not isinstance(diagnostics, bool):
        raise ValueError("cross and diagnostics must be booleans")
    weights = _cross_weights(geo, cross_weights)
    cross_max = geo.max_cross_degree if weights is None else _max_by_graph(geo, _weighted_cross_degree(geo, weights))
    eta_value = _graph_coefficient(geo, eta, geo.eta, "eta", active=geo.max_intra_degree > 0)
    gamma_value = _graph_coefficient(geo, gamma, _safe_step(cross_max), "gamma", active=cross_max > 0)
    gain = _gain(cross_gain, h)
    y0 = replicate(geo, h)
    y1 = y0 - _copy_scale(geo, apply_intra(geo, y0), eta_value)
    y2 = y1 - gain * _copy_scale(geo, _apply_cross(geo, y1, weights), gamma_value) if cross else y1
    y3 = y2 - _copy_scale(geo, apply_intra(geo, y2), eta_value) if second_intra else y2
    result = merge(geo, y3)
    if not diagnostics:
        return result
    stages: dict[str, Any] = {
        "replicated": y0, "after_intra_1": y1, "after_cross": y2,
        "cross_gain": gain, "eta": eta_value, "gamma": gamma_value,
    }
    if second_intra:
        stages["after_intra_2"] = y3
    return result, stages


def sandwich(
    geo: CouplingGeometry, h: Tensor, *, cross: bool = True,
    cross_gain: float | Tensor = 1.0, eta: float | Tensor | None = None,
    gamma: float | Tensor | None = None, cross_weights: Tensor | None = None,
    diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """M (I-eta A) (I-gamma gain K) (I-eta A) R h.

    A is identical before and after K. Turning cross off retains both internal
    steps. Defaults guarantee each individual fixed update is a contraction;
    explicit step overrides are validated but remain the caller's experiment.
    """
    return _sandwich_stages(
        geo, h, cross=cross, cross_gain=cross_gain, eta=eta, gamma=gamma,
        cross_weights=cross_weights, second_intra=True, diagnostics=diagnostics,
    )


def immediate(
    geo: CouplingGeometry, h: Tensor, *, cross: bool = True,
    cross_gain: float | Tensor = 1.0, eta: float | Tensor | None = None,
    gamma: float | Tensor | None = None, cross_weights: Tensor | None = None,
    diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Cancellation control: local update, copy coupling, immediate mean merge."""
    return _sandwich_stages(
        geo, h, cross=cross, cross_gain=cross_gain, eta=eta, gamma=gamma,
        cross_weights=cross_weights, second_intra=False, diagnostics=diagnostics,
    )


def unified(
    geo: CouplingGeometry, h: Tensor, *, steps: int = 3,
    lambda_cross: float | Tensor = 1.0, step: float | Tensor | None = None,
    cross: bool = True, cross_weights: Tensor | None = None, diagnostics: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Persistent explicit steps on A+lambda K; merge once at the end.

    cross=False uses the *same* default step bound including lambda K. Thus its
    comparison with cross=True does not change the internal diffusion step.
    The loop is over the requested algorithmic steps, never nodes or seeds.
    """
    _check_values(geo, h, geo.n, "h")
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(cross, bool) or not isinstance(diagnostics, bool):
        raise ValueError("cross and diagnostics must be booleans")
    weights = _cross_weights(geo, cross_weights)
    lam = _graph_coefficient(geo, lambda_cross, torch.ones_like(geo.eta), "lambda_cross", allow_zero=True)
    degree = geo.intra_degree + lam[geo.copy_graph] * _weighted_cross_degree(geo, weights)
    max_degree = _max_by_graph(geo, degree)
    step_value = _graph_coefficient(geo, step, _safe_step(max_degree), "step", active=max_degree > 0)
    y0 = replicate(geo, h)
    y = y0
    for _ in range(int(steps)):
        action = apply_intra(geo, y)
        if cross:
            action = action + _copy_scale(geo, _apply_cross(geo, y, weights), lam)
        y = y - _copy_scale(geo, action, step_value)
    result = merge(geo, y)
    if not diagnostics:
        return result
    return result, {"replicated": y0, "after_steps": y, "steps": int(steps), "lambda_cross": lam, "step": step_value}
