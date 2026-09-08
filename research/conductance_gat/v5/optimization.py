"""Differentiable graph-specific C optimization, without an edge-output MLP.

A signed quadratic compatibility parameterizes an energy. Its positive edge
variables are optimized afresh on each graph. Exactly K updates are unrolled;
this is a finite approximation, not a claim that the optimum was reached.
No targets, dense adjacency, eigendecomposition or autograd.grad are needed.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .operator import _validate_graph_index, graph_broadcast, graph_sum


def _degree(c: Tensor, incidence: Tensor, num_nodes: int) -> Tensor:
    return (
        c.new_zeros((num_nodes, *c.shape[1:]))
        .index_add(0, incidence[0], c)
        .index_add(0, incidence[1], c)
    )


def _edge_weighted(values: Tensor, omega: Tensor) -> Tensor:
    return values * (omega[:, None] if values.ndim == 2 else omega)


def _graph_max(
    values: Tensor, edge_graph: Tensor, num_graphs: int, *, initial: float = 0.0
) -> Tensor:
    if num_graphs == 1:
        # Include the original self value, also in the gradient's tie count.
        # clamp_min/maximum after amax would give different zero-tie gradients.
        return torch.cat((values.new_full((1, *values.shape[1:]), initial), values)).amax(
            dim=0, keepdim=True
        )
    index = edge_graph[:, None].expand_as(values) if values.ndim == 2 else edge_graph
    return values.new_full((num_graphs, *values.shape[1:]), initial).scatter_reduce(
        0, index, values, reduce="amax", include_self=True
    )


def _weighted_mean(
    values: Tensor, edge_graph: Tensor, num_graphs: int, omega: Tensor, graph_mass: Tensor
) -> Tensor:
    # Graph IDs are validated at the solver boundary. Reuse the live mass:
    # detaching it would change derivatives when sampling weights need grad.
    numerator = graph_sum(
        _edge_weighted(values, omega), edge_graph, num_graphs, validate_index=False
    )
    mass = graph_mass[:, None] if values.ndim == 2 else graph_mass
    return numerator / mass.clamp_min(torch.finfo(values.dtype).tiny)


def _normalize_log_c(
    log_c: Tensor,
    edge_graph: Tensor,
    num_graphs: int,
    omega: Tensor,
    graph_mass: Tensor | None = None,
) -> Tensor:
    # Numerical shift only: no edge or C value is truncated.
    if graph_mass is None:
        _validate_graph_index(edge_graph, num_graphs)
        graph_mass = graph_sum(omega, edge_graph, num_graphs, validate_index=False)
    maxima = _graph_max(log_c, edge_graph, num_graphs, initial=-torch.inf)
    shifted = log_c - graph_broadcast(maxima, edge_graph, num_graphs, validate_index=False)
    mean = _weighted_mean(shifted.exp(), edge_graph, num_graphs, omega, graph_mass)
    return shifted - graph_broadcast(mean, edge_graph, num_graphs, validate_index=False).log()


def _energy_from_degrees(
    c: Tensor,
    delta: Tensor,
    node_graph: Tensor,
    edge_graph: Tensor,
    num_graphs: int,
    omega: Tensor,
    graph_mass: Tensor,
    degree: Tensor,
    reference: Tensor,
    counts: Tensor,
    *,
    entropy: float,
    degree_barrier: float,
) -> Tensor:
    if degree.ndim == 2 and reference.ndim == 1:
        reference = reference[:, None]
        counts = counts[:, None]
    active = reference > 0
    safe_degree = torch.where(active, degree, torch.ones_like(degree))
    safe_reference = torch.where(active, reference, torch.ones_like(reference))
    log_ratio_sum = graph_sum(
        (safe_degree / safe_reference).log(), node_graph, num_graphs, validate_index=False
    )
    return _weighted_mean(
        c * delta + entropy * (c * c.log() - c + 1), edge_graph, num_graphs, omega, graph_mass
    ) - degree_barrier * log_ratio_sum / counts.clamp_min(1)


def conductance_energy(
    c: Tensor,
    delta: Tensor,
    incidence: Tensor,
    node_graph: Tensor,
    num_graphs: int,
    omega: Tensor,
    *,
    entropy: float,
    degree_barrier: float,
) -> Tensor:
    """Return E per graph; degree-zero isolates have no barrier term.

    E = mean_omega(c*delta + entropy*(c*log(c)-c+1))
        - degree_barrier*mean_active_nodes(log(d_c/d_reference)).

    The caller maintains mean_omega(c)=1. Scaling all omega in one graph
    by a positive constant leaves both the objective and constraint unchanged.
    """
    _validate_graph_index(node_graph, num_graphs)
    edge_graph = node_graph[incidence[0]]
    degree = _degree(_edge_weighted(c, omega), incidence, node_graph.numel())
    reference = _degree(omega, incidence, node_graph.numel())
    counts = graph_sum((reference > 0).to(c.dtype), node_graph, num_graphs, validate_index=False)
    graph_mass = graph_sum(omega, edge_graph, num_graphs, validate_index=False)
    return _energy_from_degrees(
        c,
        delta,
        node_graph,
        edge_graph,
        num_graphs,
        omega,
        graph_mass,
        degree,
        reference,
        counts,
        entropy=entropy,
        degree_barrier=degree_barrier,
    )


class GraphOptimizedConductance(nn.Module):
    """Signed compatibility followed by independently normalized C optimization.

    Uses every feature channel, a graph-conditioned signed diagonal quadratic
    metric, and symmetric structural features. There is no edge MLP or
    edge-specific parameter table. The legacy default shares C; multiple C
    heads have separate context/structure metrics, energies, degrees, steps
    and gauges, vectorized along an explicit head dimension. A common node
    projection does not average the independently learned head metrics.
    Explicit relation IDs optionally select additional quadratic metrics.
    ``degree_only`` removes task-learned costs; ``entropy_exact`` solves the
    entropy/linear-cost objective analytically and requires degree barrier 0.

    ``legacy_unit`` preserves the original unit-normalized compatibility.
    ``width_scaled`` compensates the O(channels**-0.5) contrast of isotropic
    normalized features by scaling the quadratic term by sqrt(channels).
    Its raw cost is graph-weighted centered before the existing tanh bound,
    so an unidentifiable graph-wide offset cannot saturate that bound. This
    changes neither the entropy coefficient nor C's positivity/gauge and
    imposes no target C variance; learned costs can still be constant.

    The configured step size is an upper bound. Each graph gets a
    differentiable relative-curvature/log-displacement-bounded step. This
    preserves K and all edges without host-synchronized line search or
    conductance clipping. See _step for the bound.
    """

    def __init__(
        self,
        channels: int,
        *,
        mode: str = "dynamic",
        solver_steps: int = 8,
        solver_step_size: float = 0.25,
        solver_entropy: float = 1.0,
        solver_degree_barrier: float = 0.1,
        solver_cost_scaling: str = "legacy_unit",
        cost_bound: float = 2.0,
        edge_chunk_size: int = 65536,
        conductance_heads: int = 1,
        generator: str = "optimized",
        num_relations: int = 0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("channels", channels),
            ("solver_steps", solver_steps),
            ("edge_chunk_size", edge_chunk_size),
            ("conductance_heads", conductance_heads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("solver_step_size", solver_step_size),
            ("solver_entropy", solver_entropy),
            ("cost_bound", cost_bound),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(solver_degree_barrier, bool)
            or not isinstance(solver_degree_barrier, Real)
            or not math.isfinite(solver_degree_barrier)
            or solver_degree_barrier < 0
        ):
            raise ValueError("solver_degree_barrier must be finite and nonnegative")
        if mode not in {"dynamic", "fixed_one"}:
            raise ValueError(f"unsupported conductance mode: {mode}")
        if solver_cost_scaling not in ("legacy_unit", "width_scaled"):
            raise ValueError(f"unsupported solver_cost_scaling: {solver_cost_scaling}")
        if generator not in {"optimized", "degree_only", "entropy_exact"}:
            raise ValueError(f"unsupported conductance generator: {generator}")
        if generator == "entropy_exact" and solver_degree_barrier != 0:
            raise ValueError("entropy_exact requires solver_degree_barrier=0")
        if (
            isinstance(num_relations, bool)
            or not isinstance(num_relations, int)
            or num_relations < 0
        ):
            raise ValueError("num_relations must be a nonnegative integer")
        if generator == "degree_only" and num_relations:
            raise ValueError("degree_only has no learned relation costs; num_relations must be 0")
        self.channels = channels
        self.conductance_heads = conductance_heads
        self.generator = generator
        self.num_relations = num_relations
        self.mode = mode
        self.solver_steps = solver_steps
        self.solver_step_size = float(solver_step_size)
        self.solver_entropy = float(solver_entropy)
        self.solver_degree_barrier = float(solver_degree_barrier)
        self.solver_cost_scaling = solver_cost_scaling
        self.quadratic_scale = math.sqrt(channels) if solver_cost_scaling == "width_scaled" else 1.0
        self.cost_bound = float(cost_bound)
        self.edge_chunk_size = edge_chunk_size
        if mode == "dynamic" and generator != "degree_only":
            self.node_projection: nn.Linear | None = nn.Linear(channels, channels, bias=False)
            self.context_metric: nn.Linear | None = nn.Linear(
                2 * channels + 8, channels * conductance_heads
            )
            self.structure_metric: nn.Parameter | None = nn.Parameter(
                torch.empty(8) if conductance_heads == 1 else torch.empty(conductance_heads, 8)
            )
            nn.init.normal_(self.structure_metric, std=0.01)
        else:
            # A true parameter-free C=1 control, not frozen unused modules.
            self.node_projection = None
            self.context_metric = None
            self.register_parameter("structure_metric", None)
        if num_relations and mode == "dynamic":
            # A genuine relation-conditioned diagonal quadratic metric,
            # not a constant graph offset and not fabricated relation labels.
            self.relation_metric = nn.Parameter(
                torch.empty(num_relations, conductance_heads, channels)
            )
            nn.init.normal_(self.relation_metric, std=0.01)
        else:
            self.register_parameter("relation_metric", None)
        self.override: str | None = None
        self.last_scores: Tensor | None = None
        self.last_log_c: Tensor | None = None
        self.last_c: Tensor | None = None
        self.last_solver_diagnostics: dict[str, Tensor | int | float | bool | str] = {}

    def _raw_compatibility_chunk(
        self,
        projected: Tensor,
        metric: Tensor,
        tail: Tensor,
        head: Tensor,
        sample_degree: Tensor,
        full_degree: Tensor,
        edge_graph: Tensor,
        edge_relation_id: Tensor | None = None,
    ) -> Tensor:
        if self.structure_metric is None:
            raise RuntimeError("fixed C has no compatibility parameters")
        left, right = projected[tail], projected[head]
        coverage = (
            torch.stack(
                (
                    sample_degree[tail] / full_degree[tail].clamp_min(1),
                    sample_degree[head] / full_degree[head].clamp_min(1),
                ),
                dim=1,
            )
            .sort(dim=1)
            .values
        )
        inverse = (
            torch.stack(
                (
                    full_degree[tail].clamp_min(1).reciprocal(),
                    full_degree[head].clamp_min(1).reciprocal(),
                ),
                dim=1,
            )
            .sort(dim=1)
            .values
        )
        local = torch.cat(
            (
                torch.stack(
                    (
                        (sample_degree[tail] + sample_degree[head]).log1p(),
                        (sample_degree[tail] - sample_degree[head]).abs().log1p(),
                        (full_degree[tail] + full_degree[head]).log1p(),
                        (full_degree[tail] - full_degree[head]).abs().log1p(),
                    ),
                    dim=1,
                ),
                coverage,
                inverse,
            ),
            dim=1,
        )
        edge_metric = graph_broadcast(metric, edge_graph, metric.shape[0], validate_index=False)
        if self.conductance_heads == 1:
            if edge_relation_id is not None and self.relation_metric is not None:
                edge_metric = (
                    edge_metric
                    + self.relation_metric[edge_relation_id, 0].to(projected.dtype).tanh()
                )
            quadratic = ((left - right).square() * edge_metric).sum(dim=1)
            structural = (local.tanh() * self.structure_metric.to(projected.dtype)).sum(dim=1)
        else:
            if edge_relation_id is not None and self.relation_metric is not None:
                edge_metric = (
                    edge_metric + self.relation_metric[edge_relation_id].to(projected.dtype).tanh()
                )
            quadratic = torch.einsum("ed,ehd->eh", (left - right).square(), edge_metric)
            structural = local.tanh() @ self.structure_metric.to(projected.dtype).T
        if self.solver_cost_scaling == "width_scaled":
            quadratic = quadratic * self.quadratic_scale
        return quadratic + structural

    def _compatibility_chunk(
        self,
        projected: Tensor,
        metric: Tensor,
        tail: Tensor,
        head: Tensor,
        sample_degree: Tensor,
        full_degree: Tensor,
        edge_graph: Tensor,
        edge_relation_id: Tensor | None = None,
    ) -> Tensor:
        """Keep the legacy bounded-chunk path numerically unchanged."""

        raw = self._raw_compatibility_chunk(
            projected, metric, tail, head, sample_degree, full_degree, edge_graph, edge_relation_id
        )
        return self.cost_bound * torch.tanh(raw / self.cost_bound)

    def _scaled_gradient(
        self,
        log_c: Tensor,
        delta: Tensor,
        incidence: Tensor,
        edge_graph: Tensor,
        omega: Tensor,
        graph_mass: Tensor,
        active_counts: Tensor,
        num_nodes: int,
        *,
        barrier_coefficient: Tensor | None = None,
        degree: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if degree is None:
            degree = _degree(_edge_weighted(log_c.exp(), omega), incidence, num_nodes)
        inverse_sum = degree[incidence[0]].reciprocal() + degree[incidence[1]].reciprocal()
        if barrier_coefficient is None:
            barrier_coefficient = self.solver_degree_barrier * (
                graph_mass / active_counts.clamp_min(1)
            )
        coefficient = graph_broadcast(
            barrier_coefficient, edge_graph, graph_mass.numel(), validate_index=False
        )
        if log_c.ndim == 2:
            coefficient = coefficient[:, None]
        barrier = coefficient * inverse_sum
        return delta + self.solver_entropy * log_c - barrier, barrier

    def _step(
        self,
        log_c: Tensor,
        delta: Tensor,
        incidence: Tensor,
        edge_graph: Tensor,
        num_graphs: int,
        omega: Tensor,
        graph_mass: Tensor,
        active_counts: Tensor,
        num_nodes: int,
        barrier_coefficient: Tensor | None = None,
        degree: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        gradient, barrier = self._scaled_gradient(
            log_c,
            delta,
            incidence,
            edge_graph,
            omega,
            graph_mass,
            active_counts,
            num_nodes,
            barrier_coefficient=barrier_coefficient,
            degree=degree,
        )
        centered = gradient - graph_broadcast(
            _weighted_mean(gradient, edge_graph, num_graphs, omega, graph_mass),
            edge_graph,
            num_graphs,
            validate_index=False,
        )
        curvature = _graph_max(barrier, edge_graph, num_graphs)
        magnitude = _graph_max(centered.abs(), edge_graph, num_graphs)
        # Relative entropy Hessian is diag(omega/(S*c)). Cauchy-Schwarz
        # bounds the barrier Hessian by L times this, with
        # L=max_e rho*S/n*(1/d_u+1/d_v).
        # eta <= 1/(2*max|centered_gradient|) gives |log(c_new/c_old)|<=1,
        # including the gauge normalization. Degrees along this path stay
        # >=exp(-1)*old degrees, hence curvature stays <=exp(1)*L.
        # eta<=exp(-1)/L supplies a valid majorizer; entropy is proximal.
        # Clamp denominators at the threshold for the configured upper
        # bound: unlike 1/clamp(tiny), this has finite inactive derivatives
        # when curvature or gradient vanish (including rho=0).
        curvature_step = math.exp(-1) / curvature.clamp_min(math.exp(-1) / self.solver_step_size)
        displacement_step = 0.5 / magnitude.clamp_min(0.5 / self.solver_step_size)
        step = torch.minimum(curvature_step, displacement_step)
        edge_step = graph_broadcast(step, edge_graph, num_graphs, validate_index=False)
        proposal = log_c - edge_step * centered / (1 + edge_step * self.solver_entropy)
        updated = _normalize_log_c(proposal, edge_graph, num_graphs, omega, graph_mass)
        return updated, step, centered

    def forward(
        self,
        state: Tensor,
        incidence: Tensor,
        node_graph: Tensor,
        num_graphs: int,
        *,
        graph_context: Tensor,
        sample_degree: Tensor,
        full_degree: Tensor,
        edge_normalization_weight: Tensor | None = None,
        edge_relation_id: Tensor | None = None,
    ) -> Tensor:
        if state.ndim != 2 or state.shape[1] != self.channels or not state.is_floating_point():
            raise ValueError("state must be an N x channels floating tensor")
        if incidence.dtype != torch.long or incidence.ndim != 2 or incidence.shape[0] != 2:
            raise ValueError("incidence must be a 2 x E int64 tensor")
        if node_graph.shape != (state.shape[0],) or node_graph.dtype != torch.long:
            raise ValueError("node_graph must contain one int64 graph index per node")
        if isinstance(num_graphs, bool) or not isinstance(num_graphs, int) or num_graphs < 1:
            raise ValueError("num_graphs must be a positive integer")
        if graph_context.shape != (num_graphs, 2 * self.channels + 8):
            raise ValueError("graph_context must be num_graphs x (2*channels+8)")
        if sample_degree.shape != node_graph.shape or full_degree.shape != node_graph.shape:
            raise ValueError("sample_degree and full_degree must contain one value per node")
        inputs = (incidence, node_graph, graph_context, sample_degree, full_degree)
        if any(value.device != state.device for value in inputs):
            raise ValueError("all conductance inputs must share a device")
        if self.override not in {None, "ones", "mean", "shuffle"}:
            raise ValueError(f"unsupported C intervention: {self.override}")
        compute_dtype = (
            torch.float32 if state.dtype in {torch.float16, torch.bfloat16} else state.dtype
        )
        _validate_graph_index(node_graph, num_graphs)
        tail, head = incidence
        if self.num_relations:
            if (
                edge_relation_id is None
                or edge_relation_id.shape != tail.shape
                or edge_relation_id.dtype != torch.long
                or edge_relation_id.device != state.device
            ):
                raise ValueError(
                    "explicit edge_relation_id must be same-device int64 "
                    "with one ID per physical edge"
                )
            torch._assert_async(
                ((edge_relation_id >= 0) & (edge_relation_id < self.num_relations)).all(),
                "edge_relation_id is outside configured num_relations",
            )
        elif edge_relation_id is not None:
            raise ValueError("edge_relation_id supplied without explicit num_relations")
        edge_graph = node_graph[tail]
        omega = torch.ones(tail.numel(), device=state.device, dtype=compute_dtype)
        if edge_normalization_weight is not None:
            if (
                edge_normalization_weight.shape != omega.shape
                or edge_normalization_weight.device != state.device
            ):
                raise ValueError("edge_normalization_weight must match same-device edges")
            omega = edge_normalization_weight.to(compute_dtype)
        torch._assert_async(
            torch.isfinite(omega).all() & (omega > 0).all(),
            "C solver needs finite positive sampling weights",
        )
        if self.mode == "fixed_one" or self.override == "ones" or tail.numel() == 0:
            c = (
                torch.ones_like(omega)
                if self.conductance_heads == 1
                else omega.new_ones((omega.shape[0], self.conductance_heads))
            )
            self.last_scores = torch.zeros_like(c)
            self.last_log_c = torch.zeros_like(c)
            self.last_c = c.detach()
            self.last_solver_diagnostics = {
                "enabled": False,
                "executed_steps": 0,
                "reason": "edgeless_graph" if tail.numel() == 0 else "fixed_one_intervention",
                "finite_step_approximation": False,
                "solver_cost_scaling": self.solver_cost_scaling,
                "quadratic_scale": self.quadratic_scale,
            }
            return c
        if self.generator == "degree_only":
            # This ablation deliberately has no task-learned cost parameters.
            shape = (
                (tail.numel(),)
                if self.conductance_heads == 1
                else (tail.numel(), self.conductance_heads)
            )
            delta = state.new_zeros(shape, dtype=compute_dtype)
        else:
            if self.node_projection is None or self.context_metric is None:
                raise RuntimeError("dynamic compatibility parameters are unavailable")
            projected = F.normalize(self.node_projection(state).to(compute_dtype), dim=1, eps=1e-6)
            normalized_context = F.layer_norm(graph_context, (graph_context.shape[1],))
            metric = self.context_metric(normalized_context).to(compute_dtype).tanh()
            if self.conductance_heads > 1:
                metric = metric.reshape(num_graphs, self.conductance_heads, self.channels)
            sample_degree = sample_degree.to(compute_dtype)
            full_degree = full_degree.to(compute_dtype)
            chunks = []
            compatibility = (
                self._raw_compatibility_chunk
                if self.solver_cost_scaling == "width_scaled"
                else self._compatibility_chunk
            )
            for start in range(0, tail.numel(), self.edge_chunk_size):
                stop = start + self.edge_chunk_size
                arguments = (
                    projected,
                    metric,
                    tail[start:stop],
                    head[start:stop],
                    sample_degree,
                    full_degree,
                    edge_graph[start:stop],
                )
                if edge_relation_id is not None:
                    arguments = (*arguments, edge_relation_id[start:stop])
                if torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint

                    chunk = checkpoint(
                        compatibility, *arguments, use_reentrant=False, preserve_rng_state=False
                    )
                else:
                    chunk = compatibility(*arguments)
                chunks.append(chunk)
            delta = torch.cat(chunks)
        graph_mass = graph_sum(omega, edge_graph, num_graphs, validate_index=False)
        if self.solver_cost_scaling == "width_scaled":
            # Center over complete graphs, never individual memory chunks.
            # A graph-constant cost has no effect under mean_omega(C)=1;
            # removing it before tanh avoids spurious width-driven saturation.
            delta = delta - graph_broadcast(
                _weighted_mean(delta, edge_graph, num_graphs, omega, graph_mass),
                edge_graph,
                num_graphs,
                validate_index=False,
            )
            delta = self.cost_bound * torch.tanh(delta / self.cost_bound)
        reference_degree = _degree(omega, incidence, state.shape[0])
        active_counts = graph_sum(
            (reference_degree > 0).to(compute_dtype), node_graph, num_graphs, validate_index=False
        )
        barrier_coefficient = self.solver_degree_barrier * (graph_mass / active_counts.clamp_min(1))
        log_c = torch.zeros_like(delta)
        steps = []
        initial_residual = None
        previous_log_c = log_c
        initial_degree = (
            reference_degree
            if delta.ndim == 1
            else reference_degree[:, None].expand(-1, self.conductance_heads)
        )
        if self.generator == "entropy_exact":
            # Exact minimizer of the entropy + learned linear cost energy
            # under the same omega-weighted mean-one constraint (rho=0).
            initial_centered = delta - graph_broadcast(
                _weighted_mean(delta, edge_graph, num_graphs, omega, graph_mass),
                edge_graph,
                num_graphs,
                validate_index=False,
            )
            initial_residual = _weighted_mean(
                initial_centered.detach().square(), edge_graph, num_graphs, omega, graph_mass
            ).sqrt()
            log_c = _normalize_log_c(
                -delta / self.solver_entropy, edge_graph, num_graphs, omega, graph_mass
            )
        executed_steps = 0 if self.generator == "entropy_exact" else self.solver_steps
        for iteration in range(executed_steps):
            previous_log_c = log_c
            arguments = (
                log_c,
                delta,
                incidence,
                edge_graph,
                num_graphs,
                omega,
                graph_mass,
                active_counts,
                state.shape[0],
                barrier_coefficient,
                # C starts as the constant one vector, so this degree is
                # exactly the live reference degree, including omega's grad.
                initial_degree if iteration == 0 else None,
            )
            if torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                # Keep scalar iterates, not every degree/curvature/normalizer
                # intermediate from every solver step. This also applies when
                # the outer backbone checkpoint is disabled during calibration.
                log_c, step, centered = checkpoint(
                    self._step, *arguments, use_reentrant=False, preserve_rng_state=False
                )
            else:
                log_c, step, centered = self._step(*arguments)
            steps.append(step.detach())
            if initial_residual is None:
                with torch.no_grad():
                    initial_residual = _weighted_mean(
                        centered.detach().square(), edge_graph, num_graphs, omega, graph_mass
                    ).sqrt()
        c = log_c.exp()
        with torch.no_grad():
            # At C=1, entropy and log(d_c/d_reference) are exactly zero.
            initial_energy = _weighted_mean(
                delta.detach(), edge_graph, num_graphs, omega, graph_mass
            )
            final_degree = _degree(_edge_weighted(c.detach(), omega), incidence, state.shape[0])
            final_energy = _energy_from_degrees(
                c.detach(),
                delta.detach(),
                node_graph,
                edge_graph,
                num_graphs,
                omega,
                graph_mass,
                final_degree,
                reference_degree,
                active_counts,
                entropy=self.solver_entropy,
                degree_barrier=self.solver_degree_barrier,
            )
            gradient, _ = self._scaled_gradient(
                log_c.detach(),
                delta.detach(),
                incidence,
                edge_graph,
                omega,
                graph_mass,
                active_counts,
                state.shape[0],
                barrier_coefficient=barrier_coefficient,
                degree=final_degree,
            )
            centered = gradient - graph_broadcast(
                _weighted_mean(gradient, edge_graph, num_graphs, omega, graph_mass),
                edge_graph,
                num_graphs,
                validate_index=False,
            )
            residual = _weighted_mean(
                centered.square(), edge_graph, num_graphs, omega, graph_mass
            ).sqrt()
            update = _weighted_mean(
                (log_c.detach() - previous_log_c.detach()).square(),
                edge_graph,
                num_graphs,
                omega,
                graph_mass,
            ).sqrt()
            tolerance = 128 * torch.finfo(compute_dtype).eps * (1 + initial_energy.abs())
            valid = (
                torch.isfinite(c).all()
                & (c > 0).all()
                & torch.isfinite(final_energy).all()
                & torch.isfinite(residual).all()
                & (final_energy <= initial_energy + tolerance).all()
            )
            torch._assert_async(
                valid, "C optimization produced nonfinite values or increased its objective"
            )
            step_history = (
                torch.stack(steps) if steps else torch.zeros_like(initial_energy).unsqueeze(0)
            )
            self.last_solver_diagnostics = {
                "enabled": True,
                "method": "analytic_entropy"
                if self.generator == "entropy_exact"
                else "curvature_bounded_kl_proximal",
                "executed_steps": executed_steps,
                "finite_step_approximation": self.generator != "entropy_exact",
                "conductance_generator": self.generator,
                "conductance_heads": self.conductance_heads,
                "solver_cost_scaling": self.solver_cost_scaling,
                "quadratic_scale": self.quadratic_scale,
                "objective_initial": initial_energy,
                "objective_final": final_energy,
                "projected_gradient_rms_initial": initial_residual,
                "projected_gradient_rms_final": residual,
                "last_log_update_rms": update,
                "step_size_requested": self.solver_step_size,
                "step_size_min": step_history.amin(dim=0),
                "step_size_max": step_history.amax(dim=0),
                "mean_c": _weighted_mean(c.detach(), edge_graph, num_graphs, omega, graph_mass),
                "active_nodes": active_counts.detach(),
            }
        if self.override == "mean":
            c = graph_broadcast(
                _weighted_mean(c, edge_graph, num_graphs, omega, graph_mass),
                edge_graph,
                num_graphs,
                validate_index=False,
            )
        elif self.override == "shuffle" and c.numel() > 1:
            # Reverse within each graph with no Python graph loop.
            order = torch.argsort(edge_graph, stable=True)
            counts = torch.bincount(edge_graph, minlength=num_graphs)
            starts = counts.cumsum(0) - counts
            position = torch.arange(c.shape[0], device=c.device)
            reverse = 2 * starts[edge_graph[order]] + counts[edge_graph[order]] - 1 - position
            c = (
                torch.empty_like(c).scatter(0, order, c[order[reverse]])
                if c.ndim == 1
                else torch.empty_like(c).index_copy(0, order, c[order[reverse]])
            )
        self.last_scores = delta.detach()
        self.last_log_c = log_c.detach()
        self.last_c = c.detach()
        return c
