"""New topology-selection experiment; the historical V5 implementation is untouched."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from research.conductance_gat.v5.model import (
    GraphConditionedConductanceNodeClassifier,
    _static_graph_context,
    graph_context_features,
)
from research.conductance_gat.v5.operator import graph_sum, shared_head_diffusion

from .selection import EdgeSelector, selection_configuration


class EdgeSelectionOperator(nn.Module):
    """Original positive r, new physical gate z, actual diffusion with z*r.

    r is still optimized on the complete candidate B. Its solver energy and
    convergence diagnostics describe r, NOT a newly solved gated objective.
    All candidate edges are computed; masking alone is not a speedup claim.
    """

    def __init__(self, original, selection_config, *, layer):
        super().__init__()
        # Reuse already-initialized modules/parameters without a second draw.
        # Their state_dict keys and exact seed-paired values stay the same.
        for name, module in original.named_children():
            self.add_module(name, module)
        for name, parameter in original._parameters.items():
            self.register_parameter(name, parameter)
        for name in (
            "channels",
            "heads",
            "head_width",
            "conductance_mode",
            "conductance_backend",
            "conductance_heads",
            "propagation_normalization",
            "conductance_generator",
            "num_relations",
            "edge_direction",
            "propagation_filter",
            "edge_chunk_size",
        ):
            setattr(self, name, getattr(original, name))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(selection_config["selection_seed"] + layer)
            self.selector = EdgeSelector(
                self.channels, selection_config, edge_chunk_size=self.edge_chunk_size
            )
        self.last_gate = self.last_r = self.last_effective_c = None
        self.last_logits = self.last_probability = self.last_beta = None
        self.last_sampling_correction = None
        self.live_edge_graph = None
        self.live_num_graphs = None
        self.amplitude_override = None

    def forward(
        self,
        state,
        incidence,
        node_graph,
        num_graphs,
        *,
        full_degree,
        graph_structure,
        edge_normalization_weight,
        sampling_correction,
        edge_selection_topology,
        static_context=None,
        edge_relation_id=None,
    ):
        with torch.autocast(device_type=state.device.type, enabled=False):
            geometry = state.float()
            context, sample_degree, full_degree = graph_context_features(
                geometry,
                incidence,
                node_graph,
                num_graphs,
                full_degree,
                graph_structure,
                static_context=static_context,
            )
            relation = (
                {"edge_relation_id": edge_relation_id}
                if edge_relation_id is not None or self.num_relations
                else {}
            )
            r = self.estimator(
                geometry,
                incidence,
                node_graph,
                num_graphs,
                graph_context=context,
                sample_degree=sample_degree,
                full_degree=full_degree,
                edge_normalization_weight=edge_normalization_weight,
                **relation,
            )
            if self.amplitude_override is not None:
                if self.training or torch.is_grad_enabled():
                    raise RuntimeError("amplitude interventions are evaluation/no-grad only")
                if self.amplitude_override != "ones":
                    raise ValueError("unsupported amplitude intervention")
                r = torch.ones_like(r)
            gate = self.selector(
                geometry,
                incidence,
                node_graph,
                num_graphs,
                topology=edge_selection_topology,
                sample_degree=sample_degree,
                full_degree=full_degree,
            )
            effective = r * (gate[:, None] if r.ndim == 2 else gate)
            beta = self.beta_estimator(context)
        value = torch.einsum("nd,hdk->nhk", state, self.value_weight)
        propagated = shared_head_diffusion(
            value,
            effective,
            incidence,
            node_graph,
            beta,
            sampling_correction=sampling_correction,
            edge_chunk_size=self.edge_chunk_size,
            propagation_normalization=self.propagation_normalization,
            polynomial_coefficients=self.polynomial_delta,
        )
        self.last_gate, self.last_r, self.last_effective_c = (
            gate.detach(),
            r.detach(),
            effective.detach(),
        )
        self.last_logits, self.last_probability = (
            self.selector.last_logits,
            self.selector.last_probability,
        )
        self.last_beta = beta.detach()
        self.last_sampling_correction = (
            None if sampling_correction is None else sampling_correction.detach()
        )
        self.live_edge_graph, self.live_num_graphs = edge_selection_topology.edge_graph, num_graphs
        return self.output_projection(propagated.reshape(state.shape[0], self.channels))

    def clear_auxiliary_cache(self):
        self.selector.clear_auxiliary_cache()
        self.live_edge_graph = self.live_num_graphs = None


class EdgeSelectionClassifier(GraphConditionedConductanceNodeClassifier):
    """Seed-paired full V5 backbone with an explicit sibling edge-selection axis."""

    def __init__(self, in_channels, classes, *, selection_config: dict, **architecture):
        selected = selection_configuration(selection_config)
        required = {
            "conductance_backend": "optimization",
            "conductance_generator": "optimized",
            "conductance_heads": "per_head",
            "propagation_normalization": "row",
            "propagation_filter": "linear",
            "conductance_mode": "dynamic",
        }
        for name, value in required.items():
            if name in architecture and architecture[name] != value:
                raise ValueError(f"edge selection holds {name}={value} fixed across conditions")
            architecture[name] = value
        architecture.setdefault("solver_cost_scaling", "width_scaled")
        super().__init__(in_channels, classes, **architecture)
        self.selection_config = selected
        for layer, block in enumerate(self.blocks):
            block.operator = EdgeSelectionOperator(block.operator, selected, layer=layer)

    def forward(self, graph):
        self.clear_auxiliary_cache()
        x, incidence = graph.x, graph.incidence_edge_index
        if x.ndim != 2 or x.shape[1] != self.in_channels or not x.is_floating_point():
            raise ValueError("graph.x must match the configured input width")
        if (
            incidence.dtype != torch.long
            or incidence.ndim != 2
            or incidence.shape[0] != 2
            or incidence.device != x.device
        ):
            raise ValueError("physical incidence must be same-device 2 x E int64")
        topology = getattr(graph, "edge_selection_topology", None)
        if topology is None:
            raise ValueError("prepare graph.edge_selection_topology before model forward")
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            graphs = 1
        else:
            graphs = getattr(graph, "_v5_num_graphs", None)
            if (
                type(graphs) is not int
                or graphs < 1
                or batch.shape != (x.shape[0],)
                or batch.dtype != torch.long
            ):
                raise ValueError(
                    "batched selection graphs require explicit positive _v5_num_graphs"
                )
        kwargs = {
            "full_degree": getattr(graph, "full_degree", None),
            "graph_structure": getattr(graph, "graph_structure", None),
            "edge_normalization_weight": getattr(graph, "edge_normalization_weight", None),
            "sampling_correction": getattr(graph, "sampling_correction", None),
            "edge_relation_id": getattr(graph, "edge_relation_id", None),
            "edge_selection_topology": topology,
        }
        with torch.autocast(device_type=x.device.type, enabled=False):
            kwargs["static_context"] = _static_graph_context(
                x.float(),
                incidence,
                batch,
                graphs,
                kwargs["full_degree"],
                kwargs["graph_structure"],
            )
        hidden = self.encoder(self.input_norm(x))
        for block in self.blocks:
            if self.activation_checkpoint and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                hidden = checkpoint(
                    lambda value, layer=block: layer(value, incidence, batch, graphs, **kwargs),
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                hidden = block(hidden, incidence, batch, graphs, **kwargs)
        return self.decoder(self.final_norm(hidden))

    def clear_auxiliary_cache(self):
        for operator in self.operators:
            operator.clear_auxiliary_cache()

    @contextmanager
    def gate_intervention(self, mode, *, amplitude_ones=False):
        """Read-only eval intervention, restoring RNG and diagnostics even on error.

        mode is None, 'all', 'random_budget', or one frozen E tensor per layer.
        Random budget preserves the declared forest and exact chord k; it is
        intentionally unavailable for unconstrained hard-concrete. To keep
        baseline gates fixed while replacing r=1, pass detached baseline gate
        tensors explicitly: earlier layer changes otherwise change later H/z.
        This is a full-model intervention, not a fixed-H local comparison.
        """
        if any(module.training for module in self.modules()):
            raise RuntimeError("call eval() before an edge-selection intervention")
        operators = list(self.operators)
        if isinstance(mode, (list, tuple)):
            if len(mode) != len(operators) or any(not isinstance(gate, Tensor) for gate in mode):
                raise ValueError("frozen intervention requires one E tensor per layer")
            overrides = [gate.detach() for gate in mode]
        elif mode is None or isinstance(mode, str) and mode in {"all", "random_budget"}:
            overrides = [mode] * len(operators)
        else:
            raise ValueError("unsupported gate intervention")
        snapshots = [
            (
                module,
                {
                    name: value
                    for name, value in vars(module).items()
                    if name.startswith(("last_", "live_"))
                    or name in {"gate_override", "amplitude_override"}
                },
            )
            for module in self.modules()
        ]
        devices = sorted(
            {parameter.device.index for parameter in self.parameters() if parameter.is_cuda}
        )
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                for operator, override in zip(operators, overrides, strict=True):
                    operator.selector.gate_override = override
                    operator.amplitude_override = "ones" if amplitude_ones else None
                yield self
        finally:
            for module, attributes in snapshots:
                for name in list(vars(module)):
                    if (
                        name.startswith(("last_", "live_"))
                        or name in {"gate_override", "amplitude_override"}
                    ) and name not in attributes:
                        delattr(module, name)
                for name, value in attributes.items():
                    setattr(module, name, value)

    def auxiliary_loss(self, negative_targets: Tensor | None = None) -> dict[str, Tensor]:
        """Unweighted L0 and source-label loss, unavailable to model.forward.

        L0 = mean_layers(mean_graphs(sum_edges Pr(z>0))). The optional negative
        objective averages positive/negative BCE class means within each graph,
        then graphs and layers. Targets are 1 for original / 0 for corruption.
        This is corruption discrimination, not an assertion that all absent
        edges are harmful. Budgeted structure arms have no L0/negative loss.
        """
        zero = self.decoder.weight.new_zeros(())
        if self.selection_config["condition"] != "hard_concrete":
            if negative_targets is not None:
                raise ValueError(
                    "negative targets belong only to the separate corruption experiment"
                )
            return {"l0": zero, "negative": zero}
        l0, negatives = [], []
        for operator in self.operators:
            logits, probability = operator.selector.live_logits, operator.selector.live_probability
            groups, graphs = operator.live_edge_graph, operator.live_num_graphs
            if logits is None or probability is None or groups is None:
                raise RuntimeError(
                    "auxiliary_loss requires the current forward; cache was empty or cleared"
                )
            l0.append(graph_sum(probability, groups, graphs).mean())
            if negative_targets is not None:
                if (
                    negative_targets.shape != logits.shape
                    or negative_targets.device != logits.device
                ):
                    raise ValueError(
                        "negative targets must align with current same-device physical edges"
                    )
                torch._assert_async(
                    ((negative_targets == 0) | (negative_targets == 1)).all(),
                    "negative targets must be binary original/corruption labels",
                )
                target = negative_targets.to(logits.dtype)
                positive, negative = target == 1, target == 0
                config = self.selection_config
                active_logit = logits - config["hard_concrete_temperature"] * math.log(
                    -config["hard_concrete_lower"] / config["hard_concrete_upper"]
                )
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    active_logit, target, reduction="none"
                )
                pos_count = graph_sum(positive.to(logits.dtype), groups, graphs)
                neg_count = graph_sum(negative.to(logits.dtype), groups, graphs)
                pos_loss = graph_sum(losses * positive, groups, graphs) / pos_count.clamp_min(1)
                neg_loss = graph_sum(losses * negative, groups, graphs) / neg_count.clamp_min(1)
                class_count = (pos_count > 0).to(logits.dtype) + (neg_count > 0).to(logits.dtype)
                negatives.append(((pos_loss + neg_loss) / class_count.clamp_min(1)).mean())
        return {
            "l0": torch.stack(l0).mean(),
            "negative": torch.stack(negatives).mean() if negatives else zero,
        }

    def selection_metadata(self) -> dict:
        return {
            "configuration": self.selection_config,
            "gate_layout": "one shared physical-edge E gate; amplitudes E x H",
            "probability_meaning": (
                "hard-concrete: Pr(z>0); budgeted learned train: constrained logistic relaxation; "
                "budgeted eval/control or frozen override: actual gate"
            ),
            "amplitude_objective": (
                "original positive solver on full candidate support; "
                "gate not in omega or amplitude energy"
            ),
            "budget": (
                "exact floor(chord_fraction*chords) per graph in both train and eval "
                "for budgeted conditions"
            ),
            "gradient": (
                "budgeted: biased straight-through constrained-logistic gradient; "
                "hard-concrete: reparameterized stochastic relaxation"
            ),
            "cycle_context": (
                "unsigned fundamental-cycle membership edge-cycle-edge scalar means; "
                "DFS-basis dependent"
            ),
            "l0_aggregation": (
                "sum physical edges per graph, mean graphs, mean layers; "
                "unconstrained corruption arm only"
            ),
            "negative_aggregation": (
                "BCE on Pr(z>0), mean per binary class, mean present classes per graph, "
                "mean graphs, mean layers"
            ),
            "negative_targets_visible_to_forward": False,
            "automatic_speedup_claimed": False,
        }
