"""Residual-free incidence models and a DUALFormer (ICLR 2025) comparator.

DUALFormer preserves its published SA-residual/normalization mechanism. Those
are intrinsic to that comparator; no extra wrapper residual or FFN is added.
Its optional no-skip control is explicitly named, never called a reproduction.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from experiments.incidence_ablation.model import IncidenceOperator
from research.conductance_gat.edge_selection.selection import selection_configuration
from research.conductance_gat.v5.model import (
    GraphConditionedConductanceNodeClassifier,
    _static_graph_context,
)

ARMS = {
    "incidence": ("baseline", False),
    "incidence_pre_lift": ("pre_lift", False),
    "incidence_energy": ("baseline", True),
    "incidence_energy_pre_lift": ("pre_lift", True),
    "dualformer": None,
    "dualformer_no_skip_control": None,
    "gatv2": None,
    "gcn": None,
    "graphsage": None,
}
for _energy_suffix, _energy in (("", False), ("_diagonal", "diagonal"), ("_energy", True)):
    for _lift_suffix, _lift in (
        ("", "baseline"),
        ("_linear_lift", "linear_lift"),
        ("_pre_lift", "pre_lift"),
        ("_post_lift", "post_lift"),
    ):
        ARMS.setdefault("incidence" + _energy_suffix + _lift_suffix, (_lift, _energy))
for _regime in ("fixed", "shared"):
    ARMS[f"incidence_{_regime}"] = ("baseline", False)
    ARMS[f"incidence_{_regime}_energy"] = ("baseline", True)


def conductance_contract(arm):
    if not arm.startswith("incidence"):
        return None
    regime = (
        "fixed"
        if arm.startswith("incidence_fixed")
        else ("shared" if arm.startswith("incidence_shared") else "per_head")
    )
    return {
        "regime": regime,
        "conductance_mode": "fixed_one" if regime == "fixed" else "dynamic",
        "conductance_heads": "shared" if regime in {"fixed", "shared"} else "per_head",
    }


def local_gram(history, edges, conductance, edge_chunk_size, *, diagonal_only=False):
    """N x heads x pairs, including diagonal local energies and cross terms.

    Each undirected edge contributes half its weighted inner product to each
    endpoint, so summing nodes gives tr(H_k.T L H_l), without degree scaling.
    The identity assumes a common, frozen conductance and feature coordinates.
    """
    depth, nodes, heads, _ = history.shape
    if conductance.ndim == 1:
        conductance = conductance[:, None]
    from .gram import LocalGram

    chunk = max(1, (edge_chunk_size or max(edges.shape[1], 1)) // depth)
    return LocalGram.apply(history, conductance, edges, chunk, diagonal_only)


class DualAttention(nn.Module):
    """Official feature-space SA: Q softmax(K.T V / sqrt(N)), mean heads.

    Reference: JiamingZhuo/DUALFormer, commit 68fbdaf, model/sa.py.
    Heads each have hidden_channels features, as in the original code.
    """

    def __init__(self, width, heads, chunk_size=None):
        super().__init__()
        self.width, self.heads = width, heads
        self.chunk_size = chunk_size
        self.query = nn.Linear(width, width * heads)
        self.key = nn.Linear(width, width * heads)
        self.value = nn.Linear(width, width * heads)

    def forward(self, x, batch, graphs):
        if self.chunk_size is not None:
            return self.streamed(x, batch, graphs)
        q = self.query(x).reshape(-1, self.width, self.heads)
        k = self.key(x).reshape_as(q)
        v = self.value(x).reshape_as(q)
        # Each graph is independent. Vectorized disjoint batching, no mixing
        # across the PPI tissues in a physical minibatch.
        counts = torch.bincount(batch, minlength=graphs)
        if graphs == 1:
            attention = torch.einsum("nmh,ndh->mdh", k / math.sqrt(x.shape[0]), v)
            return torch.einsum("nmh,mdh->ndh", q, attention.softmax(0)).mean(-1)
        membership = F.one_hot(batch, graphs).to(k.dtype)
        attention = torch.einsum("ng,nmh,ndh->gmdh", membership, k, v)
        attention = attention / counts.to(k.dtype).sqrt()[:, None, None, None]
        queries = torch.einsum("ng,nmh->gnmh", membership, q)
        return torch.einsum("gnmh,gmdh->ndh", queries, attention.softmax(1)).mean(-1)

    def streamed(self, x, batch, graphs):
        """Exact two-pass global attention; chunking does not sample nodes."""
        counts = torch.bincount(batch, minlength=graphs).to(x.dtype)
        metric = x.new_zeros(graphs, self.width, self.width, self.heads)

        def accumulate(features, groups):
            k = self.key(features).reshape(-1, self.width, self.heads)
            v = self.value(features).reshape_as(k)
            membership = F.one_hot(groups, graphs).to(k.dtype)
            return torch.einsum("gnmh,ndh->gmdh", membership.T[..., None, None] * k[None], v)

        def apply(features, groups, attention):
            q = self.query(features).reshape(-1, self.width, self.heads)
            membership = F.one_hot(groups, graphs).to(q.dtype)
            queries = torch.einsum("ng,nmh->gnmh", membership, q)
            return torch.einsum("gnmh,gmdh->ndh", queries, attention).mean(-1)

        for start in range(0, x.shape[0], self.chunk_size):
            features, groups = (
                x[start : start + self.chunk_size],
                batch[start : start + self.chunk_size],
            )
            term = (
                checkpoint(accumulate, features, groups, use_reentrant=False)
                if torch.is_grad_enabled()
                else accumulate(features, groups)
            )
            metric = metric + term
        attention = (metric / counts.sqrt()[:, None, None, None]).softmax(1)
        outputs = []
        for start in range(0, x.shape[0], self.chunk_size):
            features, groups = (
                x[start : start + self.chunk_size],
                batch[start : start + self.chunk_size],
            )
            outputs.append(
                checkpoint(apply, features, groups, attention, use_reentrant=False)
                if torch.is_grad_enabled()
                else apply(features, groups, attention)
            )
        return torch.cat(outputs)


def graph_normalization(x, edges):
    """Static symmetric normalization, shared by every propagation layer."""
    degree = torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
    ones = degree.new_ones(edges.shape[1])
    degree.index_add_(0, edges[0], ones)
    degree.index_add_(0, edges[1], ones)
    inverse = degree.rsqrt()
    return degree, inverse[edges[0]] * inverse[edges[1]]


def normalized_graph_step(x, edges, chunk, normalization=None):
    """Symmetric GCN normalization with unit self loops, all physical edges."""
    degree, weights = graph_normalization(x, edges) if normalization is None else normalization
    out = x / degree[:, None]
    for start in range(0, edges.shape[1], chunk or max(edges.shape[1], 1)):
        ends = edges[:, start : start + (chunk or edges.shape[1])]
        weight = weights[start : start + ends.shape[1]]
        out.index_add_(0, ends[0], x[ends[1]] * weight[:, None])
        out.index_add_(0, ends[1], x[ends[0]] * weight[:, None])
    return out


class AggregationClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        classes,
        *,
        arm,
        selection_config,
        hidden_channels,
        layers,
        heads,
        dropout=0.2,
        activation_checkpoint=True,
        edge_chunk_size=None,
        dual_sa_layers=1,
        dual_alpha=0.1,
        gram_implementation="reference",
        **architecture,
    ):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown comparison model: {arm}")
        if selection_config.get("condition") != "full":
            raise ValueError("comparison requires full original edge support")
        if layers < 1 or dual_sa_layers < 1 or not 0 <= dual_alpha <= 1:
            raise ValueError("invalid depth or DUALFormer residual coefficient")
        self.arm, self.width, self.heads = arm, hidden_channels, heads
        self.depth, self.dropout = layers, dropout
        self.activation_checkpoint = activation_checkpoint
        self.edge_chunk_size = edge_chunk_size
        if gram_implementation not in {"reference", "fused"}:
            raise ValueError("unknown Gram implementation")
        self.gram_implementation = gram_implementation
        # Instantiate the common endpoints before any family-specific RNG use.
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, classes)
        self.layers = nn.ModuleList()
        self.energy_readouts = nn.ParameterList()
        self.diagonal_only = arm.startswith("incidence_diagonal")
        self.energy_intervention = None
        self.diagnostic_collector = None
        self.dual_alpha = dual_alpha
        if arm == "gatv2":
            from torch_geometric.nn import GATv2Conv

            if hidden_channels % heads:
                raise ValueError("GATv2 total width must be divisible by heads")
            self.layers.extend(
                GATv2Conv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=0.0,
                    add_self_loops=False,  # Supplied once in the cached edge support below.
                    share_weights=False,
                    residual=False,
                )
                for _ in range(layers)
            )
        elif arm == "gcn":
            from torch_geometric.nn import GCNConv

            # Cache graph normalization once in forward, shared by all layers.
            self.layers.extend(
                GCNConv(hidden_channels, hidden_channels, normalize=False, add_self_loops=False)
                for _ in range(layers)
            )
        elif arm == "graphsage":
            from torch_geometric.nn import SAGEConv

            self.layers.extend(
                SAGEConv(
                    hidden_channels,
                    hidden_channels,
                    aggr="mean",
                    normalize=False,
                    root_weight=True,
                    project=False,
                )
                for _ in range(layers)
            )
        elif arm.startswith("dualformer"):
            self.layers.extend(
                DualAttention(hidden_channels, heads, edge_chunk_size)
                for _ in range(dual_sa_layers)
            )
            self.norms = nn.ModuleList(
                nn.LayerNorm(hidden_channels) for _ in range(dual_sa_layers + 1)
            )
        else:
            old_arm, energy = ARMS[arm]
            # Build the original V5 initialization, then retain only its operators.
            # Do not use the historical edge-selection classifier: its fixed
            # per-head/dynamic contract intentionally excludes these new controls.
            selected = selection_configuration(selection_config)
            c_config = conductance_contract(arm)
            architecture.update(
                conductance_mode=c_config["conductance_mode"],
                conductance_heads=c_config["conductance_heads"],
                conductance_backend="optimization",
                conductance_generator="optimized",
                propagation_normalization="row",
                propagation_filter="linear",
            )
            architecture.setdefault("solver_cost_scaling", "width_scaled")
            original = GraphConditionedConductanceNodeClassifier(
                in_channels,
                classes,
                hidden_channels=hidden_channels,
                layers=layers,
                heads=heads,
                dropout=dropout,
                activation_checkpoint=activation_checkpoint,
                edge_chunk_size=edge_chunk_size,
                **architecture,
            )
            # Only retain the actual operator modules. The old encoder, norms,
            # SwiGLU FFNs and both external residual paths are absent.
            lift = {
                "baseline": "none",
                "linear_lift": "linear",
                "pre_lift": "pre",
                "post_lift": "post",
            }[old_arm]
            self.layers.extend(
                IncidenceOperator(operator, selected, layer=index, lift=lift, bilinear=False)
                for index, operator in enumerate(original.operators)
            )
            if energy:
                for depth in range(1, layers + 1):
                    self.energy_readouts.append(
                        nn.Parameter(
                            torch.zeros(
                                heads,
                                depth if self.diagonal_only else depth * (depth + 1) // 2,
                                hidden_channels // heads,
                            )
                        )
                    )

    def contract(self):
        dual = self.arm.startswith("dualformer")
        return {
            "model": self.arm,
            "conductance": conductance_contract(self.arm),
            "external_residual": False,
            "external_ffn": False,
            "common_encoder": "linear",
            "common_decoder": "linear",
            "hidden_channels": self.width,
            "heads": None if self.arm in {"gcn", "graphsage"} else self.heads,
            "graph_propagation_layers": self.depth,
            "dual_sa_layers": len(self.layers) if dual else 0,
            "intrinsic_dual_residual": self.arm == "dualformer",
            "intrinsic_dual_layernorm": dual,
            "dual_alpha": self.dual_alpha if dual else None,
            "activation": "ReLU",
            "dropout": self.dropout,
            "parameter_matched": False,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "energy": (
                "diagonal local Gram channels"
                if self.diagonal_only
                else "diagonal and cross-depth local Gram channels"
            )
            if len(self.energy_readouts)
            else None,
            "depth_states_are_distance_shells": False,
            "lift": ARMS[self.arm][0] if self.arm.startswith("incidence") else None,
            "energy_coordinates": "pre-lift projected values"
            if len(self.energy_readouts)
            else None,
            "gatv2": {
                "implementation": "torch_geometric.nn.GATv2Conv",
                "head_width": self.width // self.heads,
                "attention_dropout": 0.0,
                "self_loops": True,
                "share_weights": False,
                "residual": False,
                "edge_chunking": False,
            }
            if self.arm == "gatv2"
            else None,
            "gcn": {
                "implementation": "torch_geometric.nn.GCNConv",
                "normalization": "symmetric D^-1/2 (A+I) D^-1/2 cached per input graph",
                "self_loops": True,
                "edge_chunking": False,
            }
            if self.arm == "gcn"
            else None,
            "graphsage": {
                "implementation": "torch_geometric.nn.SAGEConv",
                "aggregation": "mean",
                "root_weight": True,
                "neighbor_self_loops": False,
                "project": False,
                "normalize": False,
                "edge_chunking": False,
            }
            if self.arm == "graphsage"
            else None,
            "dual_upstream_commit": "68fbdaf007af2f7d409cd435c4c48dd0e3155510" if dual else None,
            "comparison_scope": "common training protocol, not published tuned score reproduction",
        }

    def clear_auxiliary_cache(self):
        # Shared trainer interface; this suite has no auxiliary objectives.
        # Live C is local to forward_with_state, never stored across forwards.
        return None

    def auxiliary_loss(self, targets=None):
        if targets is not None:
            raise ValueError("comparison has no corruption objective")
        zero = self.decoder.weight.new_zeros(())
        return {"l0": zero, "negative": zero}

    @contextmanager
    def intervention(self, name):
        """Read-only, whole-network evaluation intervention; restore even on error."""
        if self.training or torch.is_grad_enabled() or not self.arm.startswith("incidence"):
            raise RuntimeError("incidence interventions require eval and no_grad")
        allowed = {
            "energy_off",
            "cross_off",
            "diagonal_off",
            "c_ones",
            "c_mean",
            "c_shuffle",
            "lift_second_off",
        }
        if name not in allowed:
            raise ValueError(f"unknown intervention {name}")
        prior = (
            self.energy_intervention,
            [(op.estimator.override, op.disable_lift_channel) for op in self.layers],
        )
        try:
            if name in {"energy_off", "cross_off", "diagonal_off"}:
                self.energy_intervention = name
            for op in self.layers:
                if name.startswith("c_"):
                    op.estimator.override = name[2:]
                elif name == "lift_second_off":
                    op.disable_lift_channel = True
            yield
        finally:
            self.energy_intervention = prior[0]
            for op, (override, lift_disabled) in zip(self.layers, prior[1], strict=True):
                op.estimator.override, op.disable_lift_channel = override, lift_disabled
            self.clear_auxiliary_cache()

    def forward(self, graph):
        if self.diagnostic_collector is not None and (self.training or torch.is_grad_enabled()):
            raise RuntimeError(
                "mechanism capture requires eval/no_grad; never retain training graphs"
            )
        x, edges = graph.x, graph.incidence_edge_index
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            graphs = 1
        else:
            graphs = graph._v5_num_graphs
        h = self.encoder(x)
        if self.arm in {"gatv2", "gcn", "graphsage"}:
            # The canonical incidence support contains each undirected edge once.
            # Cache both directions and exactly one self-loop/node. Avoid
            # rebuilding static graph support in every layer/epoch.
            cache_key = (
                "_comparison_gatv2_edges"
                if self.arm == "gatv2"
                else "_comparison_local_" + self.arm
            )
            cached = getattr(graph, cache_key, None)
            signature = (edges._version, h.shape[0], h.device)
            if cached is None or cached[0] is not edges or cached[1] != signature:
                nonloops = edges[:, edges[0] != edges[1]]
                nodes = torch.arange(h.shape[0], device=h.device)
                directed = torch.cat((nonloops, nonloops.flip(0)), dim=1)
                if self.arm != "graphsage":
                    directed = torch.cat((directed, torch.stack((nodes, nodes))), dim=1)
                weights = None
                if self.arm == "gcn":
                    from torch_geometric.nn.conv.gcn_conv import gcn_norm

                    directed, weights = gcn_norm(
                        directed,
                        num_nodes=h.shape[0],
                        add_self_loops=False,
                        dtype=torch.float32,
                    )
                cached = (edges, signature, directed, weights)
                setattr(graph, cache_key, cached)
            directed = cached[2]
            weights = cached[3]
            for layer in self.layers:

                def step(value, layer=layer):
                    output = (
                        layer(value, directed, weights)
                        if self.arm == "gcn"
                        else layer(value, directed)
                    )
                    return F.dropout(F.relu(output), self.dropout, self.training)

                h = (
                    checkpoint(step, h, use_reentrant=False)
                    if self.activation_checkpoint and torch.is_grad_enabled()
                    else step(h)
                )
            return self.decoder(h)
        if self.arm.startswith("dualformer"):
            h = F.dropout(F.relu(self.norms[0](h)), self.dropout, self.training)
            for index, layer in enumerate(self.layers):

                def step(value, layer=layer, norm=self.norms[index + 1]):
                    result = layer(value, batch, graphs)
                    if self.arm == "dualformer":
                        result = self.dual_alpha * result + (1 - self.dual_alpha) * value
                    return F.dropout(F.relu(norm(result)), self.dropout, self.training)

                h = (
                    checkpoint(step, h, use_reentrant=False)
                    if self.activation_checkpoint and torch.is_grad_enabled()
                    else step(h)
                )
            cached = getattr(graph, "_comparison_sgc_normalization", None)
            signature = (edges._version, h.shape[0], h.device)
            if cached is None or cached[0] is not edges or cached[1] != signature:
                cached = (edges, signature, graph_normalization(h, edges))
                graph._comparison_sgc_normalization = cached
            normalization = cached[2]
            for _ in range(self.depth):
                h = normalized_graph_step(h, edges, self.edge_chunk_size, normalization)
            return self.decoder(F.dropout(h, self.dropout, self.training))
        kwargs = {
            name: getattr(graph, name, None)
            for name in (
                "full_degree",
                "graph_structure",
                "edge_normalization_weight",
                "sampling_correction",
                "edge_relation_id",
            )
        }
        kwargs["edge_selection_topology"] = graph.edge_selection_topology
        with torch.autocast(device_type=x.device.type, enabled=False):
            kwargs["static_context"] = _static_graph_context(
                x.float(), edges, batch, graphs, kwargs["full_degree"], kwargs["graph_structure"]
            )
        history = [h]
        for index, operator in enumerate(self.layers):

            def step(*past, operator=operator, index=index):
                current = past[-1]
                output = operator.forward_with_state(current, edges, batch, graphs, **kwargs)
                value = output.message
                if len(self.energy_readouts) or self.diagnostic_collector is not None:
                    projected = torch.einsum(
                        "knd,hdw->knhw", torch.stack(past), operator.value_weight
                    )
                    # Exactly the weight used by diffusion, including correction once.
                    metric = output.effective_weight
                    pairs = (
                        torch.arange(len(past), device=value.device).expand(2, -1)
                        if self.diagonal_only
                        else torch.triu_indices(len(past), len(past), device=value.device)
                    )
                    diagonal = pairs[0] == pairs[1]
                    statistics = None
                    if (
                        self.gram_implementation == "reference"
                        or self.diagnostic_collector is not None
                    ):
                        # Diagnostics observe a separate Gram tensor. They never choose
                        # the numerical path used for the declared prediction branch.
                        statistics = local_gram(
                            projected.float(),
                            edges,
                            metric.float(),
                            self.edge_chunk_size,
                            diagonal_only=self.diagonal_only,
                        )
                    if len(self.energy_readouts):
                        mask = None
                        if self.energy_intervention is not None:
                            if self.training or torch.is_grad_enabled():
                                raise RuntimeError("energy interventions are evaluation-only")
                            mask = (
                                diagonal
                                if self.energy_intervention == "cross_off"
                                else ~diagonal
                                if self.energy_intervention == "diagonal_off"
                                else torch.zeros_like(diagonal)
                            )
                        readout = self.energy_readouts[index].float()
                        if self.gram_implementation == "fused":
                            from .gram import GramReadout

                            # Normalize outside custom autograd so unsqueeze's backward
                            # restores the real shared/fixed estimator's (E,) gradient.
                            metric_2d = metric[:, None] if metric.ndim == 1 else metric
                            if mask is not None:
                                readout = readout * mask[None, :, None]
                            chunk = max(
                                1, (self.edge_chunk_size or max(edges.shape[1], 1)) // len(past)
                            )
                            with torch.autocast(device_type=x.device.type, enabled=False):
                                extra = GramReadout.apply(
                                    projected.float(),
                                    metric_2d.float(),
                                    readout,
                                    edges,
                                    chunk,
                                    self.diagonal_only,
                                )
                        else:
                            used = statistics if mask is None else statistics * mask
                            extra = torch.einsum("nhp,hpd->nhd", used, readout)
                        branch = F.linear(
                            extra.to(value.dtype).flatten(1), operator.output_projection.weight
                        )
                    else:
                        branch = torch.zeros_like(value)
                    if self.diagnostic_collector is not None:
                        self.diagnostic_collector.record(
                            index, metric, statistics, diagonal, value, branch
                        )
                    value = value + branch
                return F.dropout(F.relu(value), self.dropout, self.training)

            h = (
                checkpoint(step, *history, use_reentrant=False)
                if self.activation_checkpoint and torch.is_grad_enabled()
                else step(*history)
            )
            history.append(h)
        return self.decoder(h)
