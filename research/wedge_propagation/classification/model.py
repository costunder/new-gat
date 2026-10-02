"""Two-layer, seed-packed classifiers for the independent Experiment 4 contract.

Every seed owns its projection, gate/node MLP and branch parameters. Forward
operations batch the seed/node/channel axes; only the fixed two layers and exact
path chunks use Python loops. Path gates and scatter messages are checkpointed
per chunk, while C mean and kappa are reduced over ALL paths/nodes. A chunk never
changes the graph, support, feature dimension or training sample count.

Dropout uses a vectorized, counter-hashed Bernoulli stream keyed by dataset,
initialization seed, epoch and layer. It is independent of condition, learning
rate, packing order and chunk size. All matrix initialization is Xavier uniform
from independently named CPU generators. No synthetic checkpoint is imported.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

CONDITIONS = (
    "mlp",
    "first_order",
    "polynomial_2",
    "fixed_wedge",
    "learned_wedge_raw",
    "learned_wedge_rms",
    "fixed_wedge_node_mlp",
    "standard_gcn",
)
_SECOND = frozenset(CONDITIONS[2:7])
_LEARNED = frozenset(("learned_wedge_raw", "learned_wedge_rms"))
_INTERVENTIONS = frozenset(
    (
        "c_identity",
        "c_position_shuffle",
        "second_branch_remove",
        "random_physical_edge_pair_correspondence",
    )
)
_Q_CORRECTIONS: dict[tuple, tuple] = {}


def _integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _named_seed(dataset: str, seed: int, name: str) -> int:
    key = f"wedge-classification-v1|{dataset}|{seed}|{name}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & ((1 << 63) - 1)


def _matrix(seeds: tuple[int, ...], dataset: str, name: str, rows: int, cols: int) -> nn.Parameter:
    value = torch.empty(len(seeds), rows, cols)
    bound = math.sqrt(6.0 / (rows + cols))
    with torch.no_grad():
        for index, seed in enumerate(seeds):
            generator = torch.Generator().manual_seed(_named_seed(dataset, seed, name))
            value[index].uniform_(-bound, bound, generator=generator)
    return nn.Parameter(value)


class _VectorGate(nn.Module):
    def __init__(
        self, seeds: tuple[int, ...], dataset: str, layer: int, features: int, hidden: int
    ):
        super().__init__()
        prefix = f"gate.{layer}"
        self.w1 = _matrix(seeds, dataset, prefix + ".w1", 4 * features, hidden)
        self.b1 = nn.Parameter(torch.zeros(len(seeds), hidden))
        self.w2 = _matrix(seeds, dataset, prefix + ".w2", hidden, 1)

    def raw_weights(self, z: Tensor, paths: Tensor, sigma: Tensor) -> Tensor:
        g1 = (z.index_select(1, paths[1]) - z.index_select(1, paths[0])) / sigma[:, None, None]
        g2 = (z.index_select(1, paths[2]) - z.index_select(1, paths[1])) / sigma[:, None, None]
        phi = torch.cat(
            (g1.abs() + g2.abs(), g1 * g2, (g2 - g1).abs(), (g1.abs() - g2.abs()).square()), dim=-1
        )
        hidden = torch.bmm(phi, self.w1) + self.b1[:, None, :]
        score = torch.bmm(hidden.relu(), self.w2).squeeze(-1)
        return score.tanh().exp()


class _NodeMLP(nn.Module):
    def __init__(
        self, seeds: tuple[int, ...], dataset: str, layer: int, features: int, hidden: int
    ):
        super().__init__()
        prefix = f"node_mlp.{layer}"
        self.w1 = _matrix(seeds, dataset, prefix + ".w1", features, 2 * hidden)
        self.b1 = nn.Parameter(torch.zeros(len(seeds), 2 * hidden))
        self.w2 = _matrix(seeds, dataset, prefix + ".w2", 2 * hidden, features)

    def forward(self, z: Tensor) -> Tensor:
        return torch.bmm((torch.bmm(z, self.w1) + self.b1[:, None, :]).relu(), self.w2)


def _lbar(graph: Any, z: Tensor) -> Tensor:
    scaled = z * graph.sd[None, :, None]
    difference = scaled.index_select(1, graph.edges[1]) - scaled.index_select(1, graph.edges[0])
    out = z.new_zeros(z.shape)
    out = out.index_add(1, graph.edges[0], -difference)
    out = out.index_add(1, graph.edges[1], difference)
    return 0.5 * out * graph.sd[None, :, None]


def _raw_l(graph: Any, z: Tensor, coefficient: Tensor | None = None) -> Tensor:
    difference = z.index_select(1, graph.edges[1]) - z.index_select(1, graph.edges[0])
    if coefficient is not None:
        difference = difference * coefficient[None, :, None]
    out = z.new_zeros(z.shape).index_add(1, graph.edges[0], -difference)
    return out.index_add(1, graph.edges[1], difference)


def _qbar(graph: Any, z: Tensor) -> Tensor:
    """Exact all-wedge Q identity, retaining SQ coordinates on both sides.

    The degree correction can be negative and is NEVER clipped: only the sum
    equals the PSD A.T A. A weakref/version-aware static cache shares coefficients
    across packed conditions without mutating the frozen source GraphData.
    """
    edges, degree = graph.edges, graph.degree
    versions = tuple(None if torch.is_inference(t) else t._version for t in (edges, degree))
    key = (id(edges), id(degree), z.dtype, z.device)
    cached = _Q_CORRECTIONS.get(key)
    if (
        None not in versions
        and cached is not None
        and cached[2] == versions
        and cached[0]() is edges
        and cached[1]() is degree
    ):
        correction = cached[3]
    else:
        correction = (degree[edges[0]] + degree[edges[1]] - 4).to(dtype=z.dtype)
        if None not in versions:

            def discard(reference):
                current = _Q_CORRECTIONS.get(key)
                if current is not None and (current[0] is reference or current[1] is reference):
                    del _Q_CORRECTIONS[key]

            _Q_CORRECTIONS[key] = (
                weakref.ref(edges, discard),
                weakref.ref(degree, discard),
                versions,
                correction,
            )
    scaled = z * graph.sq[None, :, None]
    result = _raw_l(graph, _raw_l(graph, scaled)) + _raw_l(graph, scaled, correction)
    return result * graph.sq[None, :, None] / 3.0


def _row_message(z: Tensor, indices: Tensor, coefficients: Tensor, c: Tensor) -> Tensor:
    # Row supports may contain a repeated vertex for two physical edges sharing
    # an endpoint. Summing coefficients before the transpose is automatic here.
    selected = z[:, indices, :]
    row_value = (selected * coefficients[None, :, :, None]).sum(1)
    flow = c[:, :, None] * row_value
    weighted = coefficients[None, :, :, None] * flow[:, None, :, :]
    return z.new_zeros(z.shape).index_add(1, indices.reshape(-1), weighted.flatten(1, 2))


class PackedClassifier(nn.Module):
    """Eight controlled conditions with independent leading seed parameters.

    ``graph`` contains one complete citation graph and precomputed ``sd``/``sq``
    coordinates. A forward returns ``[S,N,K]`` logits and two dictionaries of
    detached scalar/statistic tensors. ``diagnostics=True`` additionally returns
    detached projected Z, path C and unscaled branch messages for frozen audits.
    These diagnostic tensors never add another path to the training loss.
    """

    def __init__(
        self,
        condition: str,
        input_dim: int,
        classes: int,
        seeds: Sequence[int],
        hidden: int = 64,
        gate_hidden: int = 64,
        dropout: float = 0.5,
        path_chunk: int | None = None,
        checkpoint_paths: bool = True,
        dataset_name: str = "unspecified",
    ):
        super().__init__()
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}")
        self.condition = condition
        self.input_dim = _integer(input_dim, "input_dim")
        self.classes = _integer(classes, "classes", minimum=2)
        self.hidden = _integer(hidden, "hidden")
        self.gate_hidden = _integer(gate_hidden, "gate_hidden")
        self.seeds = tuple(seeds)
        if (
            not self.seeds
            or any(isinstance(s, bool) or not isinstance(s, Integral) for s in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("seeds must be a nonempty distinct integer sequence")
        self.seeds = tuple(int(s) for s in self.seeds)
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("dataset_name must be nonempty")
        self.dataset_name = dataset_name
        self.dropout = float(dropout)
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        self.path_chunk = None if path_chunk is None else _integer(path_chunk, "path_chunk")
        if not isinstance(checkpoint_paths, bool):
            raise TypeError("checkpoint_paths must be bool")
        self.checkpoint_paths = checkpoint_paths
        self.projections = nn.ParameterList(
            (
                _matrix(self.seeds, dataset_name, "projection.0", self.input_dim, self.hidden),
                _matrix(self.seeds, dataset_name, "projection.1", self.hidden, self.classes),
            )
        )
        self.gates = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        if condition in _LEARNED:
            self.gates.extend(
                _VectorGate(self.seeds, dataset_name, layer, f, self.gate_hidden)
                for layer, f in enumerate((self.hidden, self.classes))
            )
        if condition == "fixed_wedge_node_mlp":
            self.node_mlps.extend(
                _NodeMLP(self.seeds, dataset_name, layer, f, self.gate_hidden)
                for layer, f in enumerate((self.hidden, self.classes))
            )
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()
        if condition == "first_order" or condition in _SECOND:
            initial = 0.0 if condition == "first_order" else math.log(3.0)
            self.u.extend(nn.Parameter(torch.full((len(self.seeds),), initial)) for _ in range(2))
        if condition in _SECOND:
            self.v.extend(
                nn.Parameter(torch.full((len(self.seeds),), -math.log(2.0))) for _ in range(2)
            )
        keys = [
            [_named_seed(dataset_name, s, f"dropout.{layer}") & 0xFFFFFFFF for s in self.seeds]
            for layer in range(2)
        ]
        self.register_buffer(
            "dropout_keys", torch.tensor(keys, dtype=torch.int64), persistent=False
        )
        self.register_buffer("wedge_coefficients", torch.tensor((1.0, -2.0, 1.0)), persistent=False)
        self._manifest_checks: dict[tuple, tuple] = {}

    @property
    def parameters_per_seed(self) -> int:
        return sum(p.numel() for p in self.parameters()) // len(self.seeds)

    @property
    def trainable_parameters_per_seed(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) // len(self.seeds)

    def parameter_count_per_seed(self) -> int:
        return self.parameters_per_seed

    def weight_decay_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Adam coupled L2 on actual matrix weights; biases/u/v are excluded."""
        decay = float(weight_decay)
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        weights, other = [], []
        for name, parameter in self.named_parameters():
            is_weight = name.startswith("projections.") or name.endswith((".w1", ".w2"))
            (weights if is_weight else other).append(parameter)
        groups = [{"params": weights, "weight_decay": decay}]
        if other:
            groups.append({"params": other, "weight_decay": 0.0})
        return groups

    def _dropout(self, h: Tensor, epoch: int, layer: int) -> Tensor:
        if not self.training or self.dropout == 0:
            return h
        # UInt32 arithmetic represented in int64 avoids device-specific unsigned
        # shifts. The integer threshold is identical on CPU and CUDA.
        counter = torch.arange(h.shape[1] * h.shape[2], device=h.device, dtype=torch.int64)[None]
        value = (
            counter + self.dropout_keys[layer, :, None] + (epoch * 0x9E3779B9 & 0xFFFFFFFF)
        ) & 0xFFFFFFFF
        value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
        value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
        value = value ^ (value >> 16)
        keep = value < int((1.0 - self.dropout) * (1 << 32))
        return h * keep.reshape(h.shape).to(h.dtype) / (1.0 - self.dropout)

    def _chunk_ranges(self, count: int):
        size = count if self.path_chunk is None else self.path_chunk
        if count:
            yield from ((start, min(start + size, count)) for start in range(0, count, size))

    def _run_chunk(self, function, *args):
        if self.checkpoint_paths and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False)
        return function(*args)

    def _sigma(self, graph: Any, z: Tensor) -> Tensor:
        if self.condition != "learned_wedge_rms" or graph.edges.shape[1] == 0:
            return z.new_ones(z.shape[0])
        difference = z.index_select(1, graph.edges[1]) - z.index_select(1, graph.edges[0])
        energy = difference.square().mean(dim=(1, 2))
        # Replace before sqrt to avoid sqrt(0)'s infinite backward derivative.
        return torch.where(energy > 0, energy, torch.ones_like(energy)).sqrt()

    def _gate(self, graph: Any, layer: int, z: Tensor, sigma: Tensor) -> Tensor:
        values = [
            self._run_chunk(self.gates[layer].raw_weights, z, graph.paths[:, a:b], sigma)
            for a, b in self._chunk_ranges(graph.paths.shape[1])
        ]
        if not values:
            return z.new_empty((z.shape[0], 0))
        raw = torch.cat(values, dim=1)
        return raw / raw.mean(dim=1, keepdim=True)

    @staticmethod
    def _kappa(graph: Any, c: Tensor) -> Tensor:
        if graph.paths.shape[1] == 0:
            return c.new_ones(c.shape[0])
        diagonal = c.new_zeros((c.shape[0], graph.x.shape[0]))
        diagonal = diagonal.index_add(1, graph.paths[0], c)
        diagonal = diagonal.index_add(1, graph.paths[1], 4 * c)
        diagonal = diagonal.index_add(1, graph.paths[2], c)
        q = graph.qdiag
        ratio = diagonal / torch.where(q > 0, q, torch.ones_like(q))[None, :]
        # All q=0 nodes have diagonal=0; paths exist, so a positive maximum exists.
        return torch.amax(ratio, dim=1)

    def _weighted_message(
        self, graph: Any, z: Tensor, c: Tensor, kappa: Tensor, rows: dict[str, Tensor] | None = None
    ) -> Tensor:
        scaled = z * graph.sq[None, :, None]
        result = z.new_zeros(z.shape)
        count = graph.paths.shape[1]
        for start, stop in self._chunk_ranges(count):
            if rows is None:
                indices = graph.paths[:, start:stop]
                coefficients = self.wedge_coefficients[:, None].expand(3, stop - start)
            else:
                indices = rows["indices"][:, start:stop]
                coefficients = rows["coefficients"][:, start:stop]
            result = result + self._run_chunk(
                _row_message, scaled, indices, coefficients, c[:, start:stop]
            )
        return result * graph.sq[None, :, None] / (3.0 * kappa[:, None, None])

    def apply_branch(
        self,
        graph: Any,
        z: Tensor,
        c: Tensor,
        kappa: Tensor,
        random_rows: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Apply fixed C/kappa to a sparse probe, without rerunning the gate.

        This supports frozen sparse norm estimates for hold-kappa and random
        correspondence diagnostics. The random operator has no 3-support bound.
        """
        if c.shape != (len(self.seeds), graph.paths.shape[1]) or kappa.shape != (len(self.seeds),):
            raise ValueError("c/kappa must have shape [S,P] and [S]")
        return self._weighted_message(graph, z, c, kappa, random_rows)

    def _coefficients(self, layer: int, z: Tensor) -> tuple[Tensor, Tensor]:
        if self.condition == "first_order":
            return self.u[layer].sigmoid(), z.new_zeros(z.shape[0])
        if self.condition in _SECOND:
            t, r = self.u[layer].sigmoid(), self.v[layer].sigmoid()
            return t * (1 - r), t * r
        return z.new_zeros(z.shape[0]), z.new_zeros(z.shape[0])

    def _validate_manifest(self, graph: Any, tensors: tuple[Tensor, ...], kind: str) -> None:
        """Validate each immutable audit manifest once, outside steady training.

        The first frozen audit may copy its manifest to CPU for exact validation.
        Cached validation uses weak references and mutation versions, preventing
        stale acceptance or retaining all GPU manifest tensors after evaluation.
        The evaluation writer additionally proves physical pair provenance.
        """
        versions = tuple(None if torch.is_inference(t) else t._version for t in tensors)
        key = (kind, tuple(id(t) for t in tensors), graph.paths.shape[1], graph.x.shape[0])
        cached = self._manifest_checks.get(key)
        if (
            None not in versions
            and cached is not None
            and cached[1] == versions
            and all(ref() is tensor for ref, tensor in zip(cached[0], tensors, strict=True))
        ):
            return
        cpu = tuple(t.detach().cpu() for t in tensors)
        if kind == "permutation":
            if not torch.equal(cpu[0].sort().values, torch.arange(graph.paths.shape[1])):
                raise ValueError("permutation must contain every path exactly once")
        else:
            indices, coefficients = cpu
            if bool(((indices < 0) | (indices >= graph.x.shape[0])).any()):
                raise ValueError("random rows contain invalid node indices")
            if not bool(torch.isfinite(coefficients).all()):
                raise ValueError("random coefficients must be finite")
            norm_squared = coefficients.square().sum(0)
            # Coalesce coincident endpoints before the norm: a shared-center
            # pair has three distinct nodes but still four stored entries.
            for first in range(4):
                for second in range(first + 1, 4):
                    norm_squared = norm_squared + 2 * coefficients[first] * coefficients[second] * (
                        indices[first] == indices[second]
                    )
            if not torch.allclose(
                norm_squared, torch.full_like(norm_squared, 6), rtol=2e-6, atol=2e-6
            ):
                raise ValueError("random row norm must equal sqrt(6) after node coalescing")
            if not torch.allclose(
                coefficients.sum(0), torch.zeros_like(norm_squared), rtol=0, atol=2e-6
            ):
                raise ValueError("random physical incidence rows must sum to zero")
        if None not in versions:

            def discard(_reference):
                self._manifest_checks.pop(key, None)

            self._manifest_checks[key] = (tuple(weakref.ref(t, discard) for t in tensors), versions)

    @staticmethod
    def _stats(c: Tensor | None) -> dict[str, Tensor | None]:
        if c is None or c.shape[1] == 0:
            return {name: None for name in ("c_mean", "c_std", "c_min", "c_max")}
        return {
            "c_mean": c.mean(1).detach(),
            "c_std": c.std(1, correction=0).detach(),
            "c_min": c.amin(1).detach(),
            "c_max": c.amax(1).detach(),
        }

    def probe_layer(
        self,
        graph: Any,
        layer_index: int,
        z: Tensor,
        input_scale: float = 1.0,
        intervention: str | None = None,
        manifest: dict | None = None,
        kappa_mode: str = "recompute",
        diagnostics: bool = True,
        _l_message: Tensor | None = None,
    ):
        """Evaluate a fixed projected Z; return normalized T(Z) and layer details.

        C_reference and kappa_reference are calculated at the CURRENT supplied Z,
        including layer 1 after an upstream intervention. They never reuse clean
        forward layer weights. Random correspondence always holds that kappa and
        the true-wedge SQ coordinates, as required by the diagnostic contract.
        """
        layer = _integer(layer_index, "layer_index", minimum=0)
        if layer > 1:
            raise ValueError("layer_index must be 0 or 1")
        if intervention not in _INTERVENTIONS and intervention is not None:
            raise ValueError(f"unknown intervention {intervention!r}")
        if intervention is not None and self.condition not in _LEARNED:
            raise ValueError("frozen C interventions apply only to learned gate conditions")
        if kappa_mode not in ("recompute", "hold", "hold_each_layers_preintervention_kappa"):
            raise ValueError("unknown kappa_mode")
        amplitude = float(input_scale)
        if not math.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("input_scale must be finite and positive")
        expected_f = self.hidden if layer == 0 else self.classes
        if z.ndim != 3 or z.shape != (len(self.seeds), graph.x.shape[0], expected_f):
            raise ValueError("z must have shape [packed_seeds,nodes,layer_output_channels]")
        z = z * amplitude
        alpha, beta = self._coefficients(layer, z)
        sigma = self._sigma(graph, z)
        c_reference = c = None
        kappa = reference_kappa = z.new_ones(z.shape[0])
        rows = None
        l_message = _l_message
        if l_message is None and (diagnostics or self.condition == "polynomial_2"):
            l_message = (
                _lbar(graph, z)
                if self.condition == "first_order" or self.condition in _SECOND
                else None
            )
        if self.condition in _LEARNED:
            c_reference = self._gate(graph, layer, z, sigma)
            reference_kappa = self._kappa(graph, c_reference)
            c = c_reference
            if intervention == "c_identity":
                c = torch.ones_like(c_reference)
            elif intervention == "c_position_shuffle":
                if manifest is None or "permutation" not in manifest:
                    raise ValueError("c_position_shuffle requires manifest['permutation']")
                permutation = manifest["permutation"]
                if permutation.dtype != torch.long or permutation.shape != (graph.paths.shape[1],):
                    raise ValueError("permutation must be Long[P]")
                if permutation.device != z.device:
                    raise ValueError("permutation must be on the model device")
                self._validate_manifest(graph, (permutation,), "permutation")
                c = c_reference.index_select(1, permutation)
            elif intervention == "random_physical_edge_pair_correspondence":
                if manifest is None or "random_rows" not in manifest:
                    raise ValueError("random intervention requires manifest['random_rows']")
                rows = manifest["random_rows"]
                if (
                    rows["indices"].shape != (4, graph.paths.shape[1])
                    or rows["coefficients"].shape != rows["indices"].shape
                    or rows["indices"].dtype != torch.long
                ):
                    raise ValueError("random rows must contain indices/coefficients [4,P]")
                if (
                    rows["indices"].device != z.device
                    or rows["coefficients"].device != z.device
                    or rows["coefficients"].dtype != z.dtype
                ):
                    raise ValueError("random row tensors must match model device/dtype")
                self._validate_manifest(graph, (rows["indices"], rows["coefficients"]), "rows")
            hold = kappa_mode != "recompute" or rows is not None
            kappa = reference_kappa if hold else self._kappa(graph, c)
            t_message = (
                _qbar(graph, z) / kappa[:, None, None]
                if intervention == "c_identity"
                else self._weighted_message(graph, z, c, kappa, rows)
            )
        elif self.condition in ("fixed_wedge", "fixed_wedge_node_mlp"):
            c = z.new_ones((z.shape[0], graph.paths.shape[1]))
            t_message = _qbar(graph, z)
        elif self.condition == "polynomial_2":
            t_message = _lbar(graph, l_message)
        else:
            t_message = z.new_zeros(z.shape)
        if intervention == "second_branch_remove":
            t_message = torch.zeros_like(t_message)
        detail = {
            "alpha": alpha.detach(),
            "beta": beta.detach(),
            "sigma": sigma.detach(),
            "kappa": kappa.detach(),
            "kappa_reference": reference_kappa.detach(),
            "branch_norm": torch.linalg.vector_norm(t_message, dim=(1, 2)).detach(),
            **self._stats(c),
        }
        if diagnostics:
            detail.update(
                z=z.detach(),
                c=None if c is None else c.detach(),
                c_reference=None if c_reference is None else c_reference.detach(),
                l_message=None if l_message is None else l_message.detach(),
                t_message=t_message.detach(),
            )
        return t_message, detail

    def forward(
        self,
        graph: Any,
        epoch: int = 0,
        intervention: str | None = None,
        manifest: dict | None = None,
        kappa_mode: str = "recompute",
        input_scale: float = 1.0,
        diagnostics: bool = False,
    ):
        epoch = _integer(epoch, "epoch", minimum=0)
        amplitude = float(input_scale)
        if not math.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("input_scale must be finite and positive")
        parameter = self.projections[0]
        if (
            graph.x.ndim != 2
            or graph.x.shape[1] != self.input_dim
            or graph.x.device != parameter.device
            or graph.x.dtype != parameter.dtype
        ):
            raise ValueError("graph.x shape/device/dtype must match the model")
        h = (graph.x * amplitude)[None].expand(len(self.seeds), -1, -1)
        details = []
        for layer in range(2):
            z = torch.bmm(self._dropout(h, epoch, layer), self.projections[layer])
            if self.condition == "standard_gcn":
                source = z.index_select(1, graph.gcn_edges[0]) * graph.gcn_weight[None, :, None]
                u = z.new_zeros(z.shape).index_add(1, graph.gcn_edges[1], source)
                detail = {
                    "alpha": z.new_zeros(z.shape[0]),
                    "beta": z.new_zeros(z.shape[0]),
                    "sigma": z.new_ones(z.shape[0]),
                    "kappa": z.new_ones(z.shape[0]),
                    "kappa_reference": z.new_ones(z.shape[0]),
                    "branch_norm": torch.linalg.vector_norm(u, dim=(1, 2)).detach(),
                    **self._stats(None),
                }
                if diagnostics:
                    detail.update(
                        z=z.detach(), c=None, c_reference=None, l_message=None, t_message=u.detach()
                    )
                if intervention is not None:
                    raise ValueError("frozen C interventions apply only to learned gate conditions")
            else:
                l_message = (
                    _lbar(graph, z)
                    if self.condition == "first_order" or self.condition in _SECOND
                    else None
                )
                t_message, detail = self.probe_layer(
                    graph,
                    layer,
                    z,
                    intervention=intervention,
                    manifest=manifest,
                    kappa_mode=kappa_mode,
                    diagnostics=diagnostics,
                    _l_message=l_message,
                )
                alpha, beta = self._coefficients(layer, z)
                u = z
                if self.condition == "first_order" or self.condition in _SECOND:
                    u = u - alpha[:, None, None] * l_message
                if self.condition in _SECOND:
                    u = u - beta[:, None, None] * t_message
                if self.condition == "fixed_wedge_node_mlp":
                    node = self.node_mlps[layer](z)
                    u = u + beta[:, None, None] * node
                    detail["node_mlp_norm"] = torch.linalg.vector_norm(node, dim=(1, 2)).detach()
            details.append(detail)
            h = u.relu() if layer == 0 else u
        return h, details


__all__ = ["CONDITIONS", "PackedClassifier"]
