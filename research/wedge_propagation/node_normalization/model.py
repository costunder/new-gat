"""Experiment 4.2: matched gates with global or node diagonal normalization.

The fixed/global controls delegate to the immutable Experiment 4 model. Local
conditions change only the weighted second branch to S_C A.T C A S_C / 3,
where S_C(v,v)=diag(A.T C A)[v]**(-1/2) on supported vertices and zero otherwise.
Both factors and the weighted diagonal remain in the differentiation graph.
All paths contribute; chunks only limit intermediate path/channel tensors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from ..classification.model import (
    PackedClassifier as OriginalPackedClassifier,
)
from ..classification.model import _integer, _lbar, _row_message

CONDITIONS = (
    "fixed_wedge",
    "learned_wedge_raw",
    "learned_wedge_rms",
    "learned_wedge_local_raw",
    "learned_wedge_local_rms",
)
LOCAL_CONDITIONS = frozenset(CONDITIONS[3:])
_CANONICAL = {
    "learned_wedge_local_raw": "learned_wedge_raw",
    "learned_wedge_local_rms": "learned_wedge_rms",
}
_LOCAL_INTERVENTIONS = frozenset(("c_identity", "c_position_shuffle", "second_branch_remove"))


class PackedClassifier(OriginalPackedClassifier):
    """Five conditions sharing all original initialization and dropout streams.

    ``experiment_condition`` records the five-way experimental name. ``condition``
    retains the original raw/RMS gate name because the inherited forward uses it
    to select the same sigma, first branch and coefficient parameterization.
    ``normalization`` is ``global_kappa`` or ``node_diagonal``. A fixed C=I node
    operator equals the fixed global operator, so there is one fixed control.
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
        if condition not in CONDITIONS:
            raise ValueError(f"unknown node normalization condition {condition!r}")
        super().__init__(
            _CANONICAL.get(condition, condition),
            input_dim,
            classes,
            seeds,
            hidden=hidden,
            gate_hidden=gate_hidden,
            dropout=dropout,
            path_chunk=path_chunk,
            checkpoint_paths=checkpoint_paths,
            dataset_name=dataset_name,
        )
        self.experiment_condition = condition
        self.normalization = "node_diagonal" if condition in LOCAL_CONDITIONS else "global_kappa"

    @staticmethod
    def weighted_diagonal(graph: Any, c: Tensor) -> Tensor:
        """Return diag(A.T C A) with exact full-path scatter, shape [S,N]."""
        if c.ndim != 2 or c.shape[1] != graph.paths.shape[1]:
            raise ValueError("c must have shape [packed_seeds,paths]")
        diagonal = c.new_zeros((c.shape[0], graph.x.shape[0]))
        diagonal = diagonal.index_add(1, graph.paths[0], c)
        diagonal = diagonal.index_add(1, graph.paths[1], 4 * c)
        return diagonal.index_add(1, graph.paths[2], c)

    @staticmethod
    def inverse_sqrt_diagonal(diagonal: Tensor) -> Tensor:
        """Zero unsupported vertices before rsqrt, avoiding singular gradients."""
        active = diagonal > 0
        safe = torch.where(active, diagonal, torch.ones_like(diagonal))
        return safe.rsqrt() * active.to(diagonal.dtype)

    def _node_message(self, graph: Any, z: Tensor, c: Tensor, diagonal: Tensor) -> Tensor:
        scale = self.inverse_sqrt_diagonal(diagonal)
        scaled = z * scale[:, :, None]
        result = z.new_zeros(z.shape)
        for start, stop in self._chunk_ranges(graph.paths.shape[1]):
            coefficients = self.wedge_coefficients[:, None].expand(3, stop - start)
            result = result + self._run_chunk(
                _row_message,
                scaled,
                graph.paths[:, start:stop],
                coefficients,
                c[:, start:stop],
            )
        return result * scale[:, :, None] / 3.0

    def apply_node_branch(self, graph: Any, z: Tensor, c: Tensor) -> Tensor:
        """Apply supplied C with its own node diagonal; do not rerun the gate.

        Validate supplied C because this is a public frozen-intervention API.
        Native training uses the private path after exp(tanh)/mean and performs
        no additional host synchronization for these external-input checks.
        """
        if c.shape != (len(self.seeds), graph.paths.shape[1]):
            raise ValueError("c must have shape [packed_seeds,paths]")
        if z.ndim != 3 or z.shape[:2] != (len(self.seeds), graph.x.shape[0]):
            raise ValueError("z must have shape [packed_seeds,nodes,channels]")
        if c.dtype != z.dtype or c.device != z.device:
            raise ValueError("c must match z dtype/device")
        if not bool(torch.isfinite(c).all()) or not bool((c > 0).all()):
            raise ValueError("c must contain finite positive conductances")
        return self._node_message(graph, z, c, self.weighted_diagonal(graph, c))

    def _weighted_message(
        self, graph: Any, z: Tensor, c: Tensor, kappa: Tensor, rows: dict[str, Tensor] | None = None
    ) -> Tensor:
        if self.normalization != "node_diagonal":
            return super()._weighted_message(graph, z, c, kappa, rows)
        if rows is not None:
            raise ValueError("node normalization does not define random physical row interventions")
        # The inherited apply_branch signature retains kappa for API compatibility;
        # a local operator is defined by C and D_C and has no global denominator.
        return self.apply_node_branch(graph, z, c)

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
        if self.normalization != "node_diagonal":
            return super().probe_layer(
                graph,
                layer_index,
                z,
                input_scale,
                intervention,
                manifest,
                kappa_mode,
                diagnostics,
                _l_message,
            )
        layer = _integer(layer_index, "layer_index", minimum=0)
        if layer > 1:
            raise ValueError("layer_index must be 0 or 1")
        if intervention is not None and intervention not in _LOCAL_INTERVENTIONS:
            raise ValueError(f"unsupported node normalization intervention {intervention!r}")
        if kappa_mode != "recompute":
            raise ValueError(
                "node normalization requires recomputing D_C, not holding global kappa"
            )
        amplitude = float(input_scale)
        if not math.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("input_scale must be finite and positive")
        expected_f = self.hidden if layer == 0 else self.classes
        if z.ndim != 3 or z.shape != (len(self.seeds), graph.x.shape[0], expected_f):
            raise ValueError("z must have shape [packed_seeds,nodes,layer_output_channels]")
        z = z * amplitude
        alpha, beta = self._coefficients(layer, z)
        sigma = self._sigma(graph, z)
        c_reference = self._gate(graph, layer, z, sigma)
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
        diagonal = self.weighted_diagonal(graph, c)
        t_message = self._node_message(graph, z, c, diagonal)
        if intervention == "second_branch_remove":
            t_message = torch.zeros_like(t_message)
        reference_diagonal = (
            diagonal if c is c_reference else self.weighted_diagonal(graph, c_reference)
        )
        active = graph.qdiag > 0
        denominator = torch.where(active, graph.qdiag, torch.ones_like(graph.qdiag))
        global_kappa = (reference_diagonal / denominator[None]).amax(1)
        # With no paths the global reference convention is kappa=1.
        if graph.paths.shape[1] == 0:
            global_kappa = z.new_ones(z.shape[0])
        detail = {
            "alpha": alpha.detach(),
            "beta": beta.detach(),
            "sigma": sigma.detach(),
            "kappa": z.new_ones(z.shape[0]),
            "kappa_reference": global_kappa.detach(),
            "kappa_global": global_kappa.detach(),
            "branch_norm": torch.linalg.vector_norm(t_message, dim=(1, 2)).detach(),
            "weighted_diagonal_max": diagonal.amax(1).detach(),
            **self._stats(c),
        }
        if diagnostics:
            l_message = _l_message if _l_message is not None else _lbar(graph, z)
            detail.update(
                z=z.detach(),
                c=c.detach(),
                c_reference=c_reference.detach(),
                l_message=l_message.detach(),
                t_message=t_message.detach(),
                weighted_diagonal=diagonal.detach(),
                node_inv_sqrt=self.inverse_sqrt_diagonal(diagonal).detach(),
            )
        return t_message, detail


__all__ = ["CONDITIONS", "LOCAL_CONDITIONS", "PackedClassifier"]
