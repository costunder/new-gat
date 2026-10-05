"""Fixed linear comparisons, including explicitly frozen reference coefficients."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..geometry import prepare_geometry
from ..operators import apply_metric, diagonal_action, divergence, incidence
from ...local_context_coupling.operators import prepare_geometry as copy_geometry, sandwich as copy_sandwich


OPERATOR_NAMES = ("L0", "Ld", "smooth_L0", "smooth_Ld", "copy_off", "copy_on",
                  "Q_diag", "P_diag", "Q_F0", "P_F0", "Q_reference", "P_reference")
REPETITIONS = (1, 2, 4, 8, 16)


@dataclass
class FixedAction:
    name: str
    geometry: object
    coefficient: torch.Tensor | None = None
    copy: object | None = None
    pair_chunk: int | None = None
    step: torch.Tensor | None = None

    @property
    def reference(self):
        if self.name.startswith("Q_"):
            return "Q_diag"
        if self.name.startswith("P_"):
            return "P_diag"
        if self.name.startswith("copy_"):
            return "copy_off"
        return "Ld" if self.name.startswith("L") else "smooth_Ld"

    @property
    def coefficient_batches(self):
        return 1 if self.coefficient is None else self.coefficient.shape[0]

    def once(self, value):
        g = self.geometry
        if self.name.startswith("copy_"):
            shape = value.shape
            result = copy_sandwich(self.copy, value.reshape(-1, shape[-2], shape[-1]),
                                  cross=self.name == "copy_on")
            return result.reshape(shape)
        if self.name in ("L0", "smooth_L0"):
            lap = divergence(g, incidence(g, value))
        elif self.name in ("Ld", "smooth_Ld"):
            lap = diagonal_action(g, value)
        else:
            normalized = self.name.startswith("P_")
            operand = value*g.n0[:, None] if normalized else value
            multiplier = value.new_ones(value.shape[:-2]+(g.num_edges,))
            coefficient = None if self.coefficient is None else self.coefficient.expand(value.shape[:-2]+(g.num_pairs,))
            lap = apply_metric(g, operand, multiplier, coefficient, pair_chunk=self.pair_chunk)
            return value-g.tau*g.n0[:, None]*lap if normalized else lap
        if self.name.startswith("smooth_"):
            return value-self.step*lap
        return lap

    def __call__(self, value, repetitions=1):
        if repetitions not in REPETITIONS:
            raise ValueError("undeclared repetition count")
        result = value
        for _ in range(repetitions):
            result = self.once(result)
        return result

    def transpose(self, value, repetitions=1):
        if not self.name.startswith("copy_"):
            return self(value, repetitions)
        # T is D-self-adjoint, rather than Euclidean symmetric.
        counts = self.copy.copy_counts[:, None]
        return self(value/counts, repetitions)*counts


def reference_coefficient(g, reference, *, pair_chunk=None, channel_chunk=None):
    """Analytic rule frozen at every original scalar draw / citation vector.

The smooth RMS policy is exactly epsilon=1e-4. No coefficient is recomputed
when the subsequent null-space or recovery probe changes its input.
"""
    size = max(1, g.num_pairs) if pair_chunk is None else int(pair_chunk)
    width = reference.shape[-1] if channel_chunk is None else int(channel_chunk)
    if size < 1 or width < 1:
        raise ValueError("exact pair/channel chunks must be positive")
    result = reference.new_empty(reference.shape[:-2]+(g.num_pairs,))
    for start in range(0, g.num_pairs, size):
        stop = min(g.num_pairs, start+size)
        dot = reference.new_zeros(reference.shape[:-2]+(stop-start,))
        norm = torch.zeros_like(dot)
        for first in range(0, reference.shape[-1], width):
            block = reference[..., first:first+width]
            anchor = block.index_select(-2, g.pair_anchor[start:stop])
            left = block.index_select(-2, g.pair_other_left[start:stop])-anchor
            right = block.index_select(-2, g.pair_other_right[start:stop])-anchor
            dot += (left*right).sum(-1)
            norm += .5*(left.square().sum(-1)+right.square().sum(-1))
        alignment = dot/(norm+reference.shape[-1]*1e-8)
        raw = torch.tanh(2*alignment+.5*(2*g.pair_structure[start:stop, 3]-1))
        result[..., start:stop] = raw*g.pair_sign[start:stop]*g.pair_norm[start:stop]
    return result


def build_actions(topology, recipe, reference, *, pair_chunk=None, channel_chunk=None,
                  prepared_geometry=None, prepared_copy=None):
    if reference.ndim != 3:
        raise ValueError("reference must be [scalar_draw_or_vector=1,N,F]")
    if (prepared_geometry is None) != (prepared_copy is None):
        raise ValueError("prepared edge and copy geometry must be supplied together")
    if prepared_geometry is None:
        cpu = prepare_geometry(topology.to("cpu"), recipe)
        g = cpu.to(reference.device, reference.dtype)
        copy = copy_geometry(topology.to("cpu"), recipe).to(reference.device, reference.dtype)
    else:
        g, copy = prepared_geometry, prepared_copy
        if (g.recipe != recipe or copy.mode != recipe or g.n != topology.n or
            g.edges.device != reference.device or copy.intra_weights.device != reference.device or
            g.c0.dtype != reference.dtype or copy.intra_weights.dtype != reference.dtype):
            raise ValueError("prepared geometry must match recipe, nodes, reference device and dtype")
    fixed = (g.pair_sign*g.pair_norm).unsqueeze(0)
    analytic = reference_coefficient(g, reference, pair_chunk=pair_chunk, channel_chunk=channel_chunk)
    unit_degree = g.d0.new_zeros(g.n).index_add(0, g.edges[0], g.d0.new_ones(g.num_edges))
    unit_degree = unit_degree.index_add(0, g.edges[1], g.d0.new_ones(g.num_edges))
    degrees = {"smooth_L0": unit_degree, "smooth_Ld": g.d0}
    steps = {}
    for key, degree in degrees.items():
        maxima = degree.new_zeros(g.num_graphs).scatter_reduce_(0, g.node_graph, degree, reduce="amax")
        inverse = torch.where(maxima > 0, .5/maxima.clamp_min(torch.finfo(reference.dtype).tiny), 0.)
        steps[key] = inverse[g.node_graph, None]
    output = {}
    for name in OPERATOR_NAMES:
        coefficient = fixed if name.endswith("F0") else analytic if name.endswith("reference") else None
        output[name] = FixedAction(name, g, coefficient, copy if name.startswith("copy_") else None,
                                   pair_chunk, steps.get(name))
    return output


def feature_fields(case, device):
    x = case.features.to(device=device, dtype=torch.float64)
    return x.T.unsqueeze(-1) if case.feature_mode == "independent_scalar_columns" else x.unsqueeze(0)
