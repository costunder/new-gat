"""Evaluate exp scores in log coordinates; preserve the normalized Laplacian map."""

import torch
from torch import nn

from experiments.c_learning_bracket.conductance import BracketConductance
from experiments.c_learning_bracket.operator import OperatorOutput
from research.conductance_gat.v5.model import graph_context_features
from research.conductance_gat.v5.operator import _ChunkedHeadPropagation, graph_broadcast


class ChunkedSymmetricScores(torch.autograd.Function):
    """Accumulate Q/K gradients once, instead of allocating NHD per edge chunk."""

    @staticmethod
    def forward(ctx, query, key, edges, chunk_size):
        ctx.save_for_backward(query, key, edges)
        ctx.chunk_size = chunk_size
        ctx.set_materialize_grads(False)
        chunks = [
            BracketConductance.edge_score(query, key, edges[:, start : start + chunk_size])
            for start in range(0, edges.shape[1], chunk_size)
        ]
        return torch.cat(chunks) if chunks else query.new_empty((0, query.shape[1]))

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:
            return None, None, None, None
        query, key, edges = ctx.saved_tensors
        grad_query = torch.zeros_like(query) if ctx.needs_input_grad[0] else None
        grad_key = torch.zeros_like(key) if ctx.needs_input_grad[1] else None
        for start in range(0, edges.shape[1], ctx.chunk_size):
            stop = start + ctx.chunk_size
            tail, head = edges[:, start:stop]
            scale = (gradient[start:stop] / (2 * query.shape[-1])).unsqueeze(-1)
            if grad_query is not None:
                grad_query.index_add_(0, tail, scale * key[head])
                grad_query.index_add_(0, head, scale * key[tail])
            if grad_key is not None:
                grad_key.index_add_(0, tail, scale * query[head])
                grad_key.index_add_(0, head, scale * query[tail])
        return grad_query, grad_key, None, None


def require_finite(value, label):
    # Necessary validation before normalization; unlike a device assert, this
    # leaves CUDA usable so the runner can write its ordinary failure report.
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(
            f"nonfinite {label}; log-row evaluation cannot repair invalid scores"
        )


class LogConductance(BracketConductance):
    @classmethod
    def from_existing(cls, source):
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        for name in ("channels", "heads", "head_width", "edge_chunk_size", "checkpoint_edges"):
            setattr(result, name, getattr(source, name))
        result.query, result.key = source.query, source.key
        result.last_scores = None
        return result

    def forward(self, state, incidence, *unused, **unused_kwargs):
        with torch.autocast(device_type=state.device.type, enabled=False):
            geometry = state.to(self.query.weight.dtype)
            query = self.query(geometry).reshape(-1, self.heads, self.head_width)
            key = self.key(geometry).reshape_as(query)
            if self.checkpoint_edges and torch.is_grad_enabled():
                scores = ChunkedSymmetricScores.apply(query, key, incidence, self.edge_chunk_size)
            else:
                chunks = [
                    self.edge_score(query, key, incidence[:, start : start + self.edge_chunk_size])
                    for start in range(0, incidence.shape[1], self.edge_chunk_size)
                ]
                scores = torch.cat(chunks) if chunks else query.sum(-1)[:0]
        self.last_scores = scores.detach()
        return scores


def evaluate_log_c(estimator, state, edges, groups, count):
    result = estimator(state, edges, groups, count)
    # Inherited fixed_copy/c_ones use the parameter-free OnesConductance.
    return result if isinstance(estimator, LogConductance) else result.log()


def log_row_coefficients(log_c, edges, nodes, *, correction=None):
    valid = torch.isfinite(log_c).all()
    logits = log_c
    if correction is not None:
        valid = valid & (torch.isfinite(correction) & (correction > 0)).all()
        logits = logits + correction.to(log_c.dtype).log()[:, None]
    # One combined host check, instead of repeated synchronization per field.
    if not bool(valid & torch.isfinite(logits).all()):
        raise FloatingPointError("nonfinite log C or invalid sampling correction")
    tail, head = edges
    heads = logits.shape[1]
    maximum = logits.new_full((nodes, heads), -torch.inf)
    # Detaching the numerical shift is exact: it cancels in each row's ratio.
    # Shared undirected scores feed both receiver normalizations and gradients.
    for receiver in (tail, head):
        maximum.scatter_reduce_(
            0,
            receiver[:, None].expand(-1, heads),
            logits.detach(),
            reduce="amax",
            include_self=True,
        )
    to_tail = (logits - maximum[tail]).exp()
    to_head = (logits - maximum[head]).exp()
    total = logits.new_zeros((nodes, heads))
    total.index_add_(0, tail, to_tail)
    total.index_add_(0, head, to_head)
    # Topological support, independent of raw exp underflow. Isolates retain V.
    active = torch.isfinite(maximum)
    safe = torch.where(active, total, torch.ones_like(total))
    return to_tail / safe[tail], to_head / safe[head], active


class LogRowOperator(nn.Module):
    @classmethod
    def from_existing(cls, source):
        result = cls()
        for name in ("channels", "heads", "head_width", "edge_chunk_size"):
            setattr(result, name, getattr(source, name))
        result.capture_coefficients = result.capture_signals = False
        result.observed_coefficients = result.observed_signals = None
        result.estimator = LogConductance.from_existing(source.estimator)
        result.value_weight = source.value_weight
        result.output_projection = source.output_projection
        result.beta_estimator = source.beta_estimator
        return result

    def forward_with_state(
        self,
        state,
        incidence,
        node_graph,
        num_graphs,
        *,
        full_degree,
        graph_structure,
        sampling_correction,
        static_context=None,
        edge_relation_id=None,
        **unused,
    ):
        if edge_relation_id is not None:
            raise ValueError("this experiment does not consume relation labels")
        with torch.autocast(device_type=state.device.type, enabled=False):
            geometry = state.to(self.value_weight.dtype)
            log_c = evaluate_log_c(self.estimator, geometry, incidence, node_graph, num_graphs)
            context, _, _ = graph_context_features(
                geometry,
                incidence,
                node_graph,
                num_graphs,
                full_degree,
                graph_structure,
                static_context=static_context,
            )
            beta = self.beta_estimator(context)
            to_tail, to_head, active = log_row_coefficients(
                log_c,
                incidence,
                state.shape[0],
                correction=sampling_correction,
            )
        value = torch.einsum("nd,hdk->nhk", state, self.value_weight)
        dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
        original = value.to(dtype)
        propagated = _ChunkedHeadPropagation.apply(
            original,
            to_tail.to(dtype),
            to_head.to(dtype),
            incidence,
            self.edge_chunk_size,
        )
        node_beta = graph_broadcast(beta.to(dtype), node_graph, beta.shape[0]).unsqueeze(-1)
        mixed = (original + node_beta * (propagated - active.unsqueeze(-1) * original)).to(
            value.dtype
        )
        if self.capture_coefficients:
            self.observed_coefficients = (to_tail.detach(), to_head.detach())
        if self.capture_signals:
            self.observed_signals = {
                "value_projection": value.detach(),
                "neighbor_mixing": mixed.detach(),
            }
        message = self.output_projection(mixed.reshape(state.shape[0], self.channels))
        require_finite(message, "bracket propagation")
        # In this explicitly named backend, both edge fields use log coordinates.
        effective_log = (
            log_c
            if sampling_correction is None
            else (log_c + sampling_correction.to(log_c.dtype).log()[:, None])
        )
        return OperatorOutput(message, log_c, effective_log, beta)
