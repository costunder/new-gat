"""Independent Q/K, symmetric per-edge dot mean, then an unmodified exp."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class BracketConductance(nn.Module):
    def __init__(self, channels, heads, *, edge_chunk_size, checkpoint_edges=True):
        super().__init__()
        if channels < 1 or heads < 1 or channels % heads or edge_chunk_size < 1:
            raise ValueError("positive dimensions/chunk required; channels must divide heads")
        self.channels, self.heads = channels, heads
        self.head_width = channels // heads
        self.edge_chunk_size = edge_chunk_size
        self.checkpoint_edges = checkpoint_edges
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        self.last_scores = None

    @staticmethod
    def edge_score(query, key, edges):
        tail, head = edges
        return 0.5 * (query[tail] * key[head] + query[head] * key[tail]).mean(-1)

    def forward(self, state, incidence, *unused, **unused_kwargs):
        if state.ndim != 2 or state.shape[1] != self.channels:
            raise ValueError("state must have shape [nodes, channels]")
        if incidence.ndim != 2 or incidence.shape[0] != 2 or incidence.dtype != torch.long:
            raise ValueError("incidence must be int64 [2, physical_edges]")
        # Explicit FP32 under AMP; retain FP64 for mathematical gradient tests.
        with torch.autocast(device_type=state.device.type, enabled=False):
            geometry = state.to(self.query.weight.dtype)
            query = self.query(geometry).reshape(-1, self.heads, self.head_width)
            key = self.key(geometry).reshape_as(query)
            chunks = []
            for start in range(0, incidence.shape[1], self.edge_chunk_size):
                edges = incidence[:, start : start + self.edge_chunk_size]
                scores = (
                    checkpoint(
                        self.edge_score,
                        query,
                        key,
                        edges,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if self.checkpoint_edges and torch.is_grad_enabled()
                    else self.edge_score(query, key, edges)
                )
                chunks.append(scores)
            scores = torch.cat(chunks) if chunks else query.sum(-1)[:0]
            c = scores.exp()
            # Never clamp, center, shift or substitute a different positive map.
            # This device-side check avoids a host synchronization per edge chunk.
            torch._assert_async(
                (torch.isfinite(scores) & torch.isfinite(c) & (c > 0)).all(),
                "BracketConductance exp produced nonfinite/zero C; formula unchanged; stop run",
            )
        self.last_scores = scores.detach()
        return c
