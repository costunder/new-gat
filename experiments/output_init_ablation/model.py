"""Matched random directions: only initial output projection scale differs."""

import hashlib
import math

import torch

from experiments.aggregation_comparison import engine
from experiments.c_learning_bracket.model import BracketClassifier

from . import SUITE

INITIALIZATIONS = ("baseline", "kaiming_relu")


class OutputInitClassifier(BracketClassifier):
    def __init__(self, *args, output_initialization, conductance_evaluation="raw_exp", **kwargs):
        if output_initialization not in INITIALIZATIONS:
            raise ValueError("declare baseline or kaiming_relu initialization")
        super().__init__(*args, **kwargs)
        if conductance_evaluation not in {"raw_exp", "log_row"}:
            raise ValueError("unknown conductance evaluation")
        self.conductance_evaluation = conductance_evaluation
        self.output_initialization = output_initialization
        if output_initialization == "kaiming_relu":
            # Linear default U[-1/sqrt(D),1/sqrt(D)] -> U[-sqrt(6/D),sqrt(6/D)].
            # Reuse the draws, so all other initial parameters and RNG are identical.
            with torch.no_grad():
                for op in self.layers:
                    op.output_projection.weight.mul_(math.sqrt(6.0))
        if conductance_evaluation == "log_row":
            from .log_row import LogRowOperator

            self.layers = torch.nn.ModuleList(
                LogRowOperator.from_existing(op) for op in self.layers
            )

    def contract(self):
        result = super().contract()
        result.update(
            suite=SUITE,
            implementation_revision="output_init_ablation_1",
            output_initialization=self.output_initialization,
            initialization="legacy except output projection: " + self.output_initialization,
            output_initial_scale=math.sqrt(6.0)
            if self.output_initialization == "kaiming_relu"
            else 1.0,
            initialization_pairing="same draws; only output weights scaled once before training",
        )
        result["conductance_evaluation"] = self.conductance_evaluation
        if self.conductance_evaluation == "log_row":
            result.update(
                positive_map="C=exp(score), stored as log C; no score clipping",
                normalization_evaluation="receiver log-sum-exp; shared symmetric edge scores",
                conductance_output_representation="log C",
                implementation_revision="output_init_log_row_1",
            )
        return result


def non_output_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.endswith("output_projection.weight"):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_model(payload, args, device, condition="learned"):
    if condition not in {"learned", "fixed"}:
        raise ValueError("only learned and fixed C conditions are supported")
    engine.base._seed(args.model_seed)
    model = OutputInitClassifier(
        payload["graphs"][0]["x"].shape[1],
        payload["classes"],
        channels=args.hidden_channels,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        activation_checkpoint=args.activation_checkpoint,
        edge_chunk_size=args.edge_chunk_size,
        checkpoint_edges=args.checkpoint_edges,
        output_initialization=args.output_initialization,
        conductance_evaluation=getattr(args, "conductance_evaluation", "raw_exp"),
    ).to(device)
    return model if condition == "learned" else model.fixed_copy()
