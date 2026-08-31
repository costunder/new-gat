"""Train one fresh learned-C/fixed-C arm, CUDA and official caches only.

This separate suite explicitly injects its model and optimizer into the audited
factorial training loop. It never reuses old checkpoints for the training contrast
and never constructs a test loader or reports a test metric.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ..ablation import train as shared
from .model import CLearningNodeClassifier, make_optimizer
from .protocol import CONDITIONS, SUITE

DEFINITION = shared.TrainingDefinition(
    SUITE, CONDITIONS, CLearningNodeClassifier, make_optimizer, description=__doc__
)


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return shared.configuration(args)


def build_parser() -> argparse.ArgumentParser:
    parser = shared.build_parser(definition=DEFINITION)
    parser.description = __doc__
    return parser


def train_model(
    payload: dict[str, Any],
    protocol: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    return shared.train_model(payload, protocol, args, device, output, definition=DEFINITION)


def main(argv: list[str] | None = None) -> int:
    return shared.main(argv, definition=DEFINITION)


if __name__ == "__main__":
    raise SystemExit(main())
