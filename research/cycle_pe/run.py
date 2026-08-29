"""Run the graph-family structural probe for static cycle PE."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from .synthetic import PROBE_VARIANTS, make_graph_family_split, stack_probe_graphs

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class LogisticProbe:
    """A deterministic linear probe fitted on frozen static PE features."""

    mean: FloatArray
    scale: FloatArray
    weights: FloatArray
    bias: float

    def decision_function(self, features: FloatArray) -> FloatArray:
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return standardized @ self.weights + self.bias

    def predict_proba(self, features: FloatArray) -> FloatArray:
        logits = np.clip(self.decision_function(features), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


def fit_logistic_probe(
    features: FloatArray,
    targets: IntArray,
    *,
    steps: int,
    learning_rate: float,
    l2: float,
) -> LogisticProbe:
    """Fit a full-batch logistic probe without modifying the PE extractor."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("features/targets have incompatible shapes")
    if steps < 1 or learning_rate <= 0.0 or l2 < 0.0:
        raise ValueError("invalid probe optimizer configuration")
    if np.unique(y).size != 2:
        raise ValueError("the training split must contain both edge classes")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    weights = np.zeros(x.shape[1], dtype=np.float64)
    prevalence = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    bias = float(np.log(prevalence / (1.0 - prevalence)))

    for step in range(steps):
        logits = np.clip(standardized @ weights + bias, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = probabilities - y
        gradient_weights = standardized.T @ residual / len(y) + l2 * weights
        gradient_bias = float(residual.mean())
        # Mild decay makes the small deterministic optimizer insensitive to the
        # exact requested step count while preserving a fast smoke configuration.
        rate = learning_rate / np.sqrt(1.0 + step / 250.0)
        weights -= rate * gradient_weights
        bias -= rate * gradient_bias
    return LogisticProbe(mean, scale, weights, bias)


def _binary_metrics(
    targets: IntArray,
    probabilities: FloatArray,
    graph_ids: tuple[str, ...],
) -> dict[str, float]:
    y = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    predictions = (scores >= 0.5).astype(np.int64)
    positives = y == 1
    negatives = ~positives
    true_positive = int(np.count_nonzero(predictions[positives] == 1))
    true_negative = int(np.count_nonzero(predictions[negatives] == 0))
    false_positive = int(np.count_nonzero(predictions[negatives] == 1))
    false_negative = int(np.count_nonzero(predictions[positives] == 0))
    recall_positive = true_positive / max(1, true_positive + false_negative)
    recall_negative = true_negative / max(1, true_negative + false_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    f1 = 2.0 * precision * recall_positive / max(1e-12, precision + recall_positive)

    positive_scores = scores[positives]
    negative_scores = scores[negatives]
    if len(positive_scores) and len(negative_scores):
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        auroc = float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))
    else:
        auroc = float("nan")

    graph_accuracy = []
    ids = np.asarray(graph_ids, dtype=object)
    for graph_id in dict.fromkeys(graph_ids):
        mask = ids == graph_id
        graph_accuracy.append(float(np.mean(predictions[mask] == y[mask])))
    return {
        "accuracy": float(np.mean(predictions == y)),
        "balanced_accuracy": float(0.5 * (recall_positive + recall_negative)),
        "f1": float(f1),
        "auroc": auroc,
        "macro_graph_accuracy": float(np.mean(graph_accuracy)),
    }


def run_structural_probe(
    *,
    samples_per_family: int = 8,
    seed: int = 7,
    max_cycles: int = 12,
    steps: int = 1200,
    learning_rate: float = 0.15,
    l2: float = 1e-3,
) -> dict[str, Any]:
    """Compare degree-only features with three frozen static cycle PEs."""

    train_graphs, test_graphs = make_graph_family_split(
        samples_per_family=samples_per_family,
        seed=seed,
    )
    results: dict[str, Any] = {
        "scope": "static_graph_cycle_pe_only",
        "task": "edge_is_in_any_cycle",
        "split": {
            "train_families": sorted({graph.family for graph in train_graphs}),
            "test_families": sorted({graph.family for graph in test_graphs}),
            "train_graphs": len(train_graphs),
            "test_graphs": len(test_graphs),
        },
        "variants": {},
        "notes": {
            "raw": "diagnostic; column order/sign and spanning-tree dependent",
            "cycle_set": "invariant only to sign/permutation of a fixed cycle set",
            "projector_leverage": "basis-invariant prior-style baseline, not novelty",
        },
    }

    for variant in PROBE_VARIANTS:
        train = stack_probe_graphs(train_graphs, variant, max_cycles=max_cycles)
        test = stack_probe_graphs(test_graphs, variant, max_cycles=max_cycles)
        probe = fit_logistic_probe(
            train.features,
            train.targets,
            steps=steps,
            learning_rate=learning_rate,
            l2=l2,
        )
        results["variants"][variant] = {
            "feature_dim": int(train.features.shape[1]),
            "train_edges": int(train.features.shape[0]),
            "test_edges": int(test.features.shape[0]),
            "train": _binary_metrics(
                train.targets, probe.predict_proba(train.features), train.graph_ids
            ),
            "test": _binary_metrics(
                test.targets, probe.predict_proba(test.features), test.graph_ids
            ),
        }
    return results


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _read_config(config_path)
    probe_config = config.get("probe", {})
    results = run_structural_probe(
        samples_per_family=int(config.get("samples_per_family", 8)),
        seed=int(config.get("seed", 7)),
        max_cycles=int(config.get("max_cycles", 12)),
        steps=int(probe_config.get("steps", 1200)),
        learning_rate=float(probe_config.get("learning_rate", 0.15)),
        l2=float(probe_config.get("l2", 1e-3)),
    )
    if args.output is not None:
        output = args.output.expanduser()
    else:
        output = Path(config.get("output", "results/summary.json")).expanduser()
        if not output.is_absolute():
            output = config_path.parent / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
