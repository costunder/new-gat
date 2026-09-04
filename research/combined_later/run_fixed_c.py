"""MVP fixed-conductance flow completion with hard observation preservation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from chartgat.algebra import fundamental_cycle_basis, incidence_matrix
from chartgat.graphs import make_connected_graph, spanning_tree_indices
from chartgat.observability import RuntimeResourceMonitor
from research.combined_later.completion import (
    analytic_cycle_completion,
    hard_observation_affine,
)
from research.combined_later.layers import (
    OrientationEquivariantEdgeResidual,
    hard_observation_coordinate_projector,
)
from research.combined_later.synthetic import structured_cycle_flows


@dataclass
class Dataset:
    q_part: np.ndarray
    target: np.ndarray
    edge_features: np.ndarray
    anchor_train: np.ndarray
    anchor_unseen: np.ndarray
    observed: np.ndarray
    missing: np.ndarray


class PhysicalCompletionModel(nn.Module):
    """Nonlinear physical-edge refinement encoded in the active cycle chart."""

    def __init__(self, edge_feature_channels: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                OrientationEquivariantEdgeResidual(
                    channels=1,
                    hidden_channels=hidden,
                    edge_feature_channels=edge_feature_channels,
                )
                for _ in range(depth)
            ]
        )
        self.raw_steps = nn.Parameter(torch.full((depth,), -1.5))

    @staticmethod
    def _encoder(basis: Tensor, observed: Tensor) -> Tensor:
        gram = basis.T @ basis
        unconstrained = torch.linalg.solve(gram, basis.T)
        projector = hard_observation_coordinate_projector(basis, observed)
        return projector @ unconstrained

    def forward(
        self,
        basis: Tensor,
        observed: Tensor,
        q_part: Tensor,
        anchor: Tensor,
        edge_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # q_part: (samples, edges), anchor: (samples, beta)
        encoder = self._encoder(basis, observed)
        coordinates = anchor
        samples, edges = q_part.shape
        for layer, raw_step in zip(self.layers, self.raw_steps, strict=True):
            cycle = coordinates @ basis.T
            residual = layer(
                q_part.reshape(samples * edges, 1),
                cycle.reshape(samples * edges, 1),
                edge_features.reshape(samples * edges, edge_features.shape[-1]),
            ).reshape(samples, edges)
            coordinates = coordinates + torch.nn.functional.softplus(raw_step) * (
                residual @ encoder.T
            )
        return q_part + coordinates @ basis.T, coordinates


class RawCoordinateBaseline(nn.Module):
    """Negative baseline whose learned outputs are tied to one raw tree chart."""

    def __init__(self, edges: int, beta: int, feature_channels: int, hidden: int) -> None:
        super().__init__()
        width = edges * (feature_channels + 2)
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, beta),
        )

    def forward(
        self,
        basis: Tensor,
        observed: Tensor,
        q_part: Tensor,
        anchor: Tensor,
        edge_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cycle_anchor = anchor @ basis.T
        inputs = torch.cat(
            [q_part[..., None], cycle_anchor[..., None], edge_features], dim=-1
        ).flatten(start_dim=1)
        proposed = self.network(inputs)
        projector = hard_observation_coordinate_projector(basis, observed)
        coordinates = anchor + proposed @ projector.T
        return q_part + coordinates @ basis.T, coordinates


def _build_dataset(args: argparse.Namespace) -> tuple[Dataset, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    edges = make_connected_graph(args.nodes, args.extra_edges, seed=args.seed)
    incidence = incidence_matrix(args.nodes, edges)
    tree_train = spanning_tree_indices(args.nodes, edges, mode="bfs")
    basis_train = fundamental_cycle_basis(incidence, tree_train)
    for offset in range(1, 100):
        tree_unseen = spanning_tree_indices(
            args.nodes, edges, mode="random", seed=args.seed + offset
        )
        basis_unseen = fundamental_cycle_basis(incidence, tree_unseen)
        if not np.array_equal(basis_train, basis_unseen):
            break
    else:
        raise RuntimeError("failed to construct a distinct unseen spanning-tree chart")

    latent = structured_cycle_flows(
        incidence,
        num_samples=args.samples,
        seed=args.seed + 101,
        scale=args.cycle_scale,
        return_latents=True,
    )
    potentials = latent.node_features[..., 0]
    potentials -= potentials.mean(axis=1, keepdims=True)
    gradients = np.einsum("mn,sn->sm", incidence, potentials)
    static_conductance = 0.35 + np.logaddexp(0.0, rng.normal(size=incidence.shape[0]))
    q_part = gradients * static_conductance[None, :]
    target = q_part + latent.cycle_flows

    observed_count = max(1, int(round(args.observed_fraction * incidence.shape[0])))
    observed_count = min(observed_count, max(1, basis_train.shape[1] - 1))
    observed = np.sort(rng.choice(incidence.shape[0], observed_count, replace=False))
    missing = np.asarray(
        [edge for edge in range(incidence.shape[0]) if edge not in set(observed)],
        dtype=np.int64,
    )
    observation_mask = np.zeros(incidence.shape[0])
    observation_mask[observed] = 1.0
    observed_filled = np.zeros_like(target)
    observed_filled[:, observed] = target[:, observed]
    endpoint_magnitude = np.einsum(
        "mn,sn->sm", np.abs(incidence), np.abs(latent.node_features[..., 1])
    )
    edge_features = np.concatenate(
        [
            latent.edge_features,
            endpoint_magnitude[..., None],
            np.broadcast_to(static_conductance, target.shape)[..., None],
            np.broadcast_to(observation_mask, target.shape)[..., None],
            np.abs(observed_filled)[..., None],
        ],
        axis=-1,
    )

    anchor_train = np.stack(
        [
            hard_observation_affine(
                q_part[index], basis_train, observed, target[index, observed]
            ).anchor
            for index in range(args.samples)
        ]
    )
    anchor_unseen = np.stack(
        [
            hard_observation_affine(
                q_part[index], basis_unseen, observed, target[index, observed]
            ).anchor
            for index in range(args.samples)
        ]
    )
    return (
        Dataset(
            q_part=q_part,
            target=target,
            edge_features=edge_features,
            anchor_train=anchor_train,
            anchor_unseen=anchor_unseen,
            observed=observed,
            missing=missing,
        ),
        incidence,
        basis_train,
        basis_unseen,
    )


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _rmse(prediction: Tensor, target: Tensor, indices: Tensor | None = None) -> float:
    difference = prediction - target
    if indices is not None:
        difference = difference.index_select(1, indices)
    return float(torch.sqrt(torch.mean(difference.square())).item())


def _evaluate(
    model: nn.Module,
    basis: Tensor,
    observed: Tensor,
    q_part: Tensor,
    anchor: Tensor,
    features: Tensor,
    target: Tensor,
    missing: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        prediction, _ = model(basis, observed, q_part, anchor, features)
    metrics = {
        "full_rmse": _rmse(prediction, target),
        "missing_rmse": _rmse(prediction, target, missing),
        "observed_max_abs_error": float(
            torch.max(
                torch.abs(prediction.index_select(1, observed) - target.index_select(1, observed))
            ).item()
        ),
    }
    return prediction, metrics


def _run_impl(
    args: argparse.Namespace, device: torch.device
) -> tuple[pd.DataFrame, dict[str, object]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset, incidence_np, basis_train_np, basis_unseen_np = _build_dataset(args)
    samples = dataset.target.shape[0]
    train_end = int(0.7 * samples)
    valid_end = int(0.85 * samples)
    train_index = torch.arange(0, train_end, device=device)
    test_index = torch.arange(valid_end, samples, device=device)

    q_part = _tensor(dataset.q_part, device)
    target = _tensor(dataset.target, device)
    features = _tensor(dataset.edge_features, device)
    anchor_train = _tensor(dataset.anchor_train, device)
    anchor_unseen = _tensor(dataset.anchor_unseen, device)
    basis_train = _tensor(basis_train_np, device)
    basis_unseen = _tensor(basis_unseen_np, device)
    observed = torch.as_tensor(dataset.observed, dtype=torch.long, device=device)
    missing = torch.as_tensor(dataset.missing, dtype=torch.long, device=device)

    physical = PhysicalCompletionModel(features.shape[-1], args.hidden, args.depth).to(
        device=device, dtype=torch.float64
    )
    raw = RawCoordinateBaseline(
        edges=target.shape[1],
        beta=basis_train.shape[1],
        feature_channels=features.shape[-1],
        hidden=args.hidden,
    ).to(device=device, dtype=torch.float64)
    optimizers = {
        "physical": torch.optim.Adam(physical.parameters(), lr=args.learning_rate),
        "raw_coordinate": torch.optim.Adam(raw.parameters(), lr=args.learning_rate),
    }
    models: dict[str, nn.Module] = {"physical": physical, "raw_coordinate": raw}
    history: list[dict[str, float | int | str]] = []
    for epoch in range(args.epochs):
        for name, model in models.items():
            model.train()
            prediction, _ = model(
                basis_train,
                observed,
                q_part.index_select(0, train_index),
                anchor_train.index_select(0, train_index),
                features.index_select(0, train_index),
            )
            error = prediction.index_select(1, missing) - target.index_select(
                0, train_index
            ).index_select(1, missing)
            loss = error.square().mean()
            optimizers[name].zero_grad()
            loss.backward()
            optimizers[name].step()
            history.append({"epoch": epoch, "model": name, "train_mse": float(loss.item())})

    test_q = q_part.index_select(0, test_index)
    test_target = target.index_select(0, test_index)
    test_features = features.index_select(0, test_index)
    test_anchor_train = anchor_train.index_select(0, test_index)
    test_anchor_unseen = anchor_unseen.index_select(0, test_index)

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, Tensor] = {}
    for name, model in models.items():
        same_prediction, same_metrics = _evaluate(
            model,
            basis_train,
            observed,
            test_q,
            test_anchor_train,
            test_features,
            test_target,
            missing,
        )
        unseen_prediction, unseen_metrics = _evaluate(
            model,
            basis_unseen,
            observed,
            test_q,
            test_anchor_unseen,
            test_features,
            test_target,
            missing,
        )
        same_metrics["unseen_chart_missing_rmse"] = unseen_metrics["missing_rmse"]
        same_metrics["chart_variation_max_abs"] = float(
            torch.max(torch.abs(same_prediction - unseen_prediction)).item()
        )
        same_metrics["unseen_observed_max_abs_error"] = unseen_metrics["observed_max_abs_error"]
        metrics[name] = same_metrics
        predictions[name] = same_prediction

    particular_prediction = test_q
    anchor_prediction = test_q + test_anchor_train @ basis_train.T
    ridge_predictions = []
    for index in range(valid_end, samples):
        ridge_predictions.append(
            analytic_cycle_completion(
                dataset.q_part[index],
                basis_train_np,
                dataset.observed,
                dataset.target[index, dataset.observed],
                ridge=args.ridge,
            ).flow
        )
    ridge_prediction = _tensor(np.stack(ridge_predictions), device)
    for name, prediction in {
        "particular_only": particular_prediction,
        "analytic_anchor": anchor_prediction,
        "analytic_ridge": ridge_prediction,
    }.items():
        metrics[name] = {
            "full_rmse": _rmse(prediction, test_target),
            "missing_rmse": _rmse(prediction, test_target, missing),
            "observed_max_abs_error": float(
                torch.max(
                    torch.abs(
                        prediction.index_select(1, observed) - test_target.index_select(1, observed)
                    )
                ).item()
            ),
        }

    physical_prediction = predictions["physical"]
    conservation_error = physical_prediction @ _tensor(incidence_np, device) - test_q @ _tensor(
        incidence_np, device
    )
    metrics["physical"]["conservation_max_abs_error"] = float(
        torch.max(torch.abs(conservation_error)).item()
    )
    summary: dict[str, object] = {
        "experiment": "fixed_C_hard_observation_completion",
        "device": str(device),
        "nodes": args.nodes,
        "edges": int(incidence_np.shape[0]),
        "cycle_rank": int(basis_train_np.shape[1]),
        "samples": samples,
        "observed_edges": dataset.observed.tolist(),
        "observation_rank": int(np.linalg.matrix_rank(basis_train_np[dataset.observed])),
        "epochs": args.epochs,
        "physical_batch_size_samples": train_end,
        "gradient_accumulation_steps": 1,
        "data_parallel_workers": 1,
        "effective_batch_size_samples": train_end,
        "optimizer_steps": args.epochs * len(models),
        "model_parameter_counts": {
            name: {
                "total": sum(parameter.numel() for parameter in model.parameters()),
                "trainable": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            }
            for name, model in models.items()
        },
        "debug_subset_fast_mode": False,
        "metrics": metrics,
    }
    return pd.DataFrame.from_records(history), summary


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    """Execute the requested device explicitly and persist measured resource use."""

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for fixed-C completion training but is unavailable; "
            "no CPU fallback was performed"
        )
    if device.type == "cuda":
        torch.cuda.get_device_properties(device)
        torch.cuda.reset_peak_memory_stats(device)
    monitor = RuntimeResourceMonitor(device)
    monitor.start()
    try:
        history, summary = _run_impl(args, device)
    except BaseException as primary_error:
        try:
            monitor.finish(
                peak_allocated_bytes=(
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
                peak_reserved_bytes=(
                    int(torch.cuda.max_memory_reserved(device))
                    if device.type == "cuda"
                    else None
                ),
            )
        except (Exception, KeyboardInterrupt) as monitor_error:
            primary_error.add_note(
                "resource monitor cleanup also failed: "
                f"{type(monitor_error).__name__}: {monitor_error}"
            )
        raise
    resources = monitor.finish(
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
    )
    elapsed = float(resources["summary"]["observed_wall_seconds"]["value"])
    summary["resource_observability"] = resources
    summary["throughput"] = {
        "scope": "two-model full-batch optimization plus final evaluation",
        "optimizer_steps_per_second": summary["optimizer_steps"] / elapsed,
        "training_sample_presentations_per_second": (
            summary["optimizer_steps"] * summary["physical_batch_size_samples"] / elapsed
        ),
    }
    return history, summary


def _plot_history(history: pd.DataFrame, output: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for name, group in history.groupby("model"):
        axis.plot(group["epoch"], group["train_mse"], label=name)
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("missing-edge train MSE")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/combined_later/fixed_c"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nodes", type=int, default=14)
    parser.add_argument("--extra-edges", type=int, default=9)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--observed-fraction", type=float, default=0.25)
    parser.add_argument("--cycle-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="explicit execution device; CUDA unavailability is an error, never a CPU fallback",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            f"Output path already exists: {args.output_dir}; choose a new path"
        )
    history, summary = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    history.to_csv(args.output_dir / "training.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_history(history, args.output_dir / "training.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
