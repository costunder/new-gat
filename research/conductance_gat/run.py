"""Train and compare learned edge conductance against an isotropic baseline."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .model import IncidenceConductanceAttention, IsotropicConductanceAttention
from .synthetic import (
    ConductanceDataset,
    evaluate_model,
    make_conductance_dataset,
    split_excitations,
)


def _training_loss(
    model: nn.Module,
    dataset: ConductanceDataset,
    *,
    node_weight: float,
    flux_weight: float,
    conductance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted_next, diagnostics = model(
        dataset.incidence,
        dataset.potentials,
        dataset.edge_features,
        return_diagnostics=True,
    )
    node_loss = torch.mean((predicted_next - dataset.true_next_state).square())
    flux_loss = torch.mean((diagnostics["edge_flux"] - dataset.true_flux).square())
    predicted_c = diagnostics["conductance"].mean(dim=0).squeeze(-1)
    conductance_loss = torch.mean((predicted_c - dataset.true_conductance).square())
    loss = node_weight * node_loss + flux_weight * flux_loss + conductance_weight * conductance_loss
    terms = {
        "loss": float(loss.detach().cpu()),
        "node_mse": float(node_loss.detach().cpu()),
        "flux_mse": float(flux_loss.detach().cpu()),
        "conductance_mse": float(conductance_loss.detach().cpu()),
    }
    return loss, terms


def train_model(
    model: nn.Module,
    dataset: ConductanceDataset,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    node_weight: float,
    flux_weight: float,
    conductance_weight: float,
    log_every: int = 10,
) -> list[dict[str, float]]:
    if epochs < 1 or log_every < 1:
        raise ValueError("epochs and log_every must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = _training_loss(
            model,
            dataset,
            node_weight=node_weight,
            flux_weight=flux_weight,
            conductance_weight=conductance_weight,
        )
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            history.append({"epoch": float(epoch), **terms})
    return history


def resolve_device(requested: str | None) -> torch.device:
    """Resolve a portable device request without assuming the launch host."""
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {normalized!r} was requested, but this PyTorch build cannot use CUDA"
        )
    return device


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    """Record enough host information to distinguish CPU/CUDA experiment runs."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = resolve_device(config.get("device"))
    data_config = dict(config["data"])
    dataset = make_conductance_dataset(seed=seed, **data_config)
    train_data, test_data = split_excitations(
        dataset,
        train_fraction=float(config["training"]["train_fraction"]),
        seed=seed + 1,
    )
    train_data = train_data.to(device)
    test_data = test_data.to(device)

    model_config = dict(config["model"])
    common = {
        "channels": int(data_config["channels"]),
        "edge_feature_channels": int(data_config["edge_feature_channels"]),
        "step_size": dataset.step_size,
        "stability_margin": float(model_config.pop("stability_margin", 0.95)),
        "adaptive_stability": bool(model_config.pop("adaptive_stability", True)),
    }
    learned = IncidenceConductanceAttention(**common, **model_config).to(device)
    isotropic = IsotropicConductanceAttention(**common).to(device)

    training = config["training"]
    train_kwargs = {
        "epochs": int(training["epochs"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "node_weight": float(training["node_weight"]),
        "flux_weight": float(training["flux_weight"]),
        "conductance_weight": float(training["conductance_weight"]),
        "log_every": int(training["log_every"]),
    }
    learned_history = train_model(learned, train_data, **train_kwargs)
    isotropic_history = train_model(isotropic, train_data, **train_kwargs)

    summary: dict[str, Any] = {
        "scope": "incidence_conductance_attention_only",
        "seed": seed,
        "device": str(device),
        "runtime": runtime_metadata(device),
        "num_nodes": int(dataset.incidence.shape[1]),
        "num_edges": int(dataset.incidence.shape[0]),
        "train_excitations": train_data.num_excitations,
        "test_excitations": test_data.num_excitations,
        "step_size": dataset.step_size,
        "learned": evaluate_model(learned, test_data),
        "isotropic": evaluate_model(isotropic, test_data),
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _write_history(output_dir / "learned_history.csv", learned_history)
    _write_history(output_dir / "isotropic_history.csv", isotropic_history)
    portable_state = {
        name: parameter.detach().cpu() for name, parameter in learned.state_dict().items()
    }
    torch.save(portable_state, output_dir / "learned_model.pt")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="YAML experiment configuration",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime device override: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory override (defaults to the config-relative path)",
    )
    arguments = parser.parse_args()
    config_path = arguments.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if arguments.output_dir is not None:
        output_dir = arguments.output_dir.expanduser()
    else:
        output_dir = Path(config.get("output_dir", "results")).expanduser()
        if not output_dir.is_absolute():
            output_dir = config_path.parent / output_dir
    config["output_dir"] = str(output_dir)
    if arguments.device is not None:
        config["device"] = arguments.device
    run_experiment(config)


if __name__ == "__main__":
    main()
