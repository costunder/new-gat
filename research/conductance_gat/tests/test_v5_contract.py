import hashlib
import json

from research.conductance_gat.v5.model import GraphConditionedConductanceNodeClassifier
from research.conductance_gat.v5.protocol import COMPARISON_DESIGN
from research.conductance_gat.v5.report import build_comparison
from research.conductance_gat.v5.train import (
    architecture_configuration,
    build_parser,
    configuration,
    configure_phase,
    phase_schedule,
    validate_args,
)


def small_model(mode):
    return GraphConditionedConductanceNodeClassifier(
        8,
        3,
        hidden_channels=32,
        layers=1,
        heads=4,
        ffn_multiplier=2,
        conductance_mode=mode,
        activation_checkpoint=False,
    )


def test_fixed_control_has_updates_in_every_scheduled_phase():
    network = small_model("fixed_one")
    for phase in ("spatial_warmup", "conductance_calibration", "alternating", "joint"):
        state = configure_phase(network, phase, 0)
        assert state["active_parameter_groups"]
        assert "conductance" not in state["active_parameter_groups"]
    calibration = configure_phase(network, "conductance_calibration", 0)
    assert calibration["coordinate"] == "fixed_spatial_control"
    assert calibration["active_parameter_groups"] == ["backbone", "beta", "spatial_w"]


def test_schedule_always_reserves_joint_selection_epochs():
    for epochs in (4, 5, 17, 300):
        schedule = phase_schedule(epochs, [0.1, 0.1, 0.4, 0.4])
        assert schedule[-1]["name"] == "joint"
        assert schedule[-1]["length"] >= 1
        assert sum(item["length"] for item in schedule) == epochs


def test_cli_resume_and_sampling_contract():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "ogbn-arxiv",
            "--condition",
            "shared_dynamic_c",
            "--output-dir",
            "out",
            "--sampling",
            "neighbor",
            "--num-neighbors",
            "15",
            "10",
            "5",
        ]
    )
    validate_args(args)
    assert args.resume is True
    assert args.activation_checkpoint is True
    assert args.sample_seed_batch_size == 1024
    assert args.num_neighbors == [15, 10, 5]
    architecture = architecture_configuration(args)
    assert architecture["beta_parameterization"] == "sigmoid"
    assert architecture["beta_initial"] == 0.1
    assert "beta_min" not in architecture and "beta_max" not in architecture
    assert "beta_min" not in configuration(args) and "beta_max" not in configuration(args)


def test_cli_can_reproduce_historical_margin_beta_ablation():
    args = build_parser().parse_args(
        [
            "--dataset",
            "cora",
            "--condition",
            "shared_dynamic_c",
            "--output-dir",
            "out",
            "--beta-parameterization",
            "margin_sigmoid",
            "--beta-initial",
            "0.5",
            "--beta-min",
            "0.05",
            "--beta-max",
            "0.95",
        ]
    )
    validate_args(args)
    beta = {
        key: value
        for key, value in architecture_configuration(args).items()
        if key.startswith("beta_")
    }
    assert beta == {
        "beta_parameterization": "margin_sigmoid",
        "beta_initial": 0.5,
        "beta_min": 0.05,
        "beta_max": 0.95,
    }


def test_report_is_partial_safe_then_requires_complete_pairs(tmp_path):
    manifest = {
        "status": "running",
        "config": {"datasets": ["cora"], "model_seed": 0, "batch_size": 1},
        "jobs": [
            {
                "dataset": "cora",
                "condition": condition,
                "architecture": {
                    "hidden_channels": 32,
                    "layers": 1,
                    "heads": 4,
                    "ffn_multiplier": 2,
                    "dropout": 0.0,
                },
                "sampling": "full",
                "status": "pending",
                "output_dir": str(tmp_path / condition),
            }
            for condition in ("fixed_c", "shared_dynamic_c")
        ],
    }
    partial = build_comparison(tmp_path, manifest)
    assert partial["status"] == "partial"
    assert partial["rows"] == []
    for job in manifest["jobs"]:
        output = tmp_path / job["condition"]
        output.mkdir()
        checkpoint, last, history = output / "best.pt", output / "last.pt", output / "history.json"
        checkpoint.write_bytes(b"best")
        last.write_bytes(b"last")
        history.write_text("[]", encoding="utf-8")

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        cache_sha256 = "c" * 64
        source_sha256 = {"implementation.py": "s" * 64}
        initial_state_sha256 = "i" * 64
        protocol = {"data_sha256": cache_sha256, "dataset": "cora"}
        configuration = {
            **job["architecture"],
            "model_seed": 0,
            "batch_size": 1,
            "sampling": "full",
        }
        schedule = [{"name": "joint", "start_epoch": 1, "end_epoch": 4, "length": 4}]
        identity = {
            "cache_sha256": cache_sha256,
            "source_sha256": source_sha256,
            "initial_state_sha256": initial_state_sha256,
        }
        identity_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "research_suite": "conductance_graph_conditioned_v5",
                    "dataset": "cora",
                    "condition": job["condition"],
                    "model_seed": 0,
                    "evaluation_split": "validation",
                    "test_evaluated": False,
                    "validation": 0.8,
                    "metric_name": "accuracy",
                    "total_parameters": 10,
                    "trainable_parameters": 8,
                    "allocated_parameter_capacity": 10,
                    "best_epoch": 4,
                    "configuration": configuration,
                    "schedule": schedule,
                    "cache_sha256": cache_sha256,
                    "source_sha256": source_sha256,
                    "versions": {"torch": "fixture"},
                    "initial_state_sha256": initial_state_sha256,
                    "protocol": protocol,
                    "resume_identity": identity,
                    "resume_identity_sha256": identity_sha256,
                    "comparison_design": COMPARISON_DESIGN,
                    "effective_optimizer_steps_by_group": {
                        "backbone": 4,
                        "spatial_w": 4,
                        "beta": 4,
                        "conductance": 0,
                    },
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": digest(checkpoint),
                    "last_checkpoint": str(last.resolve()),
                    "last_checkpoint_sha256": digest(last),
                    "history": str(history.resolve()),
                    "history_sha256": digest(history),
                    "selected_checkpoint_recheck": {
                        "recorded": 0.8,
                        "recomputed": 0.8,
                        "delta": 0.0,
                        "non_gating": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        job["status"] = "passed"
    manifest["status"] = "passed"
    complete = build_comparison(tmp_path, manifest)
    assert complete["status"] == "passed"
    assert complete["contrasts"][0]["dynamic_minus_fixed"] == 0
