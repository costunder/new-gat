"""Paired held-out accuracy and measured cost; no assumed sampling speedup."""

import copy


def recipe(identity):
    config = copy.deepcopy(identity["configuration"])
    for name in (
        "ablation_arm",
        "conductance_mode",
        "conductance_heads",
        "sampling",
        "batch_size",
        "workers",
        "loader_workers",
        "prefetch_factor",
        "persistent_workers",
    ):
        config.pop(name, None)
    for name in ("conductance", "c_control_lift"):
        config["comparison_contract"].pop(name, None)
    return config


def summarize(manifest):
    options = manifest["identity"]["configuration"]
    dynamic = "incidence" if options["dynamic_c"] == "per_head" else "incidence_shared"
    reports = []
    for seed in options["model_seeds"]:
        keys = {
            f"{mode}_{c}": f"seed-{seed}/{mode}/{arm}"
            for mode in ("full", "sampled")
            for c, arm in (("fixed", "incidence_fixed"), ("dynamic", dynamic))
        }
        if any(key not in manifest["results"] for key in keys.values()):
            reports.append({"seed": seed, "status": "incomplete"})
            continue
        cells = {name: manifest["results"][key] for name, key in keys.items()}
        first = cells["full_fixed"]
        for cell in cells.values():
            if (
                cell["status"] != "passed"
                or cell["identity"]["protocol"] != first["identity"]["protocol"]
                or cell["shared_initial_sha256"] != first["shared_initial_sha256"]
            ):
                raise ValueError("unpaired inductive cells")
            if cell["identity"]["sources"] != first["identity"]["sources"]:
                raise ValueError("unmatched inductive sources")
            if recipe(cell["identity"]) != recipe(first["identity"]):
                raise ValueError("unmatched non-ablated training recipe")
            if cell["identity"]["sampling_recipe"] != first["identity"]["sampling_recipe"]:
                raise ValueError("unmatched declared sampling recipe")
        score = {k: r["validation"]["metric"] for k, r in cells.items()}
        effects = {
            "C_given_full_pp": 100 * (score["full_dynamic"] - score["full_fixed"]),
            "C_given_sampled_pp": 100 * (score["sampled_dynamic"] - score["sampled_fixed"]),
            "sampling_given_fixed_pp": 100 * (score["sampled_fixed"] - score["full_fixed"]),
            "sampling_given_dynamic_pp": 100 * (score["sampled_dynamic"] - score["full_dynamic"]),
            "interaction_pp": 100
            * (
                score["sampled_dynamic"]
                - score["sampled_fixed"]
                - score["full_dynamic"]
                + score["full_fixed"]
            ),
        }
        cost_key = "wall_seconds_including_setup_train_validation_checkpoints_audit"
        costs = {k: r[cost_key] for k, r in cells.items()}
        comparable = all(r["cost_comparable"] for r in cells.values())
        reports.append(
            {
                "seed": seed,
                "status": "passed",
                "validation": score,
                "effects": effects,
                "cell_wall_seconds": costs,
                "cost_comparable": comparable,
                "full_over_sampled_wall_ratio": {
                    c: costs[f"full_{c}"] / costs[f"sampled_{c}"] for c in ("fixed", "dynamic")
                }
                if comparable
                else None,
                "cost_warning": "sequential wall timings, shared-device contention possible; "
                "resumed costs are not comparable",
                "test": {name: manifest.get("test", {}).get(key) for name, key in keys.items()},
            }
        )
    return {
        "suite": "sampled_inductive_v1",
        "seeds": reports,
        "budget": "same complete supervised-node epochs; physical units and optimizer steps "
        "differ and are recorded",
        "calibration": "resource search cost separate in manifest; cell wall time includes "
        "construction/transfer/C/forward/backward/optimizer/checkpoints/evaluation",
        "invocations_including_data_loading_and_calibration": manifest.get("invocations", []),
        "scope": "PPI official held-out graph generalization, "
        "not arbitrary-distribution guarantee or speedup claim",
    }
