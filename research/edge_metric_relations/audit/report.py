"""Fixed audit scope and numerical limitations, backed by complete raw shards."""
from __future__ import annotations


def write_report(root, config, coverage, results, checks):
    uncertain = sum(row["stationarity_unresolved_channel_solves"] for row in coverage)
    observation = sum(row["observation_residual_outside_tolerance_channel_solves"] for row in coverage)
    unresolved_targets = sum(row["unresolved_target_rows"] for row in coverage)
    lines = ["# Stage A: fixed-operator information audit", "",
             f"Profile: **{config['profile']}**. Graphs: {len(coverage)//2}. All original nodes, edges and feature channels were retained.", "",
             "## What was measured", "",
             "The audit fixes every coefficient at a stated topology or original input reference. It compares 21 operator views per weighting recipe, full/target/one-hop observations, and deterministic relative noise levels 0, 1e-6, 1e-3.", "",
             "Synthetic scalar draws remain independent. Citation channels form one vector trace; signed energy/relation terms are summed over all channels before reconstruction error is computed.", "",
             "Small synthetic graphs use direct FP64 SVD with relative tolerance 1e-10. Raw collision rows contain actual x+ and x- witnesses and their observation/target residuals. This is numerical rank, rather than a symbolic rank proof.", "",
             "Citation graphs use zero-start matrix-free primal normal-equation CG: Aᵀ A x = Aᵀ y, where A is the exact masked/scaled fixed operator. Converged solutions are numerical minimum-norm least-squares estimates, including inconsistent noisy observations. They have no exact rank/identifiability claim. Both directions of near-hidden visibility include solver qualification and measured observation residuals.", "",
             f"Channel solves with unresolved true normal stationarity residual or breakdown: **{uncertain}**. Complete q/E/J target rows excluded from successful summaries because at least one original channel is unresolved: **{unresolved_targets}**. Their full-vector errors remain in raw shards with solver_converged=false; channels are never silently removed from the scientific target.", "",
             f"Channel observation residuals outside tolerance: **{observation}**. This is a separate noise/irreducible mismatch diagnostic, not solver nonconvergence. A valid noisy least-squares optimum can have nonzero observation residual.", "",
             "Direct SVD estimates use the declared numerical rank threshold and the same normal stationarity qualification. Success summaries include only complete vectors whose every channel passed the solver criterion; unresolved counts are retained beside those summaries.", "",
             "## Evidence", "",
             "- `coverage.json`: every graph/recipe and original channel count.",
             "- `raw_manifest.json`: lossless CSV.gz shards, full row counts, bytes and SHA256.",
             "- `calibration.json`: measured graph batches, target/channel/pair work chunks and VRAM.",
             "- `scheduling.json` and `hardware.json`: all visible allocated GPU jobs and resource facts.",
             "- `source_input_manifest.json` and `source_manifest.json`: input and implementation closures.",
             "- `summary.json`: descriptive successful full-vector target averages and unresolved counts; adjacent targets are not independent research replicates.", "",
             "## Scope", "",
             f"Independent DEBUG algebra/autograd checks passed: {checks['math_checks']}. Those fixture checks are separate from source-graph audit observations.", "",
             "No classifier or learned C is trained by Stage A. Selected old model checkpoint states were not supplied, so this run is a fixed-operator audit. Input-dependent learned-gate differentiation, trained-state information effects and real classification performance require their own later evidence.", "",
             "Positive definite edge metrics preserve ker(B); changing fixed coefficients does not recover information absent from a full raw incidence observation. Residual propagation P and partial observations OF have different kernels and conditioning and are audited separately.", ""]
    with (root/"REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
