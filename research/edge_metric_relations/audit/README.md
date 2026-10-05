# Stage A — fixed edge metric information audit

This stage checks which original local targets can be distinguished from a fixed operator's observations. It does not train a classifier or pretend that old selected checkpoint states were supplied.

## Operator views

| Operator | Repetitions | Meaning |
| --- | --- | --- |
| `L0` | 1 | physical unit Laplacian |
| `Ld` | 1, 2 | the same averaged physical diagonal-weight Laplacian and its square |
| `smooth_L0`, `smooth_Ld` | 1, 2, 4, 8, 16 | per-physical-graph safe smoothing with the named fixed Laplacian |
| `copy_off`, `copy_on` | 1 | the previous fixed sandwich operator, used only as an audit comparison |
| `Q_diag`, `Q_F0`, `Q_reference` | 1 | raw new physical-node operator; diagonal, topology fixture, or coefficients frozen at the original reference input |
| `P_diag`, `P_F0`, `P_reference` | 1 | `I − (1/3) N0 Q N0`; separate observations from raw Q |

There are 21 views for each of `unit` and `local_degree`. F0 has raw pair coefficient 1. The reference rule is `tanh(2 alignment + .5(2 overlap − 1))`, with smooth RMS epsilon 1e-4. Its coefficients are frozen before null-space or reconstruction inputs are changed.

FULL reads all **201 original graph snapshots** from the completed initial local-energy audit: 198 synthetic graphs, 3,168 independent scalar draws, and the complete Cora/CiteSeer/PubMed vector features. DEBUG reads the separately contracted 21 snapshots; its citation fixtures are explicitly artificial.

## Observations and targets

Each view uses full-node, each individual target-node, and each target's original 1-hop node-set observations. Every original target and directed adjacent pair is included. Targets are the actual local vector `q_v`, local energy `E_v`, and individual `J_shared`, `J_node`, `J_distinct` values. Synthetic columns are independent scalar draws; citation channels are one vector trace.

For a fixed observation `F_obs = O F`, a linear target T is identifiable for all inputs only if `ker(F_obs) ⊆ ker(T)`. A symmetric quadratic target K needs `ker(F_obs) ⊆ ker(K)`; merely testing `nᵀ K n = 0` is insufficient.

Small synthetic graphs use direct FP64 SVD with declared relative threshold 1e-10. Raw certificates export actual x+/x− vectors, observation residual and target difference, including linear q collisions. Rank is numerical with a threshold, rather than a symbolic proof. Both current-hidden/reference-visible and reference-hidden/current-visible directions are measured.

Citation graphs use zero-start matrix-free **primal** normal-equation CG and actual original-input reconstruction differences. They **do not** get an exact null rank claim. Each solver record contains every channel's coverage count, convergence count, true normal stationarity residual and observation residual. A failed solver criterion stays visible; completion means the requested audit work was processed, not that every inverse succeeded.

Noise levels 0, 1e-6 and 1e-3 use an explicitly deterministic sine probe scaled to each observed RHS norm. These probes are not presented as natural data or independent statistical samples. Zero RHS and zero target norms have explicit unavailable ratios. No ridge is silently inserted into an inconsistent singular system.

### Least squares and numerical qualification

For `A = O F / scale` and noisy observation y, the citation path solves `Aᵀ A x = Aᵀ y` from x=0. Its RHS is always in `range(Aᵀ)`; converged zero-start Krylov solutions are numerical minimum-norm least-squares estimates. The direct SVD path computes a truncated-pseudoinverse estimate at the declared relative rank threshold 1e-10. Both paths test the **true normal stationarity residual**, `||Aᵀ(Ax−y)|| / ||Aᵀy||`, at tolerance 1e-8. Zero normal RHS has an explicit zero-residual criterion and an unavailable relative denominator; no hidden epsilon changes the metric.

The separate observation residual `||Ax−y|| / ||y||` measures irreducible noise or mismatch. It can be nonzero at a valid least-squares optimum and is **not** solver nonconvergence. Raw tables preserve both norms, denominator availability, per-channel qualification and breakdown counts. Every q/E/J recovery row is qualified using every original channel of its observer. If even one constituent channel is unresolved, its complete vector target remains in raw output with `solver_converged=false` and is excluded from successful recovery averages. `summary.json` records both successful and unresolved row counts; it never averages a converged feature subset as the original vector.

CPU and CUDA tests compare packed noisy rank-deficient graphs with disconnected components and isolates against independent dense pseudoinverses. They also test zero RHS, short-iteration unresolved flags, stationarity versus irreducible observation residuals, and complete-vector summary exclusion. Conditioning and finite iteration accuracy remain measured numerical limits, not exact citation identifiability proofs.

The preserved earlier DEBUG attempt 02 used a dual equation with an inconsistent noisy RHS. Its historical summaries include unresolved iterates and must not be used as validated noisy least-squares results. The current solver and its output contract supersede that attempt; earlier files are unchanged.

## Resources and provenance

All original inputs and topology snapshots are hash checked and cached once through the strict existing snapshot reader. Synthetic physical graphs are batched by node count for operator construction. Target observations, scalar draws and vector channels are tensor packed. Measured physical graph batches and target/channel/pair chunks change work memory only. CPU cache worker candidates are measured. Independent graph/recipe jobs use all visible allocated GPUs.

FULL requires a Linux CUDA server and explicit `CUDA_VISIBLE_DEVICES`. Output directories must be new. `terminal.log`, `hardware.json`, `calibration.json`, `scheduling.json`, `coverage.json`, input/code manifests, all lossless CSV.gz shards and completion are preserved. Failures record `failure.json` without ending the shell or deleting earlier results.

Selected trained model checkpoints are unavailable in the received evidence. `--trained-source-dir` explicitly rejects that unsupported scope instead of using fake states. Post-training frozen-state audits belong to the classification pipeline.

## Server command

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=0 \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.edge_metric_relations.audit.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --output-dir "results/edge-metric-A-$(date +%Y%m%d-%H%M%S)"
```

Use the actual allocated GPU IDs in `CUDA_VISIBLE_DEVICES`. All listed IDs are used. The original source path must identify the complete initial audit with its original manifests and numeric snapshots.

`summary.json` is descriptive, not a classification score or a confidence interval from independent seeds. `raw_manifest.json` names every evidence shard. `REPORT.md` states solver uncertainty and audit scope. No FULL experiment has been run locally.
