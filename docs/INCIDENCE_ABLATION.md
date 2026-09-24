# Independent incidence, cross-hop and lifting ablations

This is a **fresh training suite**, separate from the historical edge-selection
experiment. It never imports old checkpoints or result rows. Official cached data
and splits are shared; training state, calibration, manifests, audits and output
paths are independent. Implementation lives in `experiments/incidence_ablation/`
so adding it does not change the historical `research/`, `scripts/`, or
`src/chartgat/` source snapshots.

## Experiment matrix

The default matrix contains eight independently trained conditions:

| Lift | Cross-hop off | Cross-hop on |
| --- | --- | --- |
| Original diffusion | `baseline` | `bilinear` |
| Linear duplicate lift | `linear_lift` | `bilinear_linear_lift` |
| Quadratic lift before diffusion | `pre_lift` | `bilinear_pre_lift` |
| Quadratic lift after diffusion | `post_lift` | `bilinear_post_lift` |

Let `V = H W`, `P = I - beta D^-1 L`, and `L = B^T C B`. The four
operator outputs, before the original output projection, are:

- Baseline: `P V`.
- Linear capacity control: `P [V,V] R`.
- Pre-lift: `P [V,V²] R`.
- Post-lift: `[PV,(PV)²] R`.

Here squaring is elementwise, `R` is trainable per head, and `P` uses the
original positive optimized conductances and full physical candidate support.
All three lift arms have identical added parameter counts. They initialize
`R=[I;0]`, so their initial functions match the baseline. Base parameters and RNG
are seed-paired exactly. The original encoder, layer norms and SwiGLU remain
nonlinear in every arm: this is a **lift-placement** comparison, not a claim to
compare an entirely linear network against a nonlinear network.

At depth `l`, cross-hop arms use **every preceding depth state**, normalized by
the current block and projected into the current head coordinates. For every
unordered pair `k<m`, they learn coefficients on
`<B H_k, B H_m>/sqrt(head_width)`. Weighted edge scores accumulate at both
endpoints and divide by weighted degree. `tanh(score) * V` is added to the
propagated value. Coefficients start at zero; the first block has no pair and no
unused pair parameters. This is conditionally bilinear with frozen `C`; the
complete adaptive network is nonlinear. Depth states are not exact-distance
hop shells. In particular, this does not prove that arbitrary disjoint hop edge
spaces can be multiplied without an alignment map.

## Fixed training contract

- `reference`: hidden 256, 8 layers, 8 heads; `large`: hidden 384, 12 layers,
  8 heads. Original encoder/decoder, FFN multiplier, dropout, conductance solver,
  dataset splits and sampling policy are retained.
- Defaults retain the existing 200 reference epochs / 50 patience contract with
  `reference_updates`: if measured graph batches grow, actual epochs adjust to
  preserve optimizer-update budgets. No model, graph or data cap is added.
- Datasets and profiles must be explicitly selected. Supported datasets are
  cora, citeseer, pubmed, ppi and ogbn-arxiv. PPI uses its full official splits.
- Calibration measures **all selected arms and seeds**, including optimizer
  state, complete validation, largest graph batches and topology preparation.
  It selects a common physical batch/worker plan for each dataset/profile.
  It does not silently reduce the requested physical batch when memory is tight.
- Per-head operations, all hop pairs and disjoint graph batches run as tensor
  operations. Edge streaming and activation checkpointing bound working memory;
  every edge remains in the computation. Hop chunks divide the existing edge
  chunk budget by the number of retained states.
- One invocation uses its explicitly selected GPU. Keep paired arms on that GPU
  for calibration. Independently allocated GPUs can run separate dataset/profile
  groups concurrently using distinct run IDs; do not start multiple full-memory
  calibrations on one GPU. No additional GPU is presumed allocated.

## What the audit measures

Each completed fresh checkpoint receives five complete validation repeats,
plus a complete diagnostic validation pass. No test labels are evaluated.
The audit checks model/artifact hashes before and after and records its resources.

For every layer and every validation graph batch it records:

1. Incidence rank `N-components` and Laplacian nullity `components`, including
   isolated nodes. These are structural ranks for positive weights.
2. Frozen-metric cross-depth energy Gram matrices, using all encoder/block
   states, all edges and the final layer's mean-head conductance metric.
3. Numerical feature ranks of `X`, `[X,X²]`, `BX`, `B[X,X²]` and `[BX,(BX)²]`,
   with the Gram-eigenvalue threshold and spectral entropy rank. Feature rank
   is not the Jacobian rank or evidence that the complete model is invertible.
4. Actual centered reconstruction from edge observations, using FP64 batched
   sparse Jacobi-PCG, under clean observations and relative RMS noise `1e-3`.
5. Linear minimum-norm error, a separately labelled oracle-component-mean
   control, and quadratic pre-lift recovery error on identifiable coordinates.
   Unidentifiable component/features and observed edge energies are recorded.
   An absent denominator produces `null`, never a fabricated zero error.

For centered recovered `Y`, the constrained quadratic inverse estimates each
component/feature mean as
`sum w*(BY)*(B(X²)-B(Y²)) / (2*sum w*(BY)²)`.
Only the oracle control reads true component means. The reconstruction probe
uses the arithmetic mean of positive effective head conductances, frozen at the
forward pass, including sampling correction. It is explicitly a diagnostic
metric, not an assertion that averaging heads reproduces their trained operator.

Constant component/features cannot be identified from `B[X,X²]`; near-constant
ones can be unstable. Identifiability uses a reported relative edge-energy
threshold of `1e-12`. Noise can make an actually constant coordinate appear to
have variation; compare clean/noisy counts and errors together. `phi(BX)` cannot
restore discarded component means. In contrast, the trained propagation is
`P`, which retains constants: **the incidence reconstruction result is not an
inverse of P, the encoder, or the full network**. A rank increase by itself is
not sufficient evidence for recovery.

Additional evaluation interventions disable the trained cross-hop branch or
remove the second lift channel, one at a time. These use complete validation,
no retraining, and report prediction flips and score changes. Their effects
include distribution shift; primary comparisons are the freshly retrained arms.

PCG nonconvergence fails the audit explicitly and preserves completed training.
The runner accepts `--cg-tolerance` (default `1e-7`) and `--cg-iterations`
(default 2000). These audit-only options can change on resume to rerun the audit
without retraining. Tolerance is logged; a looser tolerance is not equivalent
accuracy evidence. Scientific training settings and source hashes remain strict.

## Server commands

### A100 MIG 10GB

Use the existing `portable` hardware recipe for the new independent run:
FP32, TF32 disabled, checkpointing enabled, PPI physical batch candidates starting
at the established portable batch of 2 and increasing through measured calibration.
This differs from the A6000 recipe (BF16 and baseline batch 8); compare all eight
new arms within this portable recipe, not directly against old A6000 scores.
The model remains 8 layers / hidden 256 / 8 heads on `reference`, and all official
PPI data are retained. `--edge-chunk-size 4096` bounds temporary edge tensors;
it is not graph sampling. The 200-reference-epoch update budget is retained for
the portable recipe. Calibration must leave at least 2 GiB measured free headroom.
No successful fit on a real 10GB slice has been measured locally.

Run in the shell where your A100 MIG instance is already allocated. Preserve
its `CUDA_VISIBLE_DEVICES` (often a MIG UUID); do not replace it with the old
physical GPU index 6. The command below verifies that visible device 0 is an
A100 with 8–11 GiB capacity before training. A failed check only stops the
command chain; it does not close the shell or modify GPU/MIG settings.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK /home/aicompetition07/.conda/envs/new-gat/bin/python -B -c 'import torch; p=torch.cuda.get_device_properties(0); g=p.total_memory/2**30; print(p.name, round(g,2), "GiB", flush=True); assert "A100" in p.name.upper() and 8 <= g <= 11, "Visible cuda:0 is not the allocated A100 MIG 10GB; check the allocation"' &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -B -m experiments.incidence_ablation \
  --run-id incidence-ppi-reference-mig10gb-seed0-v1 \
  --datasets ppi --profiles reference --model-seeds 0 \
  --device cuda:0 --hardware-profile portable \
  --edge-chunk-size 4096 --activation-checkpoint --min-free-gb 8
```

This starts all eight new arms, preceded by all-arm resource calibration. If no
candidate fits with the required headroom, it reports failure and preserves
evidence; it never switches to a smaller model, partial dataset or physical batch 1.
Repeating this exact command resumes the same independent run.

### A6000 48GB (different recipe)

From the repository root, preview the full eight-arm PPI/reference matrix:

```bash
/home/aicompetition07/.conda/envs/new-gat/bin/python -B -m experiments.incidence_ablation \
  --run-id incidence-ppi-reference-seed0-v1 \
  --datasets ppi --profiles reference --model-seeds 0 --dry-run
```

Run that explicitly selected group on the previously selected visible GPU 6:

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=6 \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -B -m experiments.incidence_ablation \
  --run-id incidence-ppi-reference-seed0-v1 \
  --datasets ppi --profiles reference --model-seeds 0 \
  --device cuda:0 --hardware-profile a6000-48gb --activation-checkpoint
```

This is eight full trainings, not the old 130/160-condition sweep. The first
stage is measured resource calibration, so it is not an immediate performance
report. Repeating the identical command verifies completed jobs and resumes
incomplete jobs from committed epoch boundaries.

Outputs: `results/incidence_ablation/<run-id>/comparison.md`, `manifest.json`,
and per-arm checkpoints/history/audit logs. The old edge-selection run directory
is never used. Use a new run ID if changing datasets, profiles, seeds or arms.
`--profiles reference large --datasets cora citeseer pubmed ppi ogbn-arxiv`
selects the full 80-training matrix per seed; nothing chooses that expensive
matrix implicitly. A single-seed, validation-only result cannot support SOTA
or seed-robust significance claims.

## Local verification and limits

Synthetic CPU tests cover all eight arms, common initialization, equality to
the original full-support baseline, finite gradients and optimizer updates,
checkpoint recomputation, state restoration, complete training epochs and CPU
RNG/optimizer continuation, dense bilinear/Gram identities, orientation
invariance, pseudoinverse agreement, disconnected/constant recovery, noise,
full diagnostic observer execution and independent source/recipe contracts.
These are explicitly debug fixtures, not reduced final training profiles.

The local PyTorch build is CPU-only. No real-data GPU training, final evaluation,
server resource calibration or PPI accuracy claim has been produced by these
tests. The server must measure throughput and memory before research runs.
