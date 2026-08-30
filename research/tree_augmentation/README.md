# Spanning-tree augmentation for static Cycle PE

This folder is an **independent extension of the static Cycle PE track**.  It
tests whether resampling the spanning tree during training improves robustness
to a previously unseen fundamental-cycle chart.

It deliberately does **not** contain or import:

- learned conductance or a matrix `C`;
- GAT/conductance layers;
- node potentials or gradient-flow states;
- flow-completion objectives.

The only shared code used here is graph/incidence algebra and spanning-tree
sampling from `src/chartgat`.

## Scope

For a connected graph with incidence matrix `B` and cycle rank

```text
beta = m - n + 1,
```

a spanning tree `T` defines a full fundamental-cycle basis
`F_T in R^(m x beta)`.  If `T` and `T'` are two charts, the transition

```text
a_T' = M_(T' <- T) a_T
F_T' M_(T' <- T) = F_T
```

is lossless.  The implementation certifies pairwise reconstruction and the
cocycle law

```text
M_(T'' <- T') M_(T' <- T) = M_(T'' <- T).
```

## Independent downstream paper protocol

`paper.py` adds a separate graph-level protocol with no conductance, potential,
flow-completion, or cross-track model import:

- Wilson loop-erased random walks for true unweighted uniform spanning trees;
- explicit BFS/DFS random-root charts, plus the separately named legacy
  `random_priority_kruskal` sampler;
- a sign-even, orientation-gauge-safe masked DeepSets encoder that batches
  different edge counts and every cycle rank, including `beta = 0`;
- exact length-3/4/5/6 simple-cycle counts computed from the physical graph
  before any chart is sampled;
- graph-first train/validation/ID-test/OOD-test splits;
- fixed-BFS versus multi-chart training with identical optimizer-update counts;
- multi-chart training on random-root BFS/DFS only, with Wilson UST excluded as
  a genuinely held-out sampler family;
- the full ID/OOD graph x fresh-seen-BFS/held-out-Wilson 2x2 evaluation,
  including mean, worst-chart, chart-spread, and prediction-flip metrics.

The fixed model trains on one root-0 BFS chart per graph. The multi model trains
on a finite bank of random-root BFS/DFS charts. Both are evaluated on fresh BFS
charts (a family seen by both conditions) and on fresh Wilson UST charts (a
family seen by neither). Thus the Wilson axis is a sampler-family OOD test, not
merely a new draw from a family used by the multi model. The exact output keys
are `*_graph_fresh_chart_seen_family` and
`*_graph_fresh_chart_unseen_family`.

Wilson draws are not rejected merely because the resulting physical tree also
happens to be obtainable from BFS. Such rejection would condition and bias the
UST distribution. Exact tree overlap between the two axes is allowed and
reported; held-out here means sampler-family exposure, not disjoint output
support.

The encoder consumes `abs(F)`, `F^2`, and normalized cycle support. It is
therefore invariant to an arbitrary incidence-row orientation, cycle-column
direction, aligned edge/cycle ordering, and a node relabeling that preserves the
same physical tree. BFS/DFS root and neighbor ordering can select a different
tree after an arbitrary relabeling; that remains an intentional chart shift,
not an exact end-to-end graph-permutation-invariance claim.

Core uses the full scientific cycle-count protocol. CSL and ZINC use public PyG adapters and
emit actionable dependency, network, and cache-path errors when PyG or a
download is unavailable. The ZINC adapter preserves the integer atom types and
one categorical bond type per canonical undirected edge in its verified cache.
The model combines those chart-invariant chemistry embeddings with its topology
and cycle-chart representation; it is not a topology-only ZINC baseline.

### Linux/CUDA setup

Use a Linux workstation or server with an NVIDIA GPU, directly from a local
terminal or through any SSH client. Neither MobaXterm nor tmux is required.
Follow the [root README](../../README.md) for the supported hardware/software
requirements and Conda installation. From the repository root, create a dedicated environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n new-gat python=3.11 pip -y
conda activate new-gat
bash scripts/setup_gpu.sh
```

If you already created this project's environment using the
[root README](../../README.md), skip creation and activate that same environment.
Do not install into `base`, a shared environment, or another project's environment.
Setup installs the repository's exact CUDA/package pins, including the public PyG
adapters, then verifies the lock, package compatibility, and CUDA. The default
wheel channel is `cu126`; do not change it between reproductions of the same run.
Tests are optional: `RUN_TESTS=1 bash scripts/setup_gpu.sh`.

Every `python` command below assumes this Conda environment is active. Run paper
and test commands from the repository root. In each new terminal, run the
`source` and `conda activate new-gat` commands again; no environment recreation
is needed. See the root README for data paths, GPU allocation, and wheel selection.

### Exact CLI

Randomness is split into four auditable axes:

| flag | tree-track role |
|---|---|
| `--data-seed` | generated core graphs/cache; ZINC cache namespace only |
| `--split-seed` | deterministic CSL fold assignment |
| `--chart-seed` | fixed/multi chart banks and fresh evaluation charts |
| `--model-seed` | Torch initialization and minibatch sampling |

ZINC retains its official train/validation/test split, so changing data or split
seeds does not change its records or split membership. Core constructs its
generated records and graph splits from the data seed. The standalone `--seed`
flag remains backward compatible: missing data/split/chart axes fall back to the
resolved data seed, and a missing model seed falls back to `--seed`. Every run
manifest and summary records the resolved `seed_axes`; no single mixed seed is
reported as the experiment identity.

For independent paper axes, pass all four explicitly:

```bash
python -m research.tree_augmentation.paper \
  --suite core --data-root ./data/tree-augmentation \
  --output-dir results/tree-core-axes --device cuda:0 \
  --data-seed 11 --split-seed 13 --chart-seed 17 --model-seed 19 \
  --amp --batch-size 16 --workers 4 --pin-memory --non-blocking
```

Prepare the deterministic offline cache without training:

```bash
python -m research.tree_augmentation.paper \
  --suite core --data-root ./data/tree-augmentation \
  --output-dir results/tree-core-prepared --seed 17 --prepare-only --workers 0
```

Run the full CUDA path with autocast/GradScaler, pinned host transfers, and
non-blocking copies enabled by the configuration:

```bash
python -m research.tree_augmentation.paper \
  --suite core --data-root ./data/tree-augmentation \
  --output-dir results/tree-core-cuda --device cuda:0 --seed 17 \
  --amp --batch-size 16 --workers 4 --pin-memory --non-blocking
```

Optional adapters use `--suite csl`, `--suite zinc`, or `--suite all`. The GPU
setup above installs their pinned PyG dependencies. A missing verified processed
cache is network-safe by default: pass
`--allow-download` explicitly to permit the PyG adapters to access their public
dataset endpoints. No generated stand-in replaces a missing public dataset.
`--workers` is passed directly to every
training and evaluation `DataLoader`. With `--suite all`, each suite gets its
own subdirectory and the output root gets an aggregate manifest; all three are
attempted even if an optional adapter fails.

Every processed cache has a SHA256 manifest and fixed split graph IDs. Each run
writes `summary.json`, `manifest.json`, CPU-portable fixed/multi checkpoints,
runtime/device metadata, AMP effectiveness, elapsed time, and peak CUDA
allocated/reserved memory. Existing non-empty output directories are rejected.

## Lossless and lossy modes are separate

All enabled experiments use every cycle coordinate (`k = beta`).  Passing
`k < beta` raises `NotImplementedError`.  The lossy top-k/chord-sparsification
idea is recorded as a disabled future extension and is never reported as
lossless tree augmentation.

## Verification

Run unit tests from the repository root:

```bash
python -m pytest research/tree_augmentation/tests -q
```
