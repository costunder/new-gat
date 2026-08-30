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

## Default public-data execution

`reproduce.sh` now runs **CSL and ZINC-12K only** through the master's `benchmark`
suite. Both compare fixed-BFS versus multi-chart with unchanged graph labels and
the same prediction model. Neither conductance models nor the separate PE
baseline comparison are imported. The generated CycleCount protocol below is
available only through the explicit supplementary `core`/`all` suites.
CSL currently uses one fixed stratified 90/30/30 partition, not a full five-fold
published-score reproduction. ZINC preserves its official 10k/1k/1k split.

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

## Reproduce this track

Complete the environment installation and dataset preparation in the
[root README](../../README.md). With the project's Conda environment active,
run this script from the repository root:

```bash
bash research/tree_augmentation/reproduce.sh
```

The script runs CSL and ZINC independently on CUDA, with model
seeds `0,1,2,3,4` and data/split/chart seeds fixed to `0`. It runs only this
track's fixed-chart and multi-chart comparisons. Dataset and result locations,
run identifiers, and shared overrides follow the root README. Missing or damaged
public data is an error; training does not download a substitute.

### Seed axes and runtime controls

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

The reproduction script passes all four seed axes explicitly. CUDA training
uses AMP, pinned host transfers, and non-blocking copies by default.
The standalone module supports `core`, `csl`, `zinc`, and `all` suites; the
reproduction script selects `all`. Setup installs the pinned PyG dependencies.
Only the data-preparation step explicitly permits public downloads with
`--allow-download`; no generated stand-in replaces a missing public dataset.
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
