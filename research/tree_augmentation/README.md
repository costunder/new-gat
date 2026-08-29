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

The legacy smoke experiment compares:

1. a raw static Cycle-PE probe trained on one fixed tree;
2. the same probe trained with multiple BFS/DFS/random-priority Kruskal charts;
3. both probes evaluated on a held-out unseen tree;
4. the chart-invariant cycle-projector diagonal as an analytic oracle.

Its diagnostic target is the static, chart-independent edge cycle leverage
`diag(P_cycle)`, where

```text
P_cycle = F_T (F_T^T F_T)^(-1) F_T^T.
```

The learned probe intentionally receives raw chart coordinates.  It is a small
diagnostic for augmentation, not a claim that augmentation is an exact
substitute for an invariant architecture.

This projector-derived target is never used as the paper headline.  It remains
available only to preserve the original algebra/pipeline smoke test.

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

Full non-tiny core, CSL, and ZINC runs are paper-table eligible; every `--tiny`
run remains pipeline validation only. CSL and ZINC use public PyG adapters and
emit actionable dependency, network, and cache-path errors when PyG or a
download is unavailable. The ZINC adapter preserves the integer atom types and
one categorical bond type per canonical undirected edge in its verified cache.
The model combines those chart-invariant chemistry embeddings with its topology
and cycle-chart representation; it is not a topology-only ZINC baseline.

The core path uses the repository's base dependencies (`pip install -e .`).
The public adapters additionally require a PyTorch/CUDA-compatible PyG install;
`pip install -e '.[paper]'` installs the declared Python extras after the
server's CUDA-enabled PyTorch environment has been selected.

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
  --suite core --data-root /data/tree-augmentation \
  --output-dir results/tree-core-axes --device cuda:0 \
  --data-seed 11 --split-seed 13 --chart-seed 17 --model-seed 19 \
  --amp --batch-size 16 --workers 4 --pin-memory --non-blocking
```

Prepare the deterministic offline cache without training:

```bash
python -m research.tree_augmentation.paper \
  --suite core --data-root /data/tree-augmentation \
  --output-dir results/tree-core-prepared --seed 17 --prepare-only --workers 0
```

Run the tiny CPU fixture used by tests:

```bash
python -m research.tree_augmentation.paper \
  --suite core --data-root /data/tree-augmentation \
  --output-dir results/tree-core-tiny --device cpu --seed 17 \
  --tiny --no-amp --batch-size 4 --workers 0
```

Run the full CUDA path with autocast/GradScaler, pinned host transfers, and
non-blocking copies enabled by the configuration:

```bash
python -m research.tree_augmentation.paper \
  --suite core --data-root /data/tree-augmentation \
  --output-dir results/tree-core-cuda --device cuda:0 --seed 17 \
  --amp --batch-size 16 --workers 4 --pin-memory --non-blocking
```

Optional adapters use `--suite csl`, `--suite zinc`, or `--suite all`. Install
PyG with wheels matching the server's PyTorch/CUDA build before requesting
them. A missing verified processed cache is network-safe by default: pass
`--allow-download` explicitly to permit the PyG adapters to access their public
dataset endpoints. `--tiny` limits converted records and optimizer updates, but
still uses the real adapter and split. `--workers` is passed directly to every
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

## Legacy projector smoke run

Run the default experiment from any working directory after installation:

```bash
python -m research.tree_augmentation.run
```

Run its tests from the repository root:

```bash
python -m pytest research/tree_augmentation/tests -q
```

Use a different configuration with `--config`.  The default run writes only to
`research/tree_augmentation/results/summary.json`.
