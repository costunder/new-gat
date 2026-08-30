# Independent Research Track A: Sparse Incidence Conductance Attention

This directory is a self-contained executable paper track for

\[
H \xrightarrow{B} BH
\xrightarrow{C_\theta} C_\theta(BH,x_E)BH
\xrightarrow{B^\top} B^\top C_\theta(BH,x_E)BH,
\qquad c_e>0.
\]

The paper path never materializes a dense incidence matrix. For each oriented
edge `tail -> head`, `sparse.py` gathers `H[head] - H[tail]`; two PyTorch
`index_add_` operations scatter signed flux back to the incident nodes. Graphs
of different sizes are concatenated into one `PackedGraphBatch`, so memory is
linear in nodes and edges and the same code runs on CPU or CUDA.

All experiments use the `paper.py` entry point.

## Linux/CUDA setup

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
The setup script installs the repository's exact CUDA/package pins, including
PyG/OGB dependencies for PascalVOC-SP and ogbg-molhiv, then verifies the lock,
package compatibility, and CUDA. The default wheel channel is `cu126`; do not
change it between reproductions of the same run. Tests are optional:
`RUN_TESTS=1 bash scripts/setup_gpu.sh`.

Every `python` command below assumes this Conda environment is active and the
working directory is the repository root. In each new terminal, run the
`source` and `conda activate new-gat` commands again; no environment recreation
is needed. See the root README for data paths, GPU allocation, and wheel selection.
The runner checks CUDA availability before doing any work and gives an explicit
error when a requested optional dependency or prepared public dataset is absent.

## One paper entry point

Prepare deterministic core data without training:

```bash
python -m research.conductance_gat.paper \
  --suite core --prepare-only \
  --data-root ./data/conductance \
  --output-dir ./results/conductance-prepare \
  --data-seed 17
```

Run the core paper suite on CUDA:

```bash
python -m research.conductance_gat.paper \
  --suite core \
  --data-root ./data/conductance \
  --output-dir ./results/conductance-core-seed17 \
  --device cuda --data-seed 17 --model-seed 17 --batch-size 16 --amp
```

Randomness is separated into `--data-seed`, `--split-seed`, `--chart-seed`, and
`--model-seed`. Generated S1--S4 graphs, excitations, trajectories, labels, and
cache keys use only `data_seed`; model initialization and training-loader
shuffling use only `model_seed`. The current generated protocol assigns splits
as part of data generation, so `split_seed` is recorded but marked not
applicable independently. This conductance track has no tree-chart sampling, so
`chart_seed` is also marked not applicable. Official PascalVOC-SP/MolHIV split
and chart axes are explicitly `not_applicable` in `summary.json`.

Standalone `--seed 17` remains compatible: every omitted axis resolves to 17.
For auditable paper runs, pass data and model axes explicitly.

If GPU memory is limited, lower `--batch-size`. CUDA defaults to AMP, pinned
DataLoader memory, and non-blocking host-to-device transfer; all are explicit:
`--amp/--no-amp`, `--pin-memory/--no-pin-memory`, `--batch-size`, and
`--workers` (alias: `--num-workers`). CPU disables AMP and pinned transfer. Runtime JSON records the
CUDA device/runtime and peak allocated/reserved memory. An OOM error reports the
batch-size recovery command instead of silently falling back to CPU.

## Implemented core datasets

| ID | Implemented protocol | Claim tested |
|---|---|---|
| S1 static identification | Shared positive static edge law; full profile has 42/9/9 independently seeded, graph-ID-disjoint train/validation/test graphs (70/15/15), plus new excitations on training graph IDs. | Held-ID recovery is not fixed-edge-ID memorization; canonical topology/feature/conductance uniqueness is not certified. |
| S2 topology/size OOD | Train/validation on ER-like/RGG-like graph generators with 16--32 nodes; test only grid/barbell graphs with 48--96 nodes. Exact cross-split isomorphism deduplication is not implemented. | Sparse variable-graph topology and size transfer. |
| S3 nonlinear rollout | Positive `c_e=f(x_e, abs(H_v-H_u))`; there is one trajectory per graph, so graph-ID disjointness also makes trajectories disjoint. Full evaluation reports horizons 1/5/10/50. | Held-graph rollout for one initial condition per graph; unseen-initial-condition and unseen-graph effects are not separated. |
| S4 robustness | Every train/validation/test split contains all 18 known-condition cells: contrast 1/10/100 × active-node fraction 1/0.25 × SNR infinity/40/20 dB, with independently seeded graph IDs. | Conditional factor-grid held-ID empirical recovery, not blind contrast identification or factor OOD. |

S1's validator checks graph-ID separation, split cardinality, tensor validity, and
cache checksums. It does not compute canonical topology hashes or cross-split
edge-feature/conductance-content hashes. Independently generated graph IDs should
therefore not be described as certified non-isomorphic physical graphs.

S4 exposes the operating contrast directly as the fourth edge feature,
`log10(contrast)/2`. Its target conductance also normalizes the base edge scores
with the minimum and maximum over the whole graph, whereas the learned estimator
is edge-local and receives no graph-level min/max statistic. S4 error consequently
mixes inverse-problem difficulty with this function-class mismatch; it is a
conditional empirical recovery diagnostic, not a formal identifiability result.

The headline model is `full`, with
`c=f(x_E,abs(BH),(BH)^2)`, trained under the `node_only` objective. It reads
only observed node messages; per-edge flux targets are neither required nor
read by that training loss. The isotropic `C=cI`, edge-only `c=f(x_E)`, and
gradient-only `c=f(abs(BH))` predictive ablations use the same `node_only`
objective and sparse residual operator. Two full-model objective comparisons
are kept separate from the headline: `full_flux_supervised` uses `flux_only`
and is a per-edge-label supervised ceiling, while `full_joint` is an explicit
joint-supervision ablation. JSON and `history.csv` record the objective for
every trained core baseline.

Core ablations share the training protocol and hidden width, but they do not
have matched parameter budgets: the isotropic model has one learned scalar and
the edge-only, gradient-only, and full MLPs have different input widths. Core
results currently do not emit per-baseline parameter counts, so their gaps
cannot be interpreted as input-information effects alone.

The exact conductance oracle is evaluated everywhere. S1 and S4 additionally
report two transductive identification ceilings on the evaluation excitations:
per-edge closed-form flux LS reads observed edge flux, while node-message NNLS
estimates nonnegative edge conductances from observed node messages without
reading flux labels. They are labelled
`transductive_same-evaluation-excitations_identification_ceiling` and
`transductive_same-evaluation-node-messages_nnls_ceiling`, respectively, and
are never presented as held-graph learned baselines. NNLS may materialize its
small CPU diagnostic design matrix; no learned model or CUDA training path
materializes `B`.

Reported metrics include graph-macro flux/node-message/next-state relative L2,
log-conductance RMSE, Pearson and Spearman conductance correlation, excited-edge
coverage, state variation of predicted conductance, stability-cap activation,
and S3 rollout norm/dissipation diagnostics. S4 additionally writes metrics for
every factorial cell.

## Required paper public benchmarks (optional loader dependencies)

Preparation is network-opt-in. No official dataset class is instantiated for a
download unless `--allow-download` is supplied:

```bash
python -m research.conductance_gat.paper \
  --suite public --prepare-only --allow-download \
  --data-root ./data/conductance \
  --output-dir ./results/conductance-public-prepare

python -m research.conductance_gat.paper \
  --suite public --device cuda --amp \
  --data-root ./data/conductance \
  --output-dir ./results/conductance-public-seed17
```

- [LRGB PascalVOC-SP](https://github.com/vijaydwivedi75/lrgb): official
  train/validation/test split and node macro-F1.
- [OGB ogbg-molhiv](https://ogb.stanford.edu/docs/graphprop/): official scaffold
  split, OGB AtomEncoder/BondEncoder, and graph ROC-AUC.

Both public tasks train five custom one-layer comparisons through the same data
adapter, split, hidden width, node encoder, readout/head, and optimizer:
no-message MLP, sparse GCN, edge-aware GAT, GINE, and the incidence-conductance
model. Shared active node-encoder/head tensors receive identical initialization.
GAT, GINE, and conductance use the same edge encoder; no-message and GCN freeze
and skip it because those backbones do not consume edge features. JSON records
active trainable `parameter_count`, but backbone budgets are not matched and
these implementations are not tuned reference benchmark configurations.
PascalVOC train/validation CE is weighted by node label count rather than graph
count.

Reciprocal PyG arcs are collapsed into one physical undirected edge and
self-loops (zero incidence rows) are removed before the conductance operator.
Categorical reciprocal conflicts are rejected; continuous directional attributes
are averaged into an orientation-free physical-edge attribute.
The adapter marker under `--data-root` records source URLs, split counts, and
required processed files. Later runs without `--allow-download` verify those
files before constructing a PyG dataset, so a damaged cache cannot silently
trigger a network request. Missing public data is never replaced by generated stand-ins.

## Deterministic cache and outputs

Core cache directories are keyed by a canonical hash of schema version,
generator version, `data_seed`, and profile. Each `manifest.json` has:

- the exact request and graph IDs in every split;
- a tensor-content SHA-256 independent of serialization metadata;
- the serialized `core.pt` SHA-256, verified on every cache load.

There is no timestamp in a cache manifest. A changed generator version receives
a new cache key rather than reusing stale data.

Training writes only to `--output-dir`. The path must be new or empty; the
runner refuses a non-empty directory before data preparation and leaves its
existing artifacts untouched:

- `summary.json`: resolved seed axes and applicability, configuration,
  manifests, claims, complete metrics, runtime, CUDA/AMP and peak-memory metadata;
- `metrics.csv`: flattened machine-readable result metrics;
- `history.csv`: per-suite/per-baseline objective, train loss, and validation loss;
- `models.pt`: CPU-portable state dictionaries.

Preparation-only writes `prepare_summary.json`.

## Verification

```bash
python -m pytest research/conductance_gat/tests -q
```

Tests cover sparse-vs-explicit algebra, orientation gauge invariance, positivity,
variable-graph isolation/mass conservation, objective isolation from flux labels,
node-message NNLS recovery, S1--S4 split leakage, topology/size OOD boundaries,
deterministic checksummed cache reload, reciprocal public edge adaptation,
unit-test adapter fixtures and CLI artifact handling.

## Files

- `sparse.py`: dense-`B`-free gather/scatter layer and variable-graph packing;
- `paper_data.py`: deterministic S1--S4 generation, splits, and cache manifests;
- `public_data.py`: official PyG/OGB adapters;
- `paper.py`: prepare/train/evaluate CLI, baselines, metrics, AMP, JSON/CSV;
- `tests/`: algebra, datasets, adapter, cache, and CLI regression tests.

This directory tests only the incidence-conductance-attention hypothesis. It
does not import or combine the other research tracks.
