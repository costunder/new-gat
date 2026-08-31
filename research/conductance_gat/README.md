# Independent Research Track A: Sparse Incidence Conductance Attention

Gate weight decay와 정규화의 2×2 원인 비교는 [별도 실험 폴더](ablation/README.md)에 있다.
PPI·ogbn-arxiv에서 seed 0 하나로 4조건을 새로 학습하며, 아래 기존 benchmark는 변경하지 않는다.
완료된 [GPU 결과와 해석](../../docs/CONDUCTANCE_FACTORIAL_FINDINGS.md)을 바탕으로,
다음 [C-learning 실험](c_learning/README.md)은 node-degree 아래 learned C/fixed C=1을
4개의 fresh training으로 비교한다. 기존 node-degree checkpoint의 평균-C 개입은
별도의 읽기 전용 검사다. 이 두 경로에도 Cycle PE/Tree나 외부 비교 모델을 섞지 않는다.

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
linear in nodes and edges. Numerical operator unit tests can run on CPU;
benchmark training requires CUDA and never silently falls back to CPU.

## Reproduce this track

Complete the environment installation and dataset preparation in the
[root README](../../README.md). With the project's Conda environment active,
run this script from the repository root:

```bash
bash research/conductance_gat/reproduce.sh
```

The script trains **only our conductance model on datasets used by GAT/GATv2**:
**Cora, CiteSeer, PubMed, PPI, and ogbn-arxiv**.
It uses CUDA and model seed `0` by default, with data/split/chart seeds fixed to `0`.
Pass `--model-seeds 0,1,2,3,4` to the reproduction script when an explicit
five-seed sweep is required.
It executes only this track, through its own `benchmark.py` entry point. Dataset and
result locations, run identifiers, and shared overrides follow the root README.
Missing or damaged public data is an error; training does not download a substitute.

## Our model on original-paper datasets

| Dataset from competing papers | Official split | Target and metric |
|---|---|---|
| Cora / CiteSeer / PubMed ([GAT](https://arxiv.org/abs/1710.10903)) | Planetoid `public` masks, unchanged | Paper subject; node accuracy |
| PPI ([GAT](https://arxiv.org/abs/1710.10903), [GATv2](https://arxiv.org/abs/2105.14491)) | Separate 20/2/2 train/validation/test graphs | 121 protein functions; global node-label micro-F1 |
| ogbn-arxiv ([GATv2](https://arxiv.org/abs/2105.14491)) | OGB temporal split, unchanged | 40 paper subject classes; node accuracy |

Every dataset is run only with our positive incidence **conductance** model.
The repository does **not** implement or train standalone competing models.
GAT/GATv2 and other published table results are external comparison references.
There is **no Cycle PE and no spanning-tree augmentation** in this track.

Our model uses a linear input encoder, two conductance layers of width 64,
layer normalization/ELU/dropout, and a linear prediction head. Dropout is 0.5.
Adam uses learning rate 0.005 and weight decay 0.0005, for at most 200 epochs
with validation patience 50. These controls and trainable parameter counts
are recorded in every run. There is no competitor selector or attention-head option.

Sharing original-paper datasets does **not** make these runs reproductions of
the papers' architectures, tuning budgets or reported scores. Compare published
tables externally only after checking split, preprocessing, training and metric
compatibility. Our ogbn-arxiv training is full-batch; GATv2 used GraphSAINT.
PPI uses graph minibatches, BCEWithLogitsLoss, and a fixed zero-logit threshold.
Only validation chooses the saved checkpoint; test is evaluated once afterward.
No claim of novelty or outperformance follows from simply running this suite.

Preparation downloads only the named official public sources. Verified caches
live under `data/paper/conductance_gat/matched_benchmark_v1/<dataset>/`.
Each cache records raw-source file checksums, prepared tensor SHA256 and official
split fingerprints; training verifies the cached tensors before using them.
Missing, partial or corrupt caches fail rather than generating replacement data.

Each per-seed run writes `manifest.json` and `metrics.json`; individual
`<dataset>/conductance/` directories contain `best.pt`, `history.json`, and
`metrics.json`. Output schema version 2 stores `datasets.<dataset>.models.conductance`.
Manifests record all expected/completed dataset runs, model/data
protocol, optimizer settings, software/GPU versions and implementation checksums.
Changing `data_seed`, `split_seed` or `chart_seed` does not alter official fixed
data/splits; these axes are explicitly recorded as not applicable. CUDA scatter
operations can remain nondeterministic, so seeded runs are not a bitwise guarantee.

## Diagnose a completed benchmark

Use [the checkpoint diagnostic guide](../../docs/CONDUCTANCE_DIAGNOSTICS.md) to inspect
training history, train/validation performance, learned conductance, and each node's
neighbor mixing weight without retraining. `scripts/diagnose_conductance.sh` uses
the active Conda environment and GPU inference on existing official caches only.
The full audit also computes train-label gradients without an optimizer step;
it checks one model seed and automatically writes a separate readable/JSON report.
It does not modify the model or original results, and does not re-evaluate test labels.
An optional validation-only graph-bypass intervention uses the same checkpoint;
it is not a separately trained baseline or proof of causation.

The supplied five-seed benchmark aggregates, seed-0 GPU diagnostics, and completed
single-seed factorial training results are recorded in
[experiment status](../../docs/EXPERIMENT_STATUS.md). The new C-learning runs and
node-degree mean-C intervention have no supplied GPU results yet.

## Supplementary suites (not the default matched benchmark)

The existing `paper.py` `core`/`all` suites remain explicitly selectable for
mechanistic S1--S4 diagnostics and additional PascalVOC-SP/ogbg-molhiv tasks.
They are **not substitutes for the GAT/GATv2 datasets above** and are not run by
the default benchmark preparation/reproduction scripts. Their original protocol
details follow; they must not be pooled into the matched-benchmark headline.

## Supplementary seed axes and runtime controls

Randomness is separated into `--data-seed`, `--split-seed`, `--chart-seed`, and
`--model-seed`. Generated S1--S4 graphs, excitations, trajectories, labels, and
cache keys use only `data_seed`; model initialization and training-loader
shuffling use only `model_seed`. The current generated protocol assigns splits
as part of data generation, so `split_seed` is recorded but marked not
applicable independently. This conductance track has no tree-chart sampling, so
`chart_seed` is also marked not applicable. Official PascalVOC-SP/MolHIV split
and chart axes are explicitly `not_applicable` in `summary.json`.

The standalone module retains `--seed` as a compatibility fallback for omitted
axes. The reproduction script passes the independent axes explicitly.

If GPU memory is limited, lower `--batch-size`. CUDA defaults to AMP, pinned
DataLoader memory, and non-blocking host-to-device transfer; all are explicit:
`--amp/--no-amp`, `--pin-memory/--no-pin-memory`, `--batch-size`, and
`--workers` (alias: `--num-workers`). CPU disables AMP and pinned transfer. Runtime JSON records the
CUDA device/runtime and peak allocated/reserved memory. An OOM error reports the
batch-size recovery command instead of silently falling back to CPU.

## Supplementary synthetic datasets

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

## Supplementary public benchmarks (optional loader dependencies)

These public benchmarks belong only to the explicit legacy `all` suite. Preparation is
network-opt-in: no official dataset class is instantiated for a download unless
`--allow-download` is supplied during the root README's data-preparation step.

- [LRGB PascalVOC-SP](https://github.com/vijaydwivedi75/lrgb): official
  train/validation/test split and node macro-F1.
- [OGB ogbg-molhiv](https://ogb.stanford.edu/docs/graphprop/): official scaffold
  split, OGB AtomEncoder/BondEncoder, and graph ROC-AUC.

Both public tasks train only the incidence-conductance model with its node and
edge encoders and prediction head. Standalone competitor models have been removed
from both the default and supplementary execution paths. JSON records active
trainable `parameter_count`; supplementary public results retain the legacy key
`baselines.conductance_model` solely for output compatibility.
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

## Supplementary deterministic cache and outputs

Core cache directories are keyed by a canonical hash of schema version,
generator version, `data_seed`, and profile. Each `manifest.json` has:

- the exact request and graph IDs in every split;
- a tensor-content SHA-256 independent of serialization metadata;
- the serialized `core.pt` SHA-256, verified on every cache load.

There is no timestamp in a cache manifest. A changed generator version receives
a new cache key rather than reusing stale data.

Supplementary `paper.py` training writes only to `--output-dir`. The path must be new or empty; the
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

- `benchmark_data.py`: Cora/CiteSeer/PubMed/PPI/ogbn-arxiv adapters and verified real caches;
- `benchmark.py`: our conductance-only CUDA benchmark on original-paper datasets;
- `sparse.py`: dense-`B`-free gather/scatter layer and variable-graph packing;
- `paper_data.py`: deterministic S1--S4 generation, splits, and cache manifests;
- `public_data.py`: official PyG/OGB adapters;
- `paper.py`: supplementary conductance models/own ablations, metrics, AMP, JSON/CSV;
- `tests/`: algebra, datasets, adapter, cache, and CLI regression tests.

This directory tests only the incidence-conductance-attention hypothesis. It
does not import or combine the other research tracks.
