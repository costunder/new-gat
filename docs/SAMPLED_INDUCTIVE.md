# Shared C learning, sampled training, and unseen graphs

This independent `sampled_inductive_v1` study implements the original research
question on the current residual-free incidence backbone. The existing 19-arm
aggregation experiment remains a separate representation/energy/lift study.
Neither a spectral interpretation nor successful unit tests establish a speedup.

## Prespecified four-cell design

| Training support | C=1 | Learned C |
| --- | --- | --- |
| Complete training graphs | `full/incidence_fixed` | `full/incidence` |
| Sampled independent contexts | `sampled/incidence_fixed` | `sampled/incidence` |

Learned C defaults to per-head. `--dynamic-c shared` selects a shared edge scalar
instead, for all learned-C cells. Energy and lift are fixed **off** in all four
cells, so this study targets the core operator rather than introducing another
energy factorial. No external residual, FFN or LayerNorm is added. The common
linear encoder/decoder, graph-specific beta and value projections remain.
Reference retains 8 layers / 256 features / 8 heads; large retains 12 / 384 / 8.
The same seed pairs the common initialization, including encoder/decoder.
Fixed C has no trainable C generator. Parameter counts are reported; the models
are not parameter matched.

The learned parameters define the finite-K C calculation, not a permanent
parameter table keyed by graph edge IDs. Each new context or held-out graph
constructs its own C from features and incidence. Prediction remains sparse
incidence-based diffusion without eigendecomposition. Labels are used in the
supervised loss/metrics; they are not supplied to the C solver.

Precisely, the input is the graph incidence B and node features H. The solver
produces edge conductances c = Solve_K(B,H;theta), with C = diag(c); C is not
the input graph. A sampled context uses L = B^T diag(omega * c) B, where omega
is the declared sampling correction (one on a full graph). Training updates
the shared theta through this calculation. Evaluation holds theta fixed and
calculates c for the new graph; it does not reinitialize or relearn theta.

## Genuine graph-held-out evaluation

This first implementation uses the official PPI train/validation/test graph
splits, checked for disjoint graph IDs and complete graph accounting. Training
contexts come only from training graphs. Validation always uses complete,
previously unseen validation graphs and selects the best epoch. Five complete
validation rechecks must reproduce the selected TP/FP/FN/total counts exactly.

`--evaluate-test` evaluates every cell only after all four cells for all requested
seeds finish and pass validation audit. The entire selected checkpoint matrix is
hash-locked before test labels are evaluated. No optimizer update or checkpoint
selection uses test results. C is recomputed on the new graph with theta frozen;
state hashes before/after evaluation must match. This is generalization to the
official held-out tissue graphs, not a guarantee about arbitrary graph domains
or a claim of disjoint biological entity IDs across tissues.

PPI is retained here because it provides independent official graphs for this
specific hypothesis. It is not presented as sufficient evidence of modern SOTA.
The separate incidence-only runner now also permits `cluster` and
`cluster_disjoint` training on its existing transductive datasets, including
ogbn-arxiv. Such a node split is **not** renamed an unseen-graph experiment.
Global comparators remain full-context comparisons.

## Sampling and supervision contract

`--context-seeds` is required: there is no hidden small sampling default. Every
training node appears exactly once as a supervised seed per epoch. Training
nodes are shuffled deterministically by seed, epoch and source graph. Each seed
group uses the existing V5 randomized cluster expansion with declared fanouts,
then keeps only actual induced edges. Each context has independent copies of
its nodes, and contexts are physically processed as a PyG disjoint-union batch.
Labels of additional context nodes are excluded from the loss.

The declared context budget is seeds times `(1 + sum(fanouts))`. This is the
explicit sampling law, not an OOM fallback. The graph/model/epoch contract is
never reduced automatically. Degree-ratio boundary correction is inherited
from V5: square root of endpoint full/sample degree ratios, clipped to [1,64].
It is an approximation, not inverse inclusion probability or an unbiased
full-graph estimator. Original-graph degrees and structural summaries are
available from the training graph and do not use validation/test graphs.

Each epoch checks exact seed coverage and reports source graph IDs, context
count, saturated contexts, node/edge instances, unique node/edge coverage, and
repeat fractions. Worker-count changes preserve context construction and seed
order. CSR structures and full-graph topology plans are cached in persistent
workers. Context topology is reconstructed for the actual sampled incidence.

## Training budget and measured resources

All cells use the same **complete supervised-node epoch budget**, default 200,
with no early stopping in this independent study. Validation still selects the
best checkpoint. This preserves the full requested data/epoch budget. Sampling
changes the unit and number of physical batches, hence optimizer update counts
are recorded and **not claimed equal** across support modes. Sampling effects
therefore include this optimization-schedule difference. They are not a pure
fixed-update estimator comparison.

Physical graph and context batches are measured separately in their actual
units; within each support mode both C regimes and all requested seeds must fit
the same selected batch/worker plan. Multiple batch and worker candidates are
required. Full graph candidates start at the historical profile's physical
batch floor (portable: 2). Context batch candidates are explicit. No automatic
reduction below a declared candidate occurs. Unsafe candidates remain recorded;
if none fits, execution stops with existing evidence preserved.

Every probe includes actual backward, optimizer state/update, complete epochs
(at least two warmup updates, five measured updates and three measured seconds)
and full validation. It records peak allocated/reserved VRAM, actual input
counts, and throughput. Selection minimizes the worst cell's measured epoch
plus validation time among the requested safe candidates. The 10% free-memory
reserve is a calibration rule, not a universal OOM guarantee.

CPU workers perform sampling/topology/collation with prefetch and pinning;
model forward, C solver, loss, backward and updates run on CUDA. Epoch timings
include loader wait/construction exposure and transfer. C-solver CUDA events
include checkpoint recomputation and overlap the forward/backward stage events;
these nested times must **not** be added together. Cell wall time includes
setup, training, validation, checkpoint I/O and final audit. Shared dataset
loading and calibration are included in the invocation wall ledger, separate
from the cell ratio. Parallel worker CPU seconds are reported as overlapping
work, not added to wall time. GPU/CPU/RAM resource telemetry accompanies training.

Four scores produce conditional C effects, conditional sampling effects and
the difference-in-differences interaction in `comparison.json`. This report
checks paired sources, initialization, data and non-ablated recipe fields.
It also reports full/sample wall ratios. A resumed cell is excluded from cost
ratios because interrupted work cannot be reconstructed exactly. Shared-device
contention remains a measurement limitation; speed or accuracy gains are not
assumed. A single seed is not a significance test.

## Checkpoints and execution

Best state, last state, optimizer, epoch history and Python/NumPy/CPU/CUDA RNG
are committed together in one atomic `last.pt`. Resume restores an epoch
boundary and rejects changed source/data/configuration/allocation identities.
Resume also verifies calibration-record checksums and recomputes the selected
resource plan from those measurements; missing or inconsistent evidence fails.
Existing historical experiments and checkpoints are not imported or overwritten.

First inspect the explicit recipe without loading data or starting training:

```bash
cd /home/aicompetition07/new-gat &&
/home/aicompetition07/.conda/envs/new-gat/bin/python -B -m experiments.sampled_inductive \
  --run-id sampled-inductive-ppi-reference-gpu4-seed0-v1 \
  --profile reference --model-seeds 0 --hardware-profile portable \
  --context-seeds 32 --context-batches 32 64 128 \
  --graph-batches 2 4 8 16 20 --worker-candidates 2 4 \
  --num-neighbors 15 10 --edge-chunk-size 4096 --dry-run
```

The example declares 32 supervised seeds/context, a maximum cluster expansion
budget of 832 nodes/context, and 1024/2048/4096 nominal supervised seeds per full
physical context batch. These are explicit hypotheses/resource candidates,
not locally calibrated PPI or MIG measurements. Runtime coverage reports expose
saturation/overlap; the code never silently adjusts the context size.

After making this revision available on the server, run the same configuration
on the user's physical GPU 4, adding the final test evaluation:

```bash
cd /home/aicompetition07/new-gat &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=4 \
/home/aicompetition07/.conda/envs/new-gat/bin/python -B -m experiments.sampled_inductive \
  --run-id sampled-inductive-ppi-reference-gpu4-seed0-v1 \
  --profile reference --model-seeds 0 --hardware-profile portable \
  --context-seeds 32 --context-batches 32 64 128 \
  --graph-batches 2 4 8 16 20 --worker-candidates 2 4 \
  --num-neighbors 15 10 --edge-chunk-size 4096 \
  --device cuda:0 --min-free-gb 8 --evaluate-test
```

`--calibration-only` measures resources without final training. Repeat the same
command to resume; a changed allocation or recipe needs a fresh run ID. There
is no session termination, GPU reset, CPU fallback or old-results replacement.

## Gram backward correction (aggregation v4)

`gram.py` implements the exact first derivatives analytically. It allocates the
full history gradient and C gradient once, computes chunk-local differences and
pair derivatives, and scatters them into those buffers. Forward and backward
both retain all nodes, physical edges and declared depth pairs. FP64 independent
dense formulas and finite-difference gradcheck cover shared/per-head C and
diagonal/full Gram. CUDA profiling checks backward, not just no-grad forward.

Floating-point reduction order differs from v3: old/new forward remains equal
in the regression fixtures, while gradients agree numerically rather than
bitwise. FP32 is used for production Gram geometry even with BF16 dense layers.
The custom autograd supports first-order training only; it explicitly does not
provide differentiable double backward or an inverse/Jacobian research claim.
The source change requires fresh v4 aggregation runs; v3 evidence is preserved.

The local CUDA backward probe uses N=4000, E=16000, K=4, H=4 and d=8.
Requested chunks 256 and 4096 caused respectively 500 and 32 full-history
gradient allocations in the old indexing backward; the new backward allocates
that buffer once in both cases. Those old logical buffer totals are
1,024,000,000 and 65,536,000 cumulative bytes, **not peak VRAM**. The profiler
also records allocator-rounded bytes separately. This isolates one operation;
it does not establish whole-model throughput or a PPI training speedup.

Local tests use explicitly synthetic graphs and CUDA reference-sized models.
No real PPI training, official test score, new-graph accuracy gain, end-to-end
speed advantage or A100 MIG 10 GB fit has been established locally. Exact test
logs and allocation measurements accompany the current review bundle.
