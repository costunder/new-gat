# Aggregation comparison v3: mechanism controls and evidence

This records the v3 mechanism design and its historical verification. V4 keeps
these controls and adds the backward allocation correction and separate
sampled-inductive study described in [SAMPLED_INDUCTIVE.md](SAMPLED_INDUCTIVE.md).
Use the current review bundle's verification manifest for current test totals.

The second review was based on the old six-condition package. V2 already added
the 12-cell incidence matrix, GATv2, score reproduction checks and scatter-copy
fix. V3 implements the additional missing conductance controls, diagnostics,
interventions and paired effect reports. It retains reference 8 layers / width
256 / 8 heads, large 12 / 384 / 8, full graph support and the existing 200-epoch
reference update budget. Historical incidence-ablation files/results are unchanged.

## Retrained conditions

The default is 19 models per dataset/profile/seed:

- Per-head dynamic C: energy none / diagonal / diagonal+cross crossed with
  lift none / linear / nonlinear-pre / nonlinear-post (12 models).
- At **no lift**, C fixed-one / shared-dynamic / per-head-dynamic crossed with
  energy off / diagonal+cross. Two per-head cells overlap the above matrix,
  hence four additional models.
- GATv2, DUALFormer, and the explicitly modified DUALFormer no-skip control.

Fixed C has no trainable estimator parameters. Shared C has one scalar per
physical edge; the head-specific value projections and beta remain present.
Per-head C has one scalar per edge/head. No extra residual, FFN or normalization
is added to our models or GATv2. DUALFormer retains its documented intrinsic
normalization; its no-skip control is separately named.

These controls are **not parameter matched**. Parameters are recorded for every
cell and contrast. They also do not implement the full 36-cell C x energy x lift
factorial, so a C-regime effect cannot be asserted across all lift placements.
All selected arms participate in the same measured physical-batch/worker plan.
Raw inherited CLI arguments retain the legacy parser's dynamic/per-head defaults;
the required `ablation_arm` chooses the actual C regime. `configuration` and
`model_contract.conductance` explicitly record the effective mode/head count.

## Diagnostics on every validation graph

`audit.py` first reproduces the best checkpoint's integer metric counts on five
complete validation passes. It then performs a separate diagnostic pass and
applicable interventions; diagnostic and restored passes must reproduce those
same selected counts. No validation subset or node/edge cap is introduced.
For a transductive graph, mechanism statistics include its context nodes outside
the validation-label mask; the evaluation metric still uses validation labels
only. PPI statistics include all nodes/edges of the validation tissue graphs.

`mechanisms.diagnostics.rows` contains each layer's:

- C mean, RMS, standard deviation, mean absolute value, minimum, maximum and
  histogram **per head**. Shared C is repeated across heads only for reporting.
  Histogram intervals are `[-inf,.25), [.25,.5), ..., [4,8), [8,inf)`.
- Diagonal and cross Gram moments per head. Observations pool every node and
  applicable depth pair across all batches; graph means are not averaged equally.
  First-layer cross statistics have zero observations, explicitly marked absent.
  Diagonal-only architectures report no cross channels rather than inventing them.
- Whole-layer base-message L2 norm, energy-branch L2 norm, and their ratio,
  accumulated over all validation nodes/features. A zero denominator produces
  `null` with `zero_message_norm=true`, never a fabricated ratio.

Gram coordinates are the layer's **pre-lift linear value projections**. Their
metric is the same live C used for diffusion, common to all depth pairs at that
layer. Diagnostics use GPU tensor reductions; only small accumulated head-level
statistics move to CPU at reporting time. During training, history records the
first update's gradient L2 norms each epoch for encoder, decoder and per-layer
conductance, beta, value projection, output projection, lift projection and
energy readout. Missing gradients fail; a zero norm is reported as zero.
Zero-initialized energy readouts mean extra energy-to-C gradients initially
vanish; C still receives gradients from diffusion.

## Inference interventions

Applicable interventions are `energy_off`, `cross_off`, `diagonal_off`,
`c_ones`, `c_mean`, `c_shuffle`, and `lift_second_off`. They run on the full
validation split without updates. `c_mean` replaces C with its graph-local
weighted mean. `c_shuffle` is deterministic edge-order reversal **within each
graph**, moving all head values together; it is not a random-seed ensemble.
C changes affect both diffusion and the energy branch. The second lift channel
is removed without reinitializing its learned projection.

Every intervention reports integer-count validation evidence and the percentage
point difference from the untouched model. Context managers restore flags even
if evaluation fails. A final actual forward/evaluation verifies restoration,
and the audit checks that checkpoint/state/artifact hashes are unchanged.

These are whole-network inference interventions: downstream states and C are
recomputed. They are not frozen-H local effects, retrained controls, inverse
reconstruction proofs, or causal identification of performance improvements.
Separate retraining cells supply the conditional comparisons below.

## Matched effects and interactions

`effects.json` and `comparison.md` report validation percentage-point differences
within exactly the same dataset, profile and model seed. Every contributing
cell must pass training **and** audit. Source hashes, data/split hashes, shared
encoder/decoder initialization, learning budget and non-ablated recipe fields
must agree. Missing cells remain explicitly incomplete; no external comparator
or different seed fills an ablation cell.

Reported contrasts include diagonal-minus-none, full-minus-diagonal,
full-minus-none at each lift; linear-minus-none, pre-minus-linear,
pre-minus-post and post-minus-linear at each energy setting; no-lift C-regime
differences; and the corresponding energy interactions. For example:

`100 * [(score(full,pre)-score(none,pre)) - (score(full,none)-score(none,none))]`.

The sign states whether the energy difference is larger with pre-lift. It is
not a significance test. All contributing scores, coefficients and parameter
counts are stored. A single model seed remains a single-seed validation result.

## Resources, verification and scope

Optimizer-inclusive calibration now also executes the complete mechanism audit
before reading peak VRAM. Candidate cost includes its measured time; a candidate
without mechanism-path evidence is rejected. Thus collecting Gram channels for
a no-energy control cannot silently escape the GPU memory probe.

Model verification uses the local RTX 5070 Ti (16 GB), CUDA PyTorch and `.venv-gpu`.
Synthetic graphs are explicitly test-only; no final dataset/model/budget is
reduced. CUDA tests cover the 19 FP32/BF16 model paths, C-regime gradients,
observation-weighted statistics, intervention semantics/restoration, whole-audit
training completion, calibration, and historical per-head operator initialization.
Control-plane tests cover recipe matching, missing cells, contrast/interaction
signs and calibration evidence. Exact completed test totals belong in the fresh
review bundle's `VERIFICATION.json` and JUnit evidence.

Local verification on 2026-09-26: **109 tests passed** (63 CUDA model/diagnostic
tests and 46 control-plane tests). After removing a redundant GPU zero buffer,
forwarding the C regime explicitly at model construction, and formatting the
contract text, **68 affected-path tests passed again** (22 CUDA, 46 control-plane).
Ruff and Git whitespace checks passed. The bundle includes both JUnit files:
`results/aggregation-v3-verified-20260926-03.xml` and `-04.xml`. Repeated tests
are not counted as additional independent experiments. PyG emitted upstream
deprecation warnings; neither run had failures or skips.

No full benchmark training/test evaluation or A100 MIG 10 GB fit is established
by these tests. Distance-shell `B_r`, bipartite boundary-value solvers,
whole-network inverse/Jacobian diagnostics and new-backbone corruption/forest/
heterogeneous-edge factors remain separate designs, **not implemented here**.
Feature-rank growth does not establish invertibility. Use a fresh v3 run ID;
existing v1/v2 and historical PPI evidence cannot be resumed or pooled into v3.
