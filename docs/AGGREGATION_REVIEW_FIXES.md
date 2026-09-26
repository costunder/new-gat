# Aggregation comparison: review corrections (2026-09-26)

This historical correction record describes suite v2 at commit `8f4602d`.
The current suite is v3; its additional C controls and mechanism diagnostics are
documented in [AGGREGATION_MECHANISMS.md](AGGREGATION_MECHANISMS.md). The v2 test
counts below apply to that earlier revision, not automatically to v3.
It does not reinterpret the completed historical PPI eight-condition experiment.
Existing historical source files, checkpoints, results and allocation contracts
are preserved. Use a new comparison run ID; source/contract checks deliberately
reject resuming v1 with changed implementation.

## Corrected defects

1. **Selected validation could fail to reproduce and still pass.** Selection
   now saves the full validation evidence in history and best.pt. Reloaded
   evaluation must reproduce that evidence before metrics.json can be published.
   The read-only integrity inspector also checks scalar scores against counts,
   selected history against checkpoint evidence, and final evidence against
   selection. Every audit repeat checks against both selection and final reload.
   Accuracy stores correct/total; PPI stores TP/FP/FN/total binary decisions.
   Exact count equality is required. Absolute score tolerance is 1e-12 and
   relative tolerance zero, fixed before execution and included in identity.
   Consistent but different scores (including selected 0.7 / rechecked 0.1),
   same-ratio/different-count evidence, NaN and infinity are rejected.

2. **Each edge chunk copied the full node output.** `local_gram` and
   `normalized_graph_step` now use an in-place accumulation buffer. There is no
   reduction in edges, nodes, depth pairs, dimensions, batch or update budget.
   CUDA differential checks compare old/new output, feature gradient and live-C
   gradient, including nested non-reentrant checkpoint recomputation. FP64,
   FP32 and BF16 Gram comparisons agree exactly in the checked cases.

3. **The production training entry point crashed before training.** An actual
   CUDA `train_model` completion test exposed missing optimizer group `name`
   and `parameter_names`, required by existing `optimizer_metadata`. These are
   now supplied without changing parameter ordering, optimizer, learning rate
   or weight decay. The test executes four explicitly synthetic CUDA epochs,
   best-checkpoint reload, integrity inspection and five real CUDA audits. A
   separate injected incorrect final evaluation verifies that no passing
   metrics file is published and both checkpoints remain available.

## Experimental controls now connected

The original four incidence models remain. Added diagonal-only controls and
linear/post-lift controls complete **3 energy choices x 4 lift choices = 12**
conditions on the same residual-free backbone. Diagonal-only contains each
preceding depth's energy. Diagonal+cross also contains every unordered pair of
preceding depths. Readouts start at zero. All 12 initial functions are checked
for equality on the CUDA reference configuration. Gram coordinates remain
the linear value projection, before lift; no inverse-map guarantee is implied.

GATv2 is implemented using PyG GATv2Conv in the common linear input/output
wrapper: reference 8 layers, width 256 total, 8 heads, 32 features/head,
ReLU/common feature dropout, no residual/FFN/LayerNorm, zero attention dropout,
separate source/target projections. Both directions and exactly one self-loop
are cached for each graph and reused. The cache is invalidated when edge tensor
identity/version, node count or device changes. CUDA comparisons against native
PyG self-loop handling check exact output, input and parameter gradients.

With the two existing DUALFormer conditions, the default matrix has **15 arms**.
`--arms` allows an explicitly selected subset; no subset is silently introduced.
Reference stays 8/256/8; large stays 12/384/8; production epochs stay 200.
Every selected arm participates in common physical batch/worker calibration.
Summary tables now include parameter counts and explicitly label the comparison
as a common-recipe experiment, not a parameter-matched or paper-reproduction claim.

For the synthetic reference input (50 input features, 7 output classes), measured
parameter counts are 10,674,375 for incidence, 10,705,095 for incidence+energy,
1,071,623 for GATv2 and 1,594,887 for DUALFormer. These differences remain real;
adding GATv2 does not make the comparison capacity-matched. Real dataset counts
are emitted from each constructed model, not copied from this synthetic example.

## GPU evidence

Environment: RTX 5070 Ti 16 GB, one GPU; Python 3.13.2,
torch 2.13.0+cu130, torch-geometric 2.8.0.post1 (`.venv-gpu`).
Preflight: 16 logical CPU cores, approximately 44.96 GB available RAM and
9,541 MiB free GPU memory at observation. These are local measurements, not the
server's A100 MIG allocation. Model computation and gradient tests use CUDA.
Small metadata/audit-control fixtures test file integrity and error handling.

The final suite is reproducible with:

```powershell
.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider -o junit_family=legacy tests/test_aggregation_comparison_integrity.py tests/test_aggregation_validation.py tests/test_aggregation_comparison_cuda.py tests/test_aggregation_review_cuda.py
```

Raw results are kept under `results/debug-review-fix-20260926/`:

- `tests-03.xml`: first 77 passing checks, before the extra production-entry
  test uncovered the optimizer metadata defect.
- `gat-final-04.xml`, `reload-05.xml`: failed intermediate tests retained for
  traceability (first an incomplete synthetic fixture, then the real optimizer
  metadata failure).
- `reload-06.xml`: injected final-score failure is correctly blocked.
- `final-07.xml`: 41 targeted regression checks passed after the optimizer fix,
  including genuine CUDA training completion and five audits.
- `verified-08.xml`: final full suite on the corrected implementation.

Final result: **80 passed, 0 failed, 0 skipped** in 245.79 seconds: 46 CUDA
checks and 34 metadata/error-handling checks. All 15 reference-size models
completed FP32 and BF16 training/optimizer-resume smoke tests. Ruff lint and
format checks passed. PyG emitted upstream JIT/typing deprecation warnings;
these did not skip tests or disable checks.

The synthetic CUDA Gram profile uses N=4,000, E=16,000, depth=4, heads=4,
head width=8 and requested edge chunk size=256 (effective 64 at depth 4).
The 0.64 MB output previously caused 500 out-of-place index_add calls allocating
320,000,000 cumulative device bytes. After correction those counts are **0**.
This is cumulative allocation attributed to that operator, not total peak VRAM.
The full profiler report is embedded in the JUnit `cuda_profile` property.
Single profiler-instrumented timings are diagnostic only, not throughput claims.

## Remaining boundaries

- No real benchmark training/test evaluation was performed in this correction.
  Synthetic checks do not establish predictive superiority or MIG 10GB fit.
- Parameter counts differ across models. A common recipe does not establish
  superiority at equal capacity or after equal-budget per-model tuning.
- GATv2 uses full PyG edge operations with layer checkpointing; its edges are
  not chunked by `--edge-chunk-size`. Actual calibration may reject a device
  allocation; it must not silently shrink the model or graph.
- DUALFormer retains the previously declared common-recipe adaptation (including
  SGC depth), and its native residual is removed only in the labelled control.
- Distance-shell incidence, bipartite boundary solves, explicit energy-statistic
  inputs to the C generator and whole-network inverse/Jacobian analysis remain
  distinct unimplemented designs, not implied by the completed local-Gram work.

See [the execution protocol](AGGREGATION_COMPARISON.md) for the fresh v2 command.
