# Stage A verification

## Implemented

- Complete 21-view, two-weight, three-observation contract with fixed coefficients and all original target/channel coverage.
- Individual linear q and symmetric quadratic E/J certificates, actual collision vectors, both visibility directions.
- FP64 direct SVD for synthetic graphs; zero-start primal least-squares CG for citation graphs, with separate true normal-stationarity and observation residuals.
- Successful summary qualification uses every original channel; one unresolved channel excludes the complete vector target from successful averages while retaining its raw error.
- Correct vector-trace channel accumulation before energy error; exact resource chunks and lossless evidence shards.
- Strict original source snapshot validation, all visible allocated GPU scheduling, source/input hash closures and exclusive output.

## Executed DEBUG checks

- CPU and CUDA independent root algebra/gate checks: **96 each**. CUDA maximum dense action absolute error: **7.21645e-16**.
- The root checks include dense/sparse action and energy, SPD/PSD and residual norm bounds, orientation invariance, zero-offdiagonal equivalence, activated-gate generic/zero/constant-input gradchecks, two actual DEBUG CE/Adam updates, and the **frozen-coefficient value action's** two-hop support. Adaptive coefficient-generator Jacobian support is a separate question.
- **50 unit tests passed**, including five CUDA tests. They include individual target equations, off-diagonal quadratic ambiguity, direct/iterative minimum-norm agreement, exact vector-trace chunking, graph-batched versus isolated operators, every GPU job's exact graph/recipe coverage, actual q collision vectors, both scalar/vector small-fixture engines through all 21 views and raw output coverage, the actual original snapshot reader's FULL/DEBUG profile calls, and rejection of completion records that omit the corrected least-squares scope.
- The updated CPU/CUDA solver checks compare noisy rank-deficient graphs with disconnected components and isolates against independent dense pseudoinverses, including a Euclidean nonsymmetric copy operator's exact transpose. Two independent leading replicas and multiple channels are solved together. They verify true normal stationarity, nonzero irreducible observation residuals, zero RHS without an epsilon denominator, finite short-iteration unresolved flags, and exclusion of whole vector targets if any channel is unresolved.
- All 10 owned Python source files passed AST parsing. The complete original-source 21-graph DEBUG integration is a separate orchestration check and is not claimed from these unit tests.

The small fixture labels, graph sizes and artificial citation fixtures are explicitly DEBUG. They provide implementation evidence, not scientific accuracy or a completed FULL audit. FULL execution uses the unchanged 201 original snapshots on the server. Selected old trained model states were not provided; there is no trained-state audit claim.

The first whole-pipeline DEBUG attempt exposed a positional profile argument in the source contract reader before any source graphs were processed. The call was corrected to `input_contract(profile=args.profile)`, and both real JSON-backed profile contracts were added to the unit tests. The failed output directory is preserved; integration resumes in a new namespace.

## Preserved historical 21-graph DEBUG attempt 02

[Pipeline DEBUG attempt 02, Stage A](../../../results/edge-metric-pipeline-DEBUG-20261005-02/A/completion.json) completed in **336.393 seconds** on CPU. All 18 synthetic graphs, 72 independent scalar inputs and three artificial citation fixtures were processed, for 42 graph/weight combinations. All original nodes, edges and channels and all 21 views/three observations/three noise levels were retained. This is DEBUG implementation evidence, not the actual 201-graph FULL study.

The independent readback at that time checked its scientific/input hashes, every compressed raw file's SHA256/byte count, complete row counts, all graph/weight identities and finite reconstructed numeric fields. Evidence contains:

- 2,184,084 reconstruction target rows;
- 150,066 solver rows;
- 7,686 numerical near-collision rows;
- 173,988 direct SVD certificate rows, including exported actual q/E/J collision vectors.

All raw `actual_data` flags are false. The code digest is `c3445bcea0a0e8cbaabb4fa7d44213cddfd724dd04d807a0e1f9e32e53876be5`; the source snapshot digest is `da71f1e9cc4f953a12ab0b572df7b81ff610012bc0cf5ff4098110b0a5a75346`. Existing source files, snapshots and the earlier failed results were preserved.

### Residual-status interpretation

The shared residual criterion was not met in **5,255 channel solves**. This consists of 995 matrix-free CG cases (56 clean, 939 noisy) and 4,260 direct SVD cases (385 clean, 3,875 noisy).

For direct SVD, the result is the least-squares estimate at the declared numerical rank threshold. An inconsistent noisy observation cannot have zero residual, and discarded small singular values can also create residual. The shared flag is therefore an observation-residual test, rather than SVD iteration convergence.

That historical citation implementation solved the noisy dual equation `A Aᵀ b = y`, which can be inconsistent. Its recorded final iterate is not a guaranteed noisy least-squares solution. Historical descriptive summaries include these iterates, so they establish complete pipeline processing, not reliable superiority under noisy recovery. Its files and counts remain unchanged.

## Current least-squares correction

After attempt 02's entire A/B/C DEBUG pipeline finished, the solver was changed to primal `Aᵀ A x = Aᵀ y` from zero. Direct SVD and CG now use a true normal-stationarity qualification; observation residuals are recorded separately as possible irreducible noise. Each raw target row includes the complete original vector's solver qualification. Failed iterations remain explicit and are excluded from successful recovery summaries. Near-collision records qualify both directions' solvers and retain actual observation differences.

The current audit configuration schema is `edge-metric-observability-audit-v2`; all 21 views, both recipes, full/target/one-hop coverage, original graph counts and channel meanings, noise levels and resource calibration remain unchanged. Completion records separate `stationarity_unresolved_channel_solves`, `observation_residual_outside_tolerance_channel_solves` and `unresolved_target_rows`. The whole 21-graph pipeline was rerun in the new attempt 03 below. No prior results were overwritten.

## Current whole 21-graph DEBUG attempt 03

[Attempt 03 Stage A completion](../../../results/edge-metric-pipeline-DEBUG-20261005-03/A/completion.json) records **302.899 seconds** on CPU. It processed all 21 DEBUG graphs and 42 graph/recipe combinations with all original nodes, edges, channels, 21 views, three observation types and three noise levels. Actual citation data was not used, and all raw `actual_data` flags are false. No classifier or trained-state audit was run by Stage A.

The independent final readback verified current source and original input hashes, all four compressed raw shards' SHA256/byte counts, every row count, graph/recipe coverage and finite reconstruction/solver fields. The output contains the same complete requested coverage as attempt 02:

| Raw table | Rows |
| --- | ---: |
| Reconstruction targets | 2,184,084 |
| Solver observations | 150,066 |
| Numerical near-collision records | 7,686 |
| Direct SVD certificates | 173,988 |

Every q/E/J raw row's full-vector solver qualification and its normal/observation residual maxima were matched to its exact observer's solver record. The successful summaries were independently recomputed from all raw rows and matched exactly: **2,179,610 successful target rows** and **4,474 unresolved complete-vector rows**. The latter remain in raw evidence and are excluded from successful averages. No converged channel subset replaced an original vector.

There are **73 unresolved normal-stationarity channel solves**, all in direct SVD with full observations of deep smoothing operators (8 or 16 repetitions). Their noise-level split is 29 clean, 15 at 1e-6 and 29 at 1e-3. **Matrix-free CG has zero unresolved channel solves and zero breakdowns** in this DEBUG run, including noisy observations. Direct SVD has no breakdown flags either. The remaining SVD residual failures are explicit numerical accuracy/conditioning limits, not successful inverses.

The separate observation-residual diagnostic is outside tolerance in **5,718 channel solves**. These are not counted as solver nonconvergence. All 7,686 near-collision rows have both numerical solvers qualified; actual observation residuals remain recorded and do not become exact null-space or rank proofs.

The current scientific digest is `86a98b7601d6dab9f1be575c25307cae9fe529ebd1b34a8e09b26cf4648f1c60`. The original snapshot digest remains `da71f1e9cc4f953a12ab0b572df7b81ff610012bc0cf5ff4098110b0a5a75346`. Earlier attempts' files were preserved. FULL with 201 original graphs and actual citations has not been executed locally; this record verifies the DEBUG pipeline and corrected least-squares implementation.
