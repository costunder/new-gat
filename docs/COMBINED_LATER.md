# Combined prototype — postponed

The code directory `research/combined_later/` contains the earlier prototype that combined:

- learned incidence conductance;
- persistent cycle coordinates;
- hard-constrained flow completion;
- tree-chart-equivariant nonlinear updates.

It is deliberately excluded from the root smoke pipeline. It must not be used
as evidence for either independent contribution until the Conductance GAT and
Cycle PE tracks have each been evaluated on their own.

Historical outputs were moved to `results/combined_later/`.

The small historical `certification.json`, `fixed_c/` (summary, CSV, PNG), and
`identifiability/` (summary, CSV, PNG) outputs are now tracked byte-for-byte.
The duplicate `fixed_c_smoke/` output remains ignored. These are archived
synthetic/prototype observations, not fresh benchmark results or evidence for
the current V5/edge-selection implementation. Original local output-path
metadata is retained; no missing run commit or new reproduction is inferred.

Optional historical checks can be run explicitly:

```powershell
.\.venv\Scripts\python.exe -m pytest research\combined_later\tests -q
.\.venv\Scripts\python.exe -m research.combined_later.run_certify
.\.venv\Scripts\python.exe -m research.combined_later.run_identifiability
.\.venv\Scripts\python.exe -m research.combined_later.run_fixed_c --epochs 40
```
