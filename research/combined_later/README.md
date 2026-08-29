# Combined prototype — postponed

This directory contains the earlier prototype that combined:

- learned incidence conductance;
- persistent cycle coordinates;
- hard-constrained flow completion;
- tree-chart-equivariant nonlinear updates.

It is deliberately excluded from the root smoke pipeline. It must not be used
as evidence for either independent contribution until the Conductance GAT and
Cycle PE tracks have each been evaluated on their own.

Historical outputs were moved to `results/combined_later/`.

Optional historical checks can be run explicitly:

```powershell
.\.venv\Scripts\python.exe -m pytest research\combined_later\tests -q
.\.venv\Scripts\python.exe -m research.combined_later.run_certify
.\.venv\Scripts\python.exe -m research.combined_later.run_identifiability
.\.venv\Scripts\python.exe -m research.combined_later.run_fixed_c --epochs 40
```

