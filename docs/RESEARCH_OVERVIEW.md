# Research-track boundaries

The repository is a monorepo only for dependency reuse. Each scientific claim
has its own folder, runner, configuration, tests, and outputs.

| Track | Active claim | Allowed | Forbidden |
|---|---|---|---|
| `conductance_gat` | learned incidence conductance attention | `B`, node/edge features, positive `C_theta`, `B.T C_theta B` | `F_T`, cycle coordinates, tree charts, flow completion |
| `cycle_pe` | static cycle-space positional encoding | `B`, static `F_T`, cycle-set/static PE, structural targets | learned `C`, potential state, sample circulation coefficients, flow completion |
| `tree_augmentation` | spanning-tree resampling for static Cycle PE | multiple full-beta `F_T`, exact chart transitions, unseen-tree evaluation | conductance attention, potentials, flow completion, lossy truncation in the core experiment |
| `combined_later` | postponed integration only | previous combined prototype | inclusion in active headline results |

No result from `combined_later` may be reported as evidence for either active
main contribution.

## Default own-model benchmarks

- Conductance GAT: Cora/CiteSeer/PubMed/PPI from GAT, ogbn-arxiv from GATv2;
  only the proposed incidence-conductance model is trained on the official splits.
- Cycle PE: ZINC-12K from SignNet/PEARL and Peptides-struct from PEARL;
  only our cycle-set PE model is trained, without learned conductance or external PE implementations.
- Tree augmentation: public CSL and ZINC-12K; fixed-vs-multi-tree is an ablation of our own model.

The default `benchmark` suite excludes generated datasets. Existing `core`/`all`
suites remain supplementary experiments, not substitutes for the competitors' datasets.
Competitor scores are cited from published tables, not generated here. Label them as published
references and disclose split/metric/training differences; they are not paired reruns with our seeds.
