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

