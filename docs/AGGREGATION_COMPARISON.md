# Independent aggregation comparison

This suite implements the follow-up to the shared research discussion. It never
modifies `experiments/incidence_ablation` or reuses its checkpoints. Original GAT
is deliberately not a comparator. The implemented recent external architecture
is **DUALFormer, ICLR 2025**, verified against the authors' repository at commit
`68fbdaf007af2f7d409cd435c4c48dd0e3155510`:

- Paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/128911cc894d57bcae78074a9551c132-Abstract-Conference.html
- Code and MIT license: https://github.com/JiamingZhuo/DUALFormer

This is not a claim that DUALFormer is the newest or best model as of 2026.
M3Dphormer (NeurIPS 2025) was inspected as another candidate but is **not implemented
in this suite**. Its cluster/global token construction and additional global-node
supervision must be reproduced and accounted for before a valid comparison.

## What is implemented

| CLI arm | Actual computation |
| --- | --- |
| `incidence` | Original learned conductance diffusion, no external residual/FFN |
| `incidence_pre_lift` | Same, with `[V,V²]` before diffusion and learned projection |
| `incidence_energy` | Same, adding explicit diagonal and cross-depth local Gram channels |
| `incidence_energy_pre_lift` | Energy channels and pre-lift together |
| `dualformer` | Published feature-space self-attention, its intrinsic residual/LayerNorm, then SGC propagation |
| `dualformer_no_skip_control` | Explicitly modified DUALFormer control with the intrinsic attention skip removed |

No model receives an extra wrapper residual or FFN. Our models use a linear
encoder, repeated incidence operators followed by ReLU/dropout, and a linear
decoder. The old input/output LayerNorm and per-block SwiGLU are not retained.
The diffusion's own `I - beta D^-1 L` is retained as the definition of that
operator; it is not a wrapper skip.

DUALFormer's original attention residual belongs to its published architecture.
Its no-skip control is labelled separately and must not be represented as the
original published model. Both retain the original attention normalization.
These rows answer different questions; do not pool them or use the modified
control alone to claim a win against the original architecture.

Every model receives the same official splits, precision, measured physical
batch, reference update budget, selected seeds, learning rate and AdamW weight
decay. Common encoder/decoder initialization is seed-paired. Operators have
different parameters; this is **not parameter matched** and not a reproduction
of each paper's independently tuned score. Total parameters and exact intrinsic
components are recorded under `model_contract` in configuration and metrics.

Reference keeps width 256, 8 heads and 8 graph propagation layers; large keeps
384, 8 and 12. DUALFormer uses its official default of one global attention
layer and alpha 0.1 before those graph propagation layers. Each DUALFormer
head has the full hidden width, as in its official implementation, rather than
silently dividing that width by the head count. All families use the common
dropout from the existing profile; this is explicitly a common-recipe evaluation.

For the energy branch, each layer projects all preceding depth states into its
current head coordinates. For all pairs k <= l, including the diagonal:

`Gamma_i[k,l] = 1/2 sum_{j adjacent i} c_ij <v_i[k]-v_j[k], v_i[l]-v_j[l]>`.

These are retained as separate node/head/pair channels and linearly mapped into
the output feature coordinates. Their readout initializes to zero, so the
initial function agrees with the corresponding no-energy incidence model.
The conductance tensor remains live: task gradients reach both the energy
readout and conductance estimator. The metric is common to each pair at a layer.
Unlike the old experiment, diagonal local energy is included, so cancelling
neighbor messages can still be distinguished. This is not an energy-maximization
loss and does not claim that a feature Gram rank increase proves invertibility.

Depth states are not exact-distance shells. Distance-specific incidence B_r,
bipartite boundary-value solvers and whole-network inverse/Jacobian diagnostics
are **not implemented here**. The shared conversation describes those as distinct
designs; the present trainable design is the local energy/first-order-message
branch. Historical reconstruction probes remain in the old independent suite.

## Execution

Full graph support is required. The global attention baseline must not receive
sampled subgraphs masquerading as its original global context. DUALFormer uses
exact two-pass node streaming and checkpoint recomputation: no node is omitted.
PPI disjoint batches compute separate attention statistics per tissue graph.
All-arm optimizer-inclusive calibration, complete validation, strict artifact
hashes and epoch-boundary resume are retained from the existing runner.

Dry-run the comparison on **ogbn-arxiv**, not PPI as the primary benchmark:

```bash
cd /home/aicompetition07/new-gat &&
/home/aicompetition07/.conda/envs/new-gat/bin/python -B -m experiments.aggregation_comparison \
  --run-id aggregation-arxiv-reference-gpu4-seed0-v1 \
  --datasets ogbn-arxiv --profiles reference --model-seeds 0 \
  --hardware-profile portable --sampling full --edge-chunk-size 4096 --dry-run
```

Train only after this code is available on that server:

```bash
cd /home/aicompetition07/new-gat &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=4 \
/home/aicompetition07/.conda/envs/new-gat/bin/python -B -m experiments.aggregation_comparison \
  --run-id aggregation-arxiv-reference-gpu4-seed0-v1 \
  --datasets ogbn-arxiv --profiles reference --model-seeds 0 \
  --device cuda:0 --hardware-profile portable --sampling full \
  --edge-chunk-size 4096 --activation-checkpoint --min-free-gb 8
```

This performs six fresh trainings, not the old eight-arm run. It uses no existing
training evidence. GPU 4 means the user's allocated physical GPU, mapped to
visible cuda:0 as in the successful earlier initialization.

A real A100 MIG 10GB fit has **not** been measured locally. Calibration may
reject the full graph recipe; it never reduces graph size, model size, physical
batch or training budget to manufacture a successful run. CPU tests are
synthetic verification only, not full training or evaluation. The output is
validation-only; test performance and multi-seed conclusions remain unmeasured.

The default learning rate remains 0.0005. `--learning-rate` records an explicit
common-recipe variation in a new run ID. It does not automatically tune models.
Use equal validation-only search budgets before drawing competitive conclusions.
Five repeated evaluations measure numerical repeatability, not five model seeds.
