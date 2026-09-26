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
batch or training budget to manufacture a successful run. CPU and CUDA tests are
synthetic verification only, not full training or evaluation. The output is
validation-only; test performance and multi-seed conclusions remain unmeasured.

The default learning rate remains 0.0005. `--learning-rate` records an explicit
common-recipe variation in a new run ID. It does not automatically tune models.
Use equal validation-only search budgets before drawing competitive conclusions.
Five repeated evaluations measure numerical repeatability, not five model seeds.

## Explicit CUDA smoke verification

The local machine has an RTX 5070 Ti with 16 GB VRAM. The default `.venv` was
incorrectly installed with CPU-only PyTorch. It has been repaired from
2.14.0+cpu to 2.14.0+cu130, preserving its existing PyTorch release. Use this
default CUDA environment for local model execution. The separate `.venv-gpu`
also remains available with Python 3.13.2, torch 2.13.0+cu130 and the remaining
packages pinned in `requirements-lock.txt`. Neither environment changes the
Linux production installation profile or GPU driver.

```powershell
.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
.venv/Scripts/python.exe -m pytest tests/test_aggregation_comparison_cuda.py -v
```

The CUDA suite covers all six reference-size models (8 graph layers, width 256,
8 heads) in FP32 and BF16. Four synthetic graphs are processed together, with
256 total nodes and 512 undirected edges. The tests execute the actual training
loop, backward, optimizer update, validation, checkpoint/optimizer restoration,
and continued training with restored RNG states. Separate CUDA checks compare
streamed attention outputs/gradients with the dense formula and local energy
with the global bilinear identity. JUnit properties record the GPU, precision,
parameter count, peak allocation and first-step CUDA event duration.
Missing CUDA is a test failure, not a skip. Parameters, model inputs and every
trainable parameter's gradient are explicitly checked to reside on CUDA.
Production training already rejects CPU execution and has no CPU fallback;
CPU work is reserved for loading/preprocessing, scheduling and reporting.
Checkpoint equivalence checks enable deterministic CUDA algorithms and set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` within the test process. This separates
restoration errors from GPU atomic-reduction ordering; it does not relax the
parameter tolerances or change the production training recipe. See the
[PyTorch deterministic algorithms contract](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html).

These are explicit synthetic smoke tests, not measured real-data throughput,
benchmark training, or evidence that the complete recipe fits an A100 MIG 10GB.
