# 데이터셋 및 평가 계약

## 기본 실행: 각 연구의 경쟁 논문과 같은 데이터셋

기본 세 트랙의 `prepare_data.sh`, `reproduce.sh`, 트랙별 `reproduce.sh`는
`benchmark` suite를 사용한다. 별도 Conductance v2/v3 runner에는 `--suite` 옵션이 없다.
기본 실행에서는 아래 공개 데이터만 사용하며 S1–S4/CycleCount를 생성하지 않는다.
현재 기본 model seed는 사용자 요청대로 `0` 하나이며 공식 데이터·분할·전체 학습 크기는 유지한다.
명시적 seed 목록은 선택 사항이다. 단일 seed의 std/CI는 추정하지 않으며 기존 5-seed 결과는 보존한다.

| 트랙 | 기본 데이터 | 경쟁 논문과의 연결 | 이 저장소에서 실행하는 모델 | 지표 |
|---|---|---|---|---|
| Conductance GAT | Cora, CiteSeer, PubMed | GAT 원 논문의 인용 그래프, Planetoid public mask | 우리 conductance만 | accuracy |
| Conductance GAT | PPI | GAT 원 논문의 공식 20/2/2 그래프 split | 위와 동일 | micro-F1 |
| Conductance GAT | ogbn-arxiv | GATv2의 OGB node prediction, 공식 시간 split | 위와 동일 | accuracy |
| Cycle PE | ZINC-12K | SignNet와 PEARL 공통, 공식 10,000/1,000/1,000 | 우리 cycle-set PE만 | MAE |
| Cycle PE | Peptides-struct | PEARL 부록 K.2, LRGB 공식 split·11개 target | 위와 동일 | MAE |
| Tree augmentation | CSL, ZINC-12K | 공개 구조·분자 데이터에서 고정 tree와 다중 tree 비교 | 같은 모델의 fixed-BFS vs multi-chart | accuracy / MAE |

**우리 모델만 실행**한다. 외부 비교 방법은 논문 표의 수치를 출처와 함께 인용한다.
기본 GAT/PE benchmark는 validation으로 checkpoint를 선택한 뒤 test를 평가하고 모델 크기와 비용을
별도로 기록한다. 논문 인용 수치는 우리 seed별 실행값이나 paired 통계에 넣지 않는다.
표를 비교할 때 데이터 버전·split·지표·추가 입력·파라미터 예산·학습 조건을 확인하고,
다른 부분은 표 주석에 남긴다. 특히 현재 ogbn-arxiv full-batch 설정은 GATv2 논문의
GraphSAINT 설정과 다르므로 동일 학습 조건의 재실험으로 표현하지 않는다.
트랙 간 모델을 결합하거나 GAT 트랙에 새 cycle PE를 주입하지 않는다.

기저벡터 입력 [Cycle PE v2](research/cycle_pe/v2/README.md)는 ZINC-12K와 Peptides-struct의
같은 공식 원본·split을 사용하되 전체 좌영공간 기저를 별도 cache에 저장한다. 기본 실행은
여전히 통계형 `cycle_set` v1이다. v2는 이 소스 버전에 포함되지만, 제공된 5-seed 결과를 v2 결과로
해석하면 안 된다. 실행 결과와 진단의 범위는 [실험 상태](docs/EXPERIMENT_STATUS.md)를 따른다.

Conductance의 별도 [v2](research/conductance_gat/v2/README.md)와
[v3](research/conductance_gat/v3/README.md)는 기존 matched benchmark cache의 Cora,
CiteSeer, PubMed, ogbn-arxiv와 공식 split을 그대로 읽는다. 기본은 두 버전 모두
ogbn-arxiv와 model seed 0을 쓰며, v2는 `direct_c`/`fixed_c`, v3는
`relative_c`/`fixed_c`의 각 두 번을 새로 학습한다.
V2의 C는 고정 topology의 엣지별 파라미터이고 v3는 공유 상대-C 생성기다. 둘 다 현재 별도
runner에서는 transductive 네 데이터만 받으며 PPI를 받지 않는다. 이 제한은 v3 생성기가
원리상 새 그래프에 적용 불가능하다는 뜻이 아니라, 이번 inductive 전이 protocol이 없다는 뜻이다.
두 실행은 validation으로 checkpoint를 선택하고 **test를 평가하지 않으며**, 위 기본 v1
benchmark의 test 점수나 기존 C-learning 결과를 재사용하지 않는다. V2/v3의 실제 GPU 결과는
아직 수령하지 않았다.

논문 원문: [GAT](https://arxiv.org/pdf/1710.10903),
[GATv2](https://arxiv.org/pdf/2105.14491),
[SignNet](https://arxiv.org/html/2202.13013v4),
[PEARL](https://arxiv.org/pdf/2502.01122).

Alchemy는 SignNet의 공개 index 파일에서 중복 및 train/test 겹침이 발견되어 기본 실행에
추가하지 않았다. 임의로 다시 나눈 split을 원 논문과 동일하다고 표시하지 않는다.
검토한 원본은 [고정된 upstream revision의 Alchemy split](https://github.com/cptq/SignNet-BasisNet/tree/07f31187823ff8d42ed2f61eabe54344aea7cf24/Alchemy)이다.

아래 S1–S4, CycleCount, BREC 및 PascalVOC-SP/molhiv는 기존 `core`/`all` suite의
**보조 연구 계약**이다. 기본 benchmark와 구분해서 읽어야 한다.

## 보조 실험의 상태 및 계약

이 문서는 세 독립 연구 트랙에서 **현재 코드로 실행되는** 데이터, split, 비교군,
metric, leakage guard를 정리한다. 계획과 구현을 혼동하지 않도록 다음 상태를 구분한다.

- `implemented`: loader/generator, runner, metric, cache manifest가 코드에 있음
- `planned`: 아이디어만 있고 paper runner에는 없음
- `blocked`: 별도 claim을 위해 필요하지만 현재 범위에는 미구현
- `code_ready`: 위 구현을 import/CLI 수준에서 실행 가능
- `cached_data_ready`: 요청한 `--data-root`에 검증 가능한 cache가 실제로 존재

현재 모든 `paper_core` entry는 `implemented`다. 이 작업 공간에는 public 원본 cache가
없으므로 `cached_data_ready`는 실행 환경에서 준비 명령을 실행하기 전까지 false다.

```bash
python scripts/check_datasets.py --profile paper --json
python scripts/check_datasets.py \
  --profile paper --data-root ./data/paper \
  --seeds 0 --require-cache --json
```

기계 판독 원본은 다음 파일이다.

- `research/conductance_gat/datasets.yaml`
- `research/cycle_pe/datasets.yaml`
- `research/tree_augmentation/datasets.yaml`

## 공통 데이터 계약

1. Physical graph split을 먼저 고정하고 excitation, trajectory, chart를 그 안에서 만든다.
2. 같은 graph의 다른 excitation/chart를 graph OOD처럼 train/test에 나누지 않는다.
3. Official split이 있는 public benchmark는 해당 split을 그대로 쓴다. CSL은 예외로
   현재 한 개의 고정 stratified 90/30/30 분할을 사용하며, 5-fold 논문 점수 재현이 아니다.
4. `--allow-download` 없이는 network 접근을 하지 않는다.
5. Generated cache는 seed/config/schema hash로 구분하고 기존 cache를 검증한다.
6. Dataset preparation과 training output을 분리하고 기존 non-empty run을 덮어쓰지 않는다.
7. Public data가 없으면 명시적인 준비·다운로드를 요구하며 가짜 대체 데이터를 만들지 않는다.
8. 모든 run은 dependency, CLI, split, source/cache hash, runtime, peak GPU memory를 기록한다.

## 1. Conductance GAT

연구 질문은 sparse incidence path

\[
H\xrightarrow{B}BH\xrightarrow{C_\theta}C_\theta BH
\xrightarrow{B^\top}B^\top C_\theta BH,
\qquad c_e>0
\]

가 한 fixed graph를 외우는 대신 heterogeneous edge law를 학습하는가이다. Cycle basis나
tree chart는 입력하지 않는다.

### Generated core

| ID | 데이터/split | 주 평가 |
|---|---|---|
| S1 | static heterogeneous multi-graph; graph ID 70/15/15, seen-graph new excitation 별도 | flux/node-message relative L2, log-C RMSE, Pearson/Spearman |
| S2 | train ER-like/RGG-like n=16–32, test grid/barbell n=48–96 | topology/size-OOD graph-macro error |
| S3 | graph당 trajectory 1개인 graph-ID-disjoint state-dependent nonlinear rollout | horizon 1/5/10/50 error, norm growth, dissipation/stability-cap violation |
| S4 | 모든 split에 공통인 contrast 1/10/100 × active-node 1/0.25 × SNR inf/40/20 dB | known-condition held-graph-ID recovery/error curve, excitation coverage |

S1–S4는 dense incidence matrix를 만들지 않는다. Oriented edge gather로 `BH`, positive
conductance로 flux, 두 번의 `index_add_`로 `B^T`를 계산하고 서로 다른 크기의 graph를
한 batch에 pack한다.

### 학습 objective와 비교군

Per-edge flux supervision이 conductance recovery를 쉽게 만들어 headline을 오염시키지
않도록 objective를 분리한다.

| 이름 | C 입력 | 학습 supervision | 역할 |
|---|---|---|---|
| `isotropic` | scalar | node message only | \(C=cI\) baseline |
| `edge_only` | edge feature | node message only | static edge law |
| `gradient_only` | \(|BH|\) | node message only | state-only law |
| `full` | edge feature + gradient | node message only | headline model |
| `full_flux_supervised` | full | per-edge flux only | supervised ceiling |
| `full_joint` | full | node + flux | objective ablation |
| `flux_ls` | analytic | same-evaluation edge flux | transductive ceiling |
| `node_message_nnls` | analytic/projected NNLS | same-evaluation node message | transductive identifiability ceiling |
| `oracle` | true C | none | data/operator oracle |

`full`이 flux target의 제거·변조에 독립적이라는 회귀 테스트가 있다. LS와 NNLS는
inductive learned baseline으로 보고하지 않는다.

### Supplementary public benchmarks (`all`, not default `benchmark`)

| 데이터 | split/task | metric | 실행 모델 |
|---|---|---|---|
| PascalVOC-SP | LRGB official split, superpixel node classification | macro-F1 | 우리 conductance |
| ogbg-molhiv | OGB official scaffold split, graph classification | ROC-AUC | 우리 conductance |

이 보조 경로도 외부 비교 모델 없이 우리 conductance의 node encoding, 한 layer의 연산,
task head를 학습하고 trainable parameter count를 기록한다. MolHIV는
OGB AtomEncoder/BondEncoder를 사용한다. Reciprocal directed arcs는 한 physical bond로
canonicalize하며 categorical attribute 불일치는 거부하고 continuous attribute는 평균한다.

S1은 independently seeded graph ID만 분리하며 canonical topology/feature/conductance-law hash
uniqueness를 검증하지 않는다. S3의 trajectory disjointness는 graph당 초기조건 하나이므로 graph
split에 종속되고 unseen-initial-condition 효과를 분리하지 못한다. S4는 contrast feature를 직접
주며 모든 factor cell이 모든 split에 있으므로 blind/OOD contrast 실험이 아니다. Truth law의
graph-global min/max 정규화와 edge-local estimator 사이 function-class mismatch도 남는다.

PGLib/MATPOWER는 `B^T C B`와 닮은 조건부 DC proxy일 뿐 real sensor conductance
validation이 아니므로 현재 paper core에서 제외되어 있다. Roman-empire는 positive
diffusion의 negative-control 후보로 planned 상태다.

## 2. Static Cycle PE

Edge-by-node incidence convention에서

\[
B\in\mathbb{R}^{m\times n},\qquad
F_T\in\mathbb{R}^{m\times\beta},\qquad
B^\top F_T=0,
\]

인 fundamental cycle basis를 topology에서 한 번 계산해 정적 edge PE로 쓴다. Learned
conductance, sample circulation coefficient, chart augmentation은 이 트랙에 없다.

`B^Tq`가 `ker(B^T)`의 circulation component를 구별하지 못하는 것과 graph topology가
Laplacian에서 사라지는 것은 다른 주장이다. Simple unweighted graph의 full `L=D-A`는 adjacency를
결정한다. 따라서 이 PE는 sample circulation 또는 “L이 잃은 topology”를 복구하는 codec이 아니라
cycle 구조를 명시적으로 제공하는 inductive bias다.

### CycleCount-OOD v4

| split | 수 | 역할 |
|---|---:|---|
| train | 10,000 | train family/size |
| validation | 2,000 | 같은 regime held-out |
| ID test | 2,000 | 최종 ID |
| size OOD | 3,000 | 더 큰 graph |
| family OOD | 3,000 | unseen generator family |

총 20,000 graph다. Exact target은 다음과 같다.

- edge: C3–C6 참여 횟수, shortest-cycle length, short-cycle congestion
- node: C3–C6 참여 횟수
- graph: C3–C6 총 개수

`graph_Ck`가 node/edge count에서 직접 재구성되는 auxiliary-label leakage를 막기 위해
edge/node/graph는 별도 task, model, head, checkpoint로 학습한다. MAE, RMSE,
train-normalized MAE, graph-macro MAE, rounded exact accuracy를 보고한다.

기본 내부 variant는 `raw`, `set`, `projector`다. `no_pe`는 명시적으로 선택할 때만 실행하는
우리 모델의 PE 제거 ablation이다. Projector는 closest-prior formulation이며
novelty로 주장하지 않는다. Raw 폭은 train max-beta만으로 정한다. OOD beta가 더 크면
그 raw split만 `not_applicable_train_fitted_width_overflow`로 기록하고 절단/test-fit하지
않으며 다른 PE 비교는 계속 평가한다.

모든 variant가 incidence/BFS tree/full `F_T`와 raw basis를 계산한다. Set 통계와 dense projector만
요청 여부에 따라 조건부로 만들기 때문에 `no_pe/raw/set`만 실행하면 dense `m×m` projector는
없지만 basis 전처리 비용은 남는다. Projector variant 자체의 O(m²) 비용과 scaling 한계도 남는다.

현재 size/family OOD는 구현됐지만 `(n,m,beta,degree sequence)` matched counterfactual은
미구현이다. 따라서 현재 결과로 degree sequence를 완전히 통제한 cycle-composition
주장을 해서는 안 된다.

1-WL-indistinguishable이면서 cycle target이 다른 known-contrast, raw/set의 isomorphic relabeling
및 spanning-tree shift robustness는 미구현이다. 외부 PE 모델은 실행 범위 밖이며 논문 표로
인용 비교한다. Suite preparation time은 variant별로 분리되지 않고
CPU RSS도 기록하지 않으므로 projector preprocessing 효율 비교는 아직 불완전하다.

Cycle PE CLI는 data/split/chart/model seed를 독립 축으로 기록한다. CycleCount 생성과 cache
identity는 `data_seed`, supervised 초기화와 minibatch shuffle은 `model_seed`만 사용한다.
현재 CycleCount split은 generator-defined regime라 `split_seed`가 적용되지 않고, static BFS
fundamental-basis PE에는 chart sampling이 없어 `chart_seed`도 `not_applicable`이다. ZINC는
fixed public records와 official PyG split이라 data/split seed가 `not_applicable`이고 model seed만
학습에 사용된다. `--seed`는 생략한 축의 하위호환 fallback이다.

### BREC v3

- 공식 400 non-isomorphic pairs와 여섯 category 사용
- RPC q=32, threshold=72.34
- official search seeds `100,200,...,1000`
- 공식 Hotelling 계산: `D_mean.T @ pinv(cov(D)) @ D_mean`; q 배수 없음
- distinguishability와 isomorphic reliability를 별도 계산
- official mode는 batch 16, 20 epochs, LR/weight decay `1e-4`, float32, no AMP,
  no clipping, no shuffle와 seed별 전체 400-pair 순차 실행을 강제
- upstream 호환 field인 seed별 `Correct`, `Fail`, `Real_correct`를 보존
- 모든 seed가 complete하고 reliability failure가 0일 때만 `global_valid=true`; 이는 저장소가
  추가한 보수적 gate이며 upstream BREC metric이 아님
- 상수/제어 흐름은 reference에 정적으로 맞췄지만 upstream differential numerical parity는 미검증
- 공식 any-seed union은 없음
- 기존 any-seed 집계는 `--brec-protocol custom`의 `custom_pairwise_union`으로만 제공
- 바깥 data/split/chart/model seed는 BREC official 학습에 사용하지 않으며 내부 열 search
  seed를 manifest의 별도 protocol axis로 기록

`--allow-download`일 때만 GraphPKU Release ZIP을 받아 path traversal, symlink, 크기를
검사하고 `brec_v3.npy` 하나만 추출한다. Official mode는 q=32, 400 pairs, 51,200
records를 검사한다. ZIP/NPY SHA256은 provenance로 저장하지만 upstream canonical pin으로
표현하지 않는다. Custom BREC도 사용자가 제공한 실제 artifact에만 실행하며 자동 fixture는 없다.

Official full run은 pair/seed별 incremental checkpoint와 resume가 없고 전체 loop 뒤 결과를
기록한다. 4 variants 전체 실행은 최대 16,000 pair trainings이므로 중단 복구와 wall-clock은
현재 paper 실행의 운영상 한계다.

Master `suite=all`은 CycleCount와 ZINC를 model seed마다 실행하지만 BREC는 내부 official
10-seed run을 정확히 한 번만 실행한다.

### ZINC-12K

PyG `ZINC(subset=True)` official 10,000/1,000/1,000 split, atom/bond categorical feature,
graph regression target을 사용한다. MAE/RMSE를 공통 PE backbone에서 비교한다. Raw 폭은
동일하게 train-only fit이다.

AQSOL은 별도 scaffold-OOD claim을 낼 때 필요한 blocked 확장이고, Peptides는 optional
scaling 확장이다.

## 3. Spanning-tree augmentation

이 트랙은 conductance와 무관하다. 같은 graph의 full-\(\beta\) fundamental basis를 여러
tree chart로 바꿔 학습하는 것이 unseen chart robustness를 높이는지 검증한다.

### Sampler와 공정성

- single BFS
- random-root BFS/DFS (multi-chart training bank)
- Wilson loop-erased **uniform spanning tree** (training에서 제외한 sampler-family OOD)
- `random_priority_kruskal`: 이름을 분리한 non-uniform legacy baseline

Graph split을 먼저 고정하고 각 graph 내부에서 chart를 만든다. Fixed-BFS와 multi-chart
모델의 optimizer update 수를 동일하게 맞춘다. 평가 quadrant는 다음과 같다.

| | `fresh_chart_seen_family` | `fresh_chart_unseen_family` |
|---|---|---|
| ID graph | fresh random-root BFS | fresh held-out Wilson UST |
| OOD graph | fresh random-root BFS | fresh held-out Wilson UST |

Fixed condition은 root-0 BFS 하나, multi condition은 random-root BFS/DFS finite bank만 학습한다.
따라서 BFS 평가는 두 condition 모두에게 seen family이고 Wilson은 두 condition 모두에게 실제
unseen sampler family다. Wilson을 multi train에 섞은 뒤 held-out family라고 부르지 않는다.
Wilson이 우연히 BFS와 같은 physical tree를 뽑았다는 이유로 reject하지 않는다. 그렇게 하면
UST 분포가 BFS output에 조건부로 바뀌기 때문이다. 두 축의 exact-tree overlap은 artifact에
기록하고, held-out은 output support가 아니라 sampler-family exposure를 뜻한다.

Mean task error/accuracy, worst-chart metric, prediction spread/std, flip rate를 기록한다.
`beta=0`을 포함한 variable-edge/variable-beta masked batch를 지원한다. Core는 graph-level
C3–C6 count를 downstream target으로 사용하며 projector-derived target은 headline에
포함하지 않는다.

Encoder의 chart-coordinate 입력은 `|F|`, `F²`, normalized cycle support로 sign-even하다. 같은
physical tree를 edge orientation, row/column order, cycle-column sign 또는 node label만 바꿔
표현한 경우 prediction이 불변이다. 다만 node relabeling 때문에 BFS/DFS가 다른 physical tree를
선택하면 그것은 별도 chart shift이며 exact end-to-end permutation invariance로 주장하지 않는다.

### CSL과 ZINC

- CSL: deterministic label-stratified five-fold 3/1/1 protocol, 10-class accuracy
- ZINC-12K: official split, MAE, fixed-BFS 대 multi-chart

Tree ZINC adapter는 PyG atom `x`와 reciprocal bond `edge_attr`를 integer category 그대로
canonical undirected edge 순서에 보존한다. Model은 symmetric endpoint atom embedding과
bond embedding을 topology/chart encoder와 결합한다. 즉 topology-only ZINC가 아니다.
같은 molecule의 모든 chart에서 chemistry tensor는 고정된다.

Lossy `k<beta` chord selection이나 MST sparsification은 이 core protocol에 없다.
모든 chart는 full cycle rank를 보존한다. BREC chart stress는 optional planned extension이다.

## Cache 및 실행

전체 cache 준비:

```bash
bash scripts/paper.sh \
  --suite all --prepare-only --allow-download \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --run-id prepare-all-v1
```

준비 후 strict 확인:

```bash
python scripts/check_datasets.py \
  --profile paper \
  --data-root ./data/paper \
  --seeds 0 \
  --require-cache
```

`--require-cache`는 파일 존재 검사가 아니다. 각 registry entry의 read-only validator가
요청한 모든 seed의 schema/profile, split graph IDs/counts, feature/target shape와 finite
값, data/manifest checksum을 검사하고 public full cache의 필수 split과 official
cardinality를 강제한다. Network 접근이나 cache 생성은 하지 않는다. 결과는 `valid`,
`missing`, `incomplete`, `corrupt`, `wrong_request`로 구분된다. 기본 데이터 경로는
저장소의 `data/paper`이며 기본 세 트랙 결과는 각 트랙의 `results/paper/<run-id>`에 저장된다.
별도 Conductance 결과는 `results/conductance_gat/v2/<run-id>/`와
`results/conductance_gat/v3/<run-id>/`에 저장된다.
Dataset checker의 `--seeds`는 `--data-seeds` 호환 alias이고 model seed가 아니다. 독립
split cache 축은 `--split-seeds`로 검증한다.

Repository가 직접 생성하는 cache는 unique same-directory temporary file에 쓴 뒤 flush와
`fsync`, temporary parse/validation, `os.replace` 순으로 publish한다. Data와 manifest가
쌍인 경우 manifest를 마지막 commit marker로 publish한다. Training manifest는 여기에
code/config hash, CLI, dependency, GPU/runtime/artifact hash를 추가한다.

## 공식 출처

- BREC: https://github.com/GraphPKU/BREC
- ZINC/CSL benchmark protocol: https://www.jmlr.org/papers/v24/22-0567.html
- PyG ZINC loader: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html
- LRGB PascalVOC-SP: https://github.com/vijaydwivedi75/lrgb
- OGB graph property prediction: https://ogb.stanford.edu/docs/graphprop/
- CycleNet: https://proceedings.mlr.press/v231/yan24b.html
- PGLib-OPF: https://github.com/power-grid-lib/pglib-opf
