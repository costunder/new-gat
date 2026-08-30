# NEW GAT 연구 프로젝트 Hand-off

작성 기준일: 2026-08-30 (Asia/Seoul)

이 문서는 외부 ChatGPT 또는 연구 리뷰어가 저장소를 처음 받아도 수학적 가설, 구현 경계,
데이터 계약, 실행법, 검증 범위와 미완료 항목을 혼동하지 않도록 만든 인수인계 문서다.
원문 코드는 같은 폴더의 `code_summary.md`에 파일별로 들어 있다.
처음 서버에서 설치·실행하는 사용자는 [README.md](README.md)의 순서를 따른다.
이 문서는 실행 입문서가 아니라 연구·구현 교차검토용이다.

## 0. 리뷰어가 먼저 알아야 할 판정

1. 활성 연구는 세 개이며 서로 독립이다.
   - `research/conductance_gat`: positive scalar edge conductance를 학습하는 sparse incidence
     operator.
   - `research/cycle_pe`: topology에서 미리 계산한 static cycle-space edge PE.
   - `research/tree_augmentation`: 같은 graph의 full-cycle-rank fundamental basis를 여러
     spanning-tree chart로 바꾸는 augmentation.
2. 위 세 연구를 결합한 모델은 아직 없다. `research/combined_later`는 격리된 과거
   prototype이며 paper runner가 import하거나 실행하지 않는다.
3. 구현과 가설 입증은 다르다. 현재 코드·CLI·fixture·artifact 회귀 테스트는 통과했지만,
   실제 Linux CUDA 서버에서 official public dataset 전체 학습 결과는 아직 생성하지 않았다.
4. tiny run, legacy smoke, CPU wiring run은 논문 결과가 아니다.
5. dataset registry의 `implemented/code_ready`는 adapter와 runner가 있다는 뜻이다. 현재 로컬에
   official public cache가 있다는 뜻은 아니다.
6. novelty는 코드만으로 확정할 수 없다. 특히 conductance operator는 learned symmetric
   diffusion 계열, projector PE는 기존 cycle-space projector 계열, tree chart augmentation은
   gauge/frame augmentation 계열과 문헌 대조가 필요하다.
7. 2026-08-29 두 차례 외부 교차감사의 코드 gate를 반영했다. Strict cache, BREC official
   mode, Wilson exposure, tree orientation gauge, 독립 seed 축에 이어 두 번째 감사의 exact GPU
   constraints, suite-aware preflight, closed paper-metric registry도 구현했다. 반면 “통합 모델을
   active로 승격”하라는 항목은 결함 수정으로 받아들이지 않았다. 사용자가 요구한 현재 범위는
   세 독립 연구이고 결합은 이후 단계이기 때문이다.

### 코드 스냅샷

- 파일: `code_summary.md`
- 포함 파일: 90개
- 크기: 878,608 bytes, 23,460 lines (`str.splitlines()` 기준)
- SHA-256: `3756DA9F3C4F672082624C976058C6A1033718E2C3BAB123C3B98351B6BFCB55`
- 포함: 모든 Python source/test, TOML/YAML, Bash/PowerShell script, requirements, `.gitignore`
- 제외: `.venv*`, data/cache, run artifact, `egg-info`, README류 설명 문서

이 디렉터리는 Git repository로 초기화되어 있으며 원격은
`https://github.com/costunder/new-gat.git`이다. 서버에서는 실행 전에 `git rev-parse HEAD`를
확인한다. Paper runner도 source revision과 dirty 상태를 manifest에 기록한다.

## 1. 공통 수학 관례와 원래 문제의식

### 1.1 Incidence convention

연결된 무방향 graph에 임의의 edge orientation을 고정한다. 코드의 incidence matrix는
edge-by-node convention이다.

\[
B\in\mathbb{R}^{m\times n},\qquad
B_{e,u}=-1,\quad B_{e,v}=+1
\]

edge `e=(u→v)`에 대해 node signal `p`의 edge gradient는

\[
(Bp)_e=p_v-p_u
\]

이고 edge flow `q`의 node aggregation/divergence는 `B^Tq`다. Unweighted node Laplacian은

\[
L=B^TB=D-A.
\]

연결 graph에서

\[
\operatorname{rank}(B)=n-1,\qquad
\dim\ker(B^T)=m-n+1=\beta.
\]

따라서 사용자가 말한 “B의 좌영공간”은 정확히 `ker(B^T)`이며 edge circulation/cycle
space다. `F_T∈R^{m×β}`는 spanning tree `T`에 대한 full-column-rank fundamental cycle
basis이고 코드가 `B^TF_T=0`을 검사한다.

### 1.2 비가역성과 pseudoinverse에 대한 정확한 해석

`B^Tq=b`가 consistent라면 일반해는

\[
q=q_{\min}+F_Ta,
\]

이다. `q_min=(B^T)^+b`는 최소노름 particular solution일 뿐 원래 flow의 임의 cycle
component `F_Ta`를 복구하지 못한다. 이것은 “pseudoinverse가 완전한 원래 해를 복구한다”는
뜻이 아니다.

다만 rank deficient라는 사실만으로 방정식이 모순인 것은 아니다. `b`가 `im(B^T)`에 있으면
해가 여러 개이고, 밖에 있으면 least-squares residual이 남는다. `Lp=b`도 connected graph에서
상수 gauge 때문에 singular하지만 `sum(b)=0`이면 consistent하고 `L^+b`가 gauge-fixed
minimum-norm node solution을 준다.

즉 핵심 정보 손실은 다음과 같다.

\[
B^T(q+F_Ta)=B^Tq.
\]

node aggregation만 보면 cycle flow를 구별할 수 없다. 하지만 이것만으로 sample의 cycle
coefficient를 static topology PE에서 자동 복구할 수 있는 것은 아니다. 추가 관측, prior 또는
supervision이 필요하다.

여기서 잃는 것은 **주어진 edge flow `q`의 circulation component**이지 graph topology 자체가
아니다. Simple unweighted graph에서는 full matrix `L=D-A`의 off-diagonal entry가 adjacency를
결정하므로 cycle topology는 `L`에서 정보이론적으로 사라지지 않는다. 따라서 Static Cycle PE는
`L`이 지운 topology를 복원하는 codec이 아니라, cycle 구조를 모델이 쓰기 쉬운 edge
representation으로 제공하는 inductive bias다.

### 1.3 세 연구를 분리해야 하는 이유

| 연구 | 학습/입력 대상 | 검증하려는 것 | 검증하지 않는 것 |
|---|---|---|---|
| Conductance | `c_e>0`인 diagonal `Cθ` | node→edge→node transport law | cycle coefficient 복구, cycle PE |
| Static Cycle PE | topology-only `F_T`/cycle statistics/projector | cycle 구조를 edge representation에 주는 효용 | sample flow, learned C |
| Tree augmentation | 동일 graph의 여러 full-β `F_T` chart | chart shift에 대한 robustness | adaptive MST, sparsification, learned C |

`q=CθBp+F_Ta`를 하나의 state codec처럼 학습하는 연구는 현재 active pipeline에 없다. 결합은
각 독립 가설이 먼저 지지된 뒤 별도 연구로 해야 한다.

## 2. 저장소 지도

### Root와 공통 계층

- `README.md`: MobaXterm/Linux GPU 전체 재현 명령.
- `DATASETS.md`: 사람이 읽는 데이터·split·metric 계약.
- `pyproject.toml`: Python 3.11+, core/dev/paper dependency와 pytest/Ruff 설정.
- `requirements-lock.txt`, `constraints-cu*.txt`: Python 3.11 호환 exact top-level 연구 stack과
  CUDA 12.6/13.0/13.2별 official torch channel 계약.
- `requirements-paper.txt`: portable paper dependency가 같은 lock을 사용하게 하는 진입점.
- `scripts/setup_gpu.sh`, `scripts/verify_gpu_lock.py`: Linux GPU 환경 생성, exact package/ABI/CUDA
  runtime 검증과 transitive freeze snapshot.
- `scripts/gpu_preflight.py`: conductance/projector/tree-chart/BREC/public-PyG synthetic shape-stress.
- `scripts/paper.sh`: GPU environment Python으로 master runner 실행.
- `scripts/run_paper.py`: 세 독립 트랙을 model-seed별 subprocess로 dispatch하고 중앙 manifest 작성.
- `scripts/aggregate_paper.py`: 폐쇄형 paper metric/efficiency registry와 seed-aligned 통계.
- `scripts/generate_code_summary.py`: 외부 교차검증용 exact source snapshot 생성/검사.
- `scripts/check_datasets.py`: 세 `datasets.yaml`의 code/cache readiness 검사.
- `src/chartgat/algebra.py`: incidence, fundamental cycle basis, chart transition 등 공통 저수준 수학.
- `src/chartgat/cache.py`: same-directory temporary, fsync, validation, atomic replace cache writer.
- `src/chartgat/graphs.py`: connected graph와 tree helper.
- `src/chartgat/seeds.py`: data/split/chart/model seed 축과 legacy fallback 해석.
- `tests/`: root runner, GPU preflight, registry, algebra, cross-track import boundary 테스트.

### 활성 연구 폴더

- `research/conductance_gat/`
  - `sparse.py`: paper headline sparse operator와 packed variable-graph batch.
  - `paper_data.py`: S1–S4 generated protocols와 deterministic cache.
  - `public_data.py`: PascalVOC-SP와 ogbg-molhiv adapter.
  - `paper.py`: 독립 paper runner, models/baselines/metrics/artifacts.
  - `model.py`, `synthetic.py`, `run.py`: 과거 dense single-graph smoke 경로.
- `research/cycle_pe/`
  - `features.py`: fundamental basis, set statistics, projector 수학.
  - `paper_data.py`: CycleCount-OOD generator와 exact cycle labels.
  - `paper_adapters.py`: BREC/ZINC adapter와 안전한 download/cache 처리.
  - `paper_model.py`: 네 PE variant와 공통 graph backbone.
  - `paper_train.py`: supervised train/eval/runtime.
  - `paper.py`: core/BREC/ZINC paper runner.
  - `run.py`, `synthetic.py`, `config.yaml`: legacy structural smoke.
- `research/tree_augmentation/`
  - `augmentation.py`: full-β chart, transition, algebra certification과 legacy probe.
  - `paper_data.py`: core/CSL/ZINC graph data와 BFS/DFS/Wilson samplers.
  - `paper_model.py`: variable-edge/variable-β chart encoder와 training/evaluation.
  - `paper.py`: fixed-vs-multi independent paper runner.
  - `run.py`: legacy algebra/smoke.

### 격리된 결합 prototype

`research/combined_later/`에는 과거 flow completion, hard observation preservation, edge
residual 실험이 남아 있다. `pyproject.toml` package discovery와 active pytest 경로에서
제외되며 master paper runner도 실행하지 않는다. 외부 리뷰어는 이 코드를 active contribution과
섞어 평가하면 안 된다.

## 3. GPU/MobaXterm 재현 파이프라인

MobaXterm은 SSH terminal이므로 실제 명령은 원격 Linux GPU node에서 실행한다. Login
node에 GPU가 없다면 먼저 cluster scheduler로 GPU allocation을 받아야 한다.

### 3.1 설치

```bash
cd /path/to/NEW-GAT
nvidia-smi
bash scripts/setup_gpu.sh
```

`setup_gpu.sh`가 수행하는 작업:

1. Linux와 `nvidia-smi` 확인.
2. `.venv-gpu` 생성 또는 `USE_ACTIVE_ENV=1`일 때 활성 Conda/venv 사용.
3. driver compatibility에 따라 `cu126`, `cu130`, `cu132` 중 official channel 선택. CUDA 12.6
   미만에서 과거 torch로 자동 후퇴하지 않는다. `CUDA_WHEEL_TAG`로 지원 tag만 고정할 수 있다.
4. 선택한 `constraints-cu*.txt`와 `requirements-lock.txt`의 14개 top-level exact pin 설치.
5. `pip check`, paper dependency import-time ABI, exact version, `torch.version.cuda`,
   `torch.cuda.is_available()` 검증.
6. `.gpu-environment.json`과 `.gpu-environment.freeze.txt`에 lock hash와 실제 transitive 환경 저장.
7. sparse incidence gather → positive C → `index_add_` B^T → backward/finite 기본 검사.
8. pytest 실행.

현재 lock은 `torch==2.13.0`이고 Python 3.11 계약을 유지한다. Setup의 기본 preflight는 환경과
conductance smoke이며, master training invocation이 선택 suite에 맞는 고메모리 profile을 추가로
실행한다.

Full paper runner는 CPU fallback을 하지 않는다. CPU는 `--tiny --allow-cpu`를 함께 준 코드
fixture 검사에서만 허용된다.

### 3.2 데이터 준비와 strict cache 확인

```bash
paper_data_root=/scratch/$USER/new-gat-data
paper_results_root=/scratch/$USER/new-gat-results

bash scripts/paper.sh \
  --suite all \
  --prepare-only \
  --allow-download \
  --data-root "$paper_data_root" \
  --results-root "$paper_results_root" \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --run-id prepare-all-v1

.venv-gpu/bin/python scripts/check_datasets.py \
  --profile paper \
  --data-root "$paper_data_root" \
  --seeds 0 \
  --require-cache
```

`--allow-download`가 없으면 public endpoint를 호출하지 않는다. Generated dataset은
`data_seed`로 한 번 준비하며 model seed 수만큼 같은 cache를 중복 생성하지 않는다.
Dataset checker의 `--seeds`는 `--data-seeds` 호환 alias이고 model seed가 아니다. Data와
split cache 축이 다르면 `--data-seeds 11 --split-seeds 13`으로 각각 검사한다. Existing
non-empty output/run-id는 덮어쓰지 않는다.

Strict check는 glob 존재 검사가 아니다. Track validator가 request/schema/profile, split
cardinality와 graph IDs, tensor/target shape와 finite 값, content/artifact SHA-256, public 필수
split을 read-only로 검사하고 `valid/missing/incomplete/corrupt/wrong_request`를 구분한다.
Repository-controlled cache writer는 unique same-directory temporary에 쓴 뒤 fsync, temporary
validation, `os.replace`, manifest-last 순서로 publish한다.

### 3.3 전체 독립 실행

먼저 `--suite core --model-seeds 0` 한 seed kill test를 완료한 뒤 아래 명령을 실행한다.
Official BREC만 4 variants × 10 seeds × 400 pairs라서 `suite=all`을 첫 실행으로 쓰면 안 된다.

```bash
tmux new -s new-gat

bash scripts/paper.sh \
  --suite all \
  --device cuda \
  --data-root "$paper_data_root" \
  --results-root "$paper_results_root" \
  --model-seeds 0,1,2,3,4 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --batch-size 32 \
  --workers 4 \
  --min-free-gb 8 \
  --run-id paper-all-v1
```

Master runner의 기본값은 CUDA, model seed `0..4`, 고정 data/split/chart seed `0`이다.
각 `track×model_seed`는 독립 subprocess, output directory, stdout log를 갖는다. 한 트랙이
실패해도 기본적으로 나머지를 계속하고
`--fail-fast`일 때만 첫 실패에서 중단한다. 단 공통 GPU preflight 실패는 모든 child 실행
전에 전체를 중단한다.

Training preflight는 선택 track과 suite에서 `conductance`, `cycle-projector`, `tree-chart`, `brec`,
`public-pyg` 중 필요한 synthetic profile을 반복 실행한다. `public-pyg`는 conductance
`suite=all`에만 붙고 cycle/tree-only run에는 붙지 않는다. 각 profile은 실제 model
forward/backward를 거치고 allocated/reserved/peak/wall-time을 기록한다. 기본 envelope는
batch 32, graph당 nodes/edges/beta 64/128/64이며 root CLI의 `--preflight-*-per-graph`와
`--preflight-cycle-rank`로 올릴 수 있다. 이것은 다운로드 없는 shape-stress이지 cache 기반
dataset E2E나 GPU 성공 결과가 아니다.

`cycle-projector`라는 profile 이름은 CLI 호환을 유지하지만 실제로는 master의
`--cycle-variants`에 선택된 variant를 각각 `PaperCycleModel` forward/backward로 검사하며 dense
projector는 선택됐을 때만 materialize한다. BREC도 같은 variant마다 cosine backward와 T²
covariance/pinv를 실행한다. Full은 official batch 16/no-AMP, tiny는 custom 요청 batch/AMP를
사용한다. `--prepare-only`는 accelerator capacity 인증이 아니므로 CPU conductance smoke만
실행한다.

CycleCount와 ZINC는 model seed마다 실행한다. BREC는 자체 official search seed 10개를
내부에서 돌리므로 `suite=all`에서 `brec-official-10-seed` child로 정확히 한 번만 dispatch한다.
Master의 `--cycle-variants`, `--cycle-core-targets`, `--cycle-epochs`,
`--cycle-learning-rate`로 core 후보 축소를 할 수 있다. Official BREC는 선택 variant만 받되
optimization override는 받지 않고 고정 protocol을 유지한다.

### 3.4 중앙 및 트랙별 산출물

```text
runs/paper/<run-id>/
├── manifest.json
├── environment.txt
├── gpu-preflight.json
├── dataset-registries/
├── aggregate/
│   ├── aggregate.json
│   ├── samples.csv
│   ├── metrics.csv
│   ├── paired.csv
│   ├── efficiency.csv
│   └── failures.csv
└── logs/
```

`--results-root`를 주면 실제 모델 결과는
`<results-root>/<track>/<run-id>/...`에 분리된다. 중앙 runner는 모든 JSON이 parse 가능하고
NaN/Inf가 없는지 검사하고 dependency/source/registry hash를 기록한다.

Root 집계는 JSON의 모든 숫자를 metric으로 취급하지 않는다. Track별 폐쇄형 registry에 등록된
test metric만 data/split/chart seed를 grouping key로 유지하고 model seed에 대해 mean, sample
std, median, min/max, deterministic bootstrap 95% CI로 요약한다. Rule이 `pairable`이고 같은
model seed가 있는 condition만 right-minus-left difference와 paired Cohen's dz를 계산한다.
Seed/epoch/batch/config/history/count는 제외된다. Elapsed time, peak memory와
`trainable_active_parameters_only` count는 raw `efficiency.csv`로 분리하고 bootstrap/paired test를
하지 않는다. `ignored_numeric_fields`가 제외 수를 감사 가능하게 남긴다. 이 집계도 논문별
multiple-comparison correction이나 task-specific significance test를 대신하지 않는다.

## 4. 연구 트랙 A — Sparse Positive Conductance Operator

### 4.1 실제 가설

핵심 paper layer는 dense incidence matrix를 만들지 않고 다음을 계산한다.

\[
g_e=H_v-H_u,\qquad
c_e=\operatorname{softplus}(f_\theta(\cdot))+c_{\min},\qquad
q_e=c_eg_e,
\]

\[
H'=H-\eta B^Tq
   =H-\eta B^T\operatorname{diag}(c_\theta)BH.
\]

학습하는 것은 arbitrary `B^TCB` matrix가 아니라 edge별 positive diagonal scalar `c_e`다.
하나의 scalar conductance가 모든 hidden/state channel에 공통 적용된다. Directed/signed 또는
channel-coupled matrix conductance는 구현하지 않았다.

`full`의 conductance input은 edge feature와 `|BH|,(BH)^2`; `edge_only`는 edge feature;
실제 `gradient_only` 구현은 `|BH|`만 사용한다. Node update는 oriented gather와 두 번의
`index_add_`로 수행되며 graph 크기가 다른 sample을 packed batch로 처리한다.

Orientation을 뒤집으면 gradient/flux 부호는 바뀌지만 undirected orientation-invariant edge
feature라는 전제에서 conductance와 node update는 불변이다. `sum(B^Tq)=0`이므로 node-state
총합을 보존한다. Step은 graph별 maximum weighted degree에 따라 stability cap을 둔다.

이 구조는 original softmax-neighbor GAT라기보다 positive symmetric diffusion/transport
operator에 가깝다. 논문 명칭과 novelty claim은 GRAND류 diffusion, anisotropic diffusion,
edge-conditioned convolution, graph neural PDE 문헌과 다시 대조해야 한다.

### 4.2 학습 objective와 비교군

| 이름 | 입력/구조 | supervision | 해석 |
|---|---|---|---|
| `isotropic` | learned scalar `C=cI` | node message | diffusion baseline |
| `edge_only` | `c=f(x_E)` | node message | static edge law |
| `gradient_only` | `c=f(|BH|)` | node message | state-only law |
| `full` | edge + gradient | node message only | headline |
| `full_flux_supervised` | full | per-edge flux only | supervised neural ceiling |
| `full_joint` | full | node + flux | objective ablation |
| `flux_ls` | analytic | evaluation flux | transductive ceiling |
| `node_message_nnls` | projected NNLS | evaluation node message | identifiability ceiling |
| `oracle` | true C | none | data/operator oracle |

Headline `full`의 loss path는 flux target을 읽지 않는다. Regression test가 flux label을
삭제하거나 바꿔도 node-only loss가 동일함을 확인한다. LS/NNLS는 inductive learned
baseline으로 해석하면 안 된다.

### 4.3 Generated core S1–S4

| Suite | Full split/생성 | 핵심 평가 |
|---|---|---|
| S1 | graph 42/9/9, excitation 6/3/3; train graph의 new excitation 84개 별도 | held-graph 및 seen-graph excitation generalization, flux/node-message relative L2, log-C recovery |
| S2 | train/val: ER-like/RGG-like n=16–32, test: grid/barbell n=48–96 | topology/size OOD graph-macro error |
| S3 | graph 12/3/5, graph당 trajectory 1, horizon 50 | rollout 1/5/10/50, norm growth, dissipation/stability violation |
| S4 | contrast 1/10/100 × active fraction 1/0.25 × SNR inf/40/20 = 18 cells | known-contrast conditional recovery/error curve와 excitation coverage |

Cache spec에는 generator version, seed, tiny/full, graph IDs, content/file SHA-256가 들어가며
deterministic reload 때 모두 검증한다.

S4의 edge feature에는 `log10(contrast)/2`가 포함된다. 따라서 이것은 관측 가능한 operating
condition을 준 조건부 복원 실험이지 blind contrast identification이 아니며, unknown
conductivity-range recovery의 근거로 사용하면 안 된다. 18개 factor cell이 모두 train/validation/
test에 있으므로 unseen-contrast OOD가 아니라 held-graph-ID empirical recovery다. 또한 truth
conductance는 graph별 edge-feature min/max로 정규화되어 graph-global context에 의존하지만
headline estimator는 edge-local 입력만 보므로, S4 오차에는 identifiability뿐 아니라
function-class misspecification도 섞인다.

### 4.4 Public benchmarks

- PascalVOC-SP: PyG LRGB official train/val/test, node classification, macro-F1.
- ogbg-molhiv: OGB official scaffold split, graph classification, OGB ROC-AUC evaluator.

비교군은 no-message MLP, sparse GCN, custom single-head edge-aware GAT, custom GINE,
conductance model이다. 같은 adapter/split/hidden width/one-layer depth/head/optimizer 아래 실행하고
exact parameter count를 기록한다. MolHIV는 OGB AtomEncoder/BondEncoder를 사용한다.

다만 이들은 benchmark 논문의 reference hyperparameter configuration이 아니라 프로젝트 내부의
custom one-layer comparison이다. Parameter count를 기록하지만 budget을 강제로 같게 맞추지는
않는다.

### 4.5 산출물과 테스트

Standalone:

```bash
python -m research.conductance_gat.paper \
  --suite all --data-root /data/new-gat \
  --output-dir /results/conductance-seed0 \
  --device cuda --data-seed 0 --split-seed 0 --chart-seed 0 --model-seed 0 \
  --batch-size 32 --workers 4 --amp
```

성공 시 `summary.json`, `metrics.csv`, `history.csv`, CPU-portable `models.pt`가 생기며 runtime,
CUDA/AMP, peak memory, elapsed time을 기록한다. `summary.json`은 resolved
`data/split/chart/model` seed axes와 suite별 applicability도 기록한다. Generated core cache와
graph/excitation/trajectory 생성에는 `data_seed`만, model 초기화와 training shuffle에는
`model_seed`만 적용한다. 현재 conductance core의 split assignment는 data generation에
포함되어 별도 `split_seed`가 적용되지 않으며 chart 축도 없다. Official PascalVOC-SP와
MolHIV의 split/chart seed는 명시적으로 `not_applicable`이다. 기존 `--seed`는 명시하지 않은
축의 standalone fallback으로만 유지한다.

Conductance track test 20개가 sparse-vs-dense algebra, variable graph isolation, positivity,
orientation, mass conservation, objective leakage, S1/S2/S3/S4 split, deterministic cache,
S2 full cardinality contract, public fixture adapter, collision refusal와 tiny full CLI artifact를
검사한다.

### 4.6 반드시 재검토할 gap

1. 실제 CUDA full S1–S4 및 official PascalVOC-SP/MolHIV 결과가 없다.
2. S1/S2는 graph ID를 분리하지만 cross-split exact isomorphism/feature/C-law content hash
   guard가 없다. Registry는 더 이상 구현되지 않은 dedup을 claim하지 않는다.
3. generator 내부 명칭 `er`와 `rgg`는 엄밀한 G(n,p)/radius RGG가 아니다. 각각 connected
   recursive-tree+random pairs와 Euclidean MST+shortest pairs 형태이므로 논문에는 ER-like,
   RGG-like로 쓰거나 표준 generator로 교체해야 한다.
4. S3은 graph당 trajectory가 하나라 trajectory split이 graph split에 종속되고, unseen graph와
   unseen initial-condition 효과를 분리하지 못한다.
5. Core neural ablation은 input과 capacity가 함께 바뀌며 active parameter count를 기록하지 않는다.
   따라서 edge-only/gradient-only/full 차이를 순수 input contribution으로 해석하면 안 된다.
6. Public baseline은 custom 1-layer이고 tuning/depth/dropout/scheduler study가 없다. 사용하지
   않는 edge encoder는 이제 frozen/skipped되어 active trainable parameter count에서 제외되지만,
   backbone 사이 exact budget matching은 하지 않는다.
7. `suite=all` public 단계가 실패하면 그 child 안에서 이미 끝난 core를 독립 partial artifact로
    보존하지 않고 중앙 log만 남을 수 있다.
8. Real sensor conductance recovery dataset과 signed/directed/channel-matrix negative control이
    없다. Roman-empire는 planned, PGLib/MATPOWER는 현재 core가 아니다.

이미 교정된 항목: `gradient_only` 문서는 실제 `|BH|` 입력과 일치한다. Reciprocal
categorical conflict는 거부하고 continuous attribute는 평균하며, PascalVOC mean CE는 node
label 수로 train/validation 합산한다.

## 5. 연구 트랙 B — Static Cycle-space PE

### 5.1 실제 가설

Root-0 BFS tree로 topology에서 `F_T`를 학습 전에 한 번 계산하고 edge PE로 전달했을 때,
ordinary degree/topology feature만 쓰는 같은 backbone보다 cycle-composition/expressivity/분자
task에 도움이 되는지 검증한다. Learned conductance, node potential, sample circulation
coefficient, layer-to-layer cycle state는 없다.

### 5.2 네 PE variant

| Variant | 실제 입력 | 불변성/주의 |
|---|---|---|
| `no_pe` | zero PE | 공통 backbone control |
| `raw` | `F_T[e,:]` train-width zero padding | sign/order/tree/orientation dependent diagnostic |
| `set` | edge별 fundamental-cycle set 6 통계 | column sign/order 및 row orientation flip 불변, tree change 불변 아님 |
| `projector` | full `P=F(F^TF)^{-1}F^T` row DeepSets | invertible basis change와 orientation sign에 불변, O(m²) |

Paper projector encoder는 단순 `diag(P)`만 쓰지 않는다. 각 row의 full dense pair feature
`|P_ij|, |P_ij|², P_jj`를 encode하고 row mean/max, diagonal/row magnitude를 합친다. 반면
legacy bridge-vs-cycle smoke는 `diag(P)` leverage만 사용하며 target을 직접 드러내므로 headline
evidence에서 제외된다.

전처리의 lazy 범위는 제한적이다. 모든 variant에서 incidence, root-0 BFS tree와 full
fundamental basis `F_T`/`raw_basis`를 계산한다. 요청 여부에 따라 생략되는 것은 set 통계와 dense
projector뿐이다. 따라서 `no_pe`는 PE를 model input으로 사용하지 않지만 cycle-basis 전처리 비용까지
없애는 순수 end-to-end timing control은 아니다.

Raw width는 train split의 최대 `β`만으로 정한다. OOD graph의 `β`가 더 크면 절단하거나
test-fit하지 않고 해당 split을 N/A로 기록하며 다른 PE variant 평가는 계속한다. Validation
일부가 overflow면 compatible subset만 early stopping에 사용하고 full validation raw metric은
N/A다.

공통 backbone은 variable-edge/variable-β graph list batch, node/edge encoder, symmetric endpoint
edge update, 양방향 mean message passing, residual LayerNorm, node/edge mean-max graph pooling을
사용한다. Edge/node/graph head는 독립 task마다 새 모델로 생성된다.

### 5.3 CycleCount-OOD v4

Full split:

| split | graphs | family/size |
|---|---:|---|
| train | 10,000 | cubic regular 및 tree+chords, n=14–22 |
| validation | 2,000 | train regime held-out |
| ID test | 2,000 | train regime held-out |
| size OOD | 3,000 | regular/tree+chords, n=28–38 |
| family OOD | 3,000 | unseen small-world/local-chords |

Exact target:

- edge: C3–C6 participation, shortest containing cycle length, short-cycle congestion.
- node: C3–C6 participation.
- graph: C3–C6 total count.

Edge/node/graph target은 별도 model/head/checkpoint로 학습해 graph target이 node/edge auxiliary
label에서 직접 유도되는 leakage를 막는다. Metric은 MAE, RMSE, train-standard-deviation
normalized MAE, graph-macro MAE, rounded exact accuracy다.

Degree sequence 또는 `(n,m,β)` matched counterfactual은 아직 없다. 따라서 현재 protocol로
degree를 완전히 통제한 cycle-composition claim을 하면 안 된다.

Cycle PE도 data/split/chart/model seed 축을 분리한다. CycleCount 생성과 cache identity는
`data_seed`, supervised model 초기화와 DataLoader shuffle은 `model_seed`만 사용한다.
CycleCount의 split regime는 generator-defined라 `split_seed`는 `not_applicable`이고, static
BFS fundamental-basis PE에는 sampled chart가 없어 `chart_seed`도 `not_applicable`이다.
기존 `--seed`는 명시되지 않은 축의 호환 fallback이며 manifest가 resolve 값과 실제 axis 사용
정책을 함께 기록한다.

### 5.4 BREC v3

- Official 400 non-isomorphic pair와 6 category.
- 기본 relabel `q=32`, threshold `72.34`.
- 내부 search seed `100,200,...,1000`.
- Embedding difference `D=L-R`.
- Hotelling statistic:

\[
T^2=\bar D^T\operatorname{pinv}(\operatorname{cov}(D))\bar D
\]

q multiplier는 없다. Train distinguishability와 isomorphic reliability를 별도로 계산한다.
Official mode는 batch 16, 20 epochs, LR/weight decay `1e-4`, float32, no AMP,
no clipping, no shuffle를 강제한다. Variant와 search seed별로 한 번 seed한 뒤 400 pair를
순서대로 실행하고, upstream과 호환되는 seed별 `Correct`, `Fail`, `Real_correct`를 따로
보존한다. `global_valid=true`는 모든 seed가 complete하고 reliability failure가 0인지 보는
**저장소 자체의 보수적 gate**이며 upstream BREC metric이 아니다. Upstream search가 정의하지
않은 official any-seed union은 없다.

위 상수와 제어 흐름은 upstream reference에 정적으로 맞췄지만, 같은 model을 양쪽 runner에서
실행해 golden output을 비교하는 differential parity test는 아직 없다. 따라서
“official-protocol compatible”이라고는 할 수 있어도 bytewise/numerically identical한 upstream
execution이라고 주장하면 안 된다.
기존 union은 custom mode의 `custom_pairwise_union`으로 격리했다.
바깥 `model_seed`는 BREC official 실행에 섞지 않으며, 내부 search seed 열 개가 manifest의
별도 protocol axis다.

Adapter는 official mode에서 q=32, 400 pairs, 51,200 records를 강제하고 pair 단위 lazy
decode한다. Full download는
opt-in이며 HTTPS host, archive path traversal, Windows path, symlink, member/size/ratio와 NPY magic을
검사하고 ZIP/NPY SHA-256을 provenance로 저장한다. Upstream이 canonical SHA-256 pin을
공개했다고 주장하지 않는다. Tiny BREC은 2-pair offline custom fixture다.

중요: full `--prepare-only`는 400 pair 전체가 아니라 first/last pair만 decode/PE sanity check한다.

### 5.5 ZINC-12K

PyG `ZINC(subset=True)` official train/validation/test adapter를 사용한다. Atom은 28-way,
bond는 4-way categorical one-hot으로 변환하고 reciprocal bond type 불일치를 거부한다.
Graph regression target은 constrained solubility다. 네 PE variant를 같은 backbone에서 비교한다.

코드는 PyG adapter가 official split을 준다고 신뢰하며 loader length가 정확히
10,000/1,000/1,000인지 별도 assert하지 않는다. Tiny는 앞 32/8/8이지만 PyG/cache가 여전히
필요하다.

### 5.6 산출물과 테스트

Standalone 예시:

```bash
python -m research.cycle_pe.paper \
  --suite core --data-root /data/new-gat \
  --output-dir /results/cycle-core-seed0 \
  --device cuda --data-seed 0 --split-seed 0 --chart-seed 0 --model-seed 0 \
  --variants no_pe,raw,set,projector \
  --core-targets edge,node,graph --batch-size 64 --workers 8 --amp
```

Supervised artifact는 `<suite>/<level>/<variant>/` 아래 `model.pt`, `metrics.json`,
`history.json`, `runtime.json`을 저장한다. BREC는 `<variant>/pairs.json`과 `metrics.json`.

Master runner에서는 child output container에 suite 이름이 한 번 더 생긴다. 예를 들어 실제
core manifest는 `.../model-seed-0/core/core/manifest.json`, BREC는
`.../brec-official-10-seed/brec/...` 구조다.

Cycle track focused test 43개가 PE invariance/non-invariance, β=0, exact labels, 20k split specification,
deterministic cache, train-only raw width, variable batch, BREC layout/download/T²/reliability/seed
aggregation, variant-lazy projector, ZINC fixture와 CLI collision/partial preservation을 검사한다.

### 5.7 반드시 재검토할 gap

1. 실제 CUDA full 20k training, official BREC 400-pair E2E, real ZINC full 결과가 없다.
2. 모든 variant가 `F_T`를 계산한다. Set 통계와 projector만 요청에 따라 생략되며,
   `no_pe/raw/set`은 dense projector를 만들지 않는다. Projector variant 자체는 dense `m×m`,
   O(m²)이고 대형 graph scaling 검증이 없다.
3. Suite-level `preparation_seconds`는 여러 variant 비용이 섞이고 per-variant efficiency table은
   전처리 CPU time/RSS를 포함하지 않는다. 모든 split의 dense projector를 동시에 보관하는 비용도
   별도로 계측하지 않아 현재 표로 projector overhead를 공정하게 비교할 수 없다.
4. 네 variant의 parameter allocation은 거의 같게 생성되지만 실제 active path가 다르고 manifest에
   exact parameter count를 기록하지 않는다.
5. Auxiliary label perturbation을 통한 직접 leakage negative test는 없다. 구조적으로 target level별
   별도 모델인 것만 테스트한다.
6. Degree/`(n,m,β)` matched 또는 1-WL-indistinguishable이면서 C3–C6 target이 다른 known-contrast
   pair가 없다. 현재 CycleCount만으로 degree/size/family confound와 cycle signal을 완전히 분리할
   수 없다.
7. Raw/set은 root-0 BFS tree와 node labeling에 의존한다. Core/ZINC에는 isomorphic relabeling이나
   spanning-tree shift robustness 평가가 없어 graph-isomorphism-invariant PE라고 주장할 수 없다.
8. Raw는 큰-β OOD split 전체가 N/A가 될 수 있고, validation overflow 시 raw만 compatible subset으로
   early stopping한다. 따라서 hardest OOD에서 완전한 4-way 비교와 동일 validation-distribution
   비교가 성립하지 않을 수 있다.
9. Upstream 호환 field는 seed별 `Correct/Fail/Real_correct`다. `global_valid`는 저장소 자체 gate이고
   `custom_pairwise_union`은 custom metric이므로 둘 다 upstream official score로 표현하면 안 된다.
10. Official BREC의 상수/제어 흐름은 정적으로 맞췄지만 upstream differential/golden-output parity는
    검증하지 않았다.
11. Official BREC는 최대 4×10×400 pair training 결과를 메모리에 모은 뒤 마지막에 기록한다.
    Pair/seed 단위 incremental checkpoint와 resume가 없어 중단 시 현재 child 진행분을 같은
    run-id로 재개할 수 없다. Pair decode/PE 전처리도 variant/seed마다 반복된다.
12. No-PE 외 baseline이 부족하다. 최소한 LapPE/sign-invariant LapPE, RWSE, GIN/GINE,
    GPS/Graphormer와 CycleNet 또는 기존 cycle-space PE를 같은 split/budget으로 비교하고 통계적
    significance를 보고해야 한다.

## 6. 연구 트랙 C — Full-β Spanning-tree Chart Augmentation

### 6.1 실제 가설

같은 physical graph와 downstream label을 유지하고 spanning tree만 바꿔 full fundamental-cycle
chart `F_T∈R^{m×β}`를 여러 개 제공하면, root-0 fixed BFS chart 하나만 본 모델보다 held-out
chart와 topology OOD에 강해지는지 검증한다.

Full chart 사이에는

\[
F_{T_2}M=F_{T_1}
\]

인 invertible coordinate transform이 있고 구현은 수치적으로 integer unimodular structure와
cocycle을 인증한다. `k<β` truncation은 lossless라고 부르지 않고 명시적으로 비활성화한다.

### 6.2 Sampler와 training fairness

- fixed condition: graph당 root-0 BFS chart 1개.
- multi condition: graph당 random-root BFS/DFS만 섞은 finite chart bank. Full 기본 8개,
  tiny 3개. Wilson은 sampler-family OOD 평가를 위해 training에서 제외한다.
- `random_priority_kruskal`: Wilson과 다른 non-uniform legacy sampler이며 headline multi
  condition에는 들어가지 않는다.

Multi-chart는 매 update마다 새 tree를 online resample하는 것이 아니라 시작 시 생성한 finite
bank에서 minibatch sampling한다. Fixed와 multi는 architecture, model seed, optimizer와 optimizer
update 수(full 800, tiny 24)를 동일하게 맞춘다. Data/split/chart/model seed는 독립 축으로
manifest에 기록하며 chart bank는 chart seed, Torch 초기화와 minibatch는 model seed만 사용한다.
Distinct view 수는 1 대 K로 다르며 graph에
unique spanning tree가 충분하지 않으면 duplicate chart가 반복될 수 있다.

평가 quadrant:

| graph regime | `fresh_chart_seen_family` | `fresh_chart_unseen_family` |
|---|---|---|
| ID graph | fresh random-root BFS | fresh held-out Wilson |
| OOD graph | fresh random-root BFS | fresh held-out Wilson |

평가 BFS family는 fixed와 multi가 모두 training에서 보았고, Wilson family는 둘 다 보지 않았다.
따라서 model별로 의미가 달랐던 기존 seen/unseen 표기를 제거하고 fresh chart와 sampler-family
shift를 명시적으로 분리한다.
Wilson output이 BFS output과 같은 tree라는 이유로 reject하지 않으므로 UST를 BFS 결과에
조건부로 만들지 않는다. Exact-tree overlap은 별도 diagnostic으로 기록하며, held-out은 sampler
family exposure에 관한 용어다.

### 6.3 Model

`GraphChartView`는 physical graph, dense `F_T`, target, tree key와 optional chemistry를 보존한다.
Batch는 `[batch,max_edges,max_beta]`로 dense padding하고 masks로 variable edge/β와 β=0을
지원한다.

`VariableBetaCycleEncoder`는 각 `F[e,c]`에서 `|F[e,c]|`, `F[e,c]²`, normalized cycle
support를 encode하고 valid cycle columns에 sum/mean/max set pooling한다. 여기에 endpoint degree,
`1/n`, `1/m`, ZINC일 때 symmetric endpoint atom embedding과 bond embedding을 합쳐 edge
representation을 만들고 graph-level sum/mean/max pooling 후 예측한다.

이 sign-even 입력과 set pooling은 같은 physical tree의 edge orientation, cycle-column sign,
aligned row/column ordering, chemistry를 함께 옮긴 node relabeling에 invariant하다. 하지만
arbitrary spanning-tree basis change에는 본질적으로 invariant하지 않다. 특히 node relabeling으로
label-dependent BFS/DFS가 다른 physical tree를 선택하면 별도 chart shift다. 바로 그 chart
dependence를 multi-chart augmentation으로 줄이는 것이 가설이다. Dense padded `m×β`이므로 대형
graph scaling 결과는 없다.

### 6.4 Datasets

#### Core CycleCount-OOD v2

- Full: train 128, validation 24, ID test 40, OOD test 40.
- ID regime: n=8–12 recursive-tree+chords.
- OOD: C3–C6 여러 개를 bridge로 이은 unseen cactus cycle-chain family.
- Target: physical graph에서 chart sampling 전에 계산한 exact graph-level C3–C6 count 4-vector.
- `(n,m)` bucket 내 graph isomorphism dedup으로 split 중복 방지.

#### CSL

- PyG `GNNBenchmarkDataset(name='CSL')`.
- Label별 deterministic permutation 후 5 folds.
- fold 0–2 train, 3 validation, 4 test.
- 10-class accuracy.

#### ZINC-12K

- PyG official train/validation/test.
- Atom integer `x`와 canonical undirected bond category를 cache에 lossless 보존.
- Reciprocal category conflict, duplicate directed arc, self-loop, range error를 거부.
- 모든 chart에서 chemistry tensor는 고정되고 model prediction에 실제 영향을 주는 regression
  test가 있다.

### 6.5 Metrics, artifacts, tests

Regression은 MAE, normalized MAE, RMSE, graph-macro MAE, graph별 worst-chart MAE 평균,
prediction spread/std, rounded flip rate/exact vector accuracy를 기록한다. Classification은 accuracy,
graph-macro accuracy, worst-chart accuracy, probability std와 flip rate를 기록한다.

Standalone:

```bash
python -m research.tree_augmentation.paper \
  --suite all --data-root /data/new-gat \
  --output-dir /results/tree-seed0 \
  --device cuda --data-seed 0 --split-seed 0 --chart-seed 0 --model-seed 0 \
  --batch-size 16 --workers 4 --amp
```

성공 artifact는 `summary.json`, `manifest.json`, `fixed_bfs_model.pt`,
`multi_chart_model.pt`. Checkpoint는 CPU state dict와 target normalization/settings를 저장한다.

Tree track test 21개가 full-β lossless basis, chart transition/cocycle, Wilson UST 분포,
training/held-out sampler 분리, chart-independent target, β=0/1/2 masks, orientation/column-sign/
edge-order/same-tree-node-relabel gauge invariance, deterministic cache/split, ZINC chemistry
roundtrip/invariance/sensitivity, collision과 suite partial failure를 검사한다.

### 6.6 반드시 재검토할 gap

1. 실제 CUDA full/core/CSL/ZINC 결과가 없다.
2. Adaptive/learned MST, trainable spanning tree, MST sparsification은 구현하지 않았다.
3. Multi-chart는 finite offline bank이지 online resampling algorithm이 아니다.
4. Validation split은 cache에는 있으나 현재 fixed 800-update training의 early stopping 또는
   hyperparameter selection에 사용하지 않는다.
5. Fixed-vs-multi 같은 encoder 비교만 있고 conventional GNN/GAT/no-PE, sampler별 ablation,
   tuning 및 CI/significance가 없다.
6. `paper_headline_eligible=True`는 non-tiny이면 자동으로 붙는 artifact flag일 뿐 성능이나
   재현성 통과 판정이 아니다.
7. Full dense `[batch,max_edges,max_beta]` padding의 memory/scaling study가 없다.
8. Degree-matched OOD와 BREC chart stress는 없다.
9. CSL은 fixed-beta sanity에 가깝고 ZINC도 standard SOTA reference configuration과 비교하지
   않는다.
10. Fresh-seen은 exact train chart 재사용이 아니라 양쪽 model이 본 BFS family의 새 chart이고,
    fresh-unseen은 양쪽 model이 training에서 제외한 Wilson family다.
11. Lossy chord selection과 `k<β` compression은 의도적으로 비활성화되어 있다.

## 7. 데이터 레지스트리 상태

`scripts/check_datasets.py --profile paper --json` 기준으로 12개 paper entry의 adapter/runner가
구현되어 있다.

| Track | Generated/core | Public/all |
|---|---|---|
| Conductance | S1, S2, S3, S4 | PascalVOC-SP, ogbg-molhiv |
| Cycle PE | CycleCount-OOD v4 | BREC v3, ZINC-12K |
| Tree augmentation | CycleCount-OOD v2 multi-chart | CSL, ZINC-12K multi-chart |

현재 로컬 public cache는 없으므로 `--require-cache`를 통과했다고 기록하면 안 된다. GPU
서버에서 prepare/download 후 strict checker를 다시 실행해야 한다.

## 8. 자동 검증 상태

마지막 로컬 회귀검사:

```text
pytest -q                     140 passed
ruff check .                 All checks passed
ruff format --check .        79 files already formatted
dataset checker paper        code_ready=true, paper_benchmark_suite_complete=true
master core tiny CPU         passed (3 independent children + schema-v2 aggregate)
```

140 tests의 구성:

- Conductance track: 20.
- Cycle PE track: 43.
- Tree augmentation track: 21.
- Root algebra/runner/preflight/registry/boundary/cache/statistics/environment: 56.

이 테스트는 실제 CUDA kernel과 official dataset 전체 학습 성능을 대체하지 않는다. 로컬
master tiny wiring `reaudit-v4-core-tiny`는 `data=43, split=47, chart=53, model=41`로
통과했고 중앙 집계는 closed schema로 paper sample 1,809행, metric group 1,809개, paired
group 1,833개, 별도 efficiency 30행, failure/OOM 0을 생성했다. 금지된 configuration/seed/
history/임의 parameter count가 paper 표에 들어간 사례는 0이다. 이 숫자와 tiny 성능은 과학
결과로 사용하면 안 된다.

마지막 Windows CPU `pytest`는 exit code 0과 140 passed를 반환했지만 PyTorch serialization
구간에서 간헐적인 Windows faulthandler `access violation` 진단을 출력했다. 이 로컬 진단을
Linux CUDA 성공 또는 실패의 증거로 해석하지 말고, target 서버의 clean environment에서
동일 검사를 다시 실행해야 한다.

## 9. 논문 claim 전 우선순위

### P0 — 외부 GPU에서 결과를 만들기 전에 남은 일

1. 실제 target GPU node에서 `setup_gpu.sh`와 CUDA preflight 실행.
2. 모든 public dataset cache 준비 후 `check_datasets --require-cache` 통과.
3. 트랙별 one-model-seed kill test와 peak-memory 확인 후 후보 variant를 확정.
4. 확정 후보만 five-model-seed로 실행하고 중앙 `aggregate/`의 failure/NaN/OOM, raw path,
   mean/std/CI/paired effect를 검토.
5. Official public cache로 split size/checksum/evaluator contract를 재검증.

코드 수준 P0 교정은 완료됐다: semantic strict cache와 atomic publish, BREC official/custom
분리와 global reliability gate, Wilson train-family 제거, tree orientation gauge test, 네 seed 축,
closed root metric/efficiency 집계, exact CUDA constraints/verification, suite-aware shape-stress와
cycle candidate CLI, stale S2 full-cache cardinality(112/24/48) 계약 교정을 반영했다. 실제
CUDA/public full 결과가 없다는 경계는 그대로다.

### P1 — 강한 scientific claim 전에

1. Conductance: standard diffusion/GAT/GINE reference implementation과 tuned matched-depth/budget
   비교, real physical/sensor conductance data 또는 명확한 synthetic-only claim.
2. Cycle PE: degree/`(n,m,β)` matched counterfactual, prior cycle PE/Graph Transformer baseline,
   dense projector scaling 측정.
3. Tree augmentation: no-PE/standard GNN baseline, BFS-only/DFS-only/Wilson-only ablation, validation
   기반 selection, large-β scaling.
4. Leakage negative tests와 graph/canonical hash guard 강화.
5. 독립 reviewer가 BREC threshold/statistic/reliability와 category aggregation을 official source와
   다시 대조.

### P2 — 독립 가설 후 확장

1. Learned/adaptive spanning-tree 또는 variable tree distribution.
2. MST sparsification과 cycle coverage를 함께 다루는 별도 연구.
3. Conductance와 static cycle PE 결합.
4. 관측 edge flow가 있을 때만 particular solution + explicit cycle coefficient completion 연구.

이 확장들은 현재 구현된 contribution으로 쓰면 안 된다.

## 10. 외부 ChatGPT에게 권장하는 교차검증 질문

`code_summary.md`와 이 파일을 함께 주고 다음을 순서대로 요청하는 것이 좋다.

1. `B∈R^{m×n}` convention에서 `ker(B^T)`, `L=B^TB`, fundamental basis와 pseudoinverse 설명이
   수학적으로 정확한가?
2. 세 트랙 사이에 import, label, cache 또는 artifact를 통한 숨은 결합이 있는가?
3. Conductance `full node_only`가 실제로 flux target을 전혀 읽지 않는가?
4. S1–S4의 split이 graph/excitation/trajectory leakage를 충분히 막는가?
5. Conductance public 5-model 비교가 architecture/parameter/optimization 측면에서 공정한가?
6. CycleCount edge/node/graph task가 auxiliary-label leakage 없이 완전히 독립적인가?
7. Raw/set/projector의 invariance claim이 코드와 정확히 일치하는가?
8. BREC T², no-q convention, `isclose`, reliability와 10-seed aggregation이 official protocol과
   일치하는가? `custom_pairwise_union`이 official 결과와 완전히 분리됐는가?
9. Tree fixed-vs-multi가 optimizer exposure, distinct views와 sampler shift 측면에서 공정한가?
10. ZINC/CSL/PascalVOC/MolHIV adapter가 official split과 chemistry/feature 정보를 보존하는가?
11. CUDA/AMP/DataLoader에서 device mismatch, hidden CPU tensor, OOM 또는 nondeterminism 위험이
    있는가?
12. Existing tests가 놓친 failure mode를 우선순위와 함께 제시할 수 있는가?
13. 각 아이디어의 closest prior work와 실제 novelty boundary는 무엇인가?
14. 현재 artifact만으로 허용되는 claim과 금지해야 할 claim을 분리해 달라.

## 11. 공식 데이터/프로토콜 출처

- PyTorch installation: https://pytorch.org/get-started/locally/
- PyTorch Geometric ZINC: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html
- OGB graph property prediction: https://ogb.stanford.edu/docs/graphprop/
- LRGB PascalVOC-SP: https://github.com/vijaydwivedi75/lrgb
- BREC: https://github.com/GraphPKU/BREC
- BREC reference runner: https://github.com/GraphPKU/BREC/blob/Release/base/test_BREC.py
- Benchmarking GNNs/CSL/ZINC: https://www.jmlr.org/papers/v24/22-0567.html
- CycleNet: https://proceedings.mlr.press/v231/yan24b.html

## 12. 최종 인수인계 문장

현재 저장소는 세 아이디어를 섞지 않고 각각 실행할 수 있는 연구 scaffold와 검증 가능한
artifact pipeline을 갖췄다. 그러나 아직 논문 결과가 나온 상태는 아니다. 다음 작업자는 먼저
P0 GPU/public full run과 생성된 통계 집계를 검토하고, 그 결과가 지지하는 트랙만 독립 contribution으로
발전시켜야 한다. Adaptive MST나 세 트랙 결합은 그 이후 별도 실험으로 다룬다.
