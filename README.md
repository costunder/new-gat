# NEW GAT — 독립 연구 트랙 GPU 재현 가이드

이 저장소는 세 가설을 **서로 결합하지 않고** 각각 학습·평가한다.

1. `conductance_gat`: sparse incidence conductance attention
2. `cycle_pe`: 정적 fundamental-cycle positional encoding
3. `tree_augmentation`: spanning-tree chart augmentation

이전 결합 prototype은 `research/combined_later/`에 격리되어 있으며 paper runner가
import하거나 실행하지 않는다.

## MobaXterm에서 바로 실행

MobaXterm은 SSH terminal이다. 먼저 실제 Linux GPU node에 접속하거나 cluster의 GPU
allocation을 받은 뒤 저장소 root에서 실행한다. 첫 확인은 반드시 다음이어야 한다.

```bash
nvidia-smi
```

`nvidia-smi`가 login node에서 실패하면 그 상태에서 학습하지 말고 해당 cluster의
`srun`, `qsub`, `bsub` 등으로 GPU node를 먼저 할당한다.

### 1. CUDA 환경 설치 및 검증

```bash
bash scripts/setup_gpu.sh
```

이 스크립트는 다음을 실제로 수행한다.

- `.venv-gpu` 생성
- NVIDIA driver가 지원하는 공식 `cu126`, `cu130`, `cu132` PyTorch wheel channel 선택
- 선택한 `constraints-cu*.txt`와 `requirements-lock.txt`의 exact pin으로 설치
- PyG, OGB, SciPy, scikit-learn을 포함한 모든 top-level pin의 설치 버전 재검증
- import-time ABI와 `torch.version.cuda`가 선택한 wheel tag와 일치하는지 검증
- `torch.cuda.is_available()` 및 선택 GPU 검증
- sparse `B -> C -> B^T` gather/scatter forward, backward, finite-value 검증
- 전체 pytest 실행
- `.gpu-environment.json`과 `.gpu-environment.freeze.txt`에 실제 해석된 환경 기록

기본 lock은 `torch==2.13.0`이며 Python 3.11에서도 wheel이 존재하는 NumPy/SciPy
버전을 사용한다. 이 torch lock은 CUDA 12.6+ wheel만 지원하므로 `nvidia-smi`가 12.6
미만을 보고하면 과거 `cu118` torch로 조용히 후퇴하지 않고 설치를 중단한다.

서버가 제공하는 active Conda 환경의 **위치**를 재사용하려면:

```bash
conda activate YOUR_ENV
USE_ACTIVE_ENV=1 bash scripts/setup_gpu.sh
```

이 경우에도 setup은 기본적으로 torch와 모든 top-level exact pin을 설치·정렬한다. 기존
의존성을 전혀 바꾸지 않으려면 `USE_ACTIVE_ENV=1 SKIP_DEPS=1`을 함께 지정해야 하며, 이미
설치된 환경이 exact pin, ABI, CUDA runtime 검증을 모두 만족하지 않으면 즉시 실패한다.

driver가 여러 wheel runtime을 지원할 때 channel을 고정하려면 tag를 명시한다. 예를 들어
CUDA 13.2 호환 driver에서도 `cu126`을 의도적으로 사용할 수 있다.

```bash
CUDA_WHEEL_TAG=cu126 \
  USE_ACTIVE_ENV=1 bash scripts/setup_gpu.sh
```

허용 tag는 `cu126`, `cu130`, `cu132`뿐이며 각각 같은 이름의 constraints 파일과
공식 `download.pytorch.org` channel에 고정된다. 설치가 끝난 뒤 exact top-level pin과
실제 CUDA runtime이 다르면 setup은 성공 메시지를 내지 않는다. Transitive dependency는
setup 시점의 전체 `pip freeze`를 함께 남기므로 서버 실행 기록에 두 snapshot 파일도
보관한다. 이 Windows 작업공간에서는 constraints의 정적 일관성과 CPU 단위 테스트만
검증했으며, 각 CUDA tag의 실제 kernel 인증은 해당 GPU 서버에서 setup을 실행해야 한다.

### 2. 전체 데이터 준비

데이터와 결과를 scratch에 두는 예시다. 경로는 서버 정책에 맞게 바꾼다.

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
```

`--allow-download`는 이 단계에서만 public endpoint 접근을 허용한다. 이후 실행은
검증된 cache를 사용하며, cache가 없거나 깨졌다면 조용히 다른 데이터로 대체하지
않고 실패한다. 모델 seed sweep에서 benchmark가 바뀌지 않도록 generated dataset은
`--data-seed`로 한 번만 만들고, `--prepare-only`는 model seed 수만큼 같은 cache를 중복
생성하지 않는다.

준비 상태 확인:

```bash
.venv-gpu/bin/python scripts/check_datasets.py \
  --profile paper \
  --data-root "$paper_data_root" \
  --seeds 0 \
  --require-cache
```

이 검사는 glob 존재 여부만 보지 않는다. 각 트랙의 read-only validator가 요청 seed의
schema/profile, split cardinality, ID와 target shape/finite 값, manifest/data checksum을
검사한다. Public cache는 필수 split과 full protocol cardinality까지 확인하며 network나
cache 생성을 수행하지 않는다. 결과 상태는 `valid`, `missing`, `incomplete`, `corrupt`,
`wrong_request`로 구분된다. Tiny fixture를 검사할 때만 같은 명령에 `--tiny`를 추가한다.
여기서 dataset checker의 `--seeds`는 `--data-seeds` 호환 alias이며 model seed 목록이
아니다. Data와 split cache seed가 다르면 `--data-seeds 11 --split-seeds 13`처럼 각각 준다.

### 3. GPU 경로 짧은 인증

먼저 데이터 다운로드 없이 트랙별 실제 모델 경로를 shape-stress한다. `--profile`은
반복할 수 있고 `--profile all`은 아래 5개를 모두 고른다.

```bash
python scripts/gpu_preflight.py \
  --device cuda --require-paper-deps \
  --profile conductance \
  --profile cycle-projector \
  --profile tree-chart \
  --profile brec \
  --profile public-pyg \
  --batch-size 32 --brec-batch-size 16 \
  --nodes-per-graph 64 --edges-per-graph 128 --cycle-rank 64 \
  --cycle-variants no_pe,raw,set,projector --brec-protocol official --no-brec-amp \
  --amp --min-free-gb 8 \
  --json-out runs/preflight/paper-envelope.json
```

각 profile은 다음 production path를 forward/backward까지 실행한다.

| profile | 검사 경로 |
|---|---|
| `conductance` | packed sparse gather → learned positive conductance → scatter |
| `cycle-projector` | 선택한 Cycle variant 각각의 `PaperCycleModel`; projector는 선택됐을 때만 dense `m×m` |
| `tree-chart` | padded dense `batch×m×beta` chart → variable-beta encoder |
| `brec` | 선택 variant별 BREC model backward와 32쌍 covariance/pinv; official은 float32/no-AMP batch 16 |
| `public-pyg` | synthetic PyG `Data` → 실제 adapter/DataLoader → PascalVOC-SP와 MolHIV model |

JSON의 각 profile에는 `allocated`, `reserved`, `peak_allocated`, `peak_reserved`(bytes),
`wall_time`(seconds), host tensor/RSS와 실제 검사 shape가 기록된다. CUDA에서는 지정한
batch/shape를 그대로 쓰고 CPU `--allow-cpu`에서는 작은 코드-smoke shape로 자동 축소한다.
`--cycle-variants`가 Cycle/BREC 양쪽 model path를 정한다. Full official BREC는
`--brec-protocol official --brec-batch-size 16 --no-brec-amp`를 강제하고, tiny custom은
master의 요청 batch/AMP를 따른다. 다른 profile은 `--amp/--no-amp`를 따른다.

중요하게, 이것은 명시한 shape의 **synthetic memory/compute stress**이며 public dataset을
디스크에서 읽은 E2E 인증이나 결과 재현이 아니다. 실제 cache에서 관측한 최대
`nodes/edges/beta`가 기본값보다 크면 세 shape 인자를 올려 다시 인증해야 한다. 아래
tiny CUDA run은 adapter부터 artifact까지의 짧은 E2E이고, 둘 다 성능 결과로 쓰지 않는다.
Master runner는 선택한 track/suite/PE variant에 맞춰 이 profile들을 자동으로 고른다.
`public-pyg`는 conductance `suite=all`에만 붙고 cycle/tree-only run에는 붙지 않는다.
`--prepare-only`는 accelerator capacity 인증이 아니므로 CPU conductance code-smoke만 실행한다.
Master에서 envelope를 바꿀 때는 `--preflight-nodes-per-graph`,
`--preflight-edges-per-graph`, `--preflight-cycle-rank`를 사용한다.

```bash
bash scripts/paper.sh \
  --suite core \
  --tiny \
  --device cuda \
  --data-root "$paper_data_root" \
  --results-root "$paper_results_root" \
  --model-seeds 0 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --run-id gpu-sanity-v1
```

### 4. 한 seed kill test

BREC를 포함한 전체 실행 전에 각 트랙의 core를 한 model seed로 먼저 확인한다.

```bash
bash scripts/paper.sh \
  --suite core --device cuda \
  --data-root "$paper_data_root" \
  --results-root "$paper_results_root" \
  --model-seeds 0 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --batch-size 16 --workers 2 --min-free-gb 8 \
  --run-id core-kill-test-v1
```

Cycle 후보를 먼저 줄일 때는 master에서도 variant, target, optimization budget을 전달할 수
있다. 이 명령은 CycleCount core만 실행하며 official BREC를 시작하지 않는다.

```bash
bash scripts/paper.sh \
  --tracks cycle_pe --suite core --device cuda \
  --model-seeds 0 \
  --cycle-variants no_pe,projector \
  --cycle-core-targets graph \
  --cycle-epochs 20 --cycle-learning-rate 0.001 \
  --batch-size 16 --workers 2 \
  --run-id cycle-candidate-kill-v1
```

Official BREC는 고정 20 epochs/LR `1e-4` protocol이므로 위 optimization override를
무시하고, `--cycle-variants`로 확정한 후보만 받는다.

### 5. 전체 독립 실험

SSH가 끊겨도 계속 돌도록 `tmux` 안에서 실행하는 것을 권장한다.

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

다른 GPU를 직접 고를 때는 `--device cuda:1`을 사용한다. cluster가
`CUDA_VISIBLE_DEVICES`를 설정한다면 보통 그 allocation 안에서 `--device cuda`가
맞다. OOM이면 같은 `run-id`를 재사용하지 말고 batch size를 낮춘 새 run을 만든다.
`suite=all`은 official BREC만 해도 4 variants × 10 search seeds × 400 pairs이므로 매우
비싸다. Kill test와 트랙별 후보 축소를 통과한 뒤 실행한다.

## CPU에 대한 명확한 경계

Full paper training은 CPU로 자동 fallback하지 않는다. `scripts/run_paper.py`는 full
CPU 실행을 거부한다. CPU는 오직 코드/fixture 검증에만 명시적으로 허용된다.

```bash
python scripts/run_paper.py \
  --suite core --tiny --device cpu --allow-cpu \
  --model-seeds 0 --data-seed 0 --split-seed 0 --chart-seed 0 \
  --run-id cpu-code-check
```

이 숫자는 논문 결과가 아니다. 실제 결과에는 CUDA preflight가 통과한 run만 쓴다.

## 세 연구의 경계

| 폴더 | 독립 가설 | 포함 | 포함하지 않음 |
|---|---|---|---|
| `research/conductance_gat/` | \(H\to BH\to C_\theta BH\to B^\top C_\theta BH\) | positive learned conductance, sparse variable-graph batching | cycle basis, tree chart, flow completion |
| `research/cycle_pe/` | \(F_T\subset\ker(B^\top)\)를 정적 edge PE로 전달 | raw/set/projector 비교, cycle counting, BREC, ZINC | learned conductance, sample circulation, chart augmentation |
| `research/tree_augmentation/` | 같은 graph에서 \(T\)를 바꾼 full-\(\beta\) chart 증강 | BFS/DFS multi-chart training, held-out Wilson UST 평가 | conductance, potential, lossy \(k<\beta\) |

교차 import는 `tests/test_research_boundaries.py`가 막는다. 공통 사용이 허용된 것은
`src/chartgat/`의 저수준 incidence/tree algebra뿐이다.

## 실제 구현된 데이터 및 비교

| 트랙 | `core` | `all`에서 추가 |
|---|---|---|
| Conductance GAT | S1 multi-graph identification, S2 topology/size OOD, S3 nonlinear rollout, S4 noise×coverage×contrast | PascalVOC-SP, ogbg-molhiv |
| Static Cycle PE | 20k CycleCount-OOD, C3–C6 edge/node/graph task를 독립 학습 | BREC v3 official RPC, ZINC-12K official split |
| Tree augmentation | graph-first CycleCount multi-chart 2×2 ID/OOD×seen/unseen-chart | CSL, chemistry-preserving ZINC-12K multi-chart |

Conductance headline model은 per-edge flux label을 읽지 않고 **node message만으로**
학습한다. `full_flux_supervised`는 별도 ceiling, `full_joint`는 ablation이다. 비교에는
isotropic, edge-only, gradient-only, node-message NNLS, flux LS, oracle이 포함된다.
Public task에서는 동일 adapter/split/hidden/depth/head/optimizer 아래 no-message MLP, GCN, GAT,
GINE, conductance model을 비교하고 active parameter count를 기록한다. 계산하지 않는 edge encoder는
freeze/skip되지만 custom one-layer backbone의 budget을 정확히 같게 맞춘 비교는 아니다.
S4는 contrast를 edge condition으로 명시적으로 입력하는 `known-contrast conditional recovery`이며
blind contrast identification, unseen-contrast OOD 또는 unknown conductivity-range recovery 실험이
아니다. Graph-global min/max로 만든 truth law와 edge-local estimator의 function-class mismatch도
오차에 포함된다.

Cycle PE의 projector는 기존 계열에 가까운 prior baseline이며 새 기여라고 부르지
않는다. 모든 variant가 incidence/BFS tree/fundamental basis `F_T`를 계산하고, set 통계와
dense projector만 요청 여부에 따라 조건부로 만든다. 따라서 no-PE/raw/set 실행은 dense
projector를 만들지 않지만 no-PE도 basis 전처리 비용까지 제거하지는 않는다. Raw PE 폭은
train split만으로 결정하고 OOD의 더 큰 cycle rank를 잘라 넣거나 test 정보로 input dimension을
정하지 않는다. BREC는 distinguishability와 isomorphic reliability를 함께 판정한다. Seed별
`Correct/Fail/Real_correct`는 upstream 호환 field지만, 모든 seed의 reliability를 요구하는
`global_valid`는 저장소 자체의 보수적 gate다.

또한 `B^Tq`가 `ker(B^T)` circulation을 구별하지 못한다는 사실과 full graph Laplacian
`L=D-A`가 simple unweighted graph topology를 결정한다는 사실은 다르다. Static Cycle PE는
sample circulation 복구가 아니라 topology inductive bias다. 현재는 degree/`(n,m,β)`-matched 또는
1-WL known-contrast, raw/set relabel robustness, LapPE·RWSE·CycleNet·표준 GNN/Graph Transformer
baseline, variant별 preprocessing time/RSS와 large-graph scaling 결과가 없다. Official BREC도
incremental pair checkpoint/resume를 지원하지 않는다.

Tree protocol은 물리 graph split을 먼저 고정한 뒤 graph 내부에서 chart를 만든다.
Wilson sampler는 uniform spanning tree이고, 기존 random-priority Kruskal은 별도
non-uniform baseline으로 이름을 유지한다. ZINC cache/model은 atom과 bond categorical
feature를 보존하며 topology-only 회귀로 바꾸지 않는다.
Multi training에는 BFS/DFS만 사용하고 Wilson은 실제 held-out sampler family로 남긴다.
Encoder는 sign-even cycle membership을 사용해 같은 physical tree의 orientation/order gauge에는
불변이지만, label-dependent traversal이 다른 tree를 고르는 경우는 chart shift로 평가한다.

세부 split, metric, leakage guard와 source는 [DATASETS.md](DATASETS.md) 및 각
`research/<track>/datasets.yaml`에 있다.

## Runner가 보장하는 것

`scripts/run_paper.py`는 각 `track × model seed`를 별도 subprocess와 별도 출력 폴더에서
실행한다. 예외적으로 BREC는 자체 official 10-seed loop를 가지므로 `suite=all`에서
정확히 한 번만 dispatch한다. 따라서 바깥 5 seeds와 곱해 50회 중복 실행하지 않는다.
한 독립 run이 실패해도 기본적으로 나머지는 계속 실행하고 중앙 manifest에 실패를
남긴다. 첫 실패에서 중단하려면 `--fail-fast`를 사용한다.

- 기본 device: `cuda`
- 기본 model seeds: `0,1,2,3,4`
- 기본 data/split/chart seed: 각각 `0`
- CUDA에서 AMP 사용(BREC official은 float32/no-AMP 고정)
- 선택 suite의 synthetic shape-stress와 실제 gather/scatter backward 사전검사
- run-id 충돌 거부
- dependency, source revision, registry SHA256 snapshot
- 모든 JSON의 parse 및 non-finite number 검사
- track/model-seed별 stdout log
- 폐쇄형 paper-metric registry에 등록된 지표만 mean/std/bootstrap 95% CI와
  seed-aligned paired difference로 자동 집계
- runtime/peak-memory/active-parameter 관측은 통계 검정과 분리된 efficiency table에 기록
- 실패 및 CUDA OOM log 판정

명령만 확인하고 아무 파일도 만들지 않으려면:

```bash
bash scripts/paper.sh \
  --suite all --device cuda --model-seeds 0,1,2,3,4 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --run-id inspect-only --dry-run
```

## 결과 위치

`--results-root`를 생략하면 track별 결과가 연구 폴더 아래에 분리된다.

```text
research/<track>/results/paper/<run-id>/model-seed-<seed>/
```

`--results-root /scratch/...`를 주면 다음처럼 저장된다.

```text
/scratch/.../<track>/<run-id>/model-seed-<seed>/
```

Cycle `suite=all`은 `cycle_pe/<run-id>/model-seed-<seed>/{core,zinc}/`와
`cycle_pe/<run-id>/brec-official-10-seed/`로 더 분리된다.

중앙에는 모델을 합치지 않고 재현 metadata만 둔다.

```text
runs/paper/<run-id>/
├── manifest.json
├── environment.txt
├── gpu-preflight.json
├── dataset-registries/
├── aggregate/{aggregate.json,samples.csv,metrics.csv,paired.csv,efficiency.csv,failures.csv}
└── logs/
```

같은 `run-id`가 이미 있으면 덮어쓰지 않는다.

`metrics.csv`는 JSON의 모든 숫자를 재귀 수집하지 않는다. Track별 폐쇄형 schema에 등록된
test metric만 포함하며 각 rule이 paired comparison 가능 여부를 명시한다. Seed, epoch,
batch size, configuration, history, sample count와 임의 parameter count는 paper 통계에서
제외된다. `efficiency.csv`도 elapsed time, peak GPU memory와
`trainable_active_parameters_only`로 인증된 count만 허용하고 bootstrap/paired test를 하지
않는다. 제외된 숫자의 수는 `aggregate.json::ignored_numeric_fields`에 남는다.

## 트랙 하나만 실행

Master runner로 특정 트랙만 고르는 것이 가장 안전하다.

```bash
bash scripts/paper.sh \
  --tracks conductance_gat \
  --suite all --device cuda \
  --data-root "$paper_data_root" \
  --results-root "$paper_results_root" \
  --model-seeds 0,1,2,3,4 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --run-id conductance-v1
```

`cycle_pe`, `tree_augmentation`도 같은 방식으로 바꿔 실행한다. 각 폴더의 README에는
더 세부적인 standalone CLI와 artifact schema가 있다.

## 검증 명령

```bash
.venv-gpu/bin/python -m pytest -q
.venv-gpu/bin/ruff check .
.venv-gpu/bin/ruff format --check .
.venv-gpu/bin/python scripts/generate_code_summary.py --check
```

Dataset code 준비와 cache 준비는 구분해서 확인한다.

```bash
.venv-gpu/bin/python scripts/check_datasets.py --profile paper
.venv-gpu/bin/python scripts/check_datasets.py \
  --profile paper --data-root "$paper_data_root" \
  --seeds 0 --require-cache
```

## 현재 검증 범위

이 작업 공간에서는 Windows CPU로 수학·cache·fixture·CLI·artifact 회귀 테스트를
검증했다. 실제 public dataset 전체 다운로드와 CUDA kernel 실행은 GPU 서버에서 위
명령으로 검증해야 한다. 따라서 저장소가 CUDA 실행을 준비하고 강제하는 것과, 이미
논문 결과가 생성됐다는 것은 구분한다.

기존 소형 smoke가 필요할 때만 다음을 사용한다.

```bash
bash scripts/setup.sh
bash scripts/smoke.sh --run-id legacy-smoke-v1
```

Paper 결과에는 이 legacy smoke를 사용하지 않는다.
