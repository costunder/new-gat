# NEW GAT 연구 프로젝트 Hand-off

작성 기준일: 2026-08-30 (Asia/Seoul)

이 문서는 외부 ChatGPT 또는 연구 리뷰어가 저장소를 처음 받아도 수학적 가설, 구현 경계,
데이터 계약, 실행법, 검증 범위와 미완료 항목을 혼동하지 않도록 만든 인수인계 문서다.
원문 코드는 같은 폴더의 `code_summary.md`에 파일별로 들어 있다.
처음 설치·실행하는 사용자는 [README.md](README.md)의 순서를 따른다.
이 문서는 실행 입문서가 아니라 연구·구현 교차검토용이다.

## 0. 리뷰어가 먼저 알아야 할 판정

### 2026-08-30 서버의 NumPy 누락 오류에 대한 설치 경로 교정

서버의 활성 `new-gat` Python에서 데이터 준비 중 `ModuleNotFoundError: numpy`가 보고됐다.
`environment.yml`은 Python/pip만 만드는 bootstrap이며, 기존 `paper.sh`는 Conda interpreter만
확인한 뒤 의존성 검사 없이 runner를 시작했다. `chartgat.cache` import가 package initializer의
eager algebra import를 통해 NumPy부터 요구하여 실제 준비 전 traceback으로 종료됐다.
서버의 이전 설치 로그가 없으므로 setup을 건너뛰었는지, 설치가 중간에 실패했는지는 단정하지 않는다.

- `scripts/check_dependencies.py`는 표준 라이브러리만으로 시작해 전체 lock의 누락/버전 불일치를
  한 번에 보고하고, runtime import와 CUDA wheel 종류도 검사한다. GPU allocation 자체는 요구하지 않는다.
- `paper.sh --prepare-only`는 검증된 Conda Python으로 먼저 검사하고, 불완전하면 전체 setup을
  한 번 실행한 뒤 재검사한다. 어느 단계든 실패하면 데이터 준비로 넘어가지 않는다.
- `setup_gpu.sh`의 `SKIP_DEPS` 분기를 제거했다. 기본 설치와 자동 보완 모두 전체 고정 의존성을
  설치한다. 설치 자체의 Linux/NVIDIA 드라이버·GPU 할당·네트워크 요건은 그대로다.
- 학습 진입점은 설치 상태를 읽기 전용으로 확인하며 누락되면 run 폴더 생성 전에 설치 명령을 안내한다.
  `chartgat`의 algebra export는 lazy loading으로 바꿔 cache/도움말이 NumPy에 의존하지 않게 했다.
  `--help`/`--dry-run`은 설치·다운로드 없이 실행되며 자동 보완 대상이 아니다.
- 패키지를 볼 수 없는 `python -S` subprocess로 cache import, CLI 도움말/dry-run, 누락 오류 안내를
  검증했다. 기존 public algebra API도 유지된다. 전체 pytest **256 passed, 24 skipped**, Ruff 통과.
  생략은 Linux/Bash 동적 검사 23개와 PyG batching 검사 1개다. 기존 Windows faulthandler 진단이
  출력됐지만 pytest 프로세스는 종료 코드 0이었다. 실제 서버 설치와 CUDA 학습은 여기서 실행하지 않았다.

### 2026-08-30 최종 실행 범위: 트랙별 데이터에서 우리 모델만 실행

기본 실행은 `--suite benchmark`로 변경했다. Conductance GAT는 GAT/GATv2 논문에
나오는 Cora/CiteSeer/PubMed/PPI/ogbn-arxiv에서 우리 conductance 모델만 학습한다.
Cycle PE는 SignNet/PEARL 공통 ZINC-12K 및 PEARL의 Peptides-struct에서 우리 cycle-set
PE 모델만 학습한다. 외부 비교 모델의 구현·실행은 제거했다. 트리 증강의 CSL/ZINC
fixed-BFS vs multi-chart는 같은 우리 모델의 증강 ablation이므로 독립적으로 유지한다.

기본 실행에서 S1–S4/CycleCount 생성과 BREC는 제외했다. 기존 코드는 명시적인 `core`/`all`
보조 suite에 남아 있다. 아래의 이전 감사 내용에서 '현재 paper core'는 이 교정 전 경로를
가리키며, 새 benchmark의 실행 목록으로 읽으면 안 된다. 이 보조 suite에서도 외부
MLP/GCN/GAT/GINE 모델은 실행하지 않는다. 자체 연산의 ablation/해석적 진단은 별개다.

외부 비교 점수는 **논문 표에서 출처를 밝혀 인용**한다. 저장소가 직접 재현한 값이라고
표기하지 않으며 우리 seed별 결과와 paired 통계로 합치지 않는다. 데이터 이름뿐 아니라
버전/split/지표/입력/파라미터 예산/학습 조건도 확인하고 다른 조건을 표 주석에 명시한다.
전체 공개 데이터 다운로드/GPU 학습은 이 Windows 작업 공간에서 수행하지 않았다.
Alchemy는 upstream index의 중복·split 겹침 때문에 기본 데이터에 추가하지 않았다.

#### 새 기본 실행의 구현 경계

| 트랙 | 새 실행 모듈 | 비교·데이터 계약 |
|---|---|---|
| Conductance GAT | `research/conductance_gat/benchmark.py` | Cora/CiteSeer/PubMed public masks, PPI 20/2/2, ogbn-arxiv official split; 우리 conductance |
| Cycle PE | `research/cycle_pe/benchmark.py` | ZINC 10k/1k/1k, Peptides-struct 10873/2331/2331; 우리 cycle-set PE |
| Tree augmentation | 기존 `research/tree_augmentation/paper.py`의 `csl`·`zinc` | fixed-BFS vs multi-chart; CSL은 한 개의 stratified 90/30/30 split, ZINC는 official split |

- GAT의 `benchmark_data.py`는 원본/분할 checksum과 전처리 규칙을 저장한다.
  우리 모델은 기존 positive scalar estimator와 sparse `H - eta B^T C B H`를 사용한다.
  표준 GAT/GATv2/GCN/SAGE 생성 분기와 `--baselines` 선택 옵션은 없다.
  ogbn-arxiv 학습은 full-batch로, GATv2 논문의 GraphSAINT 설정과 다르다.
- PE의 `benchmark_data.py`는 공식 atom/bond feature와 target을 그대로 보존한다.
  `benchmark_models.py`는 기존 Cycle PE의 edge-aware message layer와 task head를 사용한다.
  BFS fundamental basis, 여섯 set statistic, GELU encoder 및 edge-PE 주입을 재사용한다.
  별도 GINE 경쟁 모델이나 LapPE/RWSE/SignNet/PEARL 구현·전처리는 없다.
  PE를 downstream 예측으로 읽는 neural layer는 우리 모델의 구성요소이며 외부 비교 모델이 아니다.
- Cycle-set은 cycle-column sign/order에는 불변이지만 chart 교체에는 불변이 아니다.
  원 논문과 같은 데이터셋을 쓰는 것과 논문 모델의 수치를 재현하는 것은 구분한다.
- 학습은 CUDA 전용이다. 기본 float32, PPI batch 2, 분자/tree batch 32, model seeds 0–4다.
  GAT/PE는 validation으로 checkpoint를 선택한 뒤 test를 한 번 평가한다. GAT는 accuracy,
  PPI는 전체 node-label micro-F1, PE는 MAE를 사용한다. 시간/메모리/파라미터도 따로 저장한다.
- Root `scripts/run_paper.py` 기본값과 다섯 Bash wrapper를 새 `benchmark`로 연결했다.
  준비는 GAT/PE/CSL/ZINC 네 child만 한 번씩 수행한다. 기본 재현은 preflight 이후
  GAT 5개 + PE 5개 + tree 10개 child이며 세 트랙 결과 폴더를 분리한다.
- Benchmark schema v2는 `datasets.<dataset>.models.conductance` 또는 `.models.cycle_set`를
  사용한다. `scripts/aggregate_paper.py`는 이 경로의 test만 성능으로 집계하고
  validation/history/외부 모델 점수/인용 수치를 제외한다. Paired 비교는 우리 모델의
  내부 ablation에만 적용하며 benchmark의 단일 모델로 외부 모델과의 통계를 만들지 않는다.
- 보조 Cycle PE 기본 variant는 raw/set/projector이고 No-PE는 명시적 옵션의 내부 ablation이다.

#### 검증 범위

- 외부 모델 제거 후 root runner/집계/registry 및 세 트랙 관련 검사: **174 passed, 1 skipped**.
  생략된 한 검사는 로컬 PyG 미설치로 실행하지 못한 데이터 batching 검사다.
- 수정 Python의 Ruff 검사 통과. 기본 재현의 20개 학습 child와 준비의 4개 child 명령은
  실제 각 트랙의 CLI parser로 검증했다. 외부 모델 선택 옵션은 거부되고, 논문 인용 점수는
  우리 결과 집계와 paired 통계에 포함되지 않는 것을 테스트했다.
- Windows 검사 중 `Windows fatal exception: access violation` 진단 출력이 있었으나
  검사 프로세스는 계속 실행되어 위 pytest 결과와 종료 코드 0을 반환했다. 이 호스트 진단을
  Linux/CUDA 검증 통과로 해석하지 않는다.
- 이 작업에서는 실제 데이터 다운로드, PyG/OGB 실제 cache 로딩, Linux Bash 실행 또는
  GPU benchmark 학습을 수행하지 않았다. CPU 학습 결과를 연구 결과로 생성하지 않았다.

### 기존 연구 및 감사 기록

1. 활성 연구는 세 개이며 서로 독립이다.
   - `research/conductance_gat`: positive scalar edge conductance를 학습하는 sparse incidence
     operator.
   - `research/cycle_pe`: topology에서 미리 계산한 static cycle-space edge PE.
   - `research/tree_augmentation`: 같은 graph의 full-cycle-rank fundamental basis를 여러
     spanning-tree chart로 바꾸는 augmentation.
2. 위 세 연구를 결합한 모델은 아직 없다. `research/combined_later`는 격리된 과거
   prototype이며 paper runner가 import하거나 실행하지 않는다.
3. 구현과 가설 입증은 다르다. 현재 코드·CLI·fixture·artifact 회귀 테스트는 통과했지만,
   실제 Linux CUDA 실행 환경에서 official public dataset 전체 학습 결과는 아직 생성하지 않았다.
4. 실험 CLI의 `--tiny`, 공개 데이터 대체용 가짜 데이터 생성, legacy smoke 실행기는 제거했다.
   테스트 내부의 작은 입력과 실제 연구용 S1–S4/CycleCount 합성 벤치마크는 별개다.
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
8. 2026-08-30 GPU 실행 환경을 전용 Conda 생성·활성화 방식으로 변경했다. 두 Bash
   entrypoint는 활성 non-base Conda Python을 검증한 뒤 설치·실행하며 venv를 생성하지 않는다.
   전체 연구 모델·데이터·평가 protocol은 유지하고, 축소 데이터 실행 경로는 제거했다.
9. 공개 재현 안내는 `environment.yml` → `setup_gpu.sh` → 데이터 준비 → 트랙별 실험으로 정리했다.
   기본 CUDA wheel은 `cu126`으로 고정하고 전체 pytest는 `RUN_TESTS=1`일 때만 실행한다.

### 코드 스냅샷

- 파일: `code_summary.md`
- 포함 파일: 98개
- 크기: 907,044 bytes, 23,951 lines (`str.splitlines()` 기준)
- SHA-256: `9A71842E17A3DC29FFEBC6B9BCB95A4C32A9803F86396FFA3A5F6D8B4AAB5A00`
- 포함: 모든 Python source/test, TOML/YAML, Bash/PowerShell script, requirements, `.gitignore`, `.gitattributes`
- 제외: `.venv*`, data/cache, run artifact, `egg-info`, README류 설명 문서

이 디렉터리는 Git repository로 초기화되어 있으며 원격은
`https://github.com/costunder/new-gat.git`이다. 실행 환경에서는 먼저 `git rev-parse HEAD`를
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

- `README.md`: Linux NVIDIA GPU 환경의 설치부터 전체 재현까지의 실행 명령.
- `DATASETS.md`: 사람이 읽는 데이터·split·metric 계약.
- `pyproject.toml`: Python 3.11+, core/dev/paper dependency와 pytest/Ruff 설정.
- `requirements-lock.txt`, `constraints-cu*.txt`: Python 3.11 호환 exact top-level 연구 stack과
  CUDA 12.6/13.0/13.2별 official torch channel 계약.
- `requirements-paper.txt`: portable paper dependency가 같은 lock을 사용하게 하는 진입점.
- `scripts/setup_gpu.sh`, `scripts/verify_gpu_lock.py`: 활성 프로젝트 전용 Conda 환경에 exact
  package 설치, ABI/CUDA runtime 검증과 transitive freeze snapshot.
- `scripts/conda_env.sh`, `scripts/verify_conda_env.py`: Linux Bash 진입점이 공유하는
  Conda/Python 검사. 비활성 환경, base 환경과 중첩된 별도 Python 환경을 거부한다.
- `scripts/gpu_preflight.py`: CUDA device, 여유 메모리, dependency import 검사. 데이터 생성·학습 없음.
- `scripts/paper.sh`: 활성 Conda 환경의 Python으로 master runner 실행.
- `scripts/run_paper.py`: 세 독립 트랙을 model-seed별 subprocess로 dispatch하고 중앙 manifest 작성.
- `scripts/prepare_data.sh`: 전체 데이터 준비 명령을 담은 실행 파일.
- `scripts/reproduce.sh`, `research/<track>/reproduce.sh`: 전체 또는 트랙별 정식 실험 실행 파일.
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
  - `model.py`: 저수준 연산 및 수학 단위 검증용 유틸리티. legacy 실행기와 production synthetic generator는 제거했다.
- `research/cycle_pe/`
  - `features.py`: fundamental basis, set statistics, projector 수학.
  - `paper_data.py`: CycleCount-OOD generator와 exact cycle labels.
  - `paper_adapters.py`: BREC/ZINC adapter와 안전한 download/cache 처리.
  - `paper_model.py`: 네 PE variant와 공통 graph backbone.
  - `paper_train.py`: supervised train/eval/runtime.
  - `paper.py`: core/BREC/ZINC paper runner.
- `research/tree_augmentation/`
  - `augmentation.py`: full-β chart, transition, algebra certification과 legacy probe.
  - `paper_data.py`: core/CSL/ZINC graph data와 BFS/DFS/Wilson samplers.
  - `paper_model.py`: variable-edge/variable-β chart encoder와 training/evaluation.
  - `paper.py`: fixed-vs-multi independent paper runner.

### 격리된 결합 prototype

`research/combined_later/`에는 과거 flow completion, hard observation preservation, edge
residual 실험이 남아 있다. `pyproject.toml` package discovery와 active pytest 경로에서
제외되며 master paper runner도 실행하지 않는다. 외부 리뷰어는 이 코드를 active contribution과
섞어 평가하면 안 된다.

## 3. Linux NVIDIA GPU 재현 파이프라인

지원 실행 환경은 Linux와 NVIDIA GPU가 있는 워크스테이션 또는 서버다. 로컬 Linux
터미널에서 직접 실행하거나 임의의 SSH client로 해당 환경에 접속해 같은 명령을 실행한다.
MobaXterm은 선택 가능한 SSH client일 뿐 의존성이 아니며, tmux도 필수 도구가 아니다.
Cluster의 login node에 GPU가 없다면 해당 cluster의 scheduler로 GPU allocation을 먼저 받는다.

### 3.1 설치

일반 사용자용 실행 순서는 [README.md](README.md), 환경별 조정은
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)에 있다. 저장소 root에서 실행한다.

```bash
conda env create -f environment.yml
conda activate new-gat
bash scripts/setup_gpu.sh
```

Setup은 활성 non-base Conda의 Python만 사용한다. 기본 CUDA wheel은 `cu126`이며,
다른 runtime은 `CUDA_WHEEL_TAG`로 명시한 경우만 선택한다.
직접 의존성은 `requirements-lock.txt`와 해당 constraints 파일의 exact pin으로 설치한다.
전이 의존성 전체를 사전에 잠근 것은 아니며 실제 설치 결과는
`.gpu-environment.json`, `.gpu-environment.freeze.txt`에 기록한다.
Version/import ABI/CUDA 검증은 유지하고 전체 pytest는 `RUN_TESTS=1`일 때만 실행한다.

삭제된 entrypoint는 `setup.sh`, `setup.ps1`, `smoke.sh`, `smoke.ps1`,
`run_all.py`와 세 트랙의 legacy `run.py`다. 설치·실험 안내는 위 단일 경로를 사용한다.
삭제된 소스는 Git 이력으로 복원 가능하다. 사용자 데이터·결과·로컬 환경은 삭제하지 않았다.

### 3.2 데이터 준비와 cache 확인

```bash
bash scripts/prepare_data.sh
python scripts/check_datasets.py --data-root data/paper --require-cache
```

기본 데이터 경로는 `data/paper/`다. 준비 단계는 모델 학습이나 CPU 시험 학습을 하지 않는다.
공개 데이터에 대한 가짜 데이터 fallback은 없다.
`--allow-download`가 없으면 public endpoint를 호출하지 않는다.
Generated benchmark는 고정 data seed로 한 번 준비하며 model seed마다 다시 생성하지 않는다.

Checker는 request/schema, split cardinality, graph IDs, tensor/target shape, finite 값,
content/artifact hash를 읽기 전용으로 검증한다. 상태는
`valid/missing/incomplete/corrupt/wrong_request`로 구분한다.
Data와 split seed가 다르면 `--data-seeds`, `--split-seeds`를 각각 지정한다.
기존 축소·가짜 public cache는 전체 benchmark cache로 통과시키지 않는다.

### 3.3 독립 실행

```bash
bash research/conductance_gat/reproduce.sh
bash research/cycle_pe/reproduce.sh
bash research/tree_augmentation/reproduce.sh
```

위 세 명령을 순서대로 실행하는 대안은
`bash scripts/reproduce.sh`다. 두 방식을 중복 실행할 필요는 없다.

기본값은 CUDA, model seeds `0..4`, data/split/chart seed `0`, batch32/workers4다.
Run ID는 실행마다 자동 생성하며 같은 ID를 덮어쓰거나 자동 resume하지 않는다.
기본 data/와 트랙별 results/는 clone에 포함되고 하위 run 디렉터리는 자동 생성된다.
트랙 실패 시 기본적으로 다른 독립 run은
계속하며 `--fail-fast`로 전체 중단을 선택할 수 있다.
공통 GPU 검사 실패는 child 학습 전에 전체를 중단한다.

GPU 사전검사는 CUDA 사용 가능 여부, device index, 현재 여유 메모리와 package import만 확인한다.
가짜 그래프 생성, tensor 학습 입력 생성, 모델 forward/backward는 수행하지 않는다.
이 검사는 실제 데이터의 메모리 적합성이나 학습 성공을 보장하지 않는다.
데이터 준비에는 GPU 검사를 실행하지 않는다.
공식 BREC는 batch16/workers0/no-AMP, 내부 seed 10개를 사용하는 단일 child다.
CycleCount/ZINC만 외부 model seeds마다 반복한다.
Master의 cycle optimizer override는 공식 BREC에 적용하지 않는다.

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

Cache spec에는 generator version, seed, full protocol, graph IDs, content/file SHA-256가 들어가며
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

이 보조 public 경로도 우리 conductance model만 실행한다. 이전 no-message MLP, sparse GCN,
custom single-head edge-aware GAT, GINE 경쟁 모델 구현은 제거했다. Parameter count를 기록하고
MolHIV의 OGB AtomEncoder/BondEncoder는 원자·결합 입력을 읽는 구성요소로 유지한다.
외부 점수는 논문 표를 인용하며 모델·학습 조건 차이를 표시한다.

### 4.5 산출물과 테스트

Standalone:

```bash
python -m research.conductance_gat.paper \
  --suite all --data-root ./data \
  --output-dir ./results/conductance-seed0 \
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
S2 full cardinality contract, real public adapter, collision refusal와 가짜 데이터 옵션 거부를
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
삭제된 legacy bridge-vs-cycle probe는 `diag(P)` leverage로 target을 직접 드러냈으므로 headline
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
  --suite core --data-root ./data \
  --output-dir ./results/cycle-core-seed0 \
  --device cuda --data-seed 0 --split-seed 0 --chart-seed 0 --model-seed 0 \
  --variants raw,set,projector \
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
12. 외부 모델은 사용자가 정한 실행 범위에서 제외한다. 관련 논문의 표를 인용하고
    split/입력/예산/평가 조건이 일치하는지 기록한다. 인용 점수에 대해 우리 seed와의
    paired significance를 주장하지 않는다.

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
  축소 실행 profile은 제거했다. Wilson은 sampler-family OOD 평가를 위해 training에서 제외한다.
- `random_priority_kruskal`: Wilson과 다른 non-uniform legacy sampler이며 headline multi
  condition에는 들어가지 않는다.

Multi-chart는 매 update마다 새 tree를 online resample하는 것이 아니라 시작 시 생성한 finite
bank에서 minibatch sampling한다. Fixed와 multi는 architecture, model seed, optimizer와 optimizer
update 수(800)를 동일하게 맞춘다. Data/split/chart/model seed는 독립 축으로
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
  --suite all --data-root ./data \
  --output-dir ./results/tree-seed0 \
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
6. `paper_headline_eligible=True`는 전체 protocol run의 artifact flag일 뿐 성능이나
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

현재 로컬 public cache는 없으므로 `--require-cache`를 통과했다고 기록하면 안 된다. 실제
실험 환경에서 prepare/download 후 strict checker를 다시 실행해야 한다.

## 8. 자동 검증 상태

2026-08-30 더미 실행 경로 제거 및 실제 재현 실행 파일 추가 후 최종 로컬 검증:

```text
pytest -q                     204 passed, 12 skipped (11.61 s, exit 0)
ruff check .                 All checks passed
ruff format --check .        77 files already formatted
README command contract      5 script entrypoints resolve to full protocols
code_summary --check         current, 89 source files
git diff --check             passed
```

트랙별 단위시험은 Conductance 27개, Cycle PE 46개, Tree augmentation 28개다.
나머지 103개는 공통 수학, cache, runner, 집계, 환경·문서 계약 검사다.
Linux/Bash 전용 12개는 현재 Windows 환경에 Bash가 없어 skip했다.
이 중 5개는 새 실행 파일의 인자·종료 코드 전달을 검증하는 Linux shell 검사다.
이 결과는 실제 Linux Conda/Bash/CUDA 실행 성공을 인증하지 않는다.

이번 검증은 실험 CLI의 더미 옵션 거부, 공개 데이터 부재·다운로드 실패의 오류 처리,
가짜/축소 cache 거부, 전체 scientific protocol 크기·seed 유지와 README 인자 계약을 확인한다.
GPU 사전검사가 sample tensor나 모델을 생성하지 않는지도 mocked hardware 단위시험으로 검사한다.
테스트용 작은 graph/adapter 입력은 tests 내부에만 있으며 공개 experiment profile로 제공하지 않는다.

전체 pytest는 exit 0으로 끝났지만 실행 중 기존 Windows faulthandler
`access violation` 진단이 다시 출력됐다. 이를 숨기거나 Linux GPU 성공의 증거로 해석하지 않는다.
지원 Linux GPU 환경에서 실제 의존성 설치, cache 준비, 전체 학습·평가를 별도로 확인해야 한다.
과거 삭제된 더미 실행기의 숫자는 현재 검증·논문 결과로 이 문서에 재사용하지 않는다.

Read-only protocol 교차검토에서는 CycleCount full specification/hash가 이전 full protocol과
동일하고, BREC의 통계·reliability·학습·집계 함수 및 공식 설정도 유지됨을 확인했다.
전체 공개 데이터 cache나 GPU 학습 결과를 이 작업에서 생성하지 않았다.

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
closed root metric/efficiency 집계, exact CUDA constraints/verification,
cycle candidate CLI, stale S2 full-cache cardinality(112/24/48) 계약 교정을 반영했다.
과거 shape-stress는 더미 모델 실행 제거에 맞춰 hardware/import 검사로 교체했다. 실제
CUDA/public full 결과가 없다는 경계는 그대로다.

### P1 — 강한 scientific claim 전에

1. Conductance: 관련 논문 표를 인용한 비교의 조건 확인, real physical/sensor conductance data
   또는 명확한 synthetic-only claim. 외부 모델을 저장소에 추가하는 작업은 현재 범위 밖이다.
2. Cycle PE: degree/`(n,m,β)` matched counterfactual, 기존 cycle PE 논문과의 수학적 차이 설명,
   dense projector scaling 측정.
3. Tree augmentation: 우리 모델의 BFS-only/DFS-only/Wilson-only ablation, validation 기반 selection,
   large-β scaling.
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
