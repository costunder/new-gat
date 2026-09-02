# NEW GAT

> 전체 문서 목록은 **[README.md](README.md)**에서 한 번에 찾을 수 있다.

발생행렬 기반 그래프 학습을 세 개의 독립 연구 트랙으로 구현한 저장소다.
각 트랙은 자체 데이터 처리, 모델, 학습, 평가 코드를 가진다. 전체 실행도 세 연구를 결합하지 않는다.

| 트랙 | 연구 내용 | 데이터 |
|---|---|---|
| [Conductance GAT](CONDUCTANCE_GAT.md) | 우리 incidence-conductance 모델 학습·평가 | Cora, CiteSeer, PubMed, PPI, ogbn-arxiv |
| [Cycle PE](CYCLE_PE.md) | 우리 cycle-set PE 모델 학습·평가 | ZINC-12K, Peptides-struct |
| [Tree Augmentation](TREE_AUGMENTATION.md) | 우리 모델의 고정 tree·다중 tree 내부 실험 | CSL, ZINC-12K |

GAT 트랙은 GAT/GATv2 논문의 데이터셋을, PE 트랙은 SignNet/PEARL 논문의 데이터셋을
사용한다. **저장소에서는 우리 모델만 학습·평가하며 외부 비교 모델을 재구현하거나 실행하지 않는다.**
다른 방법의 점수는 해당 논문 표를 출처와 함께 인용한다. 인용값은 직접 실행한 결과와 구분하고,
데이터 버전·split·지표·학습 조건이 다른 경우 차이를 명시한다. 데이터셋 이름이 같다는 것만으로
조건이 모두 일치하는 비교라고 주장하지 않는다.

Cycle PE의 좌영공간 **기저벡터 전체를 입력하는 v2**는
[별도 폴더와 실행 안내](../gpt_handoff/CYCLE_PE_V2.md)에 있다. 아래 기본 전체 실행은
기존 통계형 v1을 유지하며, v2는 자체 명령·기저 캐시·결과 폴더로 독립 실행한다.

Conductance의 **엣지별 C 자체를 직접 학습하는 v2**도
[별도 폴더](../gpt_handoff/CONDUCTANCE_V2.md)에 있다. 기존 MLP 생성 방식과 섞지 않으며,
기본 비교는 Cora/CiteSeer/PubMed/ogbn-arxiv에서 직접 C / 고정 C=1을 model seed 0으로
각각 새로 학습한다(총 8회). 그래프별 엣지 파라미터이므로 새로운 PPI 그래프에는 정의되지 않는다.

Conductance의 **상대 C 생성기 + 별도 전파 강도 학습 v3**는
[독립 폴더](../gpt_handoff/CONDUCTANCE_V3.md)에 추가했다. v2를 교체하지 않으며,
v3는 v1의 5개 데이터와 seed 0에서 자체 C=1 대조군과 새로 비교한다(총 10회).

Conductance의 **상대 C graph operator + spatial message transform v4**에 필요한 내용은
**[V4 통합 문서](../gpt_handoff/CONDUCTANCE_V4.md)** 한 곳에 모았다. 정확한 의도, 수식, 네 조건, 실행 명령, 결과 위치,
진단과 현재 검증 상태를 다른 문서에서 찾을 필요 없이 여기서 확인한다.

이 소스 버전에는 독립 v2/v3/v4, 실행 최적화, 단일 seed 기본값과 확장 진단이 포함되어 있다.
이전 진단 전용 버전(`ebf8cd1`)과의 차이 및 실제 측정 결과는
[실험 상태](../gpt_handoff/EXPERIMENT_STATUS.md)에 구분했다.

## 환경

실행 환경은 **Linux, NVIDIA GPU, Conda Python 3.11**이다.
Linux GPU 워크스테이션에서 직접 실행하거나 SSH로 GPU 서버에 접속해서 실행한다.
접속 프로그램은 실행 환경과 무관하며, MobaXterm과 tmux는 필수 의존성이 아니다.
공용 클러스터에서는 해당 시스템의 GPU 작업 할당 정책을 따른다.

필요한 준비물은 Git, Bash, Conda, NVIDIA 드라이버다. 기본 설치는 glibc 2.28 이상을
요구한다. **Ubuntu 18.04 / glibc 2.27 컨테이너는 아래 별도 설치 경로를 사용한다.**
설치기는 `nvidia-smi`의 CUDA 표시값에 따라 다음 **고정 버전 조합**을 선택한다.

| 드라이버의 CUDA 표시값 | 설치 조합 |
|---|---|
| 11.8 이상, 12.6 미만 (CUDA 12.2 서버 포함) | PyTorch 2.7.1 / CUDA 11.8 / PyG 2.7.0 |
| 12.6 이상 | PyTorch 2.13.0 / CUDA 12.6 / PyG 2.8.0.post1 |

CUDA 11.8 호환 조합은 Linux x86_64용이다. 시스템 CUDA Toolkit을 재설치하거나
서버 드라이버를 변경하지 않는다. 다른 조합의 결과는 같은 환경의 반복 실험으로 합치지 않는다.
Conda가 없다면 [Miniforge 설치 안내](https://github.com/conda-forge/miniforge#install)를 따른다.
설치와 데이터 준비에는 패키지·데이터 배포 서버에 대한 네트워크 접근이 필요하다.
네이티브 Windows/macOS 및 AMD GPU는 이 재현 환경의 지원 대상이 아니다.

### 설치

```bash
git clone https://github.com/costunder/new-gat.git
cd new-gat
conda env create -f environment.yml
conda activate new-gat
bash scripts/setup_gpu.sh
```

이후 명령은 모두 저장소 최상위 폴더에서, 설치한 Conda 환경을 활성화한 상태로 실행한다.
설치 스크립트는 연구 패키지 버전, CUDA runtime, import 호환성을 확인하고 실제 설치 내역을 저장한다.
선택한 조합을 설치 로그에 표시하며, 이미 설치된 조합은 데이터 준비·학습 때 자동 변경하지 않는다.
`conda env create`는 Python과 pip를 준비하며, **연구 의존성 전체 설치는 `setup_gpu.sh`가 수행한다.**
설치 중 오류가 나면 그 상태를 설치 완료로 보지 않는다.
이미 `new-gat` 환경이 있다면 `conda env create`를 반복하지 않고,
`conda activate new-gat` 후 `bash scripts/setup_gpu.sh`를 실행한다.

별도 CUDA runtime 선택, Conda 활성화 문제와 검사 명령은 [환경 안내](ENVIRONMENT.md)에 있다.

### Ubuntu 18.04 / glibc 2.27 컨테이너에서 설치

이 환경에서는 위 기본 설치 **대신** 명시적인 `legacy-cu118` 조합을 사용한다.
GPU가 할당된 현재 컨테이너의 저장소 폴더에서 실행한다. 기존 `new-gat` 환경을
교체하지 않도록 새 환경을 만든다. 이미 저장소를 받았다면 다시 clone할 필요 없다.

```bash
conda env create -n new-gat-legacy -f environment.yml
conda activate new-gat-legacy
bash scripts/setup_gpu.sh --profile legacy-cu118
```

설치 조합은 **PyTorch 2.6.0+cu118 / PyG 2.7.0 / Python 3.11**이다.
Linux x86_64, glibc 2.27 이상, 드라이버 CUDA 표시값 11.8 이상이 필요하다.
이미 이 전용 환경을 만들었다면 첫 줄을 생략하고 활성화·설치만 반복한다.
설치기가 다른 Torch가 들어 있는 환경은 변경하지 않고 중단한다.
설치 성공 후 아래 데이터 준비·실험 명령은 그대로 사용한다.

이 경로는 호환용 구버전이며 최신 보안 패치 환경이 아니다. PyTorch 2.6과 2.7에는
[공개된 체크포인트 로딩 취약점](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)이
있으므로 출처 불명의 `.pt`·`.pth`·pickle 파일을 읽지 않는다. 자세한 제약은
[환경 안내](ENVIRONMENT.md)에 있다. 실제 서버 설치·GPU 실행은 설치 후 별도로 확인해야 한다.

## 데이터 준비

위 표의 트랙별 공개 데이터셋을 내려받아 준비한다. **기본 실행은 자체 생성 데이터나
공개 데이터 대체용 가짜 데이터를 만들지 않는다.**

```bash
bash scripts/prepare_data.sh
```

이 명령은 활성 Conda 환경의 필수 패키지 버전과 import를 먼저 확인한다. NumPy 등 의존성이
누락되었으면 전체 설치 스크립트를 한 번 실행하고, 설치·검사가 성공한 경우에만 데이터를 준비한다.
자동 설치에도 위의 Linux/NVIDIA GPU 환경과 네트워크가 필요하다. 설치가 실패하면 데이터 준비도 중단된다.
glibc 등 호스트 호환성 오류는 자동 재설치하지 않고 중단한다.
학습 명령은 패키지를 자동 변경하지 않으며, 설치가 불완전하면 설치 명령을 안내하고 중단한다.

기본 저장 경로는 `data/paper/`다. 준비만 수행하며 모델을 학습하지 않는다.
`data/`와 트랙별 `results/` 디렉터리는 저장소에 포함되어 있고, 하위 디렉터리는 실행기가
자동 생성한다. 별도의 `export`나 `mkdir` 명령은 필요 없다.
데이터의 원본, split, 생성 규칙, 지표는 [DATASETS.md](DATASETS.md)에 정리되어 있다.

## 실험

기본 세 트랙 benchmark는 각각 독립적으로 실행하며 서로 다른 결과 폴더를 사용한다.
중간의 Conductance v2/v3/v4는 기본 세 트랙 명령에 자동 포함되지 않는 별도 후속 실험이다.

### Conductance GAT

```bash
bash research/conductance_gat/reproduce.sh
```

Gate weight decay와 정규화의 영향을 분리하는 **PPI/arxiv × 4조건 × seed 0** 후속 실험은
[독립 비교 실험 안내](CONDUCTANCE_FACTORIAL.md)를 따른다. 기존 benchmark와
결과를 섞지 않으며 조건별 새 학습 및 비교표 생성을 한 명령으로 실행한다.

그 결과를 바탕으로 진행하는 **학습 C vs 고정 C=1 비교**와 기존 checkpoint의 평균-C 검사는
[C-learning 실행 안내](CONDUCTANCE_C_LEARNING.md)에 있다. 새 학습은
PPI/arxiv × 2조건 × seed 0으로 총 4개이며, 읽기 전용 검사는 재학습과 분리한다.
`gat-c-learning-seed0-v1`의 비교 학습과 learned checkpoint 검사는 완료 보고서를 수령했다.
결과는 [C-learning 기록](CONDUCTANCE_C_LEARNING_FINDINGS.md)에 보존한다.

### Conductance v2: 엣지별 C 직접 학습

기존 공식 데이터 캐시와 Conda 환경을 사용한다. 아래 명령은
**Cora/CiteSeer/PubMed/ogbn-arxiv × 두 조건 × seed 0 = 8회**를 별도로 학습한다.
PPI는 미관측 graph의 edge별 C가 정의되지 않아 V2에서 N/A다. 기존 모델·결과를 바꾸거나
checkpoint를 재사용하지 않는다.

```bash
bash research/conductance_gat/v2/reproduce.sh --run-id gat-direct-c-v2-seed0-v1
cat results/conductance_gat/v2/gat-direct-c-v2-seed0-v1/comparison.md
```

C는 엣지별 log 파라미터에서 양수로 변환하며, C=1 초기 상태에서 시작한다.
큰 엣지 중간값을 모두 저장하지 않도록 정확한 chunked forward/backward를 사용한다.
아직 full-graph 학습이며 이웃 샘플링을 구현한 것은 아니다. 데이터 범위·추가 인자·
계산량과 메모리 해석은 [v2 실행 안내](../gpt_handoff/CONDUCTANCE_V2.md)를 따른다.
같은 run ID는 덮어쓰지 않으므로 재실행할 때는 새 ID를 사용한다.

### Conductance v3: 상대 C와 전파 강도 분리 학습

v1과 같은 공식 Cora/CiteSeer/PubMed/PPI/ogbn-arxiv 캐시에서
**상대 C 생성기 / 고정 C=1 × seed 0 = 10회**를 새로 학습한다. PPI는 공식 20/2/2
inductive graph split, batch size 2, BCEWithLogits와 global micro-F1을 사용한다.
v2와는 모델·optimizer·정규화가 다르며 별도 결과 폴더에 저장한다.

```bash
bash research/conductance_gat/v3/reproduce.sh --run-id gat-relative-c-v3-seed0-v1
cat results/conductance_gat/v3/gat-relative-c-v3-seed0-v1/comparison.md
```

공유 MLP의 score를 그래프 전체에서 중심화하고 양의 상대 C로 변환한다. 등방성 성분과의
혼합 비율 및 전파 강도를 별도 학습하며, 대칭 정규화를 사용한다. 선택된 checkpoint의
평균 C·섞은 C·C=1·전파 제거 검사도 validation만 사용해 기록한다.
대칭 정규화에서 양의 graph-constant C는 소거되므로 평균 C와 C=1은 독립 ablation이 아니라
서로 일치해야 하는 수치 검산이다.
두 버전의 점수 차이 하나를 C 학습의 단일 요인 효과로 해석하지 않는다. 수식·진단·해석 기준은
[v3 실행 안내](../gpt_handoff/CONDUCTANCE_V3.md)에 있다.

### Conductance v4: 상대 C와 spatial W 동시 학습

V4의 전체 설명과 최신 상태는 **[V4 통합 문서](../gpt_handoff/CONDUCTANCE_V4.md)**를 기준으로 한다. 바로 실행하려면
다음 명령을 사용한다.

```bash
bash research/conductance_gat/v4/reproduce.sh --run-id gat-hybrid-c-spatial-v4-seed0-v1
cat results/conductance_gat/v4/gat-hybrid-c-spatial-v4-seed0-v1/comparison.md
```

수식, 네 조건, 다섯 대조, checkpoint 개입과 해석 제한도 모두 [CONDUCTANCE_V4.md](../gpt_handoff/CONDUCTANCE_V4.md)에 있다.
기본 실행은 v1의 5개 데이터 × 네 조건 × seed 0 = 20회다. PPI는 v3와 같은 공식
inductive 계약을 사용하며, 나머지 네 데이터는 full-graph transductive 학습이다.

### Cycle PE

```bash
bash research/cycle_pe/reproduce.sh
```

### Tree Augmentation

```bash
bash research/tree_augmentation/reproduce.sh
```

기본 세 트랙 benchmark의 공통 기본값은 CUDA, model seed `0`, data/split/chart seed `0`,
workers `4`다. 별도 Conductance v2/v3/v4는 workers `0`을 사용한다. V2와 V3/V4의 네
transductive 데이터는 full-graph batch 1이고, V3/V4의 PPI는 공식 graph 전체를 보존한
minibatch 2다. 어느 경로도 neighbor sampling을 사용하지 않는다. 기본 benchmark의 PPI batch
size도 `2`, 분자·트리 데이터는 `32`이며 인용 그래프는 full-batch다.
각 트랙의 우리 모델 구성과 평가 규칙은 해당 트랙 문서에 있다.

여러 model seed를 반복하려면 실행 명령 뒤에 `--model-seeds 0,1,2,3,4`처럼 명시한다.
이 옵션은 각 트랙의 우리 benchmark에 적용된다. Cycle PE 보조 BREC official suite의 내부
`100,200,...,1000` 10-seed search는 BREC 프로토콜 축이므로 이 기본값과 별개다.
Conductance v2/v3/v4는 한 번에 model seed 하나만 받아 `--model-seed 1`처럼 단수 옵션을 쓴다.

기본 학습은 float32로 실행한다. GAT는 accuracy/PPI micro-F1, PE는 MAE,
트리 증강은 CSL accuracy/ZINC MAE를 사용하며 각 트랙 안에서 비교한다.

Conductance의 반복 GPU 동기화 제거와 Cycle PE의 연결 정보 재사용은 기본 적용된다.
기저벡터 v2는 여러 그래프의 기저 연산을 묶는 배치 인코더를 기본 사용한다.
기본 benchmark의 선택적 `--compile`과 실제 train 데이터로 실행하는 GPU 속도 비교는
[PERFORMANCE.md](PERFORMANCE.md)를 따른다. 기본 명령이나 패키지 설치를 바꿀 필요는 없다.
별도 Conductance v2/v3/v4는 이 compile 옵션을 받지 않는다.

기본 세 트랙을 순서대로 실행하려면 각 트랙의 기본 명령 **대신** 다음 명령을 사용한다.
Conductance v2/v3/v4는 이 master 명령에 포함되지 않는다.

```bash
bash scripts/reproduce.sh
```

실행 파일에는 전체 데이터에서 우리 모델을 실행하는 설정이 들어 있다. 학습 파일은 다운로드를 수행하지 않는다.
누락되거나 손상된 데이터는 오류로 보고한다. 실행마다 고유한 run ID를 자동 생성하고
기존 run을 덮어쓰거나 자동 재개하지 않는다. 한 트랙이 실패해도 다른 독립 run은 계속하며,
첫 실패에서 중단하려면 `--fail-fast`를 추가한다.

## 결과

| 경로 | 내용 |
|---|---|
| `runs/paper/<run-id>/manifest.json` | 실행 상태, 명령, seed, 소스 revision |
| `runs/paper/<run-id>/logs/` | 트랙별 실행 로그 |
| `runs/paper/<run-id>/aggregate/` | 우리 모델 지표·효율·실패 목록, 내부 ablation의 paired 비교 |
| `research/<track>/results/paper/<run-id>/` | 트랙별 평가·학습 산출물 |
| `results/conductance_gat/v2/<run-id>/` | 직접 C v2의 manifest, 조건별 checkpoint와 비교표 |
| `results/conductance_gat/v3/<run-id>/` | 상대 C v3의 manifest, 조건별 checkpoint·개입과 비교표 |
| `results/conductance_gat/v4/<run-id>/` | [V4 통합 문서](../gpt_handoff/CONDUCTANCE_V4.md#결과-확인)의 2×2 checkpoint·진단·비교표 |

전체 성공 시 터미널에 `all requested independent paper tracks passed`와 manifest 위치가 출력된다.
실패하면 종료 코드는 0이 아니며, 해당 run의 로그와 `aggregate/failures.csv`를 확인한다.
공통 환경 검사 단계에서 중단된 경우에는 집계 파일이 없을 수 있다.

Conductance GAT의 낮은 성능을 확인하려면 [기존 checkpoint 진단](CONDUCTANCE_DIAGNOSTICS.md)을
따른다. 재학습 없이 train/validation 성능과 이웃 혼합량을 확인하며, 원래 결과를 덮어쓰지 않는다.

이 실행 파일에 `--run-id experiment-name`을 추가하면 결과 이름을 직접 지정할 수 있다.
데이터나 결과를 다른 디스크에 저장하려면 `--data-root /path/to/data`,
`--results-root /path/to/results`를 사용한다.
데이터 경로는 준비와 학습에서 같아야 한다.
결과 경로를 지정해도 실행 기록과 집계는 `runs/paper/<run-id>/`에 저장된다.

S1–S4, CycleCount, BREC 및 이전 PascalVOC-SP/molhiv 실험은 기본 benchmark에 포함되지
않는다. 기존 `--suite core`/`--suite all`은 해당 보조 실험용으로만 유지한다.
기본 실행과 보조 실험의 상세 구분은 [DATASETS.md](DATASETS.md)를 참고한다.

## 재현 범위

같은 소스 revision, 데이터 cache와 checksum, seed, 실행 옵션, 설치 패키지 기록을 함께 보존한다.
직접 의존성은 버전 고정 파일을 사용하고 전이 의존성은 설치 후 snapshot에 기록한다.
Python patch 버전과 모든 전이 의존성을 잠근 환경은 아니며,
서로 다른 GPU·드라이버에서 비트 단위 동일한 결과를 보장하지 않는다.
[PyTorch 재현성 안내](https://docs.pytorch.org/docs/stable/notes/randomness.html)도 참고한다.

사용자 제공 기존 benchmark 5-seed 집계, Conductance seed 0 GPU 진단·2×2·C-learning 결과는
[실험 상태](../gpt_handoff/EXPERIMENT_STATUS.md)에 기록했다. 같은 문서에는 2026-09-02의 과거
arxiv-only Conductance v2/v3 runner `passed` 보고와 V4 partial 중단도 별도로 보존한다. 이들은
현재 확대된 V2/V3/V4 8/10/20개 전체 실행 결과가 아니다. 확대 실행의 성능 수치·전체 artifact,
Cycle PE 기저벡터 v2의 성능 수치·전체 artifact와 실행 최적화의 가속 실측은 아직 확인하지 않았다.
구현 검증 이력과 연구상 한계는
[HANDOFF.md](../gpt_handoff/HANDOFF.md), 이 소스 버전의 코드 전체는
[CODE_SUMMARY.md](../gpt_handoff/CODE_SUMMARY.md)에 있다.
