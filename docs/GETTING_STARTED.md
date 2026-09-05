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

Cycle PE V2는 구 raw-column `cycle_basis_v2`와 QR 기반 `cycle_projector_pe_v2`를
현재 경로에서 제거하고 **SE `cycle_dfs_se_v2`, PE `cycle_dfs_relative_pe_v2`**로
분리했다. PE는 동일 SE에 cycle 상대 위치 residual을 추가한 조건이다. 유일한 기본 backend는
`dfs_fundamental`이다. DFS forest와 non-tree edge parent-path로 좌영공간의 전체 cycle
기저와 cycle 순서를 만든 뒤 sparse 집계로 학습한다. QR/SVD/Gram inverse나 dense
projector를 만들지 않는다. 선택된 DFS tree에 의존하며 일반 기저변환 불변성은 주장하지 않는다.
[새 V2 계약과 실행법](../gpt_handoff/CYCLE_PE_V2.md)을 기준으로 하고 구 cache/checkpoint와
결과를 혼합하지 않는다. DFS 탐색과 출력 cycle 총 길이의 비용도 구분한다.

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

Conductance V5는 graph-conditioned **shared dynamic C**와 multi-head spatial W·head별
전파 강도 beta를 분리해 학습한다. fixed-C와 dynamic-C는 같은 architecture·초기화를 쓰지만
phase별 optimizer update 배분이 다른 strong-recipe 비교이며, sampling과 10GB MIG용 activation
checkpoint를 지원한다. beta 기본값은 hard margin 없는 sigmoid와 nominal 초기값 0.1이고,
과거 `0.05+0.90*sigmoid`는 명시적으로 선택하는 ablation으로만 남는다. 수식과 계약은
[V5 통합 문서](../gpt_handoff/CONDUCTANCE_V5.md)를 기준으로 한다.

이 소스 버전에는 독립 v2/v3/v4/v5, 새 Cycle V2, 실행 최적화, 단일 seed 기본값과
확장 진단이 포함되어 있다.
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
Conductance v2/v3/v4/v5는 기본 세 트랙 명령에 자동 포함되지 않는 별도 후속 실험이다.

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

### Conductance v5: graph-conditioned shared dynamic C

V5의 핵심 비교는 동일 architecture·seed·초기화에서 `fixed_c` strong spatial recipe와
`shared_dynamic_c` coordinate recipe를 학습하는 두 조건이다. phase별 update allocation이 달라
C 하나의 인과효과로 해석하지 않는다. `reference`는 hidden 256, 8 layers, 8 heads,
FFN multiplier 4이고, `large`는 384, 12 layers, 8 heads다. 아래는 FP32, PPI batch 2,
ogbn-arxiv seed-node batch 1024와 activation checkpoint를 쓰는 `portable` 실행이다.

```bash
python -B scripts/run_conductance_v5.py --datasets cora citeseer pubmed ppi ogbn-arxiv --profile reference --sampling auto --sample-seed-batch-size 1024 --model-seed 0 --device cuda:0 --hardware-profile portable --run-id conductance-v5-portable-reference-seed0
```

물리 GPU 3의 RTX A6000을 쓰는 직접 V5 실행은 다음과 같다. 프로세스 안에서는 이 장치가
`cuda:0`이다. 이 profile은 dense BF16/TF32, FP32 conductance geometry, PPI batch 8,
arxiv seed-node batch 2048, 더 큰 edge chunk와 prefetch를 쓴다. 전체 block checkpoint는 끄지만
dynamic-C score MLP는 gradient가 있을 때 edge chunk별로 checkpoint한다.

```bash
CUDA_VISIBLE_DEVICES=3 python -B scripts/run_conductance_v5.py --datasets cora citeseer pubmed ppi ogbn-arxiv --profile reference --model-seed 0 --device cuda:0 --sampling auto --hardware-profile a6000-48gb --min-free-gb 40 --run-id conductance-v5-a6000-gpu3-reference-seed0
```

CUDA 검사 경로는 PyTorch import 전에 오래 남은 `PYTORCH_NVML_BASED_CUDA_CHECK` 값을 내부에서
제거한다. 따라서 호출자가 `env -u ...`를 붙일 필요가 없고, 장치 매핑에는
`CUDA_VISIBLE_DEVICES`만 명시한다.

같은 implementation에서 같은 명령과 run ID를 다시 실행하면 완료 조건은 검증 후 건너뛰고, V5의 `last.pt`가 있는
미완료 조건은 epoch·optimizer·RNG 상태부터 재개한다. 자세한 식, sampling 의미와 결과 계약은
[CONDUCTANCE_V5.md](../gpt_handoff/CONDUCTANCE_V5.md)에 있다.

### Cycle PE

```bash
bash research/cycle_pe/reproduce.sh
```

새 QR-free V2의 SE/상대 PE만 dataset-aware reference/large 크기로 실행하는 GPU 3 명령이다.
두 dataset·두 profile·SE/PE 두 조건·seed 0에서 **8개 학습**을 계획하고,
각 encoding×dataset 안에서 validation으로 선택한 checkpoint만 **4회 test** 평가한다.
PE는 동일 SE에 상대 위치 residual을 추가하며 추가 trainable parameter는 없다.
이미 완료한 Conductance/Cycle V1/Tree는 실행하지 않는다. Backbone은 기존 크기를 유지하며
기본 ZINC 모델은 두 조건 모두 7,262,785 parameters다. 같은 새 source/config의 run ID 재실행은
`<child>/<dataset>/<encoding별 모델 ID>/last.pt`부터 이어지고 완료 child는 검증 후 건너뛴다.
모델 ID는 `cycle_dfs_se_v2`와 `cycle_dfs_relative_pe_v2`이며 두 checkpoint를 바꿔 끼우는 것은 거부한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=3 python -B scripts/run_cycle_scaling.py --versions v2 --profiles reference large --encodings se pe --datasets zinc12k peptides_struct --model-seeds 0 --device cuda:0 --hardware-profile a6000-48gb --min-free-gb 40 --basis-backend dfs_fundamental --run-id cycle-se-pe-a6000-gpu3-seed0-v1
```

### Tree Augmentation

```bash
bash research/tree_augmentation/reproduce.sh
```

이 직접 paper 실행의 `full` 기본 architecture는 scaling 계약의 `reference`, 즉 hidden 128과
message-passing 8층이다. 과거 hidden 64/2층 mechanism-probe 크기는 기본값에서 제거했다.
`large` hidden 256/12층은 아래 scaling runner가 별도로 학습하고 validation으로 선택하는 후보이며,
하드웨어만 보고 조용히 대체하지 않는다. Architecture와 실행 자원은 분리하므로 direct portable
batch 16은 유지하고, A6000 batch 64/workers 4는 명시적 hardware profile에서만 적용한다.

### V1을 포함한 전체 큰 모델 scaling

기본 단일 크기와 별도로 Conductance V1~V5, Cycle PE V1/새 V2, Tree fixed/multi를 모두
dataset-aware `reference`와 `large` profile에서 실행하려면 통합 Python runner를 사용한다.
파라미터 수를 억지로 같게 맞추는 실험이 아니라 각 방법 자체의 scaling curve이며, 시간 제약을
반영한 기본 전체 계획은 model seed 0 하나에서 **122 child runs / 126 model trainings**다.
Conductance 106학습, Cycle V1 4학습+V2 SE/PE 8학습, Tree 4 child의 8학습이다.
Cycle/Tree는 두 크기를
validation-only로 학습한 뒤 요청된 seed의 평균 validation으로 공통 크기를 선택하고, 선택
checkpoint만 test-only로 평가한다. Cycle은 encoding×dataset별로 선택하며 V1 포함 6회 test다. 먼저
파일을 만들지 않는 계획 검사를 권장한다. 다음은 물리 GPU 3을 프로세스의 `cuda:0`으로
매핑하는 RTX A6000 48GB용 전체 계획이다.

```bash
CUDA_VISIBLE_DEVICES=3 python -B scripts/run_rich_scaling.py --run-id rich-a6000-gpu3-seed0-v1 --profiles reference large --model-seeds 0 --device cuda:0 --hardware-profile a6000-48gb --min-free-gb 40 --cycle-v2-basis-backend dfs_fundamental --cycle-v2-encodings se pe --dry-run
```

위는 전체 matrix를 새로 요청한 경우의 참고 계획이며 이번 수정 때문에 완료한 실험을
다시 실행하라는 명령이 아니다. 실제 전체 실행이 필요한 경우에만 `--dry-run`을 제거한다.
한 트랙이 실패해도 나머지 트랙은 기본적으로
계속 실행하고 통합 상태를 failed로 남기며, 첫 실패에서 멈추려면 `--fail-fast`를 추가한다.
`--cycle-v2-basis-backend dfs_fundamental`이 유일한 기본값이다. `thin_q`와 이전
column/pair-budget/basis-execution 옵션은 제거되어 거부된다.

이미 구 `base/wide/deep/large` Conductance V1–V4 172회 표를 완료했고 이를 다시 학습하지
않으려면 같은 통합 runner에 `--conductance-versions v5`를 추가한다. 그러면 Cycle V1/V2와
Tree는 유지하면서 Conductance는 새 V5 20회만 실행해 전체 계획이 36 child runs / 40 fresh
model trainings가 된다. 새 모델만 최소 실행하려면 `--tracks conductance cycle
--conductance-versions v5 --cycle-versions v2`를 사용하며 기본 SE/PE 두 조건에서 28 child / 28학습이다. 이전 작은 profile 결과는
별도 artifact로 보존되며 새 reference/large V5와 동일 크기 비교로 자동 병합되지는 않는다.
중단 후에는 인수와 `--run-id`를 그대로 두고 같은 명령을 재실행한다. 완료 artifact를 다시
검증해 통과한 child는 건너뛰고 미완료 child만 다시 실행한다. 실행 중이던 child 하나는
V5와 새 Cycle V2라면 `last.pt`부터 이어지고, checkpoint 계약이 없는 legacy child만 처음부터
다시 시작한다. 이 보장은 같은 source/config/schema에 한정된다. 구 projector cache/checkpoint와
변경 전 V5 operator checkpoint는 현재 source로 강제로 resume하지 않는다. 새 Cycle cache는
`cycle_pe_v2_ordered_dfs_benchmark`에 별도로 준비한다. SE/PE는 이 새 데이터 캐시를 공유하지만
서로 다른 모델/실행 identity이며 checkpoint를 교환하지 않는다. 이미 전체가 완료된 run은 source·artifact·
집계 결과만 검증하고 GPU preflight나 학습 child를 다시 실행하지 않는다. 중단 후 재개할 때도
`--dry-run`을 제거한 실제 명령과 run ID를 한 글자도 바꾸지 않고 다시 실행한다.
프로필별 크기, 정확한 횟수, portable/10GB MIG와 GPU 3/A6000 실행 예시 및 결과 경로는
[전체 scaling 문서](../gpt_handoff/RICH_SCALING_EXPERIMENTS.md)에 있다.

ad041e2의 `v5-cycle-se-pe-a6000-gpu3-seed0-v1` 실행 후 수정은 V5 throughput 기록
형식과 Cycle 병렬 IPC의 메모리 소유권만 바꾼다. 모델 구조/크기/학습 설정은 그대로지만
source identity는 바뀌므로 해당 실패 run에 수정판을 강제로 resume하지 않는다.
기존 결과는 지워야만 하는 것이 아니며 새 run ID로 분리할 수 있다. 치우려면
`scripts/archive_failed_rich_run.py`의 기본 읽기 전용 계획을 확인한 뒤
`--apply`로 실패한 V5/CycleV2 연결 결과만 복구 가능한 폴더로 격리한다.
정확한 대상과 수정판 28-training 명령은 위 전체 scaling 문서의 첫 절을 따른다.

기본 세 트랙 benchmark와 `portable` scaling의 공통 기본값은 CUDA, model seed `0`,
data/split/chart seed `0`이다. 기본 benchmark의 workers는 `4`다. 별도 Conductance
V1/V3/V4/V5의 PPI graph-minibatch DataLoader도 기본 workers `4`, prefetch factor `2`,
persistent workers와 pinned/non-blocking transfer를 사용한다. V2 및 V1/V3/V4/V5의
transductive 데이터에는 DataLoader 자체가 없어서 workers `0`을 기록한다. V2와 V3/V4의 네
transductive 데이터는 full-graph batch 1이고, V3/V4의 PPI는 공식 graph 전체를 보존한
minibatch 2다. V5만 명시적으로 full/neighbor/cluster sampling을 지원하고 PPI는 full graph를
유지한다. 기본 benchmark의 PPI batch
size도 `2`, 분자·트리 데이터는 `32`이며 인용 그래프는 full-batch다.
각 트랙의 우리 모델 구성과 평가 규칙은 해당 트랙 문서에 있다.

`a6000-48gb`는 architecture 크기가 아니라 별도의 optimization/execution profile이다.
Conductance V5의 real sample/PPI batch와 BF16 실행, Cycle V1/V2의 dataset별 batch와 mixed
precision, Tree V1/V2의 batch와 동시 child 수가 portable과 달라진다. 따라서 같은 hardware profile
안의 arm/version만 주 비교로 해석하고, portable와 A6000 사이의 점수 또는 wall time 차이를
모델 효과나 GPU 하나의 효과로 직접 해석하지 않는다. 정확한 자원값과 비교 경계는 전체 scaling
문서의 표를 따른다.

통합 scaling에서 여러 model seed를 반복하려면 실행 명령 뒤에
`--model-seeds 0 1 2 3 4`처럼 공백으로 나열한다.
이 옵션은 각 트랙의 우리 benchmark에 적용된다. Cycle PE 보조 BREC official suite의 내부
`100,200,...,1000` 10-seed search는 BREC 프로토콜 축이므로 이 기본값과 별개다.
Conductance v2/v3/v4/v5는 한 번에 model seed 하나만 받아 `--model-seed 1`처럼 단수 옵션을 쓴다.

기본 `scripts/run_paper.py --suite benchmark`는 float32로 실행한다. Scaling의 `portable`은
Conductance와 Cycle에 FP32를 쓰고 Tree에는 기존 suite config의 FP16 AMP를 유지한다. A6000
profile은 Conductance V5 dense 연산에 BF16을 쓴다. Cycle V2 backbone은 BF16 지원 시 BF16,
미지원 시 FP32를 쓰고 loss scaling을 사용하지 않으며, Tree는 FP16 AMP를 명시적으로 고정한다.
Conductance geometry와 Cycle V2 sparse 집계는 float32를
유지한다. GAT는 accuracy/PPI micro-F1, PE는 MAE, 트리 증강은 CSL accuracy/ZINC MAE를
사용하며 각 트랙 안에서 비교한다.

Conductance의 반복 GPU 동기화 제거와 Cycle PE의 연결 정보 재사용은 기본 적용된다.
현재 Cycle V2는 sparse DFS 기저 membership과 실제 cycle 순서의 cos/sin을 한 physical batch로
처리한다. DFS forest 탐색은 `O(V+E)`, 명시적 basis·위치 복원·저장은 `O(V+E+nnz(Z))`, 각 sparse
집계는 `O(nnz(Z)*d)`이며 별도로 MLP/GNN 비용이 든다. SE는 sparse product 2회, PE는 6회이므로
PE가 SE보다 빠르다는 주장이 아니다. 준비/cache/forward에 QR/SVD는 없다.
기저 총 cycle 길이 nnz(Z)는 이차적으로 커질 수 있어 전체 과정이 언제나 선형이라고 주장하지 않는다.
V5 diffusion도 모든 edge-feature tensor를 backward까지 저장하지 않고 chunk별 재계산하도록
고쳤다. 실제 A6000 전체 학습·속도/VRAM 검증과 CPU 수치검사는 구분한다.
기본 benchmark의 선택적 `--compile`과 실제 train 데이터로 실행하는 GPU 속도 비교는
[PERFORMANCE.md](PERFORMANCE.md)를 따른다. 기본 명령이나 패키지 설치를 바꿀 필요는 없다.
별도 Conductance v2/v3/v4/v5는 이 compile 옵션을 받지 않는다.

기본 세 트랙을 순서대로 실행하려면 각 트랙의 기본 명령 **대신** 다음 명령을 사용한다.
Conductance v2/v3/v4/v5는 이 master 명령에 포함되지 않는다.

```bash
bash scripts/reproduce.sh
```

실행 파일에는 전체 데이터에서 우리 모델을 실행하는 설정이 들어 있다. 학습 파일은 다운로드를 수행하지 않는다.
누락되거나 손상된 데이터는 오류로 보고한다. `--run-id`를 생략하면 고유 ID를 자동 생성한다.
명시한 동일 ID를 다시 실행하면 기본적으로 same-run 재개를 시도하며, resolved 설정·Python/platform·
검증된 연구 의존성 report·전체 runtime source SHA-256·정확한 child 명령/output 계획이 모두
같을 때만 허용한다. source hash map은 각 child 전과 최종 집계 전에 다시 계산한다. 재개를 금지하려면
`--no-resume`을 쓴다. 검증된 완료 child는 output-tree SHA-256까지 다시 확인해 건너뛰고,
불완전한 산출물은 삭제하거나 덮어쓰지 않고 `.incomplete-attempt-N` sibling으로 보존한 뒤 재시도한다.
완료된 `aggregate/`도 입력 child hash와 자체 output hash를 검증해 건너뛰며, 실패했거나 현재 입력과
달라진 집계는 기존 폴더를 sibling으로 보존한 뒤 새로 만든다.
한 트랙이 실패해도 다른 독립 run은 계속하며, 첫 실패에서 중단하려면 `--fail-fast`를 추가한다.

## 결과

| 경로 | 내용 |
|---|---|
| `runs/paper/<run-id>/manifest.json` | 실행 상태, 명령, seed, runtime 환경, 소스·child·집계 SHA-256 |
| `runs/paper/<run-id>/logs/` | 트랙별 실행 로그 |
| `runs/paper/<run-id>/aggregate/` | 우리 모델 지표·효율·실패 목록, 내부 ablation의 paired 비교 |
| `research/<track>/results/paper/<run-id>/` | 트랙별 평가·학습 산출물 |
| `results/conductance_gat/v2/<run-id>/` | 직접 C v2의 manifest, 조건별 checkpoint와 비교표 |
| `results/conductance_gat/v3/<run-id>/` | 상대 C v3의 manifest, 조건별 checkpoint·개입과 비교표 |
| `results/conductance_gat/v4/<run-id>/` | [V4 통합 문서](../gpt_handoff/CONDUCTANCE_V4.md#결과-확인)의 2×2 checkpoint·진단·비교표 |
| `results/conductance_gat/v5/<run-id>/` | graph-conditioned V5의 fixed/dynamic C checkpoint·manifest·비교표 |
| `results/cycle_pe/scaling/<run-id>/` | V1/새 V2 SE·상대 PE의 encoding×dataset별 profile 선택·최종 평가 |
| `results/rich_scaling/<run-id>/` | V1–V5, Cycle V1/V2, Tree 통합 상태와 122-child/126-training 실행 검증 |

전체 성공은 child 종료 코드와 JSON 유한성만으로 판정하지 않는다. canonical child 상태,
현재 source와 일치하는 구현 SHA-256 provenance, 실제 학습의 주기적 GPU/CPU/RAM
`resource_observability`, 측정된 `*_per_second` throughput까지 모두 검증한 뒤 터미널에
`all requested independent paper tracks passed`와 manifest 위치를 출력한다.
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
현재 확대된 V2/V3/V4 8/10/20개 전체 실행 결과가 아니다. V5와 새
`cycle_dfs_se_v2`·`cycle_dfs_relative_pe_v2`는 구현·계약·CPU 회귀 검증 단계이며 두 조건의
실제 GPU 전체 학습·성능·VRAM·가속 결과는 아직 없다.
V1–V5 reference/large 126-training 계획의 성능 수치·전체 artifact와 실행 최적화의 가속 실측도
아직 확인하지 않았다.
구현 검증 이력과 연구상 한계는
[HANDOFF.md](../gpt_handoff/HANDOFF.md), 이 소스 버전의 코드 전체는
[CODE_SUMMARY.md](../gpt_handoff/CODE_SUMMARY.md)에 있다.
