# NEW GAT

발생행렬 기반 그래프 학습을 세 개의 독립 연구 트랙으로 구현한 저장소다.
각 트랙은 자체 데이터 처리, 모델, 학습, 평가 코드를 가진다. 전체 실행도 세 연구를 결합하지 않는다.

| 트랙 | 연구 내용 | 데이터 |
|---|---|---|
| [Conductance GAT](research/conductance_gat/README.md) | 학습 가능한 edge conductance 연산자 | S1–S4, PascalVOC-SP, ogbg-molhiv |
| [Cycle PE](research/cycle_pe/README.md) | 정적 cycle-space positional encoding | CycleCount-OOD, BREC v3, ZINC12K |
| [Tree Augmentation](research/tree_augmentation/README.md) | spanning-tree chart 증강 | CycleCount, CSL, ZINC12K |

## 환경

기준 실행 환경은 **Linux, NVIDIA GPU, Conda Python 3.11, PyTorch 2.13.0 / CUDA 12.6**이다.
Linux GPU 워크스테이션에서 직접 실행하거나 SSH로 GPU 서버에 접속해서 실행한다.
접속 프로그램은 실행 환경과 무관하며, MobaXterm과 tmux는 필수 의존성이 아니다.
공용 클러스터에서는 해당 시스템의 GPU 작업 할당 정책을 따른다.

필요한 준비물은 Git, Bash, Conda, CUDA 12.6 이상을 지원하는 NVIDIA 드라이버다.
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

이후 명령은 모두 저장소 최상위 폴더에서, `new-gat` 환경을 활성화한 상태로 실행한다.
설치 스크립트는 연구 패키지 버전, CUDA runtime, import 호환성을 확인하고 실제 설치 내역을 저장한다.
서버 드라이버 버전에 따라 CUDA 패키지를 자동으로 바꾸지 않는다.

별도 CUDA runtime 선택, Conda 활성화 문제와 검사 명령은 [환경 안내](docs/ENVIRONMENT.md)에 있다.

## 데이터 준비

세 트랙에 필요한 데이터를 준비한다. 공개 데이터는 내려받고, 연구용 합성 벤치마크는
고정된 생성 규칙과 seed로 생성한다. **공개 데이터가 없을 때 가짜 데이터로 대체하지 않는다.**

```bash
bash scripts/paper.sh --suite all --prepare-only --allow-download --run-id prepare
```

기본 저장 경로는 `data/paper/`다. 준비만 수행하며 모델을 학습하지 않는다.
데이터의 원본, split, 생성 규칙, 지표는 [DATASETS.md](DATASETS.md)에 정리되어 있다.

## 실험

각 연구를 독립적으로 실행한다. 아래 세 명령은 서로 다른 결과 폴더를 사용한다.

### Conductance GAT

```bash
bash scripts/paper.sh --tracks conductance_gat --suite all --run-id conductance
```

### Cycle PE

```bash
bash scripts/paper.sh --tracks cycle_pe --suite all --run-id cycle-pe
```

### Tree Augmentation

```bash
bash scripts/paper.sh --tracks tree_augmentation --suite all --run-id tree-augmentation
```

공통 기본값은 CUDA, model seeds `0,1,2,3,4`, data/split/chart seed `0`,
batch size `32`, workers `4`다. 각 트랙의 비교군과 평가 규칙은 해당 트랙 문서에 있다.

BREC는 별도의 공식 10-seed 프로토콜을 한 번 실행한다.
외부 model seed 5개로 반복하지 않으며 batch size `16`, workers `0`, float32를 사용한다.
나머지 학습에는 기본적으로 AMP를 사용한다.

세 트랙을 순서대로 실행하려면 위 세 명령 **대신** 다음 명령을 사용한다.

```bash
bash scripts/paper.sh --suite all --run-id all-tracks
```

데이터 준비가 끝난 뒤 학습 명령에는 `--allow-download`를 붙이지 않는다.
누락되거나 손상된 데이터는 오류로 보고한다. 기존 run을 덮어쓰거나 자동 재개하지 않으므로
재실행할 때는 새 `--run-id`를 지정한다. 한 트랙이 실패해도 다른 독립 run은 계속하며,
첫 실패에서 중단하려면 `--fail-fast`를 추가한다.

## 결과

| 경로 | 내용 |
|---|---|
| `runs/paper/<run-id>/manifest.json` | 실행 상태, 명령, seed, 소스 revision |
| `runs/paper/<run-id>/logs/` | 트랙별 실행 로그 |
| `runs/paper/<run-id>/aggregate/` | 지표, paired 비교, 효율, 실패 목록 |
| `research/<track>/results/paper/<run-id>/` | 트랙별 평가·학습 산출물 |

전체 성공 시 터미널에 `all requested independent paper tracks passed`와 manifest 위치가 출력된다.
실패하면 종료 코드는 0이 아니며, 해당 run의 로그와 `aggregate/failures.csv`를 확인한다.
공통 환경 검사 단계에서 중단된 경우에는 집계 파일이 없을 수 있다.

데이터나 결과를 다른 디스크에 저장하려면 `--data-root /path/to/data`,
`--results-root /path/to/results`를 사용한다.
데이터 경로는 준비와 학습에서 같아야 한다.
결과 경로를 지정해도 실행 기록과 집계는 `runs/paper/<run-id>/`에 저장된다.

## 재현 범위

같은 소스 revision, 데이터 cache와 checksum, seed, 실행 옵션, 설치 패키지 기록을 함께 보존한다.
직접 의존성은 버전 고정 파일을 사용하고 전이 의존성은 설치 후 snapshot에 기록한다.
Python patch 버전과 모든 전이 의존성을 잠근 환경은 아니며,
서로 다른 GPU·드라이버에서 비트 단위 동일한 결과를 보장하지 않는다.
[PyTorch 재현성 안내](https://docs.pytorch.org/docs/stable/notes/randomness.html)도 참고한다.

이 저장소의 구현·단위 검증과 전체 공개 데이터 GPU 실험 완료는 별개다.
현재 전체 GPU 실험이 완료됐다고 주장하지 않으며, 검증 이력과 연구상 한계는
[hand_off.md](hand_off.md)에 기록한다.
전체 소스 스냅샷은 [code_summary.md](code_summary.md)에 있다.
