# NEW GAT — 처음부터 서버에서 실행하기

이 문서는 **MobaXterm으로 Linux NVIDIA GPU 서버에 접속한 뒤, 저장소를 받아 실험을
실행하고 결과를 확인하는 순서**를 안내한다. 아래 명령은 Windows 로컬 터미널이 아니라
접속한 **Linux 서버의 Bash 터미널**에서 실행한다.

저장소: <https://github.com/costunder/new-gat>

세 연구는 서로 독립이다. 전체 실행 명령도 모델을 합치지 않고 각 트랙을 따로 실행한다.

| 트랙 | 실행하는 연구 | 코드 폴더 |
|---|---|---|
| `conductance_gat` | 학습 가능한 edge conductance 연산자 | `research/conductance_gat/` |
| `cycle_pe` | 정적 cycle-space positional encoding | `research/cycle_pe/` |
| `tree_augmentation` | spanning-tree chart 증강 | `research/tree_augmentation/` |

**처음 실행한다면 1 → 7번을 순서대로 진행한다.** 수학적 검토와 미완료 연구 항목은
[hand_off.md](hand_off.md), 데이터·평가 규칙은 [DATASETS.md](DATASETS.md)에 별도로 정리했다.

## 1. GPU 서버에 접속하고 준비물 확인

MobaXterm에서 `Session → SSH`를 선택하고 서버 주소·계정·SSH port로 접속한다.
공용 cluster라면 관리자가 안내한 방법으로 GPU node를 먼저 할당받는다. 로그인 전용
node에서 설치·학습을 시작하지 않는다. `srun`, `qsub` 등의 정확한 할당 명령은 서버마다 다르다.

접속한 서버에서 실행한다.

```bash
hostname
nvidia-smi
git --version
conda --version
tmux -V
```

다음 조건을 확인한다.

- `nvidia-smi`에 사용할 NVIDIA GPU가 보인다.
- `nvidia-smi` 상단의 `CUDA Version`이 **12.6 이상**이다. 이 값은 driver의 CUDA 호환 범위다.
- Conda 명령을 사용할 수 있다. 실험용 Python 3.11 환경은 3번에서 새로 만든다.
- Git과 tmux가 설치되어 있다.
- GitHub, PyPI, PyTorch wheel 저장소와 데이터셋 다운로드 주소에 접속할 수 있다.
- 코드·Conda 환경·데이터·결과를 쓸 수 있는 디스크 공간과 quota가 있다.

`conda: command not found`라면 먼저 서버 안내에 따라 Conda module을 불러오거나 관리자에게
Conda 사용 경로를 확인한다. Module 이름은 서버마다 다르므로 임의 이름으로 실행하지 않는다.
이 저장소의 설치 스크립트는 Conda나 NVIDIA driver를 설치하지 않는다. Conda 자체가 준비됐다고
해서 `new-gat` 환경까지 이미 있다는 뜻은 아니므로 아래 환경 생성 단계를 생략하지 않는다.

## 2. 공개 저장소 받기

공개 상태에서는 GitHub 로그인이나 token 없이 다음 명령으로 받는다.

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"
git clone https://github.com/costunder/new-gat.git
cd new-gat
```

이미 받아 둔 저장소라면 다시 clone하지 않고 그 폴더에서 업데이트한다.

```bash
git pull --ff-only
```

이후 모든 명령은 `README.md`, `scripts/`, `research/`가 보이는 **저장소 최상위 폴더**에서
실행한다. `pwd`의 마지막 부분이 `new-gat`인지 확인한다.

긴 설치·학습 중 SSH 연결이 끊겨도 작업을 유지하려면 여기서 tmux를 시작한다.

```bash
tmux new -s new-gat
```

- tmux에서 빠져나오기: `Ctrl+b`를 누른 뒤 손을 떼고 `d`.
- 다시 들어가기: `tmux attach -t new-gat`.
- tmux는 cluster의 GPU 할당 만료나 작업 시간 제한을 연장하지 않는다.

데이터와 결과를 저장할 위치를 **tmux 안에서** 지정한다. 아래 기본값은 저장소 안의
`data/`, `results/`이며 Git 업로드 대상에서 제외되어 있다.

```bash
export NEW_GAT_DATA_ROOT="$PWD/data"
export NEW_GAT_RESULTS_ROOT="$PWD/results"
mkdir -p "$NEW_GAT_DATA_ROOT" "$NEW_GAT_RESULTS_ROOT"
```

서버가 scratch 사용을 요구한다면 위 두 경로를 서버에서 허용한 경로로 바꾼다. 예를 들어
`/scratch/$USER/new-gat-data`, `/scratch/$USER/new-gat-results`를 사용할 수 있지만,
해당 경로가 실제 서버에 존재하고 쓰기 가능한지 먼저 확인한다.
새 SSH/tmux 세션에서는 두 `export` 명령을 다시 실행하고 **기존과 같은 경로**를 지정한다.

## 3. GPU 실행 환경 설치

프로젝트 전용 **Conda 환경을 처음 한 번 생성**한다. 아래 명령도 저장소 폴더의 tmux 안에서
실행한다. `source`는 현재 Bash에서 `conda activate`를 사용할 수 있게 한다.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n new-gat python=3.11 pip -y
conda activate new-gat
python --version
```

이미 이 프로젝트용 `new-gat` 환경을 만들어 두었다면 `conda create`만 건너뛰고 활성화한다.
이름이 같더라도 다른 프로젝트가 사용 중인 환경이라면 덮어쓰지 말고 별도 이름으로 만든다.
`base`나 여러 사람이 함께 쓰는 환경에는 설치하지 않는다.

활성화한 전용 환경에 GPU 의존성을 설치한다.

```bash
bash scripts/setup_gpu.sh
```

이 명령은 **현재 활성화된 Conda 환경**에 고정 버전 의존성을 설치하고, CUDA 연산 확인과
단위 테스트를 실행한다. 환경을 별도로 만들거나 시스템 Python으로 전환하지 않는다.
Conda는 전용 환경과 Python 3.11을 관리하고, GPU PyTorch는 setup이 고른 공식 CUDA wheel과
exact constraints로 설치한다. NVIDIA driver는 서버에 미리 준비되어 있어야 한다.
별도의 CUDA/cuDNN 패키지를 임의로 섞어 설치하지 않는다.
처음에는 패키지 다운로드 때문에 시간이 걸린다.
마지막에 다음 문장이 출력되어야 다음 단계로 간다.

```text
GPU environment ready. Run: bash scripts/paper.sh --help
```

실제 Python과 GPU를 한 번 더 확인한다.

```bash
python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))'
bash scripts/paper.sh --help
```

이후 `bash scripts/paper.sh`는 활성 Conda 환경의 `$CONDA_PREFIX/bin/python`을 사용한다.
직접 호출하는 `python`도 같은 환경이어야 한다. **새 SSH 창이나 새 tmux 창을 열면 실행 전에
다시 활성화**한다. 단순히 기존 tmux 창에 재접속했다면 그 안의 환경은 유지된다.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate new-gat
```

새 shell에서는 저장소 폴더로 이동하고 2번의 데이터·결과 경로 `export`도 다시 실행한다.

설치 기록은 `.gpu-environment.json`, `.gpu-environment.freeze.txt`에 남는다.
현재 lock은 `torch==2.13.0`이며 driver에 따라 `cu126`, `cu130`, `cu132` 중 하나를 선택한다.
CUDA 12.6 미만 driver에서는 설치를 중단하며 CPU로 자동 전환하지 않는다.

Wheel channel을 고정해야 할 때만 다음처럼 지정한다. 활성 Conda 환경은 그대로 사용한다.

```bash
CUDA_WHEEL_TAG=cu126 bash scripts/setup_gpu.sh
```

이미 설치된 의존성을 변경하지 않고 검증하려면 `SKIP_DEPS=1 bash scripts/setup_gpu.sh`를
사용할 수 있다. 단, 프로젝트의 editable 설치는 갱신하며 버전·ABI·CUDA가 lock과 다르면
실패한다. 처음 만든 빈 환경에서는 이 옵션을 사용하지 않는다.

## 4. 먼저 아주 작은 GPU 테스트 실행

공개 데이터셋을 내려받기 전에 세 트랙의 학습·평가·결과 저장 경로가 작동하는지 확인한다.
이 단계는 작은 자체 생성 데이터만 사용한다.

```bash
bash scripts/paper.sh \
  --suite core --tiny --device cuda \
  --data-root "$NEW_GAT_DATA_ROOT" \
  --results-root "$NEW_GAT_RESULTS_ROOT" \
  --model-seeds 0 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --batch-size 16 --workers 0 --fail-fast \
  --run-id first-gpu-smoke
```

종료 후 성공 여부를 확인한다. 출력이 `passed`여야 한다.

```bash
python -c 'import json; print(json.load(open("runs/paper/first-gpu-smoke/manifest.json"))["status"])'
```

`--tiny`는 실행 경로 점검용이다. 여기서 얻은 숫자를 정식 실험 결과로 쓰지 않는다.
같은 명령을 재실행할 때는 `--run-id first-gpu-smoke-v2`처럼 새 이름을 사용한다.

## 5. 정식 데이터셋 준비

데이터셋은 Git 저장소에 들어 있지 않다. 다음 명령이 자체 생성 데이터와 공개 benchmark를
준비한다. **이 단계에서는 학습하지 않는다.**

```bash
bash scripts/paper.sh \
  --suite all --prepare-only --allow-download \
  --data-root "$NEW_GAT_DATA_ROOT" \
  --results-root "$NEW_GAT_RESULTS_ROOT" \
  --model-seeds 0 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --run-id prepare-all-v1
```

| 트랙 | `--suite core` | `--suite all`에서 추가되는 데이터 |
|---|---|---|
| Conductance | S1–S4 자체 생성 데이터 | PascalVOC-SP, ogbg-molhiv |
| Cycle PE | CycleCount-OOD | BREC v3, ZINC-12K |
| Tree augmentation | CycleCount multi-chart | CSL, ZINC-12K |

준비가 끝나면 파일 존재뿐 아니라 split·shape·checksum까지 검사한다.

```bash
python scripts/check_datasets.py \
  --profile paper \
  --data-root "$NEW_GAT_DATA_ROOT" \
  --data-seeds 0 --split-seeds 0 \
  --require-cache
```

모든 필수 cache가 `valid`여야 한다. `missing`, `incomplete`, `corrupt`, `wrong_request`가
있으면 학습을 시작하지 말고 출력에 나온 데이터·경로·seed를 확인한다.
다운로드가 중단됐다면 같은 data root와 seed로 준비 명령을 다시 실행하되,
`--run-id prepare-all-v2`처럼 새 run ID를 준다. 손상 cache를 무조건 덮어쓰는 기능은 없다.

`--allow-download`는 공개 데이터 다운로드를 허용하는 옵션이다. 이후 학습 명령에서는
이 옵션을 빼고 검증된 cache를 사용한다. 준비 단계가 CPU 데이터 생성 작업을 수행하는
것은 정상이며, 정식 모델 학습은 CUDA에서 실행한다.

## 6. 정식 크기로 1 seed 점검

전체 5 seed를 시작하기 전에 세 트랙의 `core`를 정식 데이터 크기로 한 번 실행한다.

```bash
bash scripts/paper.sh \
  --suite core --device cuda \
  --data-root "$NEW_GAT_DATA_ROOT" \
  --results-root "$NEW_GAT_RESULTS_ROOT" \
  --model-seeds 0 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --batch-size 16 --workers 2 --min-free-gb 8 --fail-fast \
  --run-id core-one-seed-v1
```

```bash
python -c 'import json; print(json.load(open("runs/paper/core-one-seed-v1/manifest.json"))["status"])'
```

`passed`와 결과 파일을 확인한 뒤 다음 단계로 간다. 이 실행은 공개 benchmark와 BREC를
아직 학습하지 않는다. `--min-free-gb 8`은 시작 시 여유 GPU 메모리 검사값이지,
전체 학습이 8 GB 안에 들어간다는 보장은 아니다.

## 7. 정식 실험 실행

### 세 트랙을 모두 5 seeds로 실행

**주의:** `suite=all`에는 장시간의 official BREC가 포함된다. 기본 4 PE variants ×
내부 10 search seeds × 400 graph pairs를 학습하며 pair별 중단 복구/resume는 없다.
작은 GPU 테스트와 1 seed 점검을 통과하고 서버의 작업 시간 제한을 확인한 뒤 실행한다.

```bash
bash scripts/paper.sh \
  --suite all --device cuda \
  --data-root "$NEW_GAT_DATA_ROOT" \
  --results-root "$NEW_GAT_RESULTS_ROOT" \
  --model-seeds 0,1,2,3,4 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --cycle-variants no_pe,raw,set,projector \
  --batch-size 32 --workers 4 --min-free-gb 8 \
  --run-id paper-all-v1
```

각 트랙과 model seed는 별도 프로세스·결과 폴더로 순서대로 실행된다. BREC만 자체
official 10-seed protocol을 정확히 한 번 실행하며 바깥 5 seeds와 중복해서 곱하지 않는다.
한 트랙이 실패하면 기본적으로 나머지는 계속 실행하고 마지막에 실패를 보고한다.
첫 실패에서 즉시 멈추려면 `--fail-fast`를 추가한다.

### 한 트랙만 실행

아래 명령에서 `conductance_gat`을 `cycle_pe` 또는 `tree_augmentation`으로 바꾸면 된다.
run ID도 실행마다 새 이름으로 바꾼다.

```bash
bash scripts/paper.sh \
  --tracks conductance_gat --suite all --device cuda \
  --data-root "$NEW_GAT_DATA_ROOT" \
  --results-root "$NEW_GAT_RESULTS_ROOT" \
  --model-seeds 0,1,2,3,4 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --batch-size 32 --workers 4 --min-free-gb 8 \
  --run-id conductance-all-v1
```

공개 benchmark를 제외하고 자체 생성 실험만 하려면 `--suite all`을 `--suite core`로 바꾼다.
Cycle PE 후보만 짧게 비교하려면 다음처럼 graph target과 두 후보로 제한할 수 있다.

```bash
bash scripts/paper.sh \
  --tracks cycle_pe --suite core --device cuda \
  --data-root "$NEW_GAT_DATA_ROOT" \
  --results-root "$NEW_GAT_RESULTS_ROOT" \
  --model-seeds 0 \
  --data-seed 0 --split-seed 0 --chart-seed 0 \
  --cycle-variants no_pe,projector --cycle-core-targets graph \
  --cycle-epochs 20 --cycle-learning-rate 0.001 \
  --batch-size 16 --workers 2 \
  --run-id cycle-candidates-v1
```

이 후보 비교는 official BREC를 실행하지 않는다. Official BREC를 선택하면 epochs/LR,
batch 16, float32/no-AMP 등은 고정 protocol을 따른다.

## 8. 진행 상황과 결과 확인

학습 출력은 터미널과 로그 파일에 동시에 기록된다. 다른 SSH 창에서도 저장소 폴더로
이동한 뒤 다음 명령으로 확인할 수 있다.

```bash
nvidia-smi
ls runs/paper/paper-all-v1/logs/
tail -n 50 runs/paper/paper-all-v1/logs/*.log
```

실행 종료 후 전체 성공 여부를 확인한다.

```bash
python -c 'import json; print(json.load(open("runs/paper/paper-all-v1/manifest.json"))["status"])'
ls runs/paper/paper-all-v1/aggregate/
```

첫 명령이 `passed`여야 전체 명령이 성공한 것이다. `failed`이면
`aggregate/failures.csv`, `manifest.json`의 command 항목과 해당 로그를 확인한다.

| 파일/폴더 | 확인할 내용 |
|---|---|
| `runs/paper/<run-id>/manifest.json` | 전체 상태, 실제 명령, seed, source revision, 각 작업의 성공/실패 |
| `runs/paper/<run-id>/logs/` | 트랙·seed별 stdout/stderr |
| `runs/paper/<run-id>/gpu-preflight.json` | GPU 사전검사와 검사한 shape·메모리 |
| `runs/paper/<run-id>/aggregate/metrics.csv` | 등록된 test metric의 seed 평균·표준편차·95% CI |
| `runs/paper/<run-id>/aggregate/paired.csv` | seed를 맞춘 비교군 차이 |
| `runs/paper/<run-id>/aggregate/efficiency.csv` | 시간·메모리·active parameter 관측 |
| `runs/paper/<run-id>/aggregate/failures.csv` | 실패한 실행 목록 |
| `$NEW_GAT_RESULTS_ROOT/<track>/<run-id>/` | 각 트랙의 상세 결과와 checkpoint |

Cycle `suite=all` 결과는 model seed별 `core/`, `zinc/`와 별도의
`brec-official-10-seed/`로 나뉜다. 데이터 준비만 한 run에는 metric 집계가 없다.

결과를 PC로 가져올 때는 MobaXterm의 SFTP 패널에서 **트랙 결과 폴더와
`runs/paper/<run-id>/`를 함께** 받는다. 환경 재현을 위해 `.gpu-environment.json`,
`.gpu-environment.freeze.txt`도 보관한다. 데이터·결과·checkpoint는 GitHub에 자동 업로드되지 않는다.

## 9. 자주 막히는 문제

| 증상 | 할 일 |
|---|---|
| `nvidia-smi: command not found` 또는 GPU가 안 보임 | GPU node/allocation인지 확인한다. CPU로 학습을 대신 실행하지 않는다. |
| `conda: command not found` | 서버 안내에 따라 Conda module/경로를 준비한다. 모르면 관리자에게 확인한다. |
| `conda activate`가 shell 초기화를 요구함 | 3번의 `source "$(conda info --base)/etc/profile.d/conda.sh"`를 실행하고 다시 활성화한다. |
| Conda 환경 비활성화 또는 `base` 환경 오류 | 3번에서 만든 전용 환경을 `conda activate new-gat`으로 활성화한다. |
| Python 버전 오류 | 전용 환경에서 `python --version`이 3.11 이상인지 확인한다. 최초 생성 명령은 Python 3.11을 지정한다. |
| CUDA 12.6 미만 / driver 호환 오류 | 관리자에게 driver 환경을 확인한다. 현재 lock은 구형 torch로 자동 하향하지 않는다. |
| Conda 환경의 Python을 찾지 못함 | `conda activate new-gat` 후 `python --version`을 확인하고 저장소 root에서 setup을 끝낸다. |
| `ModuleNotFoundError` | 설치에 사용한 Conda 환경이 활성화됐는지 `conda info --envs`와 `which python`으로 확인한다. |
| 데이터 cache가 없다고 나옴 | 5번 준비·검사를 실행하고 data root와 data/split seed가 학습 명령과 같은지 확인한다. |
| `run id already exists` | 기존 결과를 지우지 말고 새 `--run-id`를 사용한다. 자동 resume 옵션은 없다. |
| CUDA OOM | 해당 작업을 멈추고 batch를 `32 → 16 → 8`로 낮춰 새 run ID로 실행한다. Official BREC batch는 16 고정이다. |
| `DataLoader worker ...` 또는 공유 메모리 오류 | `--workers 0`으로 다시 점검한다. |
| SSH 연결이 끊어짐 | tmux를 사용했다면 같은 서버에서 `tmux attach -t new-gat`로 복귀한다. |
| `git clone`에 인증 요구 / repository not found | 저장소가 다시 Private이 됐는지 확인한다. Private은 권한 있는 GitHub 계정 인증이 필요하다. |

GPU를 직접 고를 때는 `--device cuda:1`처럼 지정할 수 있다. Cluster가
`CUDA_VISIBLE_DEVICES`를 설정했다면 보통 할당 내부의 `--device cuda`를 사용한다.

Runner는 선택한 트랙의 synthetic GPU 사전검사를 자동 실행하지만 실제 데이터의 최대
graph 크기를 자동 추정하지 않는다. 더 큰 graph를 검사하려면
`--preflight-nodes-per-graph`, `--preflight-edges-per-graph`, `--preflight-cycle-rank`를
실제 사용할 크기에 맞춰 늘린다. 사전검사 통과가 전체 데이터의 OOM 부재를 보장하지는 않는다.

## 10. 업데이트와 Private 복귀

실행 중에는 source를 바꾸지 않는다. 다음 새 실험 전에 최신 코드를 받으려면 저장소에서
`git pull --ff-only`를 실행하고, 의존성이 바뀌었다면 setup을 다시 실행한다.

서버 실행을 마치고 저장소를 비공개로 되돌리려면 GitHub 저장소의
`Settings → General → Danger Zone → Change visibility → Change to private`에서 확인한다.

이미 clone된 코드와 서버의 결과 파일은 Private 전환으로 지워지지 않으며 로컬 학습도
계속 실행된다. 추가 `git pull`이 필요 없다면 clone 직후에 비공개로 돌려도 된다.
Private 전환 후 새 clone/pull에는 권한 있는 계정의 SSH key 또는 HTTPS 인증이 필요하다.
공개 기간에 다른 사람이 만든 복제본은 비공개 전환으로 회수되지 않는다.

## 추가 문서와 개발용 검사

- [DATASETS.md](DATASETS.md): 데이터셋, split, metric, cache 규칙
- [Conductance README](research/conductance_gat/README.md): 트랙 단독 CLI
- [Cycle PE README](research/cycle_pe/README.md): 트랙 단독 CLI
- [Tree augmentation README](research/tree_augmentation/README.md): 트랙 단독 CLI
- [hand_off.md](hand_off.md): 수학·구현 경계·검증 범위·미완료 항목을 검토하는 문서
- [code_summary.md](code_summary.md): 파일 경로별 원문 코드 스냅샷

코드를 수정한 뒤 검증할 때는 다음을 사용한다.

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python scripts/generate_code_summary.py --check
```

`research/combined_later/`는 격리된 과거 prototype이다. 위 paper 명령은 이를 실행하지 않는다.
이 저장소에는 실행 코드와 회귀 테스트가 있으며, 실제 Linux CUDA 서버에서 전체 학습을
완료한 성능 결과가 이미 포함되어 있다는 뜻은 아니다. 정식 CPU 학습으로 자동 fallback하지 않는다.
