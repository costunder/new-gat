# 실행 환경 상세

기본 설치·실험 순서는 [시작 안내](GETTING_STARTED.md)를 따른다. 아래 항목은 환경별 조정과 개발 검증용이다.

## Conda

`environment.yml`은 Python 3.11과 pip를 준비한다. 연구 패키지는 `setup_gpu.sh`가
공식 PyTorch CUDA wheel 저장소와 선택한 조합의 requirements/constraints 파일을 함께 사용해 설치한다.
Conda base 환경과 중첩된 venv는 설치 대상에서 제외한다.

`prepare_data.sh`는 NumPy를 포함한 전체 lock의 설치 버전과 실제 runtime import를 먼저
검사한다. 누락·버전 불일치·일반 import 오류가 있으면 `setup_gpu.sh`를 한 번 실행하고 재검사한다.
호스트 ABI 검사 실패는 종료 코드 `3`으로 중단하며 자동 재설치하지 않는다.
의존성 보완도 정확히 인식된 기존 profile을 유지한다. Torch가 없는 환경만 기본 자동 선택을
사용하고, 미등록/CPU/custom Torch가 이미 있으면 자동 교체하지 않고 명시적 설치를 요구한다.
설치 스크립트는 의존성을 생략하는 `SKIP_DEPS` 경로를 더 이상 제공하지 않는다.
자동 설치가 실패하면 데이터 준비도 중단하며, 설치나 다운로드를 반복 시도하지 않는다.
자동 설치는 수동 설치와 같은 Linux/NVIDIA 드라이버·GPU 할당·네트워크 조건을 요구한다.
이미 설치된 CUDA 패키지의 검사와 데이터 준비 자체에는 GPU 할당이 필요하지 않지만 학습에는 필요하다.

설치를 명시적으로 다시 실행하려면 활성 `new-gat` 환경의 저장소 폴더에서 다음을 사용한다.

```bash
bash scripts/setup_gpu.sh && bash scripts/prepare_data.sh
```

학습 실행기는 패키지를 자동 설치하지 않는다. 의존성이 없으면 interpreter 경로와 전체 누락
패키지 목록, 위 설치 명령을 알려주고 run 폴더를 만들기 전에 종료한다.
`--help`와 `--dry-run`은 연구 패키지 없이도 확인할 수 있으며 설치나 다운로드를 수행하지 않는다.

Conda 설치 시 해당 터미널의 shell 초기화까지 완료해야 한다. `conda activate`가 초기화
오류를 내면 Conda 설치 안내에 따라 shell을 초기화하고 새 터미널에서 활성화한다.
저장소 실행 명령마다 shell 초기화 코드를 붙이지 않는다.

`conda` 자체가 없다면 [Miniforge 공식 설치 안내](https://github.com/conda-forge/miniforge#install)를 따르거나
사용 중인 클러스터의 Conda module 안내를 확인한다. 저장소는 Conda나 NVIDIA 드라이버를 설치하지 않는다.

## CUDA runtime

기본값은 `auto`다. 버전 범위를 풀어 최신 패키지를 임의로 설치하는 것이 아니라,
아래 두 고정 조합 중에서만 선택한다. CUDA 12.2로 표시되는 RTX A6000 서버는 `cu118`을 선택한다.

| CUDA 표시값 / 명시적 선택 | Torch / PyG | requirements lock | constraints |
|---|---|---|---|
| 11.8 ≤ 표시값 < 12.6, 또는 `cu118` | 2.7.1 / 2.7.0 | `requirements-cu118-lock.txt` | `constraints-cu118.txt` |
| 표시값 ≥ 12.6, 또는 `cu126` | 2.13.0 / 2.8.0.post1 | `requirements-lock.txt` | `constraints-cu126.txt` |
| 명시적 `cu130` / `cu132` | 2.13.0 / 2.8.0.post1 | `requirements-lock.txt` | 해당 `constraints-cu*.txt` |

다른 직접 의존성은 두 조합에서 동일하게 고정한다. 이후 데이터 준비·학습에서는 설치된
Torch의 **정확한 버전과 CUDA tag**로 같은 profile/lock을 선택하고 모든 pin과 runtime을 검사한다.
이 단계에서는 드라이버를 다시 조회하거나 `cu118`을 기준 조합으로 재설치하지 않는다.

`nvidia-smi`의 CUDA 표시는 드라이버의 호환 수준이지 설치된 Toolkit 버전이 아니다.
[NVIDIA의 minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
때문에 표시값보다 높은 같은 CUDA major의 runtime이 항상 실행 불가능한 것은 아니다.
다만 이 설치기는 그 기능·PTX 제약에 의존하지 않는 보수적 정책으로 조합을 선택한다.
명시적으로 선택한 runtime이 표시값보다 높으면 안내하고 중단하며 다른 조합으로 몰래 바꾸지 않는다.
드라이버나 시스템 라이브러리를 설치·변경하지 않고 CPU wheel로 대체하지 않는다.

공식 [PyTorch 배포 조합](https://pytorch.org/get-started/previous-versions/)과
[PyG 2.7 지원 조합](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.7.0)을 따른다.
기본 `cu118` 조합(Torch 2.7.1)은 Linux x86_64용이며, [공식 wheel](https://download.pytorch.org/whl/cu118/torch/)의
`manylinux_2_28` 요건에 따라 glibc 2.28 이상인지 다운로드 전에 검사한다.
기준 CUDA 12.6 조합도 glibc 2.28 이상이 필요하다. 이 검사는 실제 GPU 학습 검증을 대신하지 않는다.
설치 후에는 import 검사 외에 빈 배열의 NumPy↔Torch 변환도 검사해 ABI 오류를 확인한다.
이는 의존성 검사이며 데이터셋 생성이나 모델 학습이 아니다.

다른 서버에서도 CUDA 12.2 서버와 같은 조합을 재현하려면 명시적으로 고정한다.

```bash
CUDA_WHEEL_TAG=cu118 bash scripts/setup_gpu.sh
```

`CUDA_WHEEL_TAG`를 명시한 데이터 준비·학습은 그 조합을 엄격하게 검사한다.
기본값(`auto` 또는 미지정)은 이미 설치된 조합을 유지한다.
다른 runtime으로 실행한 결과는 기준 환경과 구분해 기록한다. 설치 보고서는
`.gpu-environment.json`, 전체 패키지 snapshot은 `.gpu-environment.freeze.txt`에 저장된다.
각 실행의 `manifest.json`에도 `research_environment`에 profile ID, CUDA tag, lock checksum,
실제 직접 의존성 버전을 저장한다. 서로 다른 조합은 동일 환경의 seed 반복으로 합치지 않는다.

## Ubuntu 18.04 / Singularity: glibc 2.27 호환 설치

`GLIBC_2.28 not found`는 Conda 활성화 여부나 NumPy 누락 문제가 아니다. 실행 중인
컨테이너의 libc가 해당 Torch binary의 요구 조건보다 오래된 경우 발생한다. Conda 환경을
다시 활성화하거나 동일한 wheel을 재설치해도 컨테이너의 glibc는 바뀌지 않는다.

서버 제공 GPU 실행기가 ELF 바이너리이면 `cat`/`sed`로 내용을 읽지 않는다.
서버가 제공하지 않은 이미지 선택 옵션을 추측하거나 시스템 libc를 교체하지 않는다.
현재 GPU 할당·컨테이너 안에서 별도 Conda 환경을 만들면 기존 `new-gat` 패키지를 보존할 수 있다.

저장소 최상위 폴더에서 다음을 실행한다.

```bash
conda env create -n new-gat-legacy -f environment.yml
conda activate new-gat-legacy
bash scripts/setup_gpu.sh --profile legacy-cu118
```

`new-gat-legacy`가 이미 있으면 생성 명령만 생략한다. 기존 환경 삭제나 전체 환경
업데이트는 필요 없다. 설치기가 기존 Torch 2.7/기타 버전을 발견하면 덮어쓰지 않고
중단하므로, 다른 패키지를 쓰는 환경을 재활용하지 않는다. 공용 Conda base도 변경하지 않는다.

| 구분 | 값 |
|---|---|
| 명시적 profile | `legacy-cu118` (자동 선택하지 않음) |
| 고정 조합 | Python 3.11 / Torch 2.6.0+cu118 / PyG 2.7.0 |
| 플랫폼 | Linux x86_64 / 전체 패키지 조합은 glibc ≥2.27 |
| 드라이버 조건 | `nvidia-smi`의 CUDA 표시값 ≥11.8 |
| 직접 의존성 lock | `requirements-legacy-cu118-lock.txt` |
| constraints | `constraints-legacy-cu118.txt` |
| 설치 기록 기본 위치 | 활성 Conda prefix의 `.new-gat-environment/` |

공식 [PyTorch wheel 플랫폼 안내](https://dev-discuss.pytorch.org/t/pytorch-linux-wheels-switching-to-new-wheel-build-platform-manylinux-2-28-on-november-12-2024/2581)에
따르면 Torch 2.6의 cu118 wheel은 Manylinux2014(glibc ≥2.17)를 사용한다. 다만 이 저장소의
NumPy/SciPy 등 직접 의존성을 포함한 전체 조합은 glibc ≥2.27이 필요하다. 직접 pin의
Python 3.11/x86_64 wheel 제공 여부는 공식 배포 메타데이터로 확인했지만,
이는 실제 컨테이너에서의 설치·import·GPU 학습 성공을 대신하는 검증은 아니다.
PyG 조합은 [PyG 2.7 지원 목록](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.7.0)을 따른다.

`CUDA_WHEEL_TAG=cu118`만 지정하면 기존 **Torch 2.7.1** 조합이 선택된다. glibc 2.27에서는
반드시 `--profile legacy-cu118`을 사용한다. profile과 `CUDA_WHEEL_TAG`가 충돌하면 중단한다.
legacy 설치는 이름이 `new-gat`인 환경을 거부하고, 다른 이름이어도 기존 Torch가 있으면
정확히 `2.6.0+cu118`인 경우만 재사용한다. 기존 환경에서 실행 중인 학습을 위해
그 환경에 `pip uninstall`, Torch 교체, 시스템 라이브러리 변경을 수행하지 않는다.

설치기는 의존성 전체 설치, `pip check`, 정확한 pin/CUDA runtime, NumPy↔Torch 변환,
GPU 접근 검사를 수행한다. 모두 성공한 후 데이터 준비와 기존 트랙 명령을 실행한다.
새 환경에서도 데이터와 run 저장 위치는 동일하며 기존 run을 덮어쓰거나 자동 재개하지 않는다.
다만 Torch 2.6 결과와 2.7/2.13 결과는 동일 환경의 seed 반복으로 합치지 않는다.

기본 설치 보고서와 freeze 파일을 덮어쓰지 않도록 legacy는 활성 Conda 환경 내부의
`.new-gat-environment/.gpu-environment.json`과 `.gpu-environment.freeze.txt`에 기록한다.
`ENVIRONMENT_SNAPSHOT_DIR`를 명시한 경우에는 그 경로를 사용한다. 실행 manifest에도
`profile_id: legacy-cu118`과 정확한 버전·lock checksum을 저장한다.

### 구버전 Torch의 보안 제약

Torch 2.6과 2.7은 최신 보안 패치판이 아니다.
[PyTorch 보안 공지 GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)는
2.9.1 이하에서 조작된 체크포인트가 `weights_only=True`에서도 코드를 실행할 수 있으며
2.10에서 수정됐다고 명시한다. 공식 출처를 확인한 데이터와 직접 생성한 체크포인트만
사용하고, 출처 불명의 `.pt`·`.pth`·pickle cache를 가져오지 않는다.
별도 Conda 환경은 의존성 분리이지 보안 격리가 아니다. 최신 보안 패치 조합이 필요한
경우에는 관리자가 제공하는 더 새 Linux 컨테이너와 그에 맞는 GPU 조합을 사용해야 한다.

## 데이터 검사

이미 준비된 전체 데이터의 내용, checksum, seed 요청을 읽기 전용으로 검증한다.

```bash
python scripts/check_datasets.py --data-root data/paper --require-cache
```

오류가 나면 실제 원본 또는 cache 문제를 해결한다. 다운로드는 `--allow-download`로
명시적으로 허용한 데이터 준비 단계에서만 수행한다.

## 개발 검증

아래 명령은 연구 결과를 생성하지 않는 단위 검사다. 실험 실행 전 필수 절차가 아니다.

```bash
python -m pytest -q
python -m ruff check src scripts research tests
```

설치 때 단위 검사를 함께 실행하려면 `RUN_TESTS=1 bash scripts/setup_gpu.sh`를 사용한다.
legacy 환경에서는 같은 명령에 `--profile legacy-cu118`을 추가한다.
테스트의 작은 입력은 테스트 내부에서만 사용하며, paper CLI에 축소·가짜 데이터 실험 모드는 없다.

명령 구성만 확인하려면 데이터 다운로드나 학습 없이 dry-run할 수 있다.

```bash
bash scripts/reproduce.sh --dry-run
```

Linux/Bash 전용 검사는 해당 환경에서 실행해야 한다. 다른 OS에서 skipped된 검사를
Linux 실행 성공으로 간주하지 않는다.
