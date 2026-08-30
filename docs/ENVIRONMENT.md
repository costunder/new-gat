# 실행 환경 상세

기본 설치·실험 순서는 [README](../README.md)를 따른다. 아래 항목은 환경별 조정과 개발 검증용이다.

## Conda

`environment.yml`은 Python 3.11과 pip를 준비한다. 연구 패키지는 `setup_gpu.sh`가
공식 PyTorch CUDA wheel 저장소와 `requirements-lock.txt`, `constraints-cu126.txt`를 함께 사용해 설치한다.
Conda base 환경과 중첩된 venv는 설치 대상에서 제외한다.

`conda activate`가 초기화 오류를 내는 Bash 세션에서는 다음을 실행한 뒤 활성화한다.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate new-gat
```

`conda` 자체가 없다면 [Miniforge 공식 설치 안내](https://github.com/conda-forge/miniforge#install)를 따르거나
사용 중인 클러스터의 Conda module 안내를 확인한다. 저장소는 Conda나 NVIDIA 드라이버를 설치하지 않는다.

## CUDA runtime

기본값은 `cu126`이다. `nvidia-smi`에 더 높은 CUDA 버전이 표시되어도 동일한 reference runtime을 설치한다.
설치기는 드라이버가 선택한 runtime을 지원하는지 검사한다. CUDA Toolkit을 별도로 빌드하지 않는다.

하드웨어 요구로 다른 runtime이 필요할 때만 명시적으로 선택한다.

```bash
CUDA_WHEEL_TAG=cu130 bash scripts/setup_gpu.sh
```

`cu126`, `cu130`, `cu132`에 대응하는 constraints 파일이 있다. 다른 runtime으로 실행한 결과는
기준 환경과 구분해 기록한다. 설치 보고서는 `.gpu-environment.json`,
전체 패키지 snapshot은 `.gpu-environment.freeze.txt`에 저장된다.

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
테스트의 작은 입력은 테스트 내부에서만 사용하며, paper CLI에 축소·가짜 데이터 실험 모드는 없다.

명령 구성만 확인하려면 데이터 다운로드나 학습 없이 dry-run할 수 있다.

```bash
bash scripts/paper.sh --suite all --dry-run
```

Linux/Bash 전용 검사는 해당 환경에서 실행해야 한다. 다른 OS에서 skipped된 검사를
Linux 실행 성공으로 간주하지 않는다.
