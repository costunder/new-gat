# 200 epoch 종료 후 소스 계약 오류 복구 — 2026-10-02

대상 로그: `output-init-log-row-server-20261001-172250`,
`train-kaiming_relu`의 learned epoch 200/200, train CE 0.62667,
마지막 validation accuracy 72.5897%.
이는 마지막 epoch의 validation이며 best checkpoint 점수나 test 점수가 아니다.

## 확인된 코드 경로와 원인 후보

`train_one()`은 모든 epoch의 JSON, 관측 JSON, best checkpoint와
`learned-trained.json`을 저장한 뒤 반환한다. 이후 `main()`의
`research_contract` 재검사에서 제시된 오류가 발생했다.
해당 오류는 OOM이나 마지막 epoch 학습 실패가 아니다.
fixed 조건 및 전체 비교의 완료는 별도로 확인해야 한다.

소스 계약은 이 모델의 직접 의존성뿐 아니라 `research/` 전체 Python/YAML과
`scripts/` 전체 Python까지 포함한다. 따라서 별도의 `research/wedge_propagation`
파일 추가도 계약을 바꾼다. 최근 `a7cea24`, `320ab62`의 독립 실험 추가가
이 경로를 유발할 수 있지만, 서버의 실제 변경 내역은 시작 시 저장된
`execution_sources.zip`과 현재 파일을 비교하기 전에는 확정하지 않는다.
기존 오류에는 종료 시의 상세 차이가 기록되지 않았다.

## 복구 방식

`scripts/recover_output_init_study.py`는 기본적으로 읽기 전용 검사만 수행한다.
`--output-dir`을 주면 원본 밖의 새로운 결과 폴더에서 복구한다.

- ZIP의 모든 원래 소스 해시, study/child/calibration 계약을 검증한다.
- 허용하는 차이는 원본에 없었던 독립 `research/wedge_propagation/` 파일과
  복구 드라이버 자체의 추가뿐이다. 기존 파일 변경·삭제, 그 밖의 새 코드,
  학습 코드에서 wedge 참조가 발견되면 중단한다.
- 원래 200 epoch의 연속성, 개별 epoch JSON과 trained history의 일치,
  전체 supervised 수·step 수, 관측 파일, best 선택과 checkpoint 해시를 검증한다.
- 원본 ZIP을 별도 실행 폴더에 펼친다. Python import와 `PYTHONPATH`를 이
  소스에 묶는다. 원래 모델·수식·학습 코드·데이터·정밀도·epoch·배치를 바꾸지 않는다.
- CUDA에서 데이터 fingerprint와 초기화를 다시 확인하고, 완료된 각 모델의
  best validation 정답 수·전체 수·parameter hash·seed coverage를 재현한다.
  기존 calibration의 GPU/메모리 조건 검사도 그대로 실행한다.
- 재사용 조건끼리 초기화와 sampled context 대응을 확인한 뒤, 전혀 시작하지
  않은 조건만 원래 예산으로 학습한다. 부분 학습은 optimizer/RNG 상태가 없어
  best weight에서 임의 재개하지 않고 중단한다.
- 평가와 4조건 비교를 다시 생성하며, 원본 파일은 수정하지 않는다.

제시된 단계대로 baseline 두 조건이 이미 완료됐고 kaiming learned의 저장 파일이
정상이면, 그 세 조건을 재사용하고 **kaiming fixed 200 epoch만 새로 학습**한다.
검증 실패를 무시하는 옵션은 없다. 현재 디스크와 시작 ZIP의 비교는 과거 모든
시점의 파일 변경 이력을 증명하지는 않는다. 이 한계를 recovery audit에도 남긴다.

## 실행

학습 작업의 checkout을 갱신하지 않고 복구 파일만 Git에서 가져올 수 있다.
기존 CUDA 할당을 유지하는 같은 환경에서 실행한다.

```bash
cd /home/aicompetition07/new-gat &&
git fetch origin &&
RECOVERY_SCRIPT="$(mktemp /tmp/new-gat-recovery-XXXXXX.py)" &&
git show origin/main:scripts/recover_output_init_study.py > "$RECOVERY_SCRIPT" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u "$RECOVERY_SCRIPT" \
  --repository /home/aicompetition07/new-gat \
  --run-dir /home/aicompetition07/new-gat/results/output-init-log-row-server-20261001-172250 \
  --output-dir "/home/aicompetition07/new-gat/results/output-init-recovered-$(date +%Y%m%d-%H%M%S)"
```

`--output-dir`을 빼면 소스 차이와 조건별 완료 여부만 읽는다.
새 폴더의 `recovery.log`, `recovery-audit.json`, `*-reproduction.json`,
`*-reused.json`에 실행·검증·복사 증거를 남긴다.
`comparison.json`과 `recovery-completed.json`은 전체 성공 후 생성한다.
실패 시 `recovery-failure.json`에 traceback을 남긴다.

## 로컬 검증과 한계

RTX 5070 Ti CUDA에서 명시적인 합성 smoke를 실행했다. 모델은 8층/256채널/8헤드,
학습은 테스트에서만 2 epoch다. completed learned를 복사·재검증하고 learned를
재학습하지 않은 채 fixed만 학습한 뒤 frozen 평가까지 확인했다.
잘못된 validation 재현 기록은 거부한다.

20개 검사 통과: 소스 추가·변경·삭제 구분, ZIP 손상·경로 검사, 원본 보존,
부분 조건 차단, epoch/step/best/checkpoint 검증, 사전 pairing 검사,
실제 저장소 소스의 별도 프로세스 import 및 해시 재현, CUDA 복구 smoke.
기록: `results/output-init-recovery-tests-20261002-v3.xml`.
정적 Ruff 검사도 통과했다.

서버의 원본 결과에는 아직 접근하지 않았다. 실제 ogbn-arxiv 200 epoch 복구,
A100 MIG에서의 validation 재현과 최종 4조건 비교는 실행 명령의 검증 대상이다.
합성 smoke를 서버 복구 완료나 최종 성능 검증으로 해석하지 않는다.
