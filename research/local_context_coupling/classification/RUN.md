# 서버 본학습과 결과 확인

## 전체 실행

할당받은 GPU 번호 또는 UUID를 입력합니다. 여러 GPU를 할당받았다면 쉼표로 구분한 목록을 입력합니다.
전체 citation 그래프 3개와 여섯 조건으로 총 252회 학습, 126,000회 갱신을 새 결과 폴더에서 실행합니다.

```bash
read -r -p "할당받은 GPU 번호 또는 UUID 목록: " LOCAL_CONTEXT_CLASS_GPU &&
test -n "$LOCAL_CONTEXT_CLASS_GPU" &&
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
LOCAL_CONTEXT_CLASS_RUN="/home/aicompetition07/new-gat/results/local-context-classification-$(date +%Y%m%d-%H%M%S)" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$LOCAL_CONTEXT_CLASS_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_context_coupling.classification.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --output-dir "$LOCAL_CONTEXT_CLASS_RUN"
```

같은 터미널에 현재 단계·작업·epoch, train CE, validation CE/accuracy, rho, step 시간, 현재 seed pack의 ETA, peak VRAM을 출력합니다.
GPU·CPU·그래프 크기·파라미터 수와 seed packing·exact edge chunk 후보의 측정 결과도 기록합니다.
전체 그래프를 사용하는 모델당 batch 1과 여러 독립 seed 모델의 동시 계산을 구분합니다.
실행시간은 서버의 실제 계측으로 판단합니다.

## 완료 후 전달할 결과

같은 셸에서 아래 두 파일을 출력하면 됩니다. 매 epoch 로그를 전부 복사할 필요는 없습니다.

```bash
cat "$LOCAL_CONTEXT_CLASS_RUN/completion.json"
cat "$LOCAL_CONTEXT_CLASS_RUN/LOCAL_CONTEXT_CLASSIFICATION_SUMMARY.md"
```

기여를 확인하려면 paired/frozen/branch CSV를 사용합니다.

| 파일 | 내용 |
| --- | --- |
| `metrics.csv`, `metric_estimates.csv` | 모든 final seed·split의 원시 분류 값 270행과 요약 54행 |
| `paired_comparisons.csv` | 같은 seed에서 재학습 조건을 비교한 paired 지표 162행 |
| `interventions.csv`, `intervention_changes.csv` | Frozen gain0/gain1 원시 결과 1,080행과 paired 차이 요약 432행 |
| `branch_diagnostics.csv`, `branch_estimates.csv` | 실제 gain·θ·context·energy·matched 변화 900행과 요약 180행 |
| `frozen_evaluation_provenance.json` | Frozen 개입 전후의 모델 hash와 optimizer 갱신 0회 확인 |
| `learning_rate_selection.json`, `tuning_validation.csv`, `final_validation_selection.csv` | Validation 선택과 최종 선택 기록 |
| `test_evaluation_unlocked.json` | 전체 final 선택 후 test 평가를 연 증거 |
| `resources.csv`, `hardware.json` | 실제 후보 계측·학습 자원·배분 |
| `source.json`, `data_manifest.json`, `context_data_manifest.json`, `contract.json` | Source·원래 데이터·새 계약 hash |
| `coverage.json`, `report_checks.json`, `completion.json` | 전체 범위·보고서 검증·완료 여부 |
| `figures/cross_effects.png/.pdf` | 재학습 비교와 paired 95% 구간 |
| `figures/frozen_effects.png/.pdf` | 같은 checkpoint의 gain 개입 ΔCE |
| `figures/branch_use.png/.pdf` | Gain과 실제 층 출력 변화 |

각 학습 pack 폴더에는 epoch history, selected checkpoint, resume checkpoint와 자원 기록이 있습니다.
Selected 모델은 validation으로 정합니다. 좋은 test 개입을 골라 selected checkpoint를 바꾸지 않습니다.

## 중단 후 재개

`--resume-from`에 기존 **최상위 study 결과 폴더**를 지정하고, `--output-dir`은 새로운 폴더로 정합니다.
Source/config/code/data/seed packing과 checkpoint hash가 맞아야 재개합니다.
완료된 seed의 epoch를 다시 갱신하지 않으며 새 갱신 횟수와 계약상 전체 갱신 횟수를 따로 기록합니다.
기존 결과는 보존합니다. 복사해 실행하는 명령에 현재 원격 셸이나 세션 종료 명령을 넣지 않습니다.

## FULL과 DEBUG

FULL은 실제 citation 분류 본학습입니다. DEBUG profile은 별도 fixture로 108회 학습과 324회 갱신을 수행하는 pipeline 검사입니다.
DEBUG 결과의 actual_data=false를 확인하고 본학습 정확도로 보고하지 않습니다.
