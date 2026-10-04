# 서버 본학습과 결과 확인

## 전체 실행

아래 명령은 할당받은 GPU가 0인 경우다. 다른 GPU나 UUID를 할당받았다면 CUDA_VISIBLE_DEVICES 값을 그 할당으로 지정한다.
여러 GPU를 할당받았다면 쉼표로 구분한다. 전체 20조건, 840회 학습, 420,000회 갱신을 새 결과 폴더에서 실행한다.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
LOCAL_CONTEXT_NORM_RUN="/home/aicompetition07/new-gat/results/local-context-normalization-$(date +%Y%m%d-%H%M%S)" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=0 \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_context_coupling.normalization.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --output-dir "$LOCAL_CONTEXT_NORM_RUN"
```

기존 source 결과와 이전 classification 결과는 보존한다.
같은 터미널에 현재 단계·조건·epoch, train CE, validation CE/accuracy, gain, step 시간,
현재 seed pack의 ETA와 peak VRAM을 출력한다. 후보 packing·chunk와 CPU 준비 측정도 기록한다.
실행시간은 서버에서 측정한 처리량으로 판단한다.

## 완료 후 전달할 결과

같은 셸에서 아래 두 파일만 먼저 출력하면 된다. 전체 epoch 로그를 복사할 필요는 없다.

```bash
cat "$LOCAL_CONTEXT_NORM_RUN/completion.json"
cat "$LOCAL_CONTEXT_NORM_RUN/LOCAL_CONTEXT_NORMALIZATION_SUMMARY.md"
```

| 파일 | FULL 결과의 내용 |
| --- | --- |
| `metrics.csv`, `metric_estimates.csv` | 모든 final seed·split의 분류 지표 900행과 요약 180행 |
| `paired_comparisons.csv` | 직접 비교 52개에 대한 split·지표별 요약 936행 |
| `interaction_comparisons.csv` | 네 조건의 paired 상호작용 12개에 대한 요약 216행 |
| `interventions.csv`, `intervention_changes.csv` | 같은 checkpoint의 gain 개입 4,320행과 차이 요약 1,728행 |
| `branch_diagnostics.csv`, `branch_estimates.csv` | 실제 gain·applied S/G 에너지·matched 변화 3,480행과 요약 696행 |
| `frozen_evaluation_provenance.json` | 개입 전후 model hash와 optimizer 갱신 0회 확인 |
| `learning_rate_selection.json`, `tuning_validation.csv`, `final_validation_selection.csv` | Validation만을 사용한 LR·checkpoint 선택 |
| `test_evaluation_unlocked.json` | 모든 final 선택을 고정한 후 test 평가를 연 기록 |
| `resources.csv`, `hardware.json` | 자원·후보 계측·실제 학습·재개 여부 |
| `source.json`, `data_manifest.json`, `context_data_manifest.json`, `contract.json` | 원래 source와 데이터, 새 계약의 hash |
| `coverage.json`, `report_checks.json`, `completion.json` | 전체 실험 범위·보고서 검증·완료 여부 |
| `figures/cross_effects.png/.pdf` | 같은 C·intra에서 cross와 gain의 효과 |
| `figures/cross_normalization.png/.pdf` | Edge−graph cross 정규화 비교 |
| `figures/intra_normalization.png/.pdf` | Local−graph 내부 정규화 비교 |
| `figures/interactions.png/.pdf` | 정규화 상호작용의 paired 95% 구간 |
| `figures/frozen_effects.png/.pdf` | 같은 checkpoint에서 gain 개입 ΔCE |
| `figures/branch_use.png/.pdf` | Gain과 실제 출력 변화·applied context norm |

학습 pack 폴더에는 epoch history, validation으로 선택한 checkpoint, resume checkpoint와 자원 기록이 있다.
좋은 test 개입 하나를 골라 selected checkpoint나 사전 지정 비교를 바꾸지 않는다.

## 중단 후 재개

`--resume-from`에는 기존 최상위 study 결과 폴더를 지정하고 `--output-dir`은 새로운 폴더로 지정한다.
Source/config/code/data/seed packing과 checkpoint hash를 확인한 후 재개한다.
완료된 epoch는 다시 갱신하지 않는다. 계약상 전체 갱신과 이번 실행에서 새로 수행한 갱신을 구분한다.
기존 파일을 덮어쓰거나 현재 원격 셸·서버 세션을 종료하는 명령을 사용하지 않는다.

## FULL과 DEBUG

FULL은 전체 citation 그래프 3개의 실제 분류 본학습이다.
DEBUG는 별도 fixture의 20조건, 360회 학습, 1,080회 갱신을 수행하는 pipeline 검사다.
DEBUG 결과는 actual_data=false이며 본학습 성능으로 보고하지 않는다.

