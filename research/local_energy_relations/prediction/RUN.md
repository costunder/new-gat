# 서버 실행과 결과 확인

본학습은 서버의 할당받은 GPU에서 실행한다. 기존 고정 audit 결과는 변경하지 않는다.
아래는 새 output directory에서 전체 336 run을 실행하는 명령이다.
번호나 UUID는 실제 할당받은 GPU를 입력한다.

```bash
read -r -p "할당받은 GPU 번호 또는 UUID: " LOCAL_PREDICTION_GPU &&
test -n "$LOCAL_PREDICTION_GPU" &&
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
LOCAL_PREDICTION_RUN="/home/aicompetition07/new-gat/results/local-prediction-$(date +%Y%m%d-%H%M%S)" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$LOCAL_PREDICTION_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_energy_relations.prediction.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --output-dir "$LOCAL_PREDICTION_RUN"
```

여러 GPU를 실제로 할당받았다면 번호를 쉼표로 구분해 입력한다.
현재 프로세스에 보이는 모든 할당 GPU를 독립 job 분배에 사용한다.
사용 중인 다른 사용자의 프로세스를 중단하거나 할당 범위를 임의로 확대하지 않는다.

## 진행·완료 확인

실행한 같은 터미널에 calibration, tuning/final phase, run, epoch, CE,
validation, seconds/epoch, ETA, VRAM이 표시되고 result 폴더의 로그에도 저장된다.
측정 전 예상 소요시간을 확정하지 않는다. CPU 준비와 seed packing 후보를 실제로 측정한다.

완료 후 다음 두 출력을 보내면 된다.

```bash
cat "$LOCAL_PREDICTION_RUN/completion.json"
cat "$LOCAL_PREDICTION_RUN/LOCAL_PREDICTION_SUMMARY.md"
```

새 셸이라 변수 값이 없으면 실제 완료 경로를 지정한다.

```bash
LOCAL_PREDICTION_RUN=/home/aicompetition07/new-gat/results/local-prediction-실제완료폴더
cat "$LOCAL_PREDICTION_RUN/completion.json"
cat "$LOCAL_PREDICTION_RUN/LOCAL_PREDICTION_SUMMARY.md"
```

## 남는 결과

- 전체 설정·raw/processed data 및 과학 source hash·실제 하드웨어와 resource calibration.
- 모든 tuning/final run의 epoch 기록·validation 선택·최종 checkpoint.
- 모든 seed×조건×train/validation/test metric, E/J 제거 개입, 층별 branch 진단.
- 대응 차이·95% fixed-seed t 구간·과학 PNG/PDF와 요약.
- `coverage.json`, `contract.json`, `completion.json`; 실패 시 `failure.json`과 기존 부분 결과.

Full은 전체 336 run·168,000 update의 coverage를 확인한 뒤에만 완료로 표시한다.
Tuning의 test 평가와 test를 통한 checkpoint/LR 선택은 허용하지 않는다.
어떤 오류에도 서버·부모 셸·SSH 세션을 종료하는 명령을 사용하지 않는다.

## 완료된 run을 보존하며 재개

실패나 중단이 있었다면 기존 결과를 지우거나 덮어쓰지 않는다.
`--resume-from`에 기존 **분류** 결과 폴더를 지정하고 새 output folder로 실행한다.
같은 계약·source·data hash와 checkpoint/optimizer 상태가 확인된 run만 이어서 사용한다.

```bash
LOCAL_PREDICTION_PREVIOUS=/home/aicompetition07/new-gat/results/local-prediction-기존분류폴더
LOCAL_PREDICTION_RUN="/home/aicompetition07/new-gat/results/local-prediction-resume-$(date +%Y%m%d-%H%M%S)"
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$LOCAL_PREDICTION_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_energy_relations.prediction.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --resume-from "$LOCAL_PREDICTION_PREVIOUS" \
  --output-dir "$LOCAL_PREDICTION_RUN"
```

## DEBUG

`--profile debug`는 별도 고정 fixture와 축소된 예산으로 전체 연결을 검사한다.
실제 citation 성능으로 제출하지 않으며 FULL 설정을 바꾸지 않는다.
DEBUG와 FULL은 서로 다른 새 output directory를 사용한다.
