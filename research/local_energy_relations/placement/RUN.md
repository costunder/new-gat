# 서버 실행과 결과 확인

본학습은 서버의 실제 할당 GPU에서 실행한다. 기존 결과는 보존한다.
이번 FULL은 전체 20조건의 **840 runs / 420,000 updates**다.
첫 층만 / 출력층만 / 두 층 모두를 각각 E / J / E+J와 비교하고,
두 고정 C의 base도 새로 학습한다.

```bash
read -r -p "할당받은 GPU 번호 또는 UUID: " LOCAL_PLACEMENT_GPU &&
test -n "$LOCAL_PLACEMENT_GPU" &&
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
LOCAL_PLACEMENT_RUN="/home/aicompetition07/new-gat/results/local-placement-$(date +%Y%m%d-%H%M%S)" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$LOCAL_PLACEMENT_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_energy_relations.placement.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --output-dir "$LOCAL_PLACEMENT_RUN"
```

여러 GPU를 실제로 할당받았다면 번호를 쉼표로 구분한다.
현재 프로세스에 보이는 모든 할당 GPU로 독립 job을 분배한다.
CPU 준비, packing·정확한 chunk 후보를 실제로 측정한 뒤 선택한다.
계측한 seconds/epoch와 남은 run 예산으로 ETA를 출력한다.

## 진행과 결과

같은 실행 터미널에 phase / condition / placement / epoch / CE / validation /
seconds/epoch / ETA / VRAM이 표시된다. 같은 내용은 결과 폴더에도 저장된다.
전체 source·data·설정·coverage 확인이 끝나야 `completed: true`를 쓴다.

완료 후 아래 두 파일을 보내면 된다.

```bash
cat "$LOCAL_PLACEMENT_RUN/completion.json"
cat "$LOCAL_PLACEMENT_RUN/LOCAL_PLACEMENT_SUMMARY.md"
```

새 셸이라 변수 값이 없으면 실제 완료 경로를 지정한다.

```bash
LOCAL_PLACEMENT_RUN=/home/aicompetition07/new-gat/results/local-placement-실제완료폴더
cat "$LOCAL_PLACEMENT_RUN/completion.json"
cat "$LOCAL_PLACEMENT_RUN/LOCAL_PLACEMENT_SUMMARY.md"
```

요약은 20조건의 train/validation/test, 같은 seed의 base 대비·위치 대비,
E/J 상호작용, 분기 norm, 모든 활성 층 범위의 frozen 제거를 포함한다.
층별 제거 표에는 validation/test의 ΔCE와 95% 구간이 함께 나온다.
개입 CE가 양수면 제거가 성능을 해쳤고, 음수면 제거가 개선한 것이다.
개별 seed와 accuracy 구간 등 전체 값은 CSV에도 남긴다.

## 기존 결과를 보존하며 재개

이전 placement 결과를 `--resume-from`에 지정하고 **새 output 폴더**로 실행한다.
Source / config / data hash와 checkpoint·optimizer 상태가 일치한 run만 재사용한다.
`prediction/` 결과는 구조가 다른 placement 재개의 입력으로 사용하지 않는다.

```bash
LOCAL_PLACEMENT_PREVIOUS=/home/aicompetition07/new-gat/results/local-placement-기존폴더
LOCAL_PLACEMENT_RUN="/home/aicompetition07/new-gat/results/local-placement-resume-$(date +%Y%m%d-%H%M%S)"
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$LOCAL_PLACEMENT_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_energy_relations.placement.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --resume-from "$LOCAL_PLACEMENT_PREVIOUS" \
  --output-dir "$LOCAL_PLACEMENT_RUN"
```

`--profile debug`는 별도 fixture의 연결 검사다. FULL의 모델·데이터·예산을 바꾸지 않는다.
어떤 실패에도 서버·SSH·부모 셸을 종료하지 않으며 `failure.json`과 부분 결과를 보존한다.
