# 서버 실행과 결과 확인

이번 실행은 **로컬 내부 이차 에너지와 로컬 사이 쌍선형 관계의 고정 진단**이다.
합성 198개 + 실제 Cora/CiteSeer/PubMed 전체 3개를 사용한다.
C/W나 분류기는 학습하지 않는다. 기존 실험 결과를 입력 checkpoint로 쓰지 않는다.

## 실행: 터미널 하나

할당받은 GPU 번호 또는 UUID를 입력한다. 다른 사용자의 프로세스나 서버 설정을 바꾸지 않는다.
결과 폴더는 실행마다 새로 만든다.

```bash
read -r -p "할당받은 GPU 번호 또는 UUID: " LOCAL_ENERGY_GPU &&
test -n "$LOCAL_ENERGY_GPU" &&
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
LOCAL_ENERGY_RUN="/home/aicompetition07/new-gat/results/local-energy-$(date +%Y%m%d-%H%M%S)" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$LOCAL_ENERGY_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_energy_relations.study \
  --profile full --device cuda \
  --data-root /home/aicompetition07/new-gat/data/paper \
  --output-dir "$LOCAL_ENERGY_RUN"
```

진행 상황은 같은 터미널과 `terminal.log`에 함께 나온다.
전처리 worker, 완전한 그래프 batch, 특징 channel chunk, 관계 pair chunk를 실제 측정한다.
GPU 후보의 실제 peak가 가용 메모리의 75% 안전 예산을 넘으면 선택하지 않는다.
후보 측정 중 CUDA OOM은 기록하고 다른 정확한 chunk 후보를 계속 측정한다.
모든 후보가 실패하거나 실제 본계산에서 오류가 나면 명시적으로 실패한다.
채널/관계 chunk는 계산만 나누며 원래 특징이나 관계를 생략하지 않는다.

## 완료 후 확인

같은 터미널에서:

```bash
cat "$LOCAL_ENERGY_RUN/completion.json"
cat "$LOCAL_ENERGY_RUN/LOCAL_ENERGY_RELATIONS_SUMMARY.md"
```

`status=complete`, `profile=full`, `graphs=201`, `actual_citation_graphs=3`을 확인한다.
`classifier_training_run=false`가 정상이다. 이번 단계에는 epoch·분류 정확도가 없다.
에너지/관계/전달의 전체 요약은 `*_summary.csv`, 원래 중심·쌍·입력별 값은 raw CSV에 있다.
`figures/`에는 PNG와 PDF가 있으며 `resources.json`에는 실제 후보별 측정이 있다.
입력과 원래 노드 ID는 `inputs/`, 로컬 집합과 대응은 `topology/`에 저장한다.
완료 전 source·입력·대응 checksum을 다시 검사한다.

코드 실행에 필요한 NumPy/SciPy/Matplotlib/psutil/threadpoolctl은 기존 실험 환경과 같다.
CUDA PyTorch는 현재 서버 환경의 설치를 유지한다.
누락된 Python 패키지만 `requirements.txt`로 설치할 수 있으며 시스템 CUDA/driver 설치는 필요 없다.

## 로컬 DEBUG 검증

`config_debug.json`은 별도 21개 fixture다. FULL 결과로 해석하지 않는다.

```powershell
.venv-gpu\Scripts\python.exe -B -X utf8 -m research.local_energy_relations.study --profile debug --device cuda --data-root data/paper --output-dir results/local-energy-DEBUG-new --no-download
```

DEBUG라도 기존 결과 폴더가 있으면 덮어쓰지 않고 거부한다.
