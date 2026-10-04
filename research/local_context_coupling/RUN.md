# 서버 실행과 결과 확인

아래 명령은 새 결과 폴더에서 전체 fixed audit를 실행합니다.
기존 `results/local-energy-20261004-002815`를 source로 읽고 보존합니다.
할당받은 GPU 번호 또는 UUID를 입력합니다. GPU 번호를 임의로 정하지 않습니다.

```bash
read -r -p "할당받은 GPU 번호 또는 UUID: " LOCAL_CONTEXT_GPU &&
test -n "$LOCAL_CONTEXT_GPU" &&
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
LOCAL_CONTEXT_RUN="/home/aicompetition07/new-gat/results/local-context-$(date +%Y%m%d-%H%M%S)" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$LOCAL_CONTEXT_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_context_coupling.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --output-dir "$LOCAL_CONTEXT_RUN"
```

같은 터미널에 source 검사, hardware, CPU 준비 후보 계측, GPU batch 후보 계측,
현재 graph/channel 처리와 경과시간, 보고서 작성, 완료 범위가 출력됩니다.
전체201 graph·8,804 feature 열·두 C조건·세 reference 상태를 사용합니다.
임의 subset이나 모델 크기 fallback을 하지 않습니다.

## 실행 직후 확인

같은 셸에서 다음만 출력해 전달하면 됩니다. 긴 epoch 로그를 전부 복사할 필요는 없습니다.

```bash
cat "$LOCAL_CONTEXT_RUN/completion.json"
cat "$LOCAL_CONTEXT_RUN/LOCAL_CONTEXT_SUMMARY.md"
```

FULL은 classifier 본학습이 아니라 fixed audit입니다. 완료 출력에서도 이 범위를 확인합니다.
수치 재집계가 필요하면 다음 CSV와 provenance 파일을 함께 읽습니다.

| 파일 | 내용 |
| --- | --- |
| `channel_metrics.csv` | 전체52,824 channel/state/C 행의 연산·소거·최종 차이 진단 |
| `graph_summary.csv` | 전체1,206 graph/state/C 행, squared norm 전체 합산 후 비율 |
| `coverage.json` | 전체 graph/channel/상태/조건 포함 여부 |
| `config.json` | 실제 full/debug 계약 |
| `source_manifest.json`, `source_input_manifest.json` | 기존 source와 입력 hash |
| `source_adapter.json` | 기존 입력을 새 모델 계약으로 연결한 검증 |
| `hardware.json`, `resources.json` | 실제 할당 자원과 후보 측정/선택 |
| `completion.json`, `LOCAL_CONTEXT_SUMMARY.md` | 완료 상태와 요약 |

`--workers auto`가 기본이며 실제 CPU 준비 후보를 측정합니다.
별도 DEBUG profile은 구현 연결 확인용 작은 fixture입니다. DEBUG 통과를201 graph FULL 결과나 분류 성능으로 보고하지 않습니다.

## 실패한 경우

실패한 단계와 오류 출력, 새 결과 폴더에 남은 provenance/로그를 전달합니다.
실패한 실행을 완료로 기록하거나 기존 결과를 덮어쓰지 않습니다.
학습/평가 source를 바꾸거나 graph/channel 수를 줄여 조용히 재실행하지 않습니다.
