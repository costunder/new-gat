# 서버 실행과 결과 확인

아래 명령은 이전 완료 결과의 201개 input snapshots를 읽는다. 새 결과 폴더에 기록하고 이전 파일은 보존한다. `--source-dir`를 임의의 작은 데이터 폴더로 바꾸지 않는다.

## 전체 실험

할당받은 GPU 번호 또는 UUID를 입력한다. 진행 출력은 같은 터미널에 표시된다.

```bash
read -r -p "할당받은 GPU 번호 또는 UUID: " RECEIVER_GPU &&
test -n "$RECEIVER_GPU" &&
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
RECEIVER_SOURCE=/home/aicompetition07/new-gat/results/local-energy-20261004-002815 &&
RECEIVER_RUN="/home/aicompetition07/new-gat/results/receiver-aggregation-$(date +%Y%m%d-%H%M%S)" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$RECEIVER_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.local_energy_relations.receiver_aggregation.study \
  --profile full --device cuda \
  --source-dir "$RECEIVER_SOURCE" \
  --output-dir "$RECEIVER_RUN"
```

전부 처리한 뒤 `[complete]`와 범위 설명을 출력한다. 완료 파일이 없거나 status가 실패이면 정상 완료로 해석하지 않는다. failure 파일과 터미널 로그에 실패 단계가 남는다. 세션이나 다른 프로세스를 종료하는 복구 명령은 사용하지 않는다.

## 결과 확인

같은 터미널에서 아래 두 출력을 전달하면 된다.

```bash
cat "$RECEIVER_RUN/completion.json"
cat "$RECEIVER_RUN/RECEIVER_AGGREGATION_SUMMARY.md"
```

터미널을 다시 열었다면 완료 로그에 표시된 정확한 경로를 `RECEIVER_RUN`에 먼저 지정한다. 가장 최근 폴더를 찾다가 실패·DEBUG 결과와 섞이지 않게 한다.

주요 해석은 합에서 직접 숨는 receipt 성분과, 실제 q/receipt 복원오차를 함께 보는 것이다. 큰 상쇄 비율만으로 영구 정보 손실이라고 결론내리지 않는다. E/J 재현 검사는 Y만을 사용한 inverse의 결과이며 E/J로 역문제를 개선한 실험은 아니다.

## DEBUG 검사

DEBUG는 검증된 이전 DEBUG 입력 21개로 별도 실행한다. 전체 실험의 기본 설정과 출력 폴더에 덮어쓰지 않는다. DEBUG 입력 폴더를 지정하여 다음 모듈에 `--profile debug --device cpu` 또는 `cuda`를 전달한다. DEBUG를 full 실험이나 실제 citation 평가로 보고하지 않는다.
