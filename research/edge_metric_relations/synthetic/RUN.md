# 서버에서 실험 B 실행

이 명령은 531개 그래프, 각 16개의 독립 scalar 입력, 240개 모델의 500 epoch를 실행한다. 입력은 새 named stream에서 생성하며 A의 201개 graph source를 대체 사용하지 않는다.

FULL은 Linux 서버의 CUDA에서만 허용하며, 명시적인 `CUDA_VISIBLE_DEVICES` 할당이 필요하다. 로컬 Windows GPU에서 FULL을 실행하거나 시스템의 모든 GPU를 자동 발견해 학습하는 경로는 거부한다.

`STUDY_GPU`에는 실제 할당받은 GPU 번호 또는 MIG UUID를 넣는다. 여러 GPU를 할당받았다면 쉼표로 연결한 번호/UUID 목록을 입력한다. 기존 remote session과 다른 사용자의 작업을 종료하지 않는다.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
read -r -p "할당받은 GPU 번호 또는 MIG UUID 목록: " STUDY_GPU &&
test -n "$STUDY_GPU" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$STUDY_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.edge_metric_relations.synthetic.study \
  --profile full --device cuda \
  --output-dir "results/edge-metric-synthetic-$(date +%Y%m%d-%H%M%S)"
```

진행은 현재 터미널과 `terminal.log`에 동시에 표시한다. `GPU calibration`은 임시 복제 모델로 전체 train 자료를 측정한다. 그 갱신은 본학습 예산에 포함하지 않으며 저장한 학습 모델을 변경하지 않는다. CPU worker, 물리 graph batch, exact pair chunk는 측정 후 선택한다.

여러 GPU가 보이면 독립 job을 subprocess worker에 나누어 동시에 실행한다. 각 worker의 진행도 같은 터미널에 `[GPU 번호]` 접두어로 표시한다. input cache는 tensor와 기본형만 저장하고 `weights_only=True`로 읽는다. worker별 소스·데이터 해시와 전체 job coverage를 합치기 전에 확인한다.

## 결과 확인

```bash
B_RUN=$(ls -dt /home/aicompetition07/new-gat/results/edge-metric-synthetic-*/ | head -n 1) &&
cat "$B_RUN/completion.json" &&
cat "$B_RUN/SYNTHETIC_SUMMARY.md"
```

완료 표시는 `graphs=531`, `learned_runs=240`, `seed_optimizer_updates=120000`, `math_checks_passed=true`여야 한다. 전체 `per_graph.csv`는 152,928행, `per_realization.csv`는 2,446,848행이며 header는 별도다. 이 숫자는 고정 대조와 train-only 수식 기준선도 포함한다.

## 안전한 재개

앞 실행의 출력은 보존하고 새 출력 폴더에 이어서 계산한다. 동일한 소스와 데이터 해시가 필요하다. 완료된 job은 다시 학습하지 않고 선택 상태로 평가한다. 미완료 job은 마지막으로 저장한 50 epoch 간격 checkpoint부터 재개한다. 기존 학습과 같은 batch/chunk 순서를 자동으로 사용한다. 여러 GPU로 시작했다면 같은 worker 개수로 재개한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$STUDY_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.edge_metric_relations.synthetic.study \
  --profile full --device cuda \
  --resume-from "$B_RUN" \
  --output-dir "results/edge-metric-synthetic-resume-$(date +%Y%m%d-%H%M%S)"
```

`--batch-size`와 `--pair-chunk`는 물리 처리 순서를 명시할 때 사용할 수 있다. 그래프·realization·seed·epoch·hidden 값은 FULL 계약에서 바뀌지 않는다. OOM 시 임의 축소나 CPU fallback은 하지 않는다.

## 별도 DEBUG 실행

```bash
python -u -m research.edge_metric_relations.synthetic.study \
  --profile debug --device cpu --workers 2 \
  --output-dir "results/edge-metric-synthetic-DEBUG-$(date +%Y%m%d-%H%M%S)"
```

DEBUG는 30개 graph, 각 4개 scalar 입력, seed 2개, 3 epoch다. 여섯 조건과 두 학생·교사 recipe, 세 target를 모두 검사하며 288회 seed optimizer 갱신을 한다. hidden 64는 유지한다. DEBUG 성능을 FULL로 제출하지 않는다.
