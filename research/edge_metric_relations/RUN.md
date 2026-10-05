# 서버 실행

## 준비

FULL은 본학습을 서버에서 실행한다. Python 환경은 기존 `/home/aicompetition07/.conda/envs/new-gat/bin/python`을 사용한다. 할당된 GPU만 CUDA_VISIBLE_DEVICES에 적는다. 여러 GPU를 실제 할당받았다면 `0,1`처럼 지정하면 독립 job을 분배한다.

이전 결과 `/home/aicompetition07/new-gat/results/local-energy-20261004-002815`를 입력으로 읽는다. 그 디렉터리와 기존 checkpoint는 수정하지 않는다. 출력은 매번 새로운 디렉터리다.

할당받은 GPU가0인 경우:

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=0 \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u -B \
  -m research.edge_metric_relations.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --data-root /home/aicompetition07/new-gat/data/wedge-citation \
  --output-dir "results/edge-metric-$(date +%Y%m%d-%H%M%S)"
```

같은 터미널에 A→B→C의 진행 상황이 표시된다. 각 phase의 terminal.log에도 저장한다. 어느 phase가 실패하면 다음 phase는 시작하지 않고 traceback/failure.json을 남긴다. GPU/VRAM/CPU/RAM과 chunk/batch 후보의 실제 측정 결과를 저장한다. 전체 ETA는 측정 전 확정하지 않는다. A의 citation 수치 복원은 모든 target·특징을 사용하므로 기존 단순 고정 연산 감사보다 계산량이 많다.

## 결과 확인

```bash
EDGE_RUN=$(ls -dt /home/aicompetition07/new-gat/results/edge-metric-*/ | head -n 1)
cat "${EDGE_RUN}completion.json"
cat "${EDGE_RUN}A/completion.json"
cat "${EDGE_RUN}B/completion.json"
cat "${EDGE_RUN}C/completion.json"
cat "${EDGE_RUN}C/EDGE_METRIC_CLASSIFICATION_SUMMARY.md"
```

`completed:true`와 FULL coverage를 먼저 확인한다. C는675 final metric,3915 intervention,3060 branch 행이다. B의240 learned seed run/120000 update와 A의201graph coverage도 확인한다. calibration update는 이 예산에 포함하지 않는다. 성능·학습 여부 판단에는 요약과 `paired_comparisons.csv`, generator gradient/update history, branch diagnostics를 같이 사용한다.

## 이미 완료한 A/B를 재사용

같은 scientific source hash, profile, 전체 coverage가 일치할 때만 재사용한다. 좋은 점수일 것을 요구하지 않는다. 경로를 직접 지정한다.

```bash
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES=0 \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u -B \
  -m research.edge_metric_relations.study \
  --profile full --device cuda \
  --source-dir /home/aicompetition07/new-gat/results/local-energy-20261004-002815 \
  --data-root /home/aicompetition07/new-gat/data/wedge-citation \
  --audit-dir /absolute/path/to/completed/A \
  --synthetic-dir /absolute/path/to/completed/B \
  --output-dir "results/edge-metric-continue-$(date +%Y%m%d-%H%M%S)"
```

B/C의 중단된 학습은 각 `synthetic.study` / `classification.study`의 `--resume-from`으로 이전 결과를 읽고 새 출력으로 재개한다. 코드·config·데이터·seed packing·checkpoint를 대조한다. C 단독 FULL에도 `--source-dir`, `--mechanism-audit-dir`, `--synthetic-dir`가 필요하다.

## DEBUG

명시된 DEBUG fixture용이다. FULL config를 덮어쓰지 않으며 실제 데이터 분류 성적으로 제출하지 않는다.

```powershell
.venv-gpu\Scripts\python.exe -B -X utf8 -m pytest tests/test_edge_metric_core.py tests/test_edge_metric_classification.py tests/test_edge_metric_audit.py tests/test_edge_metric_synthetic.py tests/test_edge_metric_evaluation.py -q -p no:cacheprovider
```

전체 DEBUG pipeline에는 기존 DEBUG local-energy source가 필요하다. source adapter에서 fixture 이름과 profile을 대조한다. 새 출력을 지정하고 각 phase의 completion에서 DEBUG와 본실험을 구분한다.
