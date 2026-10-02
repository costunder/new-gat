# Experiment 3 — 고정한 경로 규칙의 새 특징 평가

Experiment 2에서 학습한 모델을 그대로 불러온다. **같은 그래프에 새로운 특징을 넣어도
목표 메시지를 만드는지**, 입력 크기를 바꿨을 때 어디서 오차가 증가하는지 확인한다.
기존 학습 결과의 C2 회수·고정 개입·teacher 연산 진단도 실제 CSV에서 함께 요약한다.

## 실험 순서와 현재 단계

| 단계 | 질문 | 상태 |
| --- | --- | --- |
| 0/1 | 경로 연산의 대수식이 맞고 L/L²와 다른 작용이 있는가? | 서버 고정 연산 실험 완료 |
| 2 | 공유 생성기로 경로별 C2와 목표 메시지를 학습하는가? | 서버 full 학습 완료. [기록](../SERVER_LEARNED_RESULTS.md) |
| **3, 이번 실행** | 고정한 모델이 새 특징과 입력 크기 변화에서도 작동하는가? | 구현·개발 검증. 서버 full 평가 실행 대상 |
| 4 | 실제 GNN 분류에서도 경로 집계가 도움이 되는가? | 후속 단계. Backbone·학습 계약을 별도로 확정 |

Experiment 2에 새 그래프·크기·구조 평가는 이미 있다. 이번에는 원래 531개 그래프의 연결을
그대로 재사용하고 특징을 바꾼다. `train + fresh`는 학습에서 본 연결에 대한 새 특징 평가다.
ID·크기 OOD·구조 OOD도 각각 유지한다.

## 실제 평가 범위

| 항목 | full 실행 |
| --- | --- |
| 원본 | 완료된 Experiment 2 폴더의 NPZ, scalar fit, selected checkpoint, CSV |
| 그래프 | 원본 **531개**, 모든 노드·엣지·unordered wedge·random pair 유지 |
| 모델 | 원본 3 target × 5 condition. Gate hidden 64와 초기화 seed 5개 유지 |
| 원본 재현 | 기존 특징에서 메시지 오차·RMSE·계수를 다시 계산해 기존 CSV와 비교 |
| 새 특징 | 그래프마다 독립 표준정규 scalar 입력 **16개**. 새 seed stream 사용 |
| 배율 | 같은 새 특징을 **0.25/0.5/1/2/4배**로 변경 |
| Split | train/validation/ID/size OOD/family OOD/family+size OOD 모두 사용 |
| 새 학습 | **0 epoch, optimizer update 0, scalar 재추정 0, checkpoint 재선택 0** |
| 그래프·특징·배율 조합 수 | 원본 8,496개 + 새 특징의 5개 배율 42,480개 = **50,976개** |
| 저장 행 | 메시지 평가 **124,254행**, 배율 진단 **103,545행** |

배율은 기준 1을 중심으로 두 번씩 절반·두 배를 적용한 범위다. 특징의 표준편차는
0.25~4, 분산은 0.0625~16으로 바뀐다. 데이터 결과를 보고 고른 배율이 아니다.
같은 X를 배율마다 재사용하므로 다섯 배율을 독립 그래프·독립 특징 draw로 세지 않는다.
Teacher와 목표 메시지는 각 배율의 실제 입력에서 다시 계산한다.
50,976개는 입력 조합 수이며, 모델·seed에 따른 반복 평가 횟수는 포함하지 않는다.

다섯 비교군은 `uLX`, `uLX+vL²X`, `βQX`, `βA.T C2(X)AX`,
`βAr.T C2(X)ArX`다. Scalar 계수와 gate 파라미터를 모두 원본에서 읽는다.
실제 수식과 판단 기준은 [MODEL_MATH.md](MODEL_MATH.md)에 있다.

## A6000 서버 실행

사용자가 완료한 원본 결과 경로를 아래 `--run-dir`에 지정했다. 새 결과 폴더에만 쓴다.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
WEDGE_PYTHON=/home/aicompetition07/.conda/envs/new-gat/bin/python &&
"$WEDGE_PYTHON" -m pip install -r research/wedge_propagation/learned/requirements.txt &&
read -r -p "할당받은 A6000 GPU 번호 또는 UUID: " WEDGE_GPU &&
test -n "$WEDGE_GPU" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$WEDGE_GPU" \
  "$WEDGE_PYTHON" -u -m research.wedge_propagation.generalization.study \
  --profile full --device cuda \
  --run-dir /home/aicompetition07/new-gat/results/wedge-learned-20261002-172132 \
  --output-dir "results/wedge-feature-$(date +%Y%m%d-%H%M%S)"
```

전체를 한 터미널에서 실행한다. `[preflight]`, CPU/GPU batch 계측, 원본 재현,
시나리오·target·condition별 진행과 오차, `[report]`, `[complete]`가 즉시 출력된다.
동일 내용은 `terminal.log`에 남는다. 추가 epoch 학습은 하지 않는다.

원본 완료 상태·설정·입력·수식 코드 SHA256·checkpoint 대응을 먼저 검사한다.
원본 CSV 재현이 허용 오차를 벗어나거나 입력·모델이 실행 중 바뀌면 실패 처리한다.
원본 재현은 저장된 split별 배치·순서·입력 shape를 복구해서 수행한다.
새 특징 평가는 처리량으로 선택한 batch를 사용한다.
재현 허용값은 atol=rtol=1e-5다. 메시지 상대 오차와 계수는 직접 비교하고,
absolute RMSE는 원본 정답 RMS+epsilon으로 나눈 뒤 비교한다.
정답의 크기를 기준으로 FP32 반올림 차이를 판단하며, raw RMSE 값도 그대로 저장한다.
모델 상태와 원본 파일의 hash는 실행 전후 정확히 같아야 한다.

CPU worker는 전체 새 특징 준비로 측정한다. GPU는 원본 batch, 두 배 batch, 전체 그래프
batch 후보에서 두 gate의 처리량·peak VRAM을 측정한다. 모든 scalar 입력과 모델 seed를
disjoint-union batch와 tensor 축에서 병렬 처리한다. 원본은 한 번 읽고 정적 연결을 cache한다.
실제 하드웨어·선택값·입력 shape·메모리·처리량은 `contract.json`에 기록한다.

## 결과 읽기

- `SYNTHETIC_GENERALIZATION_SUMMARY.md`: split별 원본↔새 특징 오차와 배율별 결과,
  원본 C2·고정 개입·teacher 진단을 실제 값으로 요약한다.
- `metrics.csv`: 모든 그래프·target·condition·seed·시나리오별 메시지 오차와 가중치 진단.
- `scale_checks.csv`: 같은 새 특징의 배율 간 C2 변화와 메시지 비례성 오차.
- `original_reproduction_layout.json`, `original_reproduction_metrics.csv`:
  복구한 원본 배치와 정답 RMS를 포함한 재평가. 재현 검사가 실패해도 측정 표를 남긴다.
- `source_metrics.csv`, `source_interventions.csv`, `source_teacher_operator_audit.csv`:
  원본 진단의 값을 새 출력 폴더에 복사한 표. 원본 파일은 그대로 보존한다.
- `fresh-a*/dataset.npz`, `fresh-a*/data_manifest.json`: 배율별 실제 전체 입력·teacher·목표·연결·seed·hash.
- `contract.json`, `source_provenance.json`, `completion.json`: 실제 처리 범위와 불변성 검사.
- PNG/PDF 4종: 새 특징 일반화, 배율별 메시지 오차, 비례성 오차, 원본 진단.

주지표는 목표 메시지 오차다. C2 변화량이나 상관만으로 성공을 판정하지 않는다.
악화·동률·정의되지 않는 값도 그대로 보고한다. 이 단계는 합성 규칙의 평가이며
실제 분류 효과는 Experiment 4에서 확인한다.

## 개발 검증

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_wedge_generalization_data.py tests/test_wedge_generalization_frozen.py \
  tests/test_wedge_generalization_report.py tests/test_wedge_generalization_study.py
python -u -m research.wedge_propagation.generalization.study \
  --profile debug --device cuda \
  --run-dir results/wedge-learned-debug-20261002-02 \
  --output-dir "results/wedge-feature-debug-$(date +%Y%m%d-%H%M%S)"
```

DEBUG는 원본 DEBUG 모델·36개 그래프·4개 scalar 입력만 허용한다.
Full과 DEBUG를 섞거나 원본 결과 안에 출력하는 실행은 거부한다.
검증 상태와 서버 미실행 범위는 [VERIFICATION.md](VERIFICATION.md)에 기록한다.
