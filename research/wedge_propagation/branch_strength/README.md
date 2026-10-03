# Experiment 4.1 — C 위치와 분기 메시지 크기 분리

Experiment 4에서 이미 선택한 분류 checkpoint를 고정해 추가로 관측한다.
**새 학습·학습률 선택·checkpoint 선택·최적 개입 선택은 없다.**
기존 Experiment 4 소스와 결과를 보존하고 새 폴더에 진단 결과를 저장한다.

## 확인할 두 질문

| 비교 | 바꾸는 것 | 유지하는 것 |
| --- | --- | --- |
| 같은 분기 크기 | C=1 또는 C 위치 shuffle | 원래 이차 분기의 Frobenius norm |
| 같은 방향, 다른 크기 | True C 메시지의 κ 분모를 1로 변경 | 이차 분기의 방향, 학습한 α·β |

원래 각 층은 `U = Z − αL̄Z − βM_C`이고,
`M_C = S_Q Aᵀ C A S_Q Z / (3κ(C))`이다.
True C/C=1과 원래 κ/분모 1을 교차해 두 변경을 분리한다.
`identity_norm_matched`와 `shuffle_norm_matched`는 현재 층 원래 메시지에 맞춰
전체 노드·channel의 norm을 seed별로 조절한다. β도 그대로이므로 βM의 크기도 같다.

이 크기 통제는 다른 분기와의 상쇄나 전체 update의 norm을 같게 만들지 않는다.
첫 층 개입 뒤 두 번째 층의 입력은 달라진다. End-to-end에서는 그 변경된 현재 Z에서
true C와 참조 κ를 새로 계산한다. 별도 fixed-Z 진단은 원래 깨끗한 Z를 고정한다.
참조 norm이 0인 비율·cosine은 undefined로 기록한다.
양수 참조 norm을 0 메시지로 맞출 수 없는 경우 오류로 중단한다.

## 전체 범위

| 항목 | Full 계약 |
| --- | --- |
| Source | 완료된 Experiment 4 full 결과 폴더 |
| 데이터 | Cora/CiteSeer/PubMed 전체 그래프, public fixed split |
| 원래 모델 | 8조건 × 3데이터 × 5seed = 120 checkpoint |
| 개입 모델 | Learned raw/RMS × 3데이터 × 5seed = 30 checkpoint |
| 유지 | 원래 두 층·hidden 64·입력·모든 물리 엣지·모든 wedge·α·β |
| 처리 | 원래 모델, true C/분모 1, C=1/원래 κ, C=1/분모 1, C=1/norm match, 분기 제거, 두 shuffle 처리 |
| Shuffle | Source에 저장된 dataset별 10개 manifest 재사용 |
| 위치 | layer_0, layer_1, both 모두 평가 |
| 새 학습 | 0 epoch, 0 optimizer update |
| 행 수 | 원래 metric 360, 개입 metric 6,750, 층 진단 4,740, fixed-Z 1,560 |

개입 변경은 단일 처리 5개 + shuffle 2종 × manifest 10개 = 25개다.
세 층 위치에서 75개 변경을 적용하며 원래 결과도 보존한다.
Manifest는 seed 안에서 평균하고 seed의 독립 반복 수를 늘리지 않는다.
분모를 없애거나 norm을 맞춘 진단에는 원래 `spectral norm ≤ 1` 보장이 그대로 적용되지 않는다.

**Test를 이미 관측한 후속 탐색이다.** 개입의 test 최고값을 새 모델 성능으로 선택하지 않는다.
이번 관측은 다음 학습 설계를 정하는 근거이며 새 split의 확인 실험을 대신하지 않는다.

## 서버 실행

Source 폴더에는 Experiment 4의 실제 full 완료 결과를 지정한다.
검증된 source dataset cache와 원래 checkpoint·manifest를 읽으며 추가 다운로드나 학습을 하지 않는다.
GPU에는 실제 할당받은 번호 또는 UUID를 넣는다.

```bash
cd /home/aicompetition07/new-gat &&
git pull --ff-only &&
BRANCH_SOURCE=$(ls -dt /home/aicompetition07/new-gat/results/wedge-classification-[0-9]*/ | head -n 1) &&
test -n "$BRANCH_SOURCE" &&
read -r -p "할당받은 GPU 번호 또는 UUID: " BRANCH_GPU &&
test -n "$BRANCH_GPU" &&
env -u PYTORCH_NVML_BASED_CUDA_CHECK CUDA_VISIBLE_DEVICES="$BRANCH_GPU" \
  /home/aicompetition07/.conda/envs/new-gat/bin/python -u \
  -m research.wedge_propagation.branch_strength.study \
  --source-dir "$BRANCH_SOURCE" --profile full --device cuda \
  --output-dir "results/wedge-branch-strength-$(date +%Y%m%d-%H%M%S)"
```

실행기는 source 완료·config/data/code/checkpoint/manifest hash와 원래 metric 재현을 확인한다.
결과는 새 폴더만 허용한다. GPU/RAM/CPU, packed seed 처리량·peak VRAM을 실제로 기록하며
모델·그래프·개입 범위를 줄여서 실행하지 않는다. 진행 상태와 실패 원인은 같은 터미널과 로그에 남긴다.
DEBUG는 source DEBUG fixture와 축소 예산을 명확히 표시하며 실제 citation 성능으로 해석하지 않는다.

## 결과 읽기

- `BRANCH_STRENGTH_SUMMARY.md`: 원래 정확도, 실제 paired 변화, 같은 크기의 가중치 배치 진단.
- `original_classification.png/.pdf`: 원래 8조건의 test accuracy·CE.
- `weight_and_denominator.png/.pdf`: C와 분모를 나눈 두 층 개입의 paired 변화.
- `norm_matched_placement.png/.pdf`: 원래 κ 유지와 실제 norm match의 차이, 세 개입 위치.
- `branch_message_geometry.png/.pdf`: fixed-Z에서 βM/Z, 메시지 norm 비율과 원래 메시지와의 cosine.
- `baseline_metrics.csv`, `treatments.csv`, `layer_diagnostics.csv`, `fixed_z.csv`: 모든 원래/변경 관측값.
- `resources.csv`, `contract.json`, `completion.json`: 실제 자원·범위·보존·완료 증거.

Accuracy 차이는 **개입 − 원래**의 percentage point이고 CE 차이는 음수가 개선이다.
같은 seed의 paired 95% t 구간은 고정 public split의 초기화 변동만 나타낸다.
새 그래프/split 불확실성이나 다중 비교 보정을 포함하지 않는다.
같은 크기에서 C 교체가 무영향이면 가중치 위치의 유익성 근거가 약하다.
방향을 유지한 강도 변경이 개선되어도 κ가 학습 실패의 원인이라고 확정할 수 없다.
분류 정확도·CE와 실제 αL/βM 크기·방향을 함께 읽어야 한다.

완료 후 아래 명령으로 summary와 완료 범위만 확인한다. 매 case 로그 전체를 복사할 필요는 없다.

```bash
BRANCH_RUN=$(ls -dt /home/aicompetition07/new-gat/results/wedge-branch-strength-[0-9]*/ | head -n 1) &&
test -n "$BRANCH_RUN" &&
cat "${BRANCH_RUN}completion.json" &&
cat "${BRANCH_RUN}BRANCH_STRENGTH_SUMMARY.md"
```

수식과 개입 정의는 [MODEL_MATH.md](MODEL_MATH.md),
로컬 검사와 서버 미실행 범위는 [VERIFICATION.md](VERIFICATION.md)에 있다.
