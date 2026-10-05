# 분류 구현 검증

## 2026-10-05 전체 DEBUG 실행

검토한 실행은 `results/edge-metric-pipeline-DEBUG-20261005-02/C`다.
완료 기록과 원시 CSV, 모든 selected checkpoint를 직접 대조했다.
과학 코드를 바꾸거나 추가 학습을 실행하지 않았다.

이 실행은 **CPU의 명시된 DEBUG fixture**다. Backbone은 2층·hidden 8,
게이트 MLP hidden은 64, run마다 3 epoch, tuning/final seed는 각각 2개다.
FULL의 2층·hidden 64·500 epoch·630 run 계약과 별도다.
DEBUG accuracy나 통계 유의성을 실제 citation 성능으로 해석하지 않는다.

| 확인 항목 | 실제 기록 |
| --- | --- |
| Tuning 독립 run | 180 |
| Final 독립 run | 90 |
| 전체 독립 run / epoch history / optimizer 갱신 | 270 / 810 / 810 |
| Final metric 행 | 270 |
| Frozen intervention metric 행 | 1,566 |
| 원래 및 frozen branch 행 | 1,224 |
| Frozen provenance pack | 45 |
| 사전 지정한 test accuracy 비교 / Holm family | 18 / 18 |
| 실제 citation 데이터 사용 / FULL 서버 학습 | 없음 / 미실행 |

근거: [completion.json](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/completion.json),
[coverage.json](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/coverage.json),
[report_checks.json](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/report_checks.json).

## 실제 학습 연결

모든 `jobs/**/selected.pt`의 SHA256가 대응하는 JSON sidecar와 일치했다.
Checkpoint의 seed별 Adam step은 모두 3이며, history는 seed별 3개 epoch를 빠짐없이 포함했다.

학습 가능한 게이트의 모든 parameter·seed slice **1,728개**에서 Adam의
`exp_avg_sq`에 0이 아닌 원소가 있었다. 두 층의 diagonal/pair MLP 각각에 해당한다.
게이트의 weight decay는 0이므로 이 값은 CE gradient가 optimizer에 전달됐다는 증거다.
이 기록은 각 parameter의 모든 원소에 gradient가 있었다는 주장은 하지 않는다.

Epoch history의 `D1/diagonal`, `F1/pair`, `F2/diagonal`, `F2/pair`,
`DA/diagonal`, `DA/pair` 그룹은 각각 **108개 기록 모두** CE gradient norm과
parameter update norm이 양수였다. 각 seed의 선택 epoch는 1~3 안에 있었다.
현재 optimizer의 누적 기록과 validation으로 선택한 과거 상태를 구분했다.

별도의 시작 전 [preflight 검사](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/preflight_debug_checks.json)는
독립 fixture에서 96개 수학 검사를 통과했다. Dense 대조 최대 절대오차는
`8.881784197001252e-16`이다. Hidden 64의 15개 classifier fixture에서도
3번의 갱신 뒤 모든 active parameter 변화와 정확한 no-op을 확인했다.
이 preflight 갱신은 위 810회 학습 예산에 포함하지 않는다.

## 평가 잠금과 frozen 개입

[평가 잠금](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/test_evaluation_unlocked.json)의
selection digest는 final worker의 90개 validation 선택 기록과 일치했다.
Learning-rate selection digest도 저장된 선택 목록과 일치했다.
전체 final 선택이 완료된 뒤에 test와 frozen 평가를 시작했다.

45개 [frozen provenance](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/frozen_evaluation_provenance.json)
pack 모두 parameter/buffer의 before/after SHA256가 동일했고 optimizer update는 0이었다.
각 pack의 layer_0/layer_1/both no-op 3개를 모두 검사했다.
No-op metric **810개 행**의 logit 변화 norm과 prediction flip 비율은 모두 0이었다.

개입 후에는 현재 hidden을 따라 게이트와 다음 층을 다시 계산한다.
같은 projected Z에서 off-diagonal만 제거하는 branch 대조는 실제 diagonal 계수를 유지한다.
Frozen 변화는 다시 학습한 조건 간 성능 차이와 별도로 기록한다.

## 에너지와 미정의 값

모든 native branch에서 `energy_total = energy_intra + energy_cross`를
float32 허용오차 안에서 확인했다. D0/D1/DA의 off-diagonal action과 cross energy는 0이었다.
DA는 pair를 diagonal 생성에 사용하는 대조이며 off-diagonal action을 가진 모델로 해석하지 않는다.
Signed cross energy의 음수도 유효한 값으로 유지했다.

Baseline branch 144개 행은 edge-metric energy와 matched-off 진단을
빈 값으로 남겼다. 없는 값을 0으로 대체하지 않았다.
이 DEBUG 입력의 native off norm은 모두 양수라 native relative ratio는 정의됐다.
0인 denominator의 nullable 처리는 별도 단위 검사에서 확인했다.

F2−D1, F2−DA, F2−P2의 두 recipe·세 fixture 대비 18개에만 Holm 보정을 적용했다.
CE·그 밖의 대비·frozen 통계는 탐색적 보조 결과로 표시했다.

## 보존과 그림 확인

모든 cached graph 파일의 SHA256가 `graph_cache.json`과 일치했다.
Source manifest 내부 digest도 일치했다. 실행 후 QA 시점의 저장소와 비교하면
`package_review.py`만 달랐으며, 분류 과학 코드와 graph cache는 변경되지 않았다.
실행 당시 manifest를 보존하며 후속 코드의 source identity로 바꿔 적지 않는다.

PNG 3종을 직접 열어 표기·축·조건·DEBUG 구분을 확인했다.
동일 그림의 PDF도 생성됐으며 기존 보고서 덮어쓰기는 거부한다.

- [Primary contrasts](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/figures/primary_contrasts.png)
- [Classification](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/figures/classification.png)
- [Branch use](../../../results/edge-metric-pipeline-DEBUG-20261005-02/C/figures/branch_use.png)

## 남은 검증

전체 DEBUG pipeline의 C 연결은 완료됐다. 실제 citation 데이터의 FULL 학습·평가,
A6000의 처리량과 peak VRAM, 분류 개선 및 독립 그래프 일반화는 아직 이 기록으로 검증하지 않았다.
CPU DEBUG만으로 여러 GPU의 실제 동시 실행이나 서버 자원 성능을 입증하지 않는다.

## 최신 기록: RUN03 재검증

`results/edge-metric-pipeline-DEBUG-20261005-03/C`의 완료 후 기록을 다시 대조했다.
C의 실제 실행 시간은 **270.89297069999157초**이며 전체 A/B/C DEBUG pipeline은
**597.4447초**에 완료됐다. CPU·2층·backbone hidden 8·게이트 hidden 64·3 epoch의
별도 DEBUG 계약을 유지했고 실제 citation FULL은 실행하지 않았다.

실행 당시 source digest:

`86a98b7601d6dab9f1be575c25307cae9fe529ebd1b34a8e09b26cf4648f1c60`

- Tuning 180회·final 90회, 체크포인트 history 810행, 새 optimizer 갱신 810회다.
- Final metric 270행·frozen intervention 1,566행·branch 1,224행을 다시 확인했다.
- 모든 selected checkpoint의 SHA256와 payload/sidecar metadata가 일치했다.
  모든 seed의 Adam step은 3이고 선택 epoch는 1~3이다.
- 두 층의 모든 adaptive parameter seed slice 1,728개에서 CE-only Adam의 비영
  second moment를 다시 확인했다. 연결되지 않은 active parameter slice는 없었다.
- Final validation 선택·LR 선택 digest가 test unlock에 기록된 digest와 일치했다.
- Frozen provenance 45개 pack 모두 before/after hash가 동일하고 optimizer 갱신은 0이다.
  No-op metric 810행의 logit 변화와 prediction flip 비율은 모두 0이다.
- Native energy 합과 D0/D1/DA의 0인 cross action, baseline의 빈 energy/relative 값이 맞았다.
  Cached graph의 SHA256도 모두 유지됐다.

근거: [최신 completion](../../../results/edge-metric-pipeline-DEBUG-20261005-03/C/completion.json),
[최신 coverage](../../../results/edge-metric-pipeline-DEBUG-20261005-03/C/coverage.json),
[최신 frozen provenance](../../../results/edge-metric-pipeline-DEBUG-20261005-03/C/frozen_evaluation_provenance.json).

RUN03 완료 후 변경된 소스는 `package_review.py`뿐이다. 학습/평가 과학 코드는 유지됐다.
후속 패키징에서는 C epoch history와 보조 import 파일을 포함하고 evidence 경로 대응을 안내한다.
실행 당시 패키징 파일은
[보존본](../../../results/edge-metric-packaging-source-DEBUG-20261005-01/package_review_before.py)에 있다.
실행 source와 후속 ZIP 생성기의 identity 차이를 숨기거나 같은 실행으로 다시 표시하지 않는다.
