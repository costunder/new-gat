# 정규화 실험 계약과 판단 기준

## 1. 질문과 변경 범위

**그래프 전체 degree bound 때문에 약해진 로컬 전파를 각 ego의 bound로 정하면,
그리고 교차 step을 엣지별 대칭 degree 가중치로 정하면, cross의 분류 기여가 달라지는가?**

전체 물리 그래프와 로컬·cross 연결은 그대로 두고 정규화만 비교한다.
모델은 2개 macro 층과 동일 projection·dropout을 유지한다. 기존 scalar E/J 실험과 이번 결과를 합치지 않는다.

## 2. 데이터

| 데이터 | 전체 노드 × 특징 | 클래스 | Train/validation/test |
| --- | --- | ---: | --- |
| Cora | 2708 × 1433 | 7 | 140/500/1000 |
| CiteSeer | 3327 × 3703 | 6 | 120/500/1000 |
| PubMed | 19717 × 500 | 3 | 60/500/1000 |

원본 완료 audit `local-energy-20261004-002815`의 그래프 201개와 입력 hash를 검증하고 보존한다.
분류는 그 source의 전체 citation 그래프 3개에서 원래 특징·물리 엣지·1홉 topology를 사용한다.
Label/public mask는 원본 hash를 고정한 raw 데이터에서 읽는다. 합성 그래프 198개는 분류 학습 대상이 아니다.
Node·edge·feature·유도 로컬 그래프·canonical cross 연결을 버리는 sampling이나 cap은 없다.

## 3. 조건과 규모

조건 ID는 `weight__intra__cross__variant`다.
Weight는 unit/local_degree, intra는 graph/local이다.
각 조합에서 `(none,off)`, `(graph,fixed)`, `(graph,learned)`, `(edge,fixed)`, `(edge,learned)`를 학습한다.
Off를 두 cross 정책에 중복 배치하지 않아 고유 조건은 20개다.
기존 6조건을 포함해 모두 새로 튜닝·학습하며 과거 성능 값은 새 결과에 합치지 않는다.

| 단계 | 학습 횟수 | 계산 |
| --- | ---: | --- |
| Tuning | 540 | 데이터셋 3개 × 조건 20개 × LR 3개 × tuning seed 3개 |
| Final | 300 | 데이터셋 3개 × 조건 20개 × final seed 5개 |
| 합계 | 840 | 학습마다 500 epoch: 420,000회 독립 모델 갱신 |

2개 macro 층, hidden dimension 64, dropout .5, Xavier projection 초기화, learned θ 초기값 0이다.
LR 후보는 .001/.003/.01, tuning seed는 101/202/303, final seed는 11/23/37/53/71이다.
Adam의 LR는 학습 동안 고정한다. Projection weight decay는 5e−4, θ weight decay는 0이다.
Float32와 TF32 off를 사용하며 early stopping·scheduler·gradient clipping은 없다.

Checkpoint는 validation CE 최저, 동률이면 validation accuracy 최고, 그다음 이른 epoch로 선택한다.
LR는 tuning seed의 선택된 validation CE 평균으로 선택하며 동률이면 작은 LR를 선택한다.
모든 final checkpoint 선택을 고정한 후 test 평가를 연다.

## 4. 사전 지정 비교

조건 간 차이는 같은 final seed에서 먼저 계산한다. Accuracy 차이는 percentage point, CE 차이는 손실 단위다.

| 비교 | 개수 | 고정하는 요소 |
| --- | ---: | --- |
| Fixed−off, learned−off, learned−fixed | 24 | 같은 C·intra, 각 graph/edge cross |
| Edge−graph cross 정규화 | 8 | 같은 C·intra·gain variant |
| Local−graph intra 정규화 | 10 | 같은 C·cross·gain variant |
| Local_degree−unit | 10 | 같은 intra·cross·gain variant; 탐색적 |
| 합계 | 52 | 모든 비교를 보고 |

추가 학습 없이 상호작용 12개를 계산한다.

- 8개: `(local cross−local off)−(graph cross−graph off)`. 내부 정책이 cross의 추가 기여를 바꾸는가?
- 4개: `(local edge−local graph)−(graph edge−graph graph)`. 내부 정책에 따라 cross 정규화의 효과가 달라지는가?

각 seed에서 네 조건의 contrast를 계산한 후 평균·표본 std·paired 95% t 구간을 구한다.
집계한 두 구간을 빼서 새로운 구간을 만들지 않는다.
각 조건은 독립적인 validation LR 선택을 포함하므로 이 비교는 그 학습 절차까지 포함한 결과다.

## 5. Frozen 개입과 실제 작용

활성 cross 16조건에서 gain0/gain1 × layer_0/layer_1/both를 평가한다.
Fixed gain1은 예상 no-op 대조이고, off 4조건은 원래 평가만 한다.
모델·buffer hash를 보존하고 optimizer 갱신은 0회다. 개입 이후 층은 바뀐 hidden으로 다시 계산한다.

Gain·θ·매 epoch의 gradient/update와 현재 투영 특징 Z의 matched Δ/off를 함께 본다.
Context ||GY₁||와 raw context ||KY₁||를 구분한다. 에너지는 적용된 S/G의 ½trace다.
Off의 graph G 에너지는 참고 진단이며 활성 cross를 뜻하지 않는다.
Degree bound와 정확한 `−ρMSGSR` residual을 기록하고, star에서 C와 local step이 상쇄될 수 있음을 검증한다.

## 6. 결과 범위와 자원

FULL 원시 결과는 분류 지표 900행, frozen 개입 4,320행, branch 진단 3,480행이다.
요약은 metric 180행, 직접 paired 비교 936행, 상호작용 216행, frozen 차이 1,728행, branch 요약 696행이다.
행 수는 독립 그래프 수나 학습 반복 수가 아니다.

모델당 전체 그래프 1개를 사용하며 accumulation과 replica 수는 각각 1이다.
독립 seed 축은 모델의 effective batch에 곱하지 않는다.
Tuning seed pack 1/2/3개, final seed pack 1/2/4/5개와 exact edge chunk 256/4096/16384/65536을 실측한다.
CPU 준비 worker 1/2/4/8개, CPU/RAM/GPU/VRAM/할당 제한과 처리시간을 기록한다.
할당된 GPU가 여러 개면 데이터셋·조건·LR 작업을 분배한다.
정적 cache, 모든 연결을 처리하는 sparse action, 고정 대칭 backward, activation checkpointing을 사용한다.
Calibration의 임시 갱신은 본학습의 시작 상태나 420,000회 예산에 섞지 않는다. OOM 후보와 자원 실패는 기록한다.

## 7. 판정 범위

주요 지표는 test accuracy, 보조 지표는 CE(L2 항 제외)다. 둘의 방향이 다르면 각각 기록한다.
Gain이 학습되는지, cross가 실제 출력을 바꾸는지, 분류에 도움이 되는지를 구분한다.
Copy 공간 및 물리 D metric에서의 contraction을 일반 Euclidean node norm이나 전체 분류기의 안정성으로 확대하지 않는다.
고정 전파는 하나의 전역 node 연산으로도 쓸 수 있다. Copy 공간을 쓴다는 이유만으로 표현력·정보 복원·신규성을 입증하지 않는다.

이전 실험에서 동일 public test를 본 뒤 설계한 후속 탐색 연구다.
Paired 구간은 고정 split에서 초기화 seed 5개의 변동이며 독립 split·새 그래프 일반화가 아니다.
다중 비교 보정은 적용하지 않는다. 0을 포함하는 구간은 동등성 증명이 아니다.

## 8. DEBUG 분리

DEBUG-Cora/CiteSeer/PubMed fixture의 노드 수는 24/30/36, 특징 수 12, 클래스 수 3이다.
Hidden dimension 8, 학습 3 epoch, LR 2개, tuning seed 2개, final seed 2개로
20조건을 모두 검사한다. Tuning 240회 + final 120회 = 360회 학습과 1,080회 갱신이다.
원시 metric/frozen/branch는 360/1,728/1,392행이다.
원본 DEBUG source의 그래프 21개를 사용하고 actual_data=false로 표시한다.
FULL 설정에 덮어쓰지 않으며 citation 본학습 성능으로 보고하지 않는다.

실제 실행 검증과 서버 FULL 실행 여부는 [VERIFICATION.md](VERIFICATION.md)에 기록한다.

